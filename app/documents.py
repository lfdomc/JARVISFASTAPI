import io
import hashlib
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import httpx
from pypdf import PdfReader

from app.config import settings
from app.chunking import crear_chunks_markdown
from app import gemini_client

logger = logging.getLogger("documents")
router = APIRouter(prefix="/api/documents", tags=["documents"])

# Mismo umbral que calibramos hoy con datos reales para la ingesta de Firecrawl
UMBRAL_RELEVANCIA_FRAGMENTO = 0.35
UMBRAL_COHERENCIA_CATEGORIA = 0.45


def _similitud_coseno(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _headers(extra: dict | None = None) -> dict:
    h = {
        "apikey": settings.SUPABASE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def _base() -> str:
    return settings.SUPABASE_URL.rstrip("/")


@router.get("/categories")
async def listar_categorias_existentes():
    """Categorías que ya existen, para autocompletar en el formulario de
    subida y evitar duplicados por error de tipeo (ej. MANUAL_X vs Manual_x)."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{_base()}/rest/v1/fragmentos_vectoriales_gdp?select=categoria&limit=5000",
            headers=_headers()
        )
        if resp.status_code != 200:
            return []
        categorias = sorted(set(f["categoria"] for f in resp.json() if f.get("categoria")))
        return categorias


@router.get("")
async def listar_documentos():
    """Lista todos los documentos activos con su conteo de fragmentos."""
    url = (
        f"{_base()}/rest/v1/documentos_gdp"
        f"?estado=eq.ACTIVE&select=id,nombre_archivo,categoria,mime_type,creado_en"
        f"&order=creado_en.desc"
    )
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, headers=_headers())
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=resp.text)
        documentos = resp.json()

        # Conteo de fragmentos por documento (una consulta por documento;
        # para pocos documentos esto es simple y suficientemente rápido —
        # si crece mucho, se puede optimizar con una vista agregada después)
        for doc in documentos:
            count_url = (
                f"{_base()}/rest/v1/fragmentos_vectoriales_gdp"
                f"?documento_id=eq.{doc['id']}&select=id"
            )
            count_resp = await client.get(
                count_url, headers=_headers({"Prefer": "count=exact", "Range": "0-0"})
            )
            content_range = count_resp.headers.get("content-range", "")
            total = content_range.split("/")[-1] if "/" in content_range else "0"
            doc["total_fragmentos"] = int(total) if total.isdigit() else 0

        return documentos


@router.delete("/{documento_id}")
async def borrar_documento(documento_id: str):
    """
    Borrado SUAVE (archivado) — nunca borra físicamente. Marca el
    documento como ARCHIVED y busqueda_hibrida_rrf ya sabe excluirlo de
    resultados. Consistente con el estándar GDP del proyecto: nunca se
    pierde el dato, solo deja de ser visible/activo.
    """
    url = f"{_base()}/rest/v1/documentos_gdp?id=eq.{documento_id}"
    payload = {"estado": "ARCHIVED"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.patch(url, json=payload, headers=_headers({"Prefer": "return=minimal"}))
        if resp.status_code not in (200, 204):
            raise HTTPException(status_code=502, detail=resp.text)
    return {"ok": True, "archivado": documento_id}


@router.get("/archived")
async def listar_documentos_archivados():
    """Lista los documentos archivados (borrado suave), para poder revisarlos y restaurarlos."""
    url = (
        f"{_base()}/rest/v1/documentos_gdp"
        f"?estado=eq.ARCHIVED&select=id,nombre_archivo,categoria,mime_type,creado_en"
        f"&order=creado_en.desc"
    )
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, headers=_headers())
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=resp.text)
        return resp.json()


@router.post("/{documento_id}/restore")
async def restaurar_documento(documento_id: str):
    """Revierte un archivado: vuelve a poner el documento como ACTIVE,
    reapareciendo en las búsquedas y en la lista principal."""
    url = f"{_base()}/rest/v1/documentos_gdp?id=eq.{documento_id}"
    payload = {"estado": "ACTIVE"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.patch(url, json=payload, headers=_headers({"Prefer": "return=minimal"}))
        if resp.status_code not in (200, 204):
            raise HTTPException(status_code=502, detail=resp.text)
    return {"ok": True, "restaurado": documento_id}


async def _crear_documento_maestro(nombre_archivo: str, contenido_markdown: str, categoria: str, mime_type: str) -> str | None:
    import hashlib
    hash_sha256 = hashlib.sha256(contenido_markdown.encode("utf-8")).hexdigest()

    url = f"{_base()}/rest/v1/documentos_gdp"
    payload = {
        "categoria": categoria,
        "estandar_interop": "INTERNAL",
        "pii_sensitivity_level": "S1",
        "pii_lifecycle_state": "ACTIVE",
        "nombre_archivo": nombre_archivo,
        "mime_type": mime_type,
        "contenido_markdown": contenido_markdown,
        "estado": "ACTIVE",
        "hash_sha256": hash_sha256,
        "creado_por": "dashboard",
        "version_major": 1,
        "version_minor": 0,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload, headers=_headers({"Prefer": "return=representation"}))
        if resp.status_code in (200, 201):
            data = resp.json()
            if data:
                return data[0]["id"]
        logger.error(f"Error creando documento maestro: {resp.status_code} {resp.text}")
    return None


async def _guardar_fragmento(texto: str, categoria: str, embedding: list, documento_id: str, metadata: dict) -> bool:
    url = f"{_base()}/rest/v1/fragmentos_vectoriales_gdp"
    payload = [{
        "documento_id": documento_id,
        "categoria": categoria,
        "contenido_chunk": texto,
        "embedding": embedding,
        "metadata": metadata,
    }]
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload, headers=_headers({"Prefer": "return=minimal"}))
        return resp.status_code in (201, 204)


def _extraer_texto_pdf(contenido_bytes: bytes) -> str:
    lector = PdfReader(io.BytesIO(contenido_bytes))
    partes = []
    for pagina in lector.pages:
        texto_pagina = pagina.extract_text() or ""
        partes.append(texto_pagina)
    return "\n\n".join(partes)


@router.post("/upload")
async def subir_documento(
    archivo: UploadFile = File(...),
    categoria: str = Form(...),
):
    """
    Acepta .pdf, .md o .txt. Extrae el texto, lo trocea, genera embeddings
    y lo guarda como un documento nuevo.

    Protecciones aplicadas (mismas que en la ingesta de Firecrawl):
    1. Duplicados: rechaza si ya existe un documento con el mismo
       contenido exacto (comparado por hash SHA-256).
    2. Filtro de relevancia por fragmento: descarta fragmentos que no se
       parecen lo suficiente al título+categoría del documento.
    3. Chequeo de coherencia categoría-contenido: avisa (no bloquea) si
       el documento no encaja con lo que ya existe en esa categoría.
    """
    nombre = archivo.filename or "documento_sin_nombre"
    categoria_final = categoria.strip().upper()
    contenido_bytes = await archivo.read()

    if nombre.lower().endswith(".pdf"):
        try:
            texto = _extraer_texto_pdf(contenido_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"No se pudo leer el PDF: {e}")
        mime_type = "application/pdf"
    elif nombre.lower().endswith((".md", ".txt")):
        texto = contenido_bytes.decode("utf-8", errors="replace")
        mime_type = "text/markdown"
    else:
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .pdf, .md o .txt")

    if not texto.strip():
        raise HTTPException(status_code=400, detail="El archivo no contiene texto extraíble")

    # PROTECCIÓN 1: duplicados por contenido exacto (hash SHA-256)
    hash_contenido = hashlib.sha256(texto.encode("utf-8")).hexdigest()
    async with httpx.AsyncClient(timeout=15.0) as client:
        dup_resp = await client.get(
            f"{_base()}/rest/v1/documentos_gdp?hash_sha256=eq.{hash_contenido}&estado=eq.ACTIVE&select=id,nombre_archivo",
            headers=_headers()
        )
        if dup_resp.status_code == 200:
            existentes = dup_resp.json()
            if existentes:
                raise HTTPException(
                    status_code=409,
                    detail=f"Este contenido ya está indexado como \"{existentes[0]['nombre_archivo']}\"."
                )

    documento_id = await _crear_documento_maestro(nombre, texto, categoria_final, mime_type)
    if not documento_id:
        raise HTTPException(status_code=502, detail="No se pudo crear el documento maestro en Supabase")

    chunks = crear_chunks_markdown(texto)

    # PROTECCIÓN 3: coherencia categoría-contenido (solo avisa, no bloquea)
    advertencia_coherencia = None
    try:
        muestra = (nombre + "\n" + texto)[:1500]
        embedding_muestra = await gemini_client.generar_embedding(muestra)
        if embedding_muestra:
            async with httpx.AsyncClient(timeout=15.0) as client:
                coh_resp = await client.post(
                    f"{_base()}/rest/v1/rpc/verificar_coherencia_categoria",
                    json={"query_embedding": embedding_muestra, "p_categoria": categoria_final},
                    headers=_headers()
                )
                if coh_resp.status_code == 200:
                    datos = coh_resp.json()
                    if datos and datos[0].get("fragmentos_comparados", 0) > 0:
                        similitud = datos[0].get("similitud_promedio", 1.0)
                        if similitud < UMBRAL_COHERENCIA_CATEGORIA:
                            advertencia_coherencia = (
                                f"Este documento no se parece mucho a lo que ya tienes en "
                                f"'{categoria_final}' ({round(similitud * 100)}% de coherencia)."
                            )
    except Exception as e:
        logger.warning(f"No se pudo verificar coherencia (no bloquea): {e}")

    # PROTECCIÓN 2: filtro de relevancia por fragmento
    vector_referencia = await gemini_client.generar_embedding(f"{categoria_final}: {nombre}")

    guardados = 0
    descartados = 0
    for i, chunk in enumerate(chunks):
        embedding = await gemini_client.generar_embedding(chunk)
        if not embedding:
            continue

        if vector_referencia:
            relevancia = _similitud_coseno(vector_referencia, embedding)
            if relevancia < UMBRAL_RELEVANCIA_FRAGMENTO:
                descartados += 1
                logger.info(f"[FILTRO {round(relevancia*100)}%] Fragmento {i+1}/{len(chunks)} descartado.")
                continue

        metadata = {"origen": nombre, "chunk_index": i + 1, "total_chunks": len(chunks)}
        exito = await _guardar_fragmento(chunk, categoria_final, embedding, documento_id, metadata)
        if exito:
            guardados += 1

    if guardados == 0 and len(chunks) > 0:
        raise HTTPException(
            status_code=422,
            detail="El documento se guardó, pero ningún fragmento pasó el filtro de relevancia. Revisa la categoría elegida."
        )

    return {
        "ok": True,
        "documento_id": documento_id,
        "nombre_archivo": nombre,
        "fragmentos_totales": len(chunks),
        "fragmentos_guardados": guardados,
        "fragmentos_descartados": descartados,
        "advertencia_coherencia": advertencia_coherencia,
    }
