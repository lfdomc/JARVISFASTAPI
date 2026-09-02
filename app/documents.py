import io
import hashlib
import logging
import uuid
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
import httpx
from pypdf import PdfReader

from app.config import settings
from app.chunking import crear_chunks_markdown, crear_chunks_con_paginas
from app.indice_extractor import extraer_indice_documento, encontrar_seccion_para_pagina, clasificar_tipo_documento
from app.calidad_ingesta import auditar_calidad_ingesta, resumen_legible, validar_indice_contra_chunks, resumen_legible_validacion_indice
from app.docling_extractor import extraer_paginas_con_docling
from app import gemini_client, logging_utils

logger = logging.getLogger("documents")
router = APIRouter(prefix="/api/documents", tags=["documents"])

# Mismo umbral que calibramos hoy con datos reales para la ingesta de Firecrawl
UMBRAL_RELEVANCIA_FRAGMENTO = 0.35
UMBRAL_COHERENCIA_CATEGORIA = 0.45


def _similitud_coseno(vec_a: list[float], vec_b: list[float]) -> float:
    """Similitud de coseno vectorizada con NumPy — antes era un bucle de
    Python puro sobre 768 dimensiones, llamado por cada fragmento en el
    filtro de coherencia y el de alta confianza. NumPy calcula esto en C,
    no interpretado — mejora real de velocidad, no cosmética."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    a, b = np.asarray(vec_a, dtype=np.float64), np.asarray(vec_b, dtype=np.float64)
    norma_a, norma_b = np.linalg.norm(a), np.linalg.norm(b)
    if norma_a == 0 or norma_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norma_a * norma_b))


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
    """Lista documentos activos y en proceso, con su conteo de fragmentos."""
    url = (
        f"{_base()}/rest/v1/documentos_gdp"
        f"?estado=in.(ACTIVE,PROCESSING,FAILED)&select=id,nombre_archivo,categoria,mime_type,creado_en,estado"
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


@router.get("/{documento_id}/fragments")
async def listar_fragmentos_documento(documento_id: uuid.UUID):
    """Fragmentos reales de un documento, para revisar la calidad del
    chunking directamente desde el dashboard."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{_base()}/rest/v1/fragmentos_vectoriales_gdp"
            f"?documento_id=eq.{documento_id}&select=id,contenido_chunk,categoria,metadata",
            headers=_headers()
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=resp.text)
        fragmentos = resp.json()
        # Ordena por chunk_index si está disponible en metadata
        fragmentos.sort(key=lambda f: (f.get("metadata") or {}).get("chunk_index", 0))
        return fragmentos


@router.delete("/{documento_id}")
async def borrar_documento(documento_id: uuid.UUID):
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
async def restaurar_documento(documento_id: uuid.UUID):
    """Revierte un archivado: vuelve a poner el documento como ACTIVE,
    reapareciendo en las búsquedas y en la lista principal."""
    url = f"{_base()}/rest/v1/documentos_gdp?id=eq.{documento_id}"
    payload = {"estado": "ACTIVE"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.patch(url, json=payload, headers=_headers({"Prefer": "return=minimal"}))
        if resp.status_code not in (200, 204):
            raise HTTPException(status_code=502, detail=resp.text)
    return {"ok": True, "restaurado": documento_id}


async def _crear_documento_maestro(nombre_archivo: str, contenido_markdown: str, categoria: str, mime_type: str, estado: str = "ACTIVE", indice_markdown: str | None = None) -> str | None:
    import hashlib
    import uuid
    hash_sha256 = hashlib.sha256(contenido_markdown.encode("utf-8")).hexdigest()
    documento_uuid = str(uuid.uuid4())

    url = f"{_base()}/rest/v1/documentos_gdp"
    payload = {
        "id": documento_uuid,
        "documento_raiz_id": documento_uuid,
        "categoria": categoria,
        "estandar_interop": "INTERNAL",
        "pii_sensitivity_level": "S1",
        "pii_lifecycle_state": "ACTIVE",
        "nombre_archivo": nombre_archivo,
        "mime_type": mime_type,
        "contenido_markdown": contenido_markdown,
        "estado": estado,
        "hash_sha256": hash_sha256,
        "creado_por": "dashboard",
        "version_major": 1,
        "version_minor": 0,
        "indice_markdown": indice_markdown,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, headers=_headers({"Prefer": "return=representation"}))
        if resp.status_code in (200, 201):
            data = resp.json()
            if data:
                return data[0]["id"]
        # CORREGIDO: antes ocultaba el error real detrás de un mensaje
        # genérico — ahora lo expone directo en la respuesta HTTP, para
        # no repetir el mismo problema de "logs escondidos" de hoy.
        detalle_real = f"Supabase respondió HTTP {resp.status_code}: {resp.text[:500]}"
        logger.error(f"Error creando documento maestro: {detalle_real}")
        raise HTTPException(status_code=502, detail=detalle_real)
    return None


async def _guardar_fragmento(texto: str, categoria: str, embedding: list, documento_id: str, metadata: dict, pagina_inicio: int | None = None, pagina_fin: int | None = None, seccion: str | None = None, tipo_contenido: str | None = None) -> bool:
    url = f"{_base()}/rest/v1/fragmentos_vectoriales_gdp"
    payload = [{
        "documento_id": documento_id,
        "categoria": categoria,
        "contenido_chunk": texto,
        "embedding": embedding,
        "metadata": metadata,
        "pagina_inicio": pagina_inicio,
        "pagina_fin": pagina_fin,
        "seccion": seccion,
        "tipo_contenido": tipo_contenido,
    }]
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload, headers=_headers({"Prefer": "return=minimal"}))
        return resp.status_code in (201, 204)


def _extraer_texto_pdf(contenido_bytes: bytes) -> list[str]:
    """Devuelve una lista con el texto de cada página por separado —
    preserva el límite real de página para poder citarla con certeza
    después, en vez de que el modelo tenga que adivinarla del pie de
    página embebido en el texto."""
    lector = PdfReader(io.BytesIO(contenido_bytes))
    return [pagina.extract_text() or "" for pagina in lector.pages]


@router.post("/upload")
async def subir_documento(
    background_tasks: BackgroundTasks,
    archivo: UploadFile = File(...),
    categoria: str = Form(...),
):
    """
    Acepta .pdf, .md o .txt. Extrae el texto rápido (síncrono, es solo
    parseo local) y valida duplicados de inmediato — pero el chunking +
    generación de embeddings (lo que de verdad tarda) corre en SEGUNDO
    PLANO, para no dejar al usuario esperando con la pantalla congelada
    en archivos grandes.

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
        # Intenta primero con Docling (mejor reconocimiento de layout y
        # tablas reales); si falla por cualquier razón (red, timeout,
        # lo que sea), cae automáticamente al extractor de pypdf que ya
        # funciona — la subida nunca se rompe por esto.
        resultado_docling = await extraer_paginas_con_docling(contenido_bytes)
        encabezados_docling = None
        if resultado_docling is not None:
            paginas = resultado_docling["paginas"]
            encabezados_docling = resultado_docling.get("encabezados") or None
            motor_extraccion = "docling"
        else:
            try:
                paginas = _extraer_texto_pdf(contenido_bytes)
                motor_extraccion = "pypdf (respaldo)"
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"No se pudo leer el PDF: {e}")
        logger.info(f"[{nombre}] Extracción con motor: {motor_extraccion}" + (f" ({len(encabezados_docling)} encabezados detectados en la estructura)" if encabezados_docling else ""))
        mime_type = "application/pdf"
    elif nombre.lower().endswith((".md", ".txt")):
        paginas = [contenido_bytes.decode("utf-8", errors="replace")]
        mime_type = "text/markdown"
        motor_extraccion = "texto plano"
        encabezados_docling = None
    else:
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .pdf, .md o .txt")

    texto = "\n\n".join(paginas)
    if not texto.strip():
        raise HTTPException(status_code=400, detail="El archivo no contiene texto extraíble")

    # PROTECCIÓN 1: duplicados por contenido exacto (hash SHA-256) — se
    # revisa YA, antes de encolar nada, para fallar rápido sin desperdiciar
    # trabajo en segundo plano.
    hash_contenido = hashlib.sha256(texto.encode("utf-8")).hexdigest()
    async with httpx.AsyncClient(timeout=15.0) as client:
        dup_resp = await client.get(
            f"{_base()}/rest/v1/documentos_gdp?hash_sha256=eq.{hash_contenido}&estado=in.(ACTIVE,PROCESSING)&select=id,nombre_archivo",
            headers=_headers()
        )
        if dup_resp.status_code == 200:
            existentes = dup_resp.json()
            if existentes:
                raise HTTPException(
                    status_code=409,
                    detail=f"Este contenido ya está indexado (o en proceso) como \"{existentes[0]['nombre_archivo']}\"."
                )

    # Detecta el índice/tabla de contenidos del documento (si tiene uno) —
    # solo tiene sentido para PDFs, que sí traen páginas reales separadas.
    resultado_indice = extraer_indice_documento(paginas) if nombre.lower().endswith(".pdf") else None
    indice_markdown = resultado_indice["markdown"] if resultado_indice else None
    if resultado_indice:
        logger.info(f"[{nombre}] Índice detectado (págs. {resultado_indice['pagina_encontrado']}-{resultado_indice['pagina_fin_indice']}, {len(resultado_indice['entradas'])} entradas) — tipo probable: {resultado_indice['tipo_documento_probable']}")
    elif encabezados_docling:
        # SIN índice impreso detectable — pero Docling ya reconoció los
        # encabezados REALES de la estructura del documento (no un texto
        # impreso, la estructura misma). Esto es algo que antes era
        # imposible: un documento sin tabla de contenidos ahora también
        # puede obtener columna 'seccion' poblada por fragmento.
        # pagina_encontrado/pagina_fin_indice quedan en None a propósito:
        # no existe una "página del índice impreso" que excluir de la
        # búsqueda, porque no hay índice impreso — los encabezados vienen
        # de la estructura, repartidos por todo el documento.
        resultado_indice = {
            "entradas": encabezados_docling,
            "tipo_documento_probable": clasificar_tipo_documento(encabezados_docling),
            "pagina_encontrado": None,
            "pagina_fin_indice": None,
        }
        logger.info(f"[{nombre}] Sin índice impreso — usando {len(encabezados_docling)} encabezados detectados por Docling en la estructura del documento.")

    # Se crea de inmediato en estado PROCESSING — visible en el dashboard
    # como "procesando", pero invisible para las búsquedas (busqueda_hibrida_rrf
    # solo considera documentos ACTIVE) hasta que termine.
    documento_id = await _crear_documento_maestro(nombre, texto, categoria_final, mime_type, estado="PROCESSING", indice_markdown=indice_markdown)
    if not documento_id:
        raise HTTPException(status_code=502, detail="No se pudo crear el documento maestro en Supabase")

    rango_paginas_indice = None
    if resultado_indice and resultado_indice.get("pagina_encontrado") and resultado_indice.get("pagina_fin_indice"):
        rango_paginas_indice = (resultado_indice["pagina_encontrado"], resultado_indice["pagina_fin_indice"])

    background_tasks.add_task(
        _procesar_documento_en_segundo_plano, documento_id, nombre, paginas, categoria_final,
        resultado_indice["entradas"] if resultado_indice else None,
        rango_paginas_indice,
    )

    return {
        "ok": True,
        "documento_id": documento_id,
        "nombre_archivo": nombre,
        "estado": "processing",
        "indice_detectado": bool(resultado_indice),
        "tipo_documento_probable": resultado_indice.get("tipo_documento_probable") if resultado_indice else None,
        "motor_extraccion": motor_extraccion,
        "mensaje": "El documento se está procesando en segundo plano. Actualiza la lista en unos momentos para ver el progreso.",
    }


async def _actualizar_estado_documento(documento_id: str, estado: str, datos_extra: dict | None = None):
    payload = {"estado": estado}
    if datos_extra:
        payload.update(datos_extra)
    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.patch(
            f"{_base()}/rest/v1/documentos_gdp?id=eq.{documento_id}",
            json=payload,
            headers=_headers({"Prefer": "return=minimal"})
        )


async def _procesar_documento_en_segundo_plano(documento_id: str, nombre: str, paginas: list[str], categoria_final: str, entradas_indice: list[dict] | None = None, rango_paginas_indice: tuple[int, int] | None = None):
    """Corre DESPUÉS de que la petición HTTP ya respondió — el chunking,
    los embeddings y el chequeo de coherencia pasan aquí, sin bloquear al
    usuario. Cada fragmento guarda su página real de origen (capturada en
    la extracción, no adivinada después) y, si el documento tiene índice,
    también su sección — dirección completa dentro del documento."""
    try:
        chunks = crear_chunks_con_paginas(paginas)
        texto_completo = "\n\n".join(paginas)

        # AUTOCHEQUEO: convierte en código automático lo que hicimos a
        # mano toda la sesión de hoy — corre con CUALQUIER documento, sin
        # importar su tipo. Si algo estructural sale mal, queda
        # registrado con evidencia concreta, sin depender de que alguien
        # se acuerde de ir a revisarlo.
        reporte_calidad = auditar_calidad_ingesta(chunks, len(paginas))
        logger.info(f"[{nombre}] {resumen_legible(reporte_calidad)}")
        if not reporte_calidad["ok"]:
            await logging_utils.registrar_error(
                "CALIDAD_INGESTA", resumen_legible(reporte_calidad), nombre,
                "Revisar el reporte completo en documentos_gdp.reporte_calidad_ingesta — no bloquea la subida."
            )

        # Cruza el ÍNDICE del documento (su propia respuesta correcta)
        # contra los fragmentos ya trocedos — sin IA, sin datos externos.
        if entradas_indice:
            reporte_validacion_indice = validar_indice_contra_chunks(entradas_indice, chunks)
            logger.info(f"[{nombre}] {resumen_legible_validacion_indice(reporte_validacion_indice)}")
            reporte_calidad["validacion_indice"] = reporte_validacion_indice
            if not reporte_validacion_indice["ok"]:
                await logging_utils.registrar_error(
                    "CALIDAD_INGESTA_INDICE", resumen_legible_validacion_indice(reporte_validacion_indice), nombre,
                    "El índice del documento menciona páginas que ningún fragmento cubre — revisar el troceo."
                )

        # PROTECCIÓN 4: el índice del documento ya se guarda aparte, de
        # forma estructurada, en indice_markdown — no debe ADEMÁS quedar
        # indexado como fragmento normal de búsqueda. Si queda, una
        # pregunta por el nombre de una sección puede encontrar la
        # ENTRADA del índice que la menciona, y citar la página física
        # del índice (ej. pág. 6) en vez de la página real del contenido
        # (ej. pág. 50) — exactamente el tipo de error que esto evita.
        if rango_paginas_indice:
            p_ini_indice, p_fin_indice = rango_paginas_indice
            antes = len(chunks)
            chunks = [
                c for c in chunks
                if not (c["pagina_inicio"] >= p_ini_indice and c["pagina_fin"] <= p_fin_indice)
            ]
            if len(chunks) < antes:
                logger.info(f"[{nombre}] Excluidos {antes - len(chunks)} fragmento(s) del índice (págs. {p_ini_indice}-{p_fin_indice}) — ya está guardado aparte, estructurado.")

        # PROTECCIÓN 3: coherencia categoría-contenido
        try:
            muestra = (nombre + "\n" + texto_completo)[:1500]
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
                                await logging_utils.registrar_error(
                                    "SUBIDA_COHERENCIA", f"Baja coherencia con '{categoria_final}' ({round(similitud*100)}%)",
                                    nombre, "Revisar la categoría elegida — no bloquea, solo avisa"
                                )
        except Exception as e:
            logger.warning(f"No se pudo verificar coherencia (no bloquea): {e}")

        # PROTECCIÓN 2: filtro de relevancia por fragmento
        vector_referencia = await gemini_client.generar_embedding(f"{categoria_final}: {nombre}")

        guardados = 0
        descartados = 0
        for i, chunk in enumerate(chunks):
            embedding = await gemini_client.generar_embedding(chunk["texto"])
            if not embedding:
                continue

            if vector_referencia:
                relevancia = _similitud_coseno(vector_referencia, embedding)
                if relevancia < UMBRAL_RELEVANCIA_FRAGMENTO:
                    descartados += 1
                    continue

            metadata = {
                "origen": nombre,
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
                "pagina_inicio": chunk["pagina_inicio"],
                "pagina_fin": chunk["pagina_fin"],
            }
            seccion = encontrar_seccion_para_pagina(entradas_indice, chunk["pagina_inicio"]) if entradas_indice else None
            exito = await _guardar_fragmento(
                chunk["texto"], categoria_final, embedding, documento_id, metadata,
                pagina_inicio=chunk["pagina_inicio"], pagina_fin=chunk["pagina_fin"], seccion=seccion,
                tipo_contenido=chunk.get("tipo_contenido"),
            )
            if exito:
                guardados += 1

        if guardados == 0 and len(chunks) > 0:
            await _actualizar_estado_documento(documento_id, "FAILED")
            await logging_utils.registrar_error(
                "SUBIDA_DOCUMENTO", "Ningún fragmento pasó el filtro de relevancia", nombre,
                "Revisar la categoría elegida o el umbral de relevancia"
            )
            return

        await _actualizar_estado_documento(documento_id, "ACTIVE", datos_extra={"reporte_calidad_ingesta": reporte_calidad})
        logger.info(f"[{nombre}] procesado en segundo plano: {guardados}/{len(chunks)} fragmentos ({descartados} descartados).")

    except Exception as e:
        await _actualizar_estado_documento(documento_id, "FAILED")
        await logging_utils.registrar_error("SUBIDA_DOCUMENTO", f"Excepción procesando en segundo plano: {e}", nombre)
