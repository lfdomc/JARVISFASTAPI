import hashlib
import uuid
import httpx
from app.config import settings
from app import logging_utils

UMBRAL_RELEVANCIA_FRAGMENTO = 0.35
UMBRAL_COHERENCIA_CATEGORIA = 0.45


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


def similitud_coseno(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def buscar_documento_duplicado(hash_contenido: str) -> dict | None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{_base()}/rest/v1/documentos_gdp?hash_sha256=eq.{hash_contenido}&estado=eq.ACTIVE&select=id,nombre_archivo",
            headers=_headers()
        )
        if resp.status_code == 200:
            filas = resp.json()
            return filas[0] if filas else None
    return None


async def crear_documento_maestro(nombre_archivo: str, contenido_markdown: str, categoria: str, mime_type: str, creado_por: str = "sistema", estado: str = "ACTIVE") -> str | None:
    documento_uuid = str(uuid.uuid4())
    hash_contenido = hashlib.sha256(contenido_markdown.encode("utf-8")).hexdigest()
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
        "hash_sha256": hash_contenido,
        "creado_por": creado_por,
        "version_major": 1,
        "version_minor": 0,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url=f"{_base()}/rest/v1/documentos_gdp", json=payload, headers=_headers({"Prefer": "return=representation"}))
        if resp.status_code in (200, 201):
            datos = resp.json()
            if datos:
                return datos[0]["id"]
        await logging_utils.registrar_error("CREAR_DOCUMENTO_MAESTRO", f"HTTP {resp.status_code}", resp.text)
    return None


async def guardar_fragmento(texto: str, categoria: str, embedding: list[float], documento_id: str | None, metadata: dict, propietario_telegram_id: str | None = None) -> bool:
    payload = [{
        "documento_id": documento_id,
        "categoria": categoria,
        "contenido_chunk": texto,
        "embedding": embedding,
        "metadata": metadata,
        "propietario_telegram_id": propietario_telegram_id,
    }]
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(f"{_base()}/rest/v1/fragmentos_vectoriales_gdp", json=payload, headers=_headers({"Prefer": "return=minimal"}))
        if resp.status_code in (201, 204):
            return True
        await logging_utils.registrar_error("GUARDAR_FRAGMENTO", f"HTTP {resp.status_code}", resp.text)
    return False


async def verificar_coherencia_categoria(embedding: list[float], categoria: str) -> tuple[float, int]:
    """Devuelve (similitud_promedio, fragmentos_comparados)."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{_base()}/rest/v1/rpc/verificar_coherencia_categoria",
            json={"query_embedding": embedding, "p_categoria": categoria},
            headers=_headers()
        )
        if resp.status_code == 200:
            datos = resp.json()
            if datos:
                return datos[0].get("similitud_promedio") or 0.0, datos[0].get("fragmentos_comparados") or 0
    return 0.0, 0


async def buscar_nota_similar(embedding: list[float], umbral: float = 0.85) -> dict | None:
    """Reutiliza buscar_nota_similar (misma RPC que Apps Script) para detectar duplicados de notas personales."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{_base()}/rest/v1/rpc/buscar_nota_similar",
            json={"query_embedding": embedding, "match_threshold": umbral},
            headers=_headers()
        )
        if resp.status_code == 200:
            datos = resp.json()
            if datos:
                return datos[0]
    return None


async def sugerir_categoria_similar(embedding: list[float]) -> str | None:
    """Reutiliza sugerir_categorias_similares (misma RPC que Apps Script)."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{_base()}/rest/v1/rpc/sugerir_categorias_similares",
            json={"query_embedding": embedding, "cantidad": 3},
            headers=_headers()
        )
        if resp.status_code == 200:
            datos = resp.json()
            if datos and datos[0].get("categoria_sugerida"):
                return datos[0]["categoria_sugerida"]
    return None


# ============================================================================
# COLA DE PROCESAMIENTO EN SEGUNDO PLANO
# El estado del propio documento (documentos_gdp.estado: PROCESSING -> ACTIVE
# o FAILED) hace las veces de "cola" — más simple que una tabla de jobs
# separada, y busqueda_hibrida_rrf ya excluye todo lo que no esté ACTIVE.
# ============================================================================

async def actualizar_estado_documento(documento_id: str, estado: str):
    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.patch(
            f"{_base()}/rest/v1/documentos_gdp?id=eq.{documento_id}",
            json={"estado": estado},
            headers=_headers({"Prefer": "return=minimal"})
        )


async def procesar_fragmentos_en_segundo_plano(documento_id: str, texto: str, categoria: str, nombre: str) -> tuple[int, int]:
    """Corre DESPUÉS de responder al usuario — chunking, embeddings y
    filtro de relevancia, sin bloquear la respuesta inicial.
    Devuelve (guardados, descartados)."""
    from app import gemini_client
    from app.chunking import crear_chunks_markdown
    try:
        chunks = crear_chunks_markdown(texto)
        vector_referencia = await gemini_client.generar_embedding(f"{categoria}: {nombre}")

        guardados = 0
        descartados = 0
        for i, chunk in enumerate(chunks):
            embedding = await gemini_client.generar_embedding(chunk)
            if not embedding:
                continue
            if vector_referencia and similitud_coseno(vector_referencia, embedding) < UMBRAL_RELEVANCIA_FRAGMENTO:
                descartados += 1
                continue
            metadata = {"origen": nombre, "chunk_index": i + 1, "total_chunks": len(chunks)}
            if await guardar_fragmento(chunk, categoria, embedding, documento_id, metadata):
                guardados += 1

        if guardados == 0 and len(chunks) > 0:
            await actualizar_estado_documento(documento_id, "FAILED")
            await logging_utils.registrar_error("PROCESAR_FRAGMENTOS_BG", "Ningún fragmento pasó el filtro de relevancia", documento_id)
        else:
            await actualizar_estado_documento(documento_id, "ACTIVE")
        return guardados, descartados
    except Exception as e:
        await actualizar_estado_documento(documento_id, "FAILED")
        await logging_utils.registrar_error("PROCESAR_FRAGMENTOS_BG", str(e), documento_id)
        return 0, 0


async def obtener_indices_documentos(documento_ids: list[str]) -> dict[str, str]:
    """Devuelve {documento_id: indice_markdown} para los documentos que
    sí tienen un índice detectado — usado en modo profundo para darle al
    modelo el mapa completo del documento antes de analizar fragmentos
    sueltos."""
    ids_validos = [d for d in set(documento_ids) if d]
    if not ids_validos:
        return {}
    filtro = ",".join(ids_validos)
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{_base()}/rest/v1/documentos_gdp?id=in.({filtro})&indice_markdown=not.is.null&select=id,indice_markdown",
            headers=_headers()
        )
        if resp.status_code == 200:
            return {fila["id"]: fila["indice_markdown"] for fila in resp.json()}
    return {}


MAX_VECINOS_TOTAL = 12
RADIO_PAGINAS_VECINAS = 1  # misma página + 1 antes + 1 después


async def expandir_contexto_con_vecinos(fragmentos: list[dict]) -> list[dict]:
    """
    Los fragmentos de la MISMA página, o de páginas inmediatamente antes/
    después, están relacionados — probablemente son continuación del
    mismo párrafo, tabla o idea que quedó partida por el troceo. Esta
    función busca esos vecinos y los agrega al contexto, marcados como
    "contexto de vecindad" (no como resultado directo de la búsqueda),
    para que el modelo tenga la imagen completa de esa zona del
    documento, no solo el fragmento aislado que ganó por similitud.

    No aplica a notas personales (sin documento_id) ni si no hay
    información de página — no hay "vecindad" que expandir ahí.
    """
    ids_ya_incluidos = {f["id"] for f in fragmentos}
    vecinos_encontrados = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        for f in fragmentos:
            if len(vecinos_encontrados) >= MAX_VECINOS_TOTAL:
                break
            documento_id = f.get("documento_id")
            pagina = f.get("pagina_inicio") or (f.get("metadata") or {}).get("pagina_inicio")
            if not documento_id or not pagina:
                continue

            p_desde = max(1, int(pagina) - RADIO_PAGINAS_VECINAS)
            p_hasta = int(pagina) + RADIO_PAGINAS_VECINAS

            resp = await client.get(
                f"{_base()}/rest/v1/fragmentos_vectoriales_gdp"
                f"?documento_id=eq.{documento_id}"
                f"&pagina_inicio=gte.{p_desde}&pagina_inicio=lte.{p_hasta}"
                f"&select=id,contenido_chunk,categoria,metadata,pagina_inicio,pagina_fin,seccion,documento_id"
                f"&order=pagina_inicio.asc",
                headers=_headers()
            )
            if resp.status_code != 200:
                continue

            for vecino in resp.json():
                if vecino["id"] in ids_ya_incluidos or len(vecinos_encontrados) >= MAX_VECINOS_TOTAL:
                    continue
                ids_ya_incluidos.add(vecino["id"])
                vecino["nombre_documento"] = f.get("nombre_documento")
                vecino["es_contexto_vecino"] = True
                vecinos_encontrados.append(vecino)

    return fragmentos + vecinos_encontrados
