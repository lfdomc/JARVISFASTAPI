"""
Cliente de Supabase (REST + RPC) usando httpx async. Traducción directa
de las funciones de VectorStore.gs que ya probamos hoy.
"""
import httpx
import logging
from app.config import settings

logger = logging.getLogger("supabase_client")


def _headers() -> dict:
    return {
        "apikey": settings.SUPABASE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def _base_url() -> str:
    """Evita dobles barras si SUPABASE_URL viene con / al final."""
    return settings.SUPABASE_URL.rstrip("/")


async def buscar_por_numero_pagina(numero_pagina: int, telegram_id_solicitante: str | None = None, limite: int = 8) -> list[dict]:
    """
    Búsqueda directa por número de página — mismo principio que
    buscar_por_numero_seccion: cuando el usuario pregunta literalmente
    "¿qué dice la página N?", es más confiable ir directo a la columna
    pagina_inicio/pagina_fin que depender de que la búsqueda semántica
    adivine cuál fragmento cubre esa página exacta.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{_base_url()}/rest/v1/fragmentos_vectoriales_gdp"
            f"?pagina_inicio=lte.{numero_pagina}&pagina_fin=gte.{numero_pagina}"
            f"&select=id,contenido_chunk,categoria,metadata,pagina_inicio,pagina_fin,seccion,documento_id"
            f"&order=pagina_inicio.asc"
            f"&limit={limite}",
            headers=_headers()
        )
        if resp.status_code != 200:
            return []
        filas = resp.json()

        documento_ids = list({f["documento_id"] for f in filas if f.get("documento_id")})
        nombres = {}
        if documento_ids:
            filtro = ",".join(documento_ids)
            resp_docs = await client.get(
                f"{_base_url()}/rest/v1/documentos_gdp?id=in.({filtro})&select=id,nombre_archivo,estado",
                headers=_headers()
            )
            if resp_docs.status_code == 200:
                nombres = {d["id"]: d for d in resp_docs.json()}

        resultado = []
        for f in filas:
            doc = nombres.get(f.get("documento_id"), {})
            if doc.get("estado") not in (None, "ACTIVE"):
                continue
            f["nombre_documento"] = doc.get("nombre_archivo", "N/A")
            f["similitud_coseno"] = 1.0
            resultado.append(f)
        return resultado


async def buscar_por_numero_seccion(numero_seccion: str, telegram_id_solicitante: str | None = None, limite: int = 8) -> list[dict]:
    """
    Búsqueda directa por la columna 'seccion' (ej. '3.5') — para cuando el
    usuario menciona un número de sección explícito. Más confiable que la
    búsqueda semántica para este caso: un fragmento de contenido puede no
    repetir literalmente "3.5 Empleo turístico" en su propio texto (ese
    título a veces queda al final del fragmento ANTERIOR), así que la
    similitud de significado puede no encontrarlo — pero la columna
    'seccion' ya lo tiene etiquetado con certeza desde la ingesta.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{_base_url()}/rest/v1/fragmentos_vectoriales_gdp"
            f"?seccion=ilike.*{numero_seccion}*"
            f"&select=id,contenido_chunk,categoria,metadata,pagina_inicio,pagina_fin,seccion,documento_id"
            f"&order=pagina_inicio.asc"
            f"&limit={limite}",
            headers=_headers()
        )
        if resp.status_code != 200:
            return []
        filas = resp.json()

        # Trae el nombre del documento para cada fragmento (la RPC normal
        # lo hace con un JOIN; aquí hay que resolverlo aparte)
        documento_ids = list({f["documento_id"] for f in filas if f.get("documento_id")})
        nombres = {}
        if documento_ids:
            filtro = ",".join(documento_ids)
            resp_docs = await client.get(
                f"{_base_url()}/rest/v1/documentos_gdp?id=in.({filtro})&select=id,nombre_archivo,estado",
                headers=_headers()
            )
            if resp_docs.status_code == 200:
                nombres = {d["id"]: d for d in resp_docs.json()}

        resultado = []
        for f in filas:
            doc = nombres.get(f.get("documento_id"), {})
            if doc.get("estado") not in (None, "ACTIVE"):  # excluye archivados/en proceso
                continue
            f["nombre_documento"] = doc.get("nombre_archivo", "N/A")
            f["similitud_coseno"] = 1.0  # match directo por sección — máxima confianza
            resultado.append(f)
        return resultado


async def buscar_contexto_semantico(
    pregunta: str,
    embedding: list[float],
    categoria: str | None = None,
    match_count: int = 5,
    telegram_id_solicitante: str | None = None,
) -> list[dict]:
    """
    Llama a busqueda_hibrida_rrf — la misma función RPC ya calibrada hoy,
    incluye el filtro de aislamiento por propietario_telegram_id.
    """
    url = f"{_base_url()}/rest/v1/rpc/busqueda_hibrida_rrf"
    payload = {
        "query_text": pregunta,
        "query_embedding": embedding,
        "match_count": match_count,
        "rrf_k": 60,
        "p_categoria": categoria,
        "p_metadata_filter": None,
        "p_telegram_id_solicitante": telegram_id_solicitante,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload, headers=_headers())
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"Error en busqueda_hibrida_rrf: HTTP {resp.status_code}: {resp.text}")
        return []


async def guardar_nota_personal(
    texto: str,
    categoria: str,
    embedding: list[float],
    telegram_id: str,
    metadata_extra: dict | None = None,
) -> bool:
    url = f"{_base_url()}/rest/v1/fragmentos_vectoriales_gdp"
    payload = [{
        "documento_id": None,
        "categoria": categoria,
        "contenido_chunk": texto,
        "embedding": embedding,
        "metadata": metadata_extra or {},
        "propietario_telegram_id": telegram_id,
    }]
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            url, json=payload,
            headers={**_headers(), "Prefer": "return=minimal"}
        )
        if resp.status_code in (201, 204):
            return True
        logger.error(f"Error guardando nota: HTTP {resp.status_code}: {resp.text}")
        return False


async def obtener_historial_conversacion(telegram_id: str, limite: int = 6) -> list[dict]:
    url = (
        f"{_base_url()}/rest/v1/historial_conversacion"
        f"?telegram_id=eq.{telegram_id}&order=creado_en.desc&limit={limite}"
    )
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=_headers())
        if resp.status_code == 200:
            filas = resp.json()
            filas.reverse()  # orden cronológico
            return [
                {"role": f["role"], "parts": [{"text": f["contenido"]}]}
                for f in filas
            ]
        return []


async def guardar_mensaje_historial(telegram_id: str, rol: str, contenido: str) -> None:
    url = f"{_base_url()}/rest/v1/historial_conversacion"
    payload = {"telegram_id": telegram_id, "role": rol, "contenido": contenido}
    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.post(url, json=payload, headers={**_headers(), "Prefer": "return=minimal"})
