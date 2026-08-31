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
