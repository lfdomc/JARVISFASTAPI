import httpx
import logging
from app.config import settings

logger = logging.getLogger("capa_logs")


def _headers() -> dict:
    return {
        "apikey": settings.SUPABASE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def _base() -> str:
    return settings.SUPABASE_URL.rstrip("/")


async def registrar_error(
    agente_origen: str,
    descripcion_falla: str,
    causa_raiz: str = "Excepción no controlada",
    accion_correctiva: str = "Revisión técnica del backend",
):
    """
    Registra un error en capa_logs (misma tabla que usa JARVIS en Apps
    Script) — así el panel "Errores recientes" del dashboard refleja la
    realidad, y no queda vacío por omisión.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_base()}/rest/v1/rpc/registrar_log_sistema",
                json={
                    "p_agente_origen": agente_origen,
                    "p_descripcion_falla": str(descripcion_falla)[:2000],
                    "p_causa_raiz": str(causa_raiz)[:2000],
                    "p_accion_correctiva": accion_correctiva,
                    "p_estado": "OPEN",
                },
                headers=_headers(),
            )
            if resp.status_code not in (200, 201, 204):
                logger.error(f"[CAPA_LOGS] Falló registrar_log_sistema (HTTP {resp.status_code}): {resp.text}")
    except Exception as e:
        logger.error(f"[CAPA_LOGS] Excepción al registrar error: {e}")
