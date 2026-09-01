from fastapi import Header, HTTPException
from app.config import settings


async def verificar_api_key(authorization: str | None = Header(default=None)):
    """
    Protege los endpoints del dashboard (/api/clients, /api/documents,
    /api/stats) con una clave compartida — sin esto, cualquiera con la
    URL pública podría ver, subir o archivar documentos.

    Si DASHBOARD_API_KEY no está configurada, no bloquea nada (para no
    romper pruebas locales por accidente) — pero se loguea una advertencia
    fuerte, porque en producción esto NO debería quedar así.
    """
    if not settings.DASHBOARD_API_KEY:
        import logging
        logging.getLogger("auth").warning(
            "⚠️ DASHBOARD_API_KEY no está configurada — el dashboard queda SIN PROTECCIÓN. "
            "Configúrala en Railway antes de compartir la URL con nadie."
        )
        return

    esperado = f"Bearer {settings.DASHBOARD_API_KEY}"
    if authorization != esperado:
        raise HTTPException(status_code=401, detail="No autorizado.")
