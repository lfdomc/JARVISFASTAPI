import httpx
import logging
from app.config import settings

logger = logging.getLogger("telegram_client")


async def indicar_escribiendo(chat_id: int) -> None:
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendChatAction"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(url, json={"chat_id": chat_id, "action": "typing"})
            if resp.status_code != 200:
                logger.warning(f"Falló indicar_escribiendo: {resp.text}")
        except Exception as e:
            logger.warning(f"Excepción en indicar_escribiendo: {e}")


async def enviar_mensaje(chat_id: int, texto: str) -> None:
    """Reintenta en texto plano si el Markdown rompe el parser de Telegram
    (mismo problema que ya resolvimos en Apps Script)."""
    if len(texto) > 4096:
        texto = texto[:4090] + "…"

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json={"chat_id": chat_id, "text": texto, "parse_mode": "Markdown"})
        if resp.status_code == 200:
            return
        logger.warning(f"Falló envío con Markdown, reintentando en texto plano: {resp.text}")
        await client.post(url, json={"chat_id": chat_id, "text": texto})
