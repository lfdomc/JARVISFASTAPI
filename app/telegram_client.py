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


async def enviar_mensaje_con_botones(chat_id: int, texto: str, teclado_inline: list) -> None:
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": texto, "reply_markup": {"inline_keyboard": teclado_inline}}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json={**payload, "parse_mode": "Markdown"})
        if resp.status_code == 200:
            return
        logger.warning(f"Falló envío con botones + Markdown, reintentando en texto plano: {resp.text}")
        await client.post(url, json=payload)


async def responder_callback(callback_query_id: str, texto_alerta: str = "") -> None:
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post(url, json={"callback_query_id": callback_query_id, "text": texto_alerta})
        except Exception as e:
            logger.warning(f"Excepción respondiendo callback: {e}")
