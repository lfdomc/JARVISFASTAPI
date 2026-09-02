import base64
import httpx
import logging
from app.config import settings
from app import logging_utils

logger = logging.getLogger("voice")


async def _descargar_archivo_telegram(file_id: str) -> bytes | None:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/getFile", params={"file_id": file_id})
        if resp.status_code != 200:
            return None
        file_path = resp.json().get("result", {}).get("file_path")
        if not file_path:
            return None

        resp_archivo = await client.get(f"https://api.telegram.org/file/bot{settings.TELEGRAM_BOT_TOKEN}/{file_path}")
        if resp_archivo.status_code != 200:
            return None
        return resp_archivo.content


async def transcribir_nota_de_voz(file_id: str) -> str | None:
    """Descarga la nota de voz de Telegram y la transcribe con Gemini
    (mismo enfoque multimodal que VoiceHandler.gs en Apps Script)."""
    audio_bytes = await _descargar_archivo_telegram(file_id)
    if not audio_bytes:
        await logging_utils.registrar_error("TRANSCRIPCION_VOZ", "No se pudo descargar el archivo de Telegram", file_id)
        return None

    from app.config import settings as cfg
    claves = cfg.obtener_pool_claves_gemini()
    if not claves:
        return None

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    payload = {
        "contents": [{
            "parts": [
                {"text": "Transcribe este audio al español. Responde ÚNICAMENTE con el texto transcrito, sin comillas ni comentarios adicionales."},
                {"inline_data": {"mime_type": "audio/ogg", "data": audio_b64}}
            ]
        }]
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        for clave in claves:
            try:
                resp = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                    headers={"x-goog-api-key": clave},
                    json=payload,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    candidatos = data.get("candidates", [])
                    if candidatos:
                        partes = candidatos[0].get("content", {}).get("parts", [])
                        if partes:
                            return partes[0].get("text", "").strip()
                elif resp.status_code == 429:
                    continue
            except Exception as e:
                logger.warning(f"[TRANSCRIPCION] Clave falló, probando la siguiente: {e}")
                continue

    await logging_utils.registrar_error("TRANSCRIPCION_VOZ", "Todas las claves fallaron al transcribir", file_id)
    return None
