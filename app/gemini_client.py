"""
Cliente de Gemini con rotación de claves y failover de modelos.
Traducción directa a Python del patrón que ya probamos y calibramos hoy
en Apps Script (incluye la lección real: el límite gratis es de 1,000
embeddings/día POR cuenta — rotar entre varias cuentas multiplica la
cuota efectiva).
"""
import httpx
import logging
from app.config import settings
from app import logging_utils

logger = logging.getLogger("gemini_client")

MODELOS_GENERACION = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
]


async def generar_embedding(texto: str) -> list[float] | None:
    claves = settings.obtener_pool_claves_gemini()
    if not claves:
        logger.error("No hay ninguna clave de Gemini configurada (GEMINI_KEY_*).")
        return None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, clave in enumerate(claves):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={clave}"
            payload = {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": texto}]},
                "outputDimensionality": 768,
            }
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    valores = data.get("embedding", {}).get("values")
                    if valores:
                        return valores
                elif resp.status_code == 429:
                    logger.warning(f"[EMBEDDING] Clave #{i+1}/{len(claves)} sin cuota (429), probando la siguiente...")
                    continue
                else:
                    logger.warning(f"[EMBEDDING] Clave #{i+1} HTTP {resp.status_code}: {resp.text}")
                    continue
            except Exception as e:
                logger.warning(f"[EMBEDDING] Clave #{i+1} excepción: {e}")
                continue

    logger.error("Todas las claves de Gemini fallaron o están sin cuota para embeddings.")
    await logging_utils.registrar_error(
        "GEMINI_API_EMBEDDING",
        "Todas las claves de Gemini fallaron o están sin cuota",
        "Ver logs de Railway para el detalle por clave",
        "Revisar cuotas en ai.dev/rate-limit o agregar más claves GEMINI_KEY_*"
    )
    return None


async def generar_respuesta(payload_contents: list[dict], usar_url_context: bool = False) -> str:
    claves = settings.obtener_pool_claves_gemini()
    if not claves:
        raise RuntimeError("No hay ninguna clave de Gemini configurada.")

    payload = {"contents": payload_contents}
    if usar_url_context:
        payload["tools"] = [{"url_context": {}}]

    ultimo_error = ""

    async with httpx.AsyncClient(timeout=60.0) as client:
        for k, clave in enumerate(claves):
            for modelo in MODELOS_GENERACION:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent"
                try:
                    resp = await client.post(url, json=payload, headers={"x-goog-api-key": clave})
                    if resp.status_code == 200:
                        data = resp.json()
                        candidatos = data.get("candidates", [])
                        if candidatos:
                            partes = candidatos[0].get("content", {}).get("parts", [])
                            if partes:
                                return partes[0].get("text", "")
                    elif resp.status_code == 429:
                        ultimo_error = f"[clave #{k+1}][{modelo}] sin cuota (429)"
                        logger.warning(ultimo_error)
                    else:
                        ultimo_error = f"[clave #{k+1}][{modelo}] HTTP {resp.status_code}: {resp.text}"
                except Exception as e:
                    ultimo_error = f"[clave #{k+1}][{modelo}] excepción: {e}"

    await logging_utils.registrar_error(
        "GEMINI_API_GENERACION",
        "Todos los modelos y todas las claves fallaron",
        ultimo_error,
        "Revisar cuotas en ai.dev/rate-limit o agregar más claves GEMINI_KEY_*"
    )
    raise RuntimeError(f"Todos los modelos y todas las claves fallaron. Último error: {ultimo_error}")
