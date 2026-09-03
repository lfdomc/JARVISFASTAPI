"""
Cliente de Gemini con rotación de claves y failover de modelos.
Traducción directa a Python del patrón que ya probamos y calibramos hoy
en Apps Script (incluye la lección real: el límite gratis es de 1,000
embeddings/día POR cuenta — rotar entre varias cuentas multiplica la
cuota efectiva).
"""
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed, retry_if_exception_type
from app.config import settings
from app import logging_utils

logger = logging.getLogger("gemini_client")

MODELOS_GENERACION = [
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",  # antes gemini-2.5-flash — descontinuado, confirmado
    # hoy con el error real de la API (ver docling_extractor.py)
    "gemini-3.5-flash",
    "gemini-3.7-flash",
]


class TodasLasClavesSinCupo(Exception):
    """Las 8 claves respondieron 429 (sin cuota) en el mismo ciclo — el
    límite por minuto se libera solo en segundos, así que reintentar con
    espera recupera fragmentos que antes se perdían en silencio."""
    pass


async def _intentar_embedding_con_todas_las_claves(texto: str, claves: list[str]) -> list[float] | None:
    """Un solo ciclo por las 8 claves. Si TODAS devuelven 429, lanza
    TodasLasClavesSinCupo para que @retry decida si reintentar el ciclo
    completo con espera."""
    todas_sin_cupo = True
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
                    todas_sin_cupo = False
                elif resp.status_code == 429:
                    logger.warning(f"[EMBEDDING] Clave #{i+1}/{len(claves)} sin cuota (429), probando la siguiente...")
                    continue
                else:
                    logger.warning(f"[EMBEDDING] Clave #{i+1} HTTP {resp.status_code}: {resp.text}")
                    todas_sin_cupo = False
            except Exception as e:
                logger.warning(f"[EMBEDDING] Clave #{i+1} excepción: {e}")
                todas_sin_cupo = False

    if todas_sin_cupo:
        raise TodasLasClavesSinCupo()
    return None


@retry(
    retry=retry_if_exception_type(TodasLasClavesSinCupo),
    # Mismas esperas de antes (5s, 15s, 30s) — solo que ahora es
    # configuración declarativa de tenacity, no un bucle escrito a mano.
    wait=wait_chain(wait_fixed(5), wait_fixed(15), wait_fixed(30)),
    stop=stop_after_attempt(4),
    reraise=False,
)
async def _generar_embedding_con_reintento(texto: str, claves: list[str]) -> list[float] | None:
    return await _intentar_embedding_con_todas_las_claves(texto, claves)


async def generar_embedding(texto: str) -> list[float] | None:
    claves = settings.obtener_pool_claves_gemini()
    if not claves:
        logger.error("No hay ninguna clave de Gemini configurada (GEMINI_KEY_*).")
        return None

    try:
        resultado = await _generar_embedding_con_reintento(texto, claves)
        if resultado:
            return resultado
    except TodasLasClavesSinCupo:
        pass  # se agotaron los reintentos — cae al registro de error de abajo

    logger.error("Todas las claves de Gemini fallaron o están sin cuota para embeddings.")
    await logging_utils.registrar_error(
        "GEMINI_API_EMBEDDING",
        "Todas las claves de Gemini fallaron o están sin cuota",
        "Ver logs de Railway para el detalle por clave",
        "Revisar cuotas en ai.dev/rate-limit o agregar más claves GEMINI_KEY_*"
    )
    return None


async def generar_respuesta(payload_contents: list[dict], usar_url_context: bool = False, system_instruction: str | None = None, temperatura: float = 0.2, response_schema: dict | None = None) -> str:
    claves = settings.obtener_pool_claves_gemini()
    if not claves:
        raise RuntimeError("No hay ninguna clave de Gemini configurada.")

    generation_config = {"temperature": temperatura}
    if response_schema:
        # Fuerza que la respuesta sea JSON con esta forma exacta — usado
        # en modo profundo para que cada afirmación declare
        # obligatoriamente de qué fragmento salió, en vez de confiar en
        # que el modelo escriba un marcador en el lugar correcto dentro
        # de texto libre.
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema

    payload = {
        "contents": payload_contents,
        "generationConfig": generation_config,
    }
    if system_instruction:
        # Las reglas de comportamiento van en el campo dedicado de Gemini
        # (no mezcladas dentro del texto de la pregunta) — el modelo está
        # entrenado para priorizar esto más que instrucciones dentro de
        # un turno normal de conversación. Mismos tokens, mejor organizados.
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
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
