"""
Estado en memoria del proceso. A diferencia de Apps Script (que necesita
CacheService porque cada ejecución es efímera), aquí el servidor de
FastAPI es un proceso continuo — un diccionario en memoria persiste
mientras el proceso siga corriendo, sin necesitar un servicio externo.

Limitación honesta: si Railway reinicia el proceso (nuevo despliegue,
caída), este estado se pierde. Para producción real con varios workers o
alta disponibilidad, esto debería migrar a Redis — por ahora, para un
solo proceso, es suficiente y más simple.
"""
import time
import uuid

TTL_IDEMPOTENCIA_SEG = 600
TTL_CATEGORIA_PENDIENTE_SEG = 600
TTL_ESPERANDO_CATEGORIA_SEG = 300
TTL_CACHE_FAQ_SEG = 10800
TTL_CACHE_CATEGORIAS_SEG = 1800  # 30 min — las categorías cambian poco, no hace falta consultarlas en cada mensaje

_updates_procesados: dict[int, float] = {}
_categorias_pendientes: dict[str, dict] = {}
_esperando_categoria_nueva: dict[str, str] = {}  # chat_id -> token
_cache_faq: dict[str, tuple[str, float]] = {}
_cache_pares_categoria: tuple[list[dict], float] | None = None


def _limpiar_vencidos(store: dict, ttl: int, ahora: float):
    vencidos = [k for k, v in store.items() if (ahora - (v[1] if isinstance(v, tuple) else v)) > ttl]
    for k in vencidos:
        del store[k]


def ya_procesado(update_id: int) -> bool:
    ahora = time.time()
    _limpiar_vencidos(_updates_procesados, TTL_IDEMPOTENCIA_SEG, ahora)
    if update_id in _updates_procesados:
        return True
    _updates_procesados[update_id] = ahora
    return False


def guardar_categoria_pendiente(datos: dict) -> str:
    token = uuid.uuid4().hex[:16]
    _categorias_pendientes[token] = {"datos": datos, "ts": time.time()}
    return token


def obtener_categoria_pendiente(token: str) -> dict | None:
    entrada = _categorias_pendientes.pop(token, None)
    if not entrada:
        return None
    if (time.time() - entrada["ts"]) > TTL_CATEGORIA_PENDIENTE_SEG:
        return None
    return entrada["datos"]


def marcar_esperando_categoria_nueva(chat_id: str, token: str):
    _esperando_categoria_nueva[chat_id] = token


def tomar_espera_categoria_nueva(chat_id: str) -> str | None:
    return _esperando_categoria_nueva.pop(chat_id, None)


def cache_faq_get(clave: str) -> str | None:
    entrada = _cache_faq.get(clave)
    if not entrada:
        return None
    valor, ts = entrada
    if (time.time() - ts) > TTL_CACHE_FAQ_SEG:
        del _cache_faq[clave]
        return None
    return valor


def cache_faq_set(clave: str, valor: str):
    _cache_faq[clave] = (valor, time.time())


def cache_pares_categoria_get() -> list[dict] | None:
    global _cache_pares_categoria
    if not _cache_pares_categoria:
        return None
    datos, ts = _cache_pares_categoria
    if (time.time() - ts) > TTL_CACHE_CATEGORIAS_SEG:
        _cache_pares_categoria = None
        return None
    return datos


def cache_pares_categoria_set(datos: list[dict]):
    global _cache_pares_categoria
    _cache_pares_categoria = (datos, time.time())
