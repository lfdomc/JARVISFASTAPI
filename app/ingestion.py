import hashlib
import httpx
from app.config import settings
from app import logging_utils

UMBRAL_RELEVANCIA_FRAGMENTO = 0.35
UMBRAL_COHERENCIA_CATEGORIA = 0.45


def _headers(extra: dict | None = None) -> dict:
    h = {
        "apikey": settings.SUPABASE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def _base() -> str:
    return settings.SUPABASE_URL.rstrip("/")


def similitud_coseno(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def buscar_documento_duplicado(hash_contenido: str) -> dict | None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{_base()}/rest/v1/documentos_gdp?hash_sha256=eq.{hash_contenido}&estado=eq.ACTIVE&select=id,nombre_archivo",
            headers=_headers()
        )
        if resp.status_code == 200:
            filas = resp.json()
            return filas[0] if filas else None
    return None


async def crear_documento_maestro(nombre_archivo: str, contenido_markdown: str, categoria: str, mime_type: str, creado_por: str = "sistema") -> str | None:
    hash_contenido = hashlib.sha256(contenido_markdown.encode("utf-8")).hexdigest()
    payload = {
        "categoria": categoria,
        "estandar_interop": "INTERNAL",
        "pii_sensitivity_level": "S1",
        "pii_lifecycle_state": "ACTIVE",
        "nombre_archivo": nombre_archivo,
        "mime_type": mime_type,
        "contenido_markdown": contenido_markdown,
        "estado": "ACTIVE",
        "hash_sha256": hash_contenido,
        "creado_por": creado_por,
        "version_major": 1,
        "version_minor": 0,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url=f"{_base()}/rest/v1/documentos_gdp", json=payload, headers=_headers({"Prefer": "return=representation"}))
        if resp.status_code in (200, 201):
            datos = resp.json()
            if datos:
                return datos[0]["id"]
        await logging_utils.registrar_error("CREAR_DOCUMENTO_MAESTRO", f"HTTP {resp.status_code}", resp.text)
    return None


async def guardar_fragmento(texto: str, categoria: str, embedding: list[float], documento_id: str | None, metadata: dict, propietario_telegram_id: str | None = None) -> bool:
    payload = [{
        "documento_id": documento_id,
        "categoria": categoria,
        "contenido_chunk": texto,
        "embedding": embedding,
        "metadata": metadata,
        "propietario_telegram_id": propietario_telegram_id,
    }]
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(f"{_base()}/rest/v1/fragmentos_vectoriales_gdp", json=payload, headers=_headers({"Prefer": "return=minimal"}))
        if resp.status_code in (201, 204):
            return True
        await logging_utils.registrar_error("GUARDAR_FRAGMENTO", f"HTTP {resp.status_code}", resp.text)
    return False


async def verificar_coherencia_categoria(embedding: list[float], categoria: str) -> tuple[float, int]:
    """Devuelve (similitud_promedio, fragmentos_comparados)."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{_base()}/rest/v1/rpc/verificar_coherencia_categoria",
            json={"query_embedding": embedding, "p_categoria": categoria},
            headers=_headers()
        )
        if resp.status_code == 200:
            datos = resp.json()
            if datos:
                return datos[0].get("similitud_promedio") or 0.0, datos[0].get("fragmentos_comparados") or 0
    return 0.0, 0


async def buscar_nota_similar(embedding: list[float], umbral: float = 0.85) -> dict | None:
    """Reutiliza buscar_nota_similar (misma RPC que Apps Script) para detectar duplicados de notas personales."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{_base()}/rest/v1/rpc/buscar_nota_similar",
            json={"query_embedding": embedding, "match_threshold": umbral},
            headers=_headers()
        )
        if resp.status_code == 200:
            datos = resp.json()
            if datos:
                return datos[0]
    return None


async def sugerir_categoria_similar(embedding: list[float]) -> str | None:
    """Reutiliza sugerir_categorias_similares (misma RPC que Apps Script)."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{_base()}/rest/v1/rpc/sugerir_categorias_similares",
            json={"query_embedding": embedding, "cantidad": 3},
            headers=_headers()
        )
        if resp.status_code == 200:
            datos = resp.json()
            if datos and datos[0].get("categoria_sugerida"):
                return datos[0]["categoria_sugerida"]
    return None
