import httpx
from fastapi import APIRouter, HTTPException
from app.config import settings

router = APIRouter(prefix="/api/stats", tags=["stats"])


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


async def _contar(client: httpx.AsyncClient, tabla: str, filtro: str = "") -> int:
    url = f"{_base()}/rest/v1/{tabla}?select=id{filtro}"
    resp = await client.get(url, headers=_headers({"Prefer": "count=exact", "Range": "0-0"}))
    content_range = resp.headers.get("content-range", "")
    total = content_range.split("/")[-1] if "/" in content_range else "0"
    return int(total) if total.isdigit() else 0


@router.get("")
async def obtener_estadisticas():
    async with httpx.AsyncClient(timeout=20.0) as client:
        # Conteos generales
        total_documentos = await _contar(client, "documentos_gdp", "&estado=eq.ACTIVE")
        total_fragmentos = await _contar(client, "fragmentos_vectoriales_gdp")
        total_notas = await _contar(client, "fragmentos_vectoriales_gdp", "&documento_id=is.null")

        # Categorías (trae solo la columna categoria y cuenta en memoria —
        # simple y suficiente mientras el volumen sea moderado)
        resp_cat = await client.get(
            f"{_base()}/rest/v1/fragmentos_vectoriales_gdp?select=categoria&limit=5000",
            headers=_headers(),
        )
        categorias = {}
        if resp_cat.status_code == 200:
            for fila in resp_cat.json():
                cat = fila.get("categoria") or "SIN_CATEGORIA"
                categorias[cat] = categorias.get(cat, 0) + 1
        categorias_lista = sorted(
            [{"categoria": k, "total": v} for k, v in categorias.items()],
            key=lambda x: x["total"], reverse=True
        )

        # Cola de procesamiento
        resp_cola = await client.get(
            f"{_base()}/rest/v1/cola_procesamiento?select=estado&limit=5000",
            headers=_headers(),
        )
        cola = {}
        if resp_cola.status_code == 200:
            for fila in resp_cola.json():
                est = fila.get("estado") or "DESCONOCIDO"
                cola[est] = cola.get(est, 0) + 1
        cola_lista = [{"estado": k, "total": v} for k, v in cola.items()]

        # Errores abiertos recientes
        resp_errores = await client.get(
            f"{_base()}/rest/v1/capa_logs?estado=eq.OPEN&select=agente_origen,descripcion_falla,timestamp_utc&order=timestamp_utc.desc&limit=10",
            headers=_headers(),
        )
        errores = resp_errores.json() if resp_errores.status_code == 200 else []

        return {
            "total_documentos": total_documentos,
            "total_fragmentos": total_fragmentos,
            "total_notas_personales": total_notas,
            "total_documento_real": total_fragmentos - total_notas,
            "categorias": categorias_lista,
            "cola_procesamiento": cola_lista,
            "errores_recientes": errores,
        }
