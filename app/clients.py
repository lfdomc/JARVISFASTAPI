from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx
from app.config import settings

router = APIRouter(prefix="/api/clients", tags=["clients"])


def _headers() -> dict:
    return {
        "apikey": settings.SUPABASE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


class ClienteCrear(BaseModel):
    nombre: str
    supabase_url: str | None = None
    supabase_key: str | None = None


@router.get("")
async def listar_clientes():
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/clientes?activo=eq.true&order=creado_en.asc"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=_headers())
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=resp.text)
        return resp.json()


@router.post("")
async def crear_cliente(cliente: ClienteCrear):
    """
    AVISO: por ahora, crear un cliente nuevo solo lo agrega a la lista —
    el backend sigue usando SIEMPRE las credenciales de SUPABASE_URL/KEY
    del entorno (tu propia base) para todas las operaciones, sin importar
    qué cliente esté "seleccionado". El enrutamiento dinámico por cliente
    (usar la Supabase de CADA cliente real) es el siguiente paso, cuando
    tengas un segundo cliente de verdad para probarlo con seguridad.
    """
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/clientes"
    payload = [cliente.model_dump()]
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            url, json=payload,
            headers={**_headers(), "Prefer": "return=representation"}
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=502, detail=resp.text)
        return resp.json()
