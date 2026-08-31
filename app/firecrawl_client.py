import re
import httpx
from app.config import settings
from app import logging_utils

PATRONES_CORTE = [
    r"\n#{1,3}\s*referencias\b",
    r"\n#{1,3}\s*enlaces externos\b",
    r"\n#{1,3}\s*v[eé]ase tambi[eé]n\b",
    r"\n#{1,3}\s*bibliograf[ií]a\b",
    r"\n#{1,3}\s*notas\b",
    r"\n#{1,3}\s*references\b",
    r"\n#{1,3}\s*external links\b",
    r"\n#{1,3}\s*see also\b",
]
MARCADORES_INICIO = ["De Wikipedia, la enciclopedia libre", "From Wikipedia, the free encyclopedia"]


def _limpiar_secciones_bajo_valor(markdown: str) -> str:
    texto = markdown
    for marcador in MARCADORES_INICIO:
        idx = texto.find(marcador)
        if idx > 0:
            texto = texto[idx:]
            break

    indice_corte = len(texto)
    for patron in PATRONES_CORTE:
        m = re.search(patron, texto, re.IGNORECASE)
        if m and m.start() < indice_corte:
            indice_corte = m.start()

    return texto[:indice_corte].strip()


async def scrapear(url: str) -> dict | None:
    if not settings.FIRECRAWL_API_KEY:
        await logging_utils.registrar_error("FIRECRAWL_SCRAPE", "FIRECRAWL_API_KEY no configurada", "N/A")
        return None

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}"},
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
            if resp.status_code != 200:
                await logging_utils.registrar_error("FIRECRAWL_SCRAPE", f"HTTP {resp.status_code}", resp.text)
                return None

            data = resp.json()
            markdown = data.get("data", {}).get("markdown")
            titulo = data.get("data", {}).get("metadata", {}).get("title") or url

            if not markdown or not markdown.strip():
                return None

            return {"markdown": _limpiar_secciones_bajo_valor(markdown), "titulo": titulo}
        except Exception as e:
            await logging_utils.registrar_error("FIRECRAWL_SCRAPE", str(e), "Excepción de conexión")
            return None
