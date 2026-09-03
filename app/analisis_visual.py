"""
Análisis visual de páginas con poco texto extraíble.

Reemplaza el enfoque anterior (marcar como "vacía" cualquier página bajo
un umbral de caracteres) por uno más certero: cuando una página tiene
poco texto, se RENDERIZA como imagen y se le pregunta a Gemini qué hay
ahí — en vez de adivinar por conteo de caracteres.

Por qué el umbral solo no alcanzaba (confirmado con pruebas reales):
- FALSOS POSITIVOS: "Capítulo 3" o "Anexos" (títulos reales y cortos)
  se marcaban como vacíos — se perdía texto real y útil.
- FALSOS NEGATIVOS: una página que es enteramente un gráfico, pero
  tiene un pie de página repetido tipo "Plan nacional de turismo de
  Costa Rica.147" (~40 caracteres), NO se marcaba — se quedaba sin
  describir aunque fuera prácticamente toda imagen.

Este módulo corrige ambos: nunca descarta texto real (se preserva y se
complementa), y decide con evidencia visual, no con un número mágico.
"""
import base64
import logging
import pypdfium2 as pdfium

from app.config import settings
from app import gemini_client

logger = logging.getLogger("analisis_visual")

# Páginas con MENOS texto que esto se analizan visualmente — más alto
# que el umbral anterior (25) a propósito: así también se capturan
# páginas que solo tienen un pie de página repetido pero son
# mayormente una imagen o gráfico.
UMBRAL_ANALISIS_VISUAL = 150

PROMPT_ANALISIS_VISUAL = (
    "Esta es una página de un documento con poco texto extraíble automáticamente. "
    "Responde en español, en 2-4 oraciones:\n"
    "- Si la página está genuinamente en blanco o casi vacía (sin gráficos, tablas ni "
    "imágenes relevantes), dilo explícitamente: 'Página en blanco.'\n"
    "- Si tiene un gráfico, tabla, infografía o imagen, descríbelo con la mayor precisión "
    "posible, incluyendo cifras y tendencias visibles si las hay.\n"
    "- Si es una portada o divisor de sección con solo un título grande, transcribe el "
    "título tal cual aparece."
)


def _renderizar_pagina_como_png(contenido_pdf: bytes, numero_pagina: int) -> bytes | None:
    """numero_pagina es 1-indexado. Devuelve PNG en bytes, o None si falla."""
    try:
        documento = pdfium.PdfDocument(contenido_pdf)
        if numero_pagina < 1 or numero_pagina > len(documento):
            return None
        pagina = documento[numero_pagina - 1]
        bitmap = pagina.render(scale=1.5)
        imagen_pil = bitmap.to_pil()
        import io
        buffer = io.BytesIO()
        imagen_pil.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"No se pudo renderizar la página {numero_pagina}: {e}")
        return None


async def _analizar_imagen_con_gemini(imagen_png: bytes) -> str | None:
    """Reutiliza gemini_client.generar_respuesta (mismo pool de claves,
    mismo reintento entre modelos) — solo se arma el content con la
    imagen en base64 en vez de solo texto."""
    if not settings.obtener_pool_claves_gemini():
        return None
    imagen_base64 = base64.b64encode(imagen_png).decode("utf-8")
    contents = [{
        "role": "user",
        "parts": [
            {"inline_data": {"mime_type": "image/png", "data": imagen_base64}},
            {"text": PROMPT_ANALISIS_VISUAL},
        ],
    }]
    try:
        return await gemini_client.generar_respuesta(contents, temperatura=0.1)
    except Exception as e:
        logger.warning(f"No se pudo analizar la imagen con Gemini: {e}")
        return None


async def analizar_paginas_con_poco_texto(contenido_pdf: bytes, paginas: list[str]) -> list[str]:
    """
    Punto de entrada. Para cada página con menos de
    UMBRAL_ANALISIS_VISUAL caracteres, la renderiza y le pide a Gemini
    que la describa — el texto real que ya hubiera (por corto que sea)
    NUNCA se descarta, solo se complementa con la descripción visual si
    Gemini encuentra algo. Si el análisis falla por cualquier razón
    (sin claves, error de red, lo que sea), esa página simplemente se
    queda con su texto original tal cual — nunca rompe la ingesta.
    """
    resultado = list(paginas)
    total_analizadas = 0
    total_con_descripcion = 0

    for i, texto in enumerate(paginas):
        texto_limpio = texto.strip()
        if len(texto_limpio) >= UMBRAL_ANALISIS_VISUAL:
            continue

        imagen = _renderizar_pagina_como_png(contenido_pdf, i + 1)
        if not imagen:
            continue

        total_analizadas += 1
        descripcion = await _analizar_imagen_con_gemini(imagen)
        if not descripcion:
            continue

        descripcion = descripcion.strip()
        if "página en blanco" in descripcion.lower() and not texto_limpio:
            # Genuinamente vacía y confirmada por Gemini — se deja constancia,
            # no se descarta a un texto vacío silencioso.
            resultado[i] = f"[Página {i + 1}: en blanco, sin contenido — confirmado por análisis visual.]\n"
        else:
            total_con_descripcion += 1
            # El texto real (por corto que sea) se PRESERVA siempre —
            # solo se complementa con lo que Gemini vio.
            resultado[i] = (
                (texto_limpio + "\n\n" if texto_limpio else "") +
                f"[Contenido visual de la página {i + 1}: {descripcion}]\n"
            )

    if total_analizadas:
        logger.info(f"[ANÁLISIS VISUAL] {total_analizadas} página(s) con poco texto analizadas, {total_con_descripcion} con descripción real generada.")

    return resultado
