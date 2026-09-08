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

AISLAMIENTO DE MEMORIA (agregado hoy): el renderizado con pypdfium2 —
una librería nativa, igual que PyTorch en Docling — corre en un PROCESO
HIJO aparte, no en el proceso principal de FastAPI. Se confirmó con
datos reales de uso de memoria en Railway que, sin este aislamiento, la
memoria subía con cada documento procesado y nunca volvía a bajar — el
mismo patrón que ya habíamos resuelto para Docling, pero que se nos
había quedado sin cubrir aquí. Se renderizan TODAS las páginas
necesarias en UNA sola llamada al proceso hijo (no una por página) para
no pagar el costo de arrancar un proceso nuevo por cada página.
"""
import base64
import logging

from app.config import settings
from app import gemini_client

logger = logging.getLogger("analisis_visual")

# Páginas con MENOS texto que esto se analizan visualmente — más alto
# que el umbral anterior (25) a propósito: así también se capturan
# páginas que solo tienen un pie de página repetido pero son
# mayormente una imagen o gráfico.
UMBRAL_ANALISIS_VISUAL = 150
TIMEOUT_RENDERIZADO_SEGUNDOS = 120  # renderizar imágenes es rápido — no
# necesita el margen amplio que le dimos a Docling (que hace layout/OCR/tablas)

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


def _proceso_hijo_renderizado(contenido_pdf: bytes, numeros_pagina: list[int], cola: "multiprocessing.Queue"):
    """
    Función objetivo del proceso hijo — abre el PDF UNA vez con
    pypdfium2 y renderiza TODAS las páginas pedidas, todo dentro de este
    proceso aparte. Cuando termina y el proceso se cierra, el sistema
    operativo recupera toda la memoria que pypdfium2 haya reservado —
    igual que ya hacemos con Docling.
    """
    import io
    import pypdfium2 as pdfium

    resultado: dict[int, bytes] = {}
    try:
        documento = pdfium.PdfDocument(contenido_pdf)
        total_paginas_doc = len(documento)
        for numero_pagina in numeros_pagina:
            if numero_pagina < 1 or numero_pagina > total_paginas_doc:
                continue
            try:
                pagina = documento[numero_pagina - 1]
                bitmap = pagina.render(scale=1.5)
                imagen_pil = bitmap.to_pil()
                buffer = io.BytesIO()
                imagen_pil.save(buffer, format="PNG")
                resultado[numero_pagina] = buffer.getvalue()
            except Exception as e:
                cola.put(("advertencia", f"No se pudo renderizar la página {numero_pagina}: {e}"))
        cola.put(("ok", resultado))
    except Exception as e:
        cola.put(("error", f"{type(e).__name__}: {e}"))


async def _renderizar_paginas_en_proceso_hijo(contenido_pdf: bytes, numeros_pagina: list[int]) -> dict[int, bytes]:
    """Lanza el proceso hijo, espera su resultado sin congelar el event
    loop (igual patrón que docling_extractor.py), y siempre limpia el
    proceso al final. Si algo falla, devuelve un diccionario vacío — el
    llamador simplemente se queda sin descripciones visuales para esas
    páginas, nunca rompe la ingesta."""
    import asyncio
    import multiprocessing

    if not numeros_pagina:
        return {}

    cola = multiprocessing.Queue()
    proceso = multiprocessing.Process(target=_proceso_hijo_renderizado, args=(contenido_pdf, numeros_pagina, cola))
    proceso.start()

    def _esperar():
        mensajes_finales = []
        # Puede haber varias advertencias en la cola antes del resultado final ("ok"/"error")
        while True:
            proceso.join(timeout=TIMEOUT_RENDERIZADO_SEGUNDOS)
            if not proceso.is_alive() and cola.empty():
                break
            if cola.empty():
                if proceso.is_alive():
                    return ("timeout", None)
                break
            estado, valor = cola.get()
            if estado == "advertencia":
                mensajes_finales.append(valor)
                continue
            return (estado, valor)
        return ("error", "El proceso hijo terminó sin devolver ningún resultado.")

    try:
        estado, valor = await asyncio.to_thread(_esperar)
        if estado == "timeout":
            logger.warning(f"[ANÁLISIS VISUAL] Tiempo de espera agotado renderizando páginas — se cancela el proceso hijo.")
            proceso.terminate()
            proceso.join(timeout=10)
            return {}
        if estado == "error":
            logger.warning(f"[ANÁLISIS VISUAL] El proceso hijo de renderizado falló ({valor}).")
            return {}
        return valor or {}
    finally:
        if proceso.is_alive():
            proceso.terminate()


async def _analizar_imagen_con_gemini(imagen_png: bytes) -> str | None:
    """Reutiliza gemini_client.generar_respuesta (mismo pool de claves,
    mismo reintento entre modelos) — solo se arma el content con la
    imagen en base64 en vez de solo texto. Corre en el proceso principal
    a propósito: es solo una llamada de red (httpx), sin ninguna
    librería nativa pesada de por medio."""
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
    UMBRAL_ANALISIS_VISUAL caracteres, la renderiza (en un proceso hijo
    aparte — ver arriba) y le pide a Gemini que la describa — el texto
    real que ya hubiera (por corto que sea) NUNCA se descarta, solo se
    complementa con la descripción visual si Gemini encuentra algo. Si
    el análisis falla por cualquier razón (sin claves, error de red, lo
    que sea), esa página simplemente se queda con su texto original tal
    cual — nunca rompe la ingesta.
    """
    resultado = list(paginas)

    paginas_a_analizar = [
        i + 1 for i, texto in enumerate(paginas)
        if len(texto.strip()) < UMBRAL_ANALISIS_VISUAL
    ]
    if not paginas_a_analizar:
        return resultado

    imagenes = await _renderizar_paginas_en_proceso_hijo(contenido_pdf, paginas_a_analizar)

    total_analizadas = 0
    total_con_descripcion = 0

    for numero_pagina in paginas_a_analizar:
        i = numero_pagina - 1
        imagen = imagenes.get(numero_pagina)
        if not imagen:
            continue

        texto_limpio = paginas[i].strip()
        total_analizadas += 1
        descripcion = await _analizar_imagen_con_gemini(imagen)
        if not descripcion:
            continue

        descripcion = descripcion.strip()
        if "página en blanco" in descripcion.lower() and not texto_limpio:
            # Genuinamente vacía y confirmada por Gemini — se deja constancia,
            # no se descarta a un texto vacío silencioso.
            resultado[i] = f"[Página {numero_pagina}: en blanco, sin contenido — confirmado por análisis visual.]\n"
        else:
            total_con_descripcion += 1
            # El texto real (por corto que sea) se PRESERVA siempre —
            # solo se complementa con lo que Gemini vio.
            resultado[i] = (
                (texto_limpio + "\n\n" if texto_limpio else "") +
                f"[Contenido visual de la página {numero_pagina}: {descripcion}]\n"
            )

    if total_analizadas:
        logger.info(f"[ANÁLISIS VISUAL] {total_analizadas} página(s) con poco texto analizadas, {total_con_descripcion} con descripción real generada.")

    return resultado
