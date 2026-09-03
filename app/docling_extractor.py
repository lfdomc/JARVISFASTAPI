"""
Extracción de PDF con Docling — motor de layout con reconocimiento real
de tablas y estructura, en vez de solo texto plano.

DISEÑO DE SEGURIDAD (importante): Docling necesita descargar modelos de
Hugging Face en su primer uso. Si esa descarga falla por cualquier razón
(red bloqueada, timeout, lo que sea), esta función devuelve None — el
llamador (documents.py) cae de vuelta al extractor de pypdf que ya
funciona hoy. La subida de un documento NUNCA debe romperse por esto.

Docling exporta cada página como Markdown, tablas incluidas en formato
de tabla Markdown real — se reutiliza TAL CUAL toda la maquinaria de
chunking.py que ya construimos (seguimiento continuo de página, unión de
palabras cortadas por guion, preservación de tablas como unidad) sin
tocarla, porque ya sabe reconocer y proteger tablas en Markdown.

DESCRIPCIÓN DE IMÁGENES (Picture Description): cuando Docling encuentra
un gráfico o infografía que no se puede leer como texto (los glifos
ilegibles /gid0013X que vimos en el plan de turismo), en vez de dejarlo
en blanco o con texto roto, le manda la imagen renderizada a Gemini por
su endpoint compatible con OpenAI y pide una descripción en español —
esa descripción queda como texto real, buscable y citable en el
fragmento. Usa UNA sola clave del pool existente (no la rotación
completa) porque esto corre poco — solo cuando hay imágenes — y a
diferencia de los embeddings, no se espera volumen alto por documento.
"""
import asyncio
import logging
import os
import tempfile
from app.config import settings

logger = logging.getLogger("docling_extractor")

TIMEOUT_SEGUNDOS = 600  # antes 180 — ahora que la extracción corre en segundo
# plano (ya no bloquea la petición HTTP del usuario), tiene sentido darle a
# Docling más margen para terminar documentos grandes en vez de rendirse
# rápido y caer a pypdf de forma innecesaria.
MODELO_DESCRIPCION_IMAGENES = "gemini-3.6-flash"  # gemini-2.5-flash quedó
# descontinuado — confirmado hoy con el error real de la API: "This model
# models/gemini-2.5-flash is no longer available to new users."
PROMPT_DESCRIPCION_IMAGENES = (
    "Describe este gráfico o imagen en 2-3 oraciones, en español. Si es un "
    "gráfico de datos (barras, líneas, torta), menciona las cifras y "
    "tendencias visibles con la mayor precisión posible. Si es una foto o "
    "ilustración, describe lo que muestra de forma concisa."
)


def _configurar_pipeline_con_descripcion_imagenes():
    """Arma las opciones de Docling con Picture Description apuntando al
    endpoint de Gemini compatible con OpenAI. Si no hay ninguna clave de
    Gemini configurada, se desactiva esta parte sin fallar — Docling
    sigue funcionando normal, solo sin describir imágenes."""
    from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionApiOptions

    opciones = PdfPipelineOptions()

    claves = settings.obtener_pool_claves_gemini()
    if claves:
        opciones.do_picture_description = True
        opciones.generate_picture_images = True  # necesario para tener la imagen que enviar
        opciones.enable_remote_services = True  # Docling bloquea llamadas a servicios
        # externos por seguridad a menos que se active explícitamente — sin esto,
        # la conversión entera fallaba de inmediato (OperationNotAllowed) y caía
        # al respaldo de pypdf, sin que ni siquiera el resto de Docling llegara
        # a correr. Confirmado con el log real: "Connections to remote services
        # is only allowed when set explicitly."
        opciones.picture_description_options = PictureDescriptionApiOptions(
            url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers={"Authorization": f"Bearer {claves[0]}"},
            params={"model": MODELO_DESCRIPCION_IMAGENES},
            prompt=PROMPT_DESCRIPCION_IMAGENES,
            timeout=45.0,
            picture_area_threshold=0.01,  # antes 0.05 (5%) por defecto — muy
            # alto para gráficos incrustados en una página con texto (no
            # ocupan página completa); con 1% se capturan también esos.
        )
    else:
        logger.warning("[DOCLING] Sin claves de Gemini configuradas — se omite la descripción de imágenes.")

    return opciones


def _convertir_sincrono(ruta_archivo: str) -> dict | None:
    """Corre la conversión real — es bloqueante y puede tardar, por eso
    se llama desde un hilo aparte (ver extraer_paginas_con_docling).

    Devuelve {"paginas": list[str], "encabezados": list[dict]} o None si
    falla. Los "encabezados" son un hallazgo extra que Docling permite y
    que antes no teníamos de ninguna forma: reconoce títulos y encabezados
    de sección REALES de la estructura del documento — no de un índice
    impreso que puede no existir. Esto significa que un documento SIN
    tabla de contenidos puede de todas formas obtener columna 'seccion'
    poblada, algo imposible con el detector de índice basado en regex."""
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling_core.types.doc import SectionHeaderItem, TitleItem, PictureItem

        opciones = _configurar_pipeline_con_descripcion_imagenes()
        conversor = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opciones)}
        )
        resultado = conversor.convert(ruta_archivo)
        documento = resultado.document

        total_paginas = documento.num_pages()
        if not total_paginas:
            return None

        paginas = []
        for n in range(1, total_paginas + 1):
            texto_pagina = documento.export_to_markdown(page_no=n) or ""
            paginas.append(texto_pagina)

        # Diagnóstico explícito de la descripción de imágenes — para
        # confirmar en el log de Railway, sin adivinar, cuántas imágenes
        # se detectaron y a cuántas se les generó descripción real.
        imagenes_totales = 0
        imagenes_con_descripcion = 0
        for item, _nivel in documento.iterate_items():
            if isinstance(item, PictureItem):
                imagenes_totales += 1
                if item.annotations:
                    imagenes_con_descripcion += 1
        logger.info(f"[DOCLING] Imágenes detectadas: {imagenes_totales}, con descripción generada: {imagenes_con_descripcion}")

        encabezados = []
        for item, _nivel_arbol in documento.iterate_items():
            if isinstance(item, (SectionHeaderItem, TitleItem)):
                pagina = item.prov[0].page_no if item.prov else None
                if pagina and item.text.strip():
                    encabezados.append({
                        "titulo": item.text.strip(),
                        "pagina": pagina,
                        "nivel": getattr(item, "level", 1) or 1,
                        "numero": None,  # Docling no numera automáticamente, solo da el texto real
                    })

        return {"paginas": paginas, "encabezados": encabezados}
    except Exception as e:
        logger.warning(f"[DOCLING] Falló la extracción ({type(e).__name__}: {e}) — se usará el respaldo de pypdf.")
        return None


async def extraer_paginas_con_docling(contenido_bytes: bytes) -> dict | None:
    """
    Punto de entrada — recibe los bytes del PDF (igual que el extractor
    de pypdf), intenta convertir con Docling en un hilo aparte (para no
    congelar el event loop de FastAPI mientras corre el modelo de
    layout), con límite de tiempo. Si algo sale mal, o se agota el
    tiempo, devuelve None — el llamador debe caer al respaldo de pypdf.

    Devuelve {"paginas": list[str], "encabezados": list[dict]} en éxito.
    """
    archivo_temporal = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contenido_bytes)
            archivo_temporal = tmp.name

        return await asyncio.wait_for(
            asyncio.to_thread(_convertir_sincrono, archivo_temporal),
            timeout=TIMEOUT_SEGUNDOS,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[DOCLING] Tiempo de espera agotado ({TIMEOUT_SEGUNDOS}s) — se usará el respaldo de pypdf.")
        return None
    except Exception as e:
        logger.warning(f"[DOCLING] Excepción inesperada ({type(e).__name__}: {e}) — se usará el respaldo de pypdf.")
        return None
    finally:
        if archivo_temporal and os.path.exists(archivo_temporal):
            os.remove(archivo_temporal)


async def extraer_paginas_con_docling(contenido_bytes: bytes) -> dict | None:
    """
    Punto de entrada — recibe los bytes del PDF (igual que el extractor
    de pypdf), intenta convertir con Docling en un hilo aparte (para no
    congelar el event loop de FastAPI mientras corre el modelo de
    layout), con límite de tiempo. Si algo sale mal, o se agota el
    tiempo, devuelve None — el llamador debe caer al respaldo de pypdf.

    Devuelve {"paginas": list[str], "encabezados": list[dict]} en éxito.
    """
    archivo_temporal = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contenido_bytes)
            archivo_temporal = tmp.name

        return await asyncio.wait_for(
            asyncio.to_thread(_convertir_sincrono, archivo_temporal),
            timeout=TIMEOUT_SEGUNDOS,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[DOCLING] Tiempo de espera agotado ({TIMEOUT_SEGUNDOS}s) — se usará el respaldo de pypdf.")
        return None
    except Exception as e:
        logger.warning(f"[DOCLING] Excepción inesperada ({type(e).__name__}: {e}) — se usará el respaldo de pypdf.")
        return None
    finally:
        if archivo_temporal and os.path.exists(archivo_temporal):
            os.remove(archivo_temporal)
