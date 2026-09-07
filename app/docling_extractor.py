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
UMBRAL_PAGINAS_PARA_LOTES = 150  # documentos con más páginas que esto se
# procesan en lotes, no de una sola pasada — confirmado con un caso real:
# un manual de 545 páginas causó un crash del contenedor por falta de
# memoria (Out of Memory) procesándolo completo de una vez. 150 es el
# tamaño más grande que hemos confirmado funcionando bien en una sola
# pasada (el plan de turismo, 157 páginas, quedó justo en el límite).
PAGINAS_POR_LOTE = 80  # tamaño de cada lote — bastante por debajo del
# umbral de arriba, para dejar margen real de memoria libre por lote.
MARCADOR_LOTE_FALLIDO = "[Docling no pudo procesar esta página como parte de un lote — el análisis visual, si aplica, debería cubrir el hueco.]"
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
    """
    DESACTIVADO hoy (do_picture_description = False): confirmado con datos
    reales que esta función de Docling detectaba 349 "imágenes" en un
    documento (con el umbral bajo a propósito para no perderse gráficos
    pequeños) — la mayoría, elementos decorativos (íconos, viñetas, logos),
    no gráficos reales. Sumado a que gemini-3.6-flash tiene un límite
    gratuito muy estricto (5-20 peticiones por período), la extracción
    completa tardó 21 minutos solo esperando cuotas.

    Ya tenemos analisis_visual.py, que hace el mismo trabajo de forma
    mucho más controlada: solo analiza páginas con POCO TEXTO (no cada
    elemento visual que Docling detecta), y ya demostró que funciona bien
    de forma independiente, con un volumen de llamadas muy inferior y
    predecible. Esta función de Docling quedaba redundante y cara — se
    deja el código listo por si se quiere retomar más adelante (ej. con
    un umbral más alto y una cuota de pago), pero no corre por defecto.
    """
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opciones = PdfPipelineOptions()
    opciones.do_picture_description = False
    return opciones


def _convertir_un_archivo(conversor, ruta_archivo: str) -> dict | None:
    """
    Convierte UN archivo PDF (puede ser el documento completo, o solo un
    lote de páginas de uno más grande) con un DocumentConverter YA
    inicializado — reutilizar el mismo conversor entre lotes evita tener
    que recargar los modelos de layout/OCR/tablas en cada lote, que sería
    un desperdicio de tiempo real.
    """
    try:
        from docling_core.types.doc import SectionHeaderItem, TitleItem, PictureItem

        resultado = conversor.convert(ruta_archivo)
        documento = resultado.document

        total_paginas = documento.num_pages()
        if not total_paginas:
            return None

        paginas = []
        for n in range(1, total_paginas + 1):
            texto_pagina = documento.export_to_markdown(page_no=n) or ""
            paginas.append(texto_pagina)

        imagenes_totales = 0
        imagenes_con_descripcion = 0
        for item, _nivel in documento.iterate_items():
            if isinstance(item, PictureItem):
                imagenes_totales += 1
                if item.annotations:
                    imagenes_con_descripcion += 1

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

        return {
            "paginas": paginas, "encabezados": encabezados,
            "imagenes_totales": imagenes_totales, "imagenes_con_descripcion": imagenes_con_descripcion,
        }
    except Exception as e:
        logger.warning(f"[DOCLING] Falló convirtiendo un archivo/lote ({type(e).__name__}: {e})")
        return None


def _dividir_pdf_en_lotes(ruta_archivo: str, paginas_por_lote: int) -> list[tuple[str, int]]:
    """
    Divide un PDF grande en varios archivos temporales más pequeños, cada
    uno con como máximo paginas_por_lote páginas. Devuelve una lista de
    (ruta_del_lote, paginas_antes_de_este_lote) — el segundo valor es el
    "offset" real que hay que sumarle a cualquier número de página que
    Docling detecte DENTRO de ese lote, para que corresponda a la página
    real del documento completo (Docling, al procesar un lote como
    archivo independiente, siempre empieza a contar desde la página 1).
    """
    from pypdf import PdfReader, PdfWriter

    lector = PdfReader(ruta_archivo)
    total = len(lector.pages)
    lotes = []
    for inicio in range(0, total, paginas_por_lote):
        fin = min(inicio + paginas_por_lote, total)
        escritor = PdfWriter()
        for i in range(inicio, fin):
            escritor.add_page(lector.pages[i])
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_lote:
            escritor.write(tmp_lote)
            lotes.append((tmp_lote.name, inicio))
    return lotes


def _convertir_sincrono(ruta_archivo: str) -> dict | None:
    """
    Corre la conversión real — es bloqueante y puede tardar, por eso se
    llama desde un hilo aparte (ver extraer_paginas_con_docling).

    Documentos grandes (más de UMBRAL_PAGINAS_PARA_LOTES páginas) se
    procesan en LOTES, no de una sola pasada — confirmado con un caso
    real: un manual de 545 páginas hizo que el contenedor se quedara sin
    memoria (Out of Memory) y se cayera, procesándolo completo de una
    vez. Cada lote se convierte, se extraen sus páginas y encabezados
    (con la página corregida al número real del documento completo), y
    el resultado del lote en sí (con todos sus datos internos de Docling)
    se libera antes de pasar al siguiente — así el pico de memoria queda
    acotado al tamaño de UN lote, no del documento entero.

    Devuelve {"paginas": list[str], "encabezados": list[dict]} o None si
    falla. Los "encabezados" son un hallazgo extra que Docling permite y
    que antes no teníamos de ninguna forma: reconoce títulos y encabezados
    de sección REALES de la estructura del documento — no de un índice
    impreso que puede no existir.
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from pypdf import PdfReader

        total_paginas_documento = len(PdfReader(ruta_archivo).pages)
        opciones = _configurar_pipeline_con_descripcion_imagenes()
        conversor = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opciones)}
        )

        if total_paginas_documento <= UMBRAL_PAGINAS_PARA_LOTES:
            resultado = _convertir_un_archivo(conversor, ruta_archivo)
            if not resultado:
                return None
            logger.info(f"[DOCLING] Imágenes detectadas: {resultado['imagenes_totales']}, con descripción generada: {resultado['imagenes_con_descripcion']}")
            return {"paginas": resultado["paginas"], "encabezados": resultado["encabezados"]}

        logger.info(f"[DOCLING] Documento grande ({total_paginas_documento} páginas) — procesando en lotes de {PAGINAS_POR_LOTE} para evitar quedarse sin memoria.")
        lotes = _dividir_pdf_en_lotes(ruta_archivo, PAGINAS_POR_LOTE)
        paginas_totales, encabezados_totales = [], []
        imagenes_totales_acum, imagenes_con_descripcion_acum = 0, 0
        try:
            for i, (ruta_lote, offset) in enumerate(lotes):
                logger.info(f"[DOCLING] Procesando lote {i + 1}/{len(lotes)} (páginas {offset + 1}-{min(offset + PAGINAS_POR_LOTE, total_paginas_documento)})...")
                resultado_lote = _convertir_un_archivo(conversor, ruta_lote)
                if resultado_lote is None:
                    # Este lote específico falló — no se pierde el documento
                    # completo por esto. Se rellena con un marcador corto
                    # (no vacío del todo) para que el análisis visual, que
                    # corre después sobre CUALQUIER página con poco texto,
                    # tenga oportunidad de cubrir el hueco de forma
                    # independiente, renderizando la página directamente.
                    cuantas = min(PAGINAS_POR_LOTE, total_paginas_documento - offset)
                    logger.warning(f"[DOCLING] Lote {i + 1}/{len(lotes)} falló — esas {cuantas} páginas quedan con un marcador para que el análisis visual las cubra.")
                    paginas_totales.extend([MARCADOR_LOTE_FALLIDO] * cuantas)
                    continue
                paginas_totales.extend(resultado_lote["paginas"])
                for enc in resultado_lote["encabezados"]:
                    encabezados_totales.append({**enc, "pagina": enc["pagina"] + offset})
                imagenes_totales_acum += resultado_lote["imagenes_totales"]
                imagenes_con_descripcion_acum += resultado_lote["imagenes_con_descripcion"]
        finally:
            for ruta_lote, _ in lotes:
                if os.path.exists(ruta_lote):
                    os.remove(ruta_lote)

        if not paginas_totales:
            return None

        logger.info(f"[DOCLING] Lotes completos — {len(lotes)} lote(s), {len(encabezados_totales)} encabezados, imágenes: {imagenes_totales_acum} detectadas / {imagenes_con_descripcion_acum} con descripción.")
        return {"paginas": paginas_totales, "encabezados": encabezados_totales}
    except Exception as e:
        logger.warning(f"[DOCLING] Falló la extracción ({type(e).__name__}: {e}) — se usará el respaldo de pypdf.")
        return None


def _proceso_hijo_docling(ruta_archivo: str, cola: "multiprocessing.Queue"):
    """
    Función objetivo del proceso hijo — corre en un proceso del sistema
    operativo COMPLETAMENTE APARTE del proceso principal de FastAPI. Todo
    lo que Docling necesita (PyTorch, los modelos de layout/OCR/tablas)
    se carga SOLO aquí dentro, nunca en el proceso principal.

    Por qué esto resuelve el gasto de memoria de raíz: PyTorch casi nunca
    le devuelve memoria al sistema operativo mientras el proceso que la
    pidió siga vivo — es un comportamiento conocido de la librería, no
    algo que se pueda "limpiar" con código dentro del mismo proceso. La
    única forma confiable de recuperar esa memoria es que el proceso
    completo termine. Al aislar Docling en su propio proceso hijo, cuando
    termina de procesar UN documento, el proceso se cierra por completo
    y el sistema operativo recupera el 100% de su memoria — garantizado,
    no una esperanza.

    Beneficio adicional real: si Docling llegara a quedarse sin memoria
    procesando un documento (como pasó hoy con el manual de 545 páginas),
    ahora solo muere ESTE proceso hijo — el proceso principal de FastAPI
    sigue vivo y atendiendo a otros usuarios sin interrupción. Antes, ese
    mismo crash tumbaba el contenedor completo.
    """
    try:
        resultado = _convertir_sincrono(ruta_archivo)
        cola.put(("ok", resultado))
    except Exception as e:
        cola.put(("error", f"{type(e).__name__}: {e}"))


async def extraer_paginas_con_docling(contenido_bytes: bytes) -> dict | None:
    """
    Punto de entrada — recibe los bytes del PDF (igual que el extractor
    de pypdf), y lanza la conversión real en un PROCESO HIJO aparte del
    sistema operativo (no un hilo) — así, cuando termina, su memoria se
    libera de verdad, sin depender de que Python decida devolverla. Si
    algo sale mal, se agota el tiempo, o el proceso hijo se cae, devuelve
    None — el llamador debe caer al respaldo de pypdf.

    A diferencia del diseño anterior con hilos (donde un timeout dejaba
    de ESPERAR pero el hilo seguía corriendo de fondo sin poder matarlo),
    con un proceso real si el tiempo se agota se le manda `.terminate()`
    — que sí lo mata de verdad y libera su memoria de inmediato.

    Devuelve {"paginas": list[str], "encabezados": list[dict]} en éxito.
    """
    import multiprocessing

    archivo_temporal = None
    proceso = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contenido_bytes)
            archivo_temporal = tmp.name

        cola = multiprocessing.Queue()
        proceso = multiprocessing.Process(target=_proceso_hijo_docling, args=(archivo_temporal, cola))
        proceso.start()

        # proceso.join() y cola.get() son bloqueantes — se corren en un
        # hilo aparte (asyncio.to_thread) SOLO para esperar sin congelar
        # el event loop de FastAPI. El trabajo pesado en sí ya está en el
        # proceso hijo, no en este hilo de espera.
        def _esperar():
            proceso.join(timeout=TIMEOUT_SEGUNDOS)
            if proceso.is_alive():
                return ("timeout", None)
            if not cola.empty():
                return cola.get()
            return ("error", f"El proceso hijo terminó (código {proceso.exitcode}) sin devolver ningún resultado — probablemente se quedó sin memoria (OOM) y el sistema operativo lo mató directamente.")

        estado, valor = await asyncio.to_thread(_esperar)

        if estado == "timeout":
            logger.warning(f"[DOCLING] Tiempo de espera agotado ({TIMEOUT_SEGUNDOS}s) — se termina el proceso hijo y se usará el respaldo de pypdf.")
            proceso.terminate()
            proceso.join(timeout=10)
            return None
        if estado == "error":
            logger.warning(f"[DOCLING] El proceso hijo falló ({valor}) — se usará el respaldo de pypdf.")
            return None
        return valor
    except Exception as e:
        logger.warning(f"[DOCLING] Excepción inesperada lanzando el proceso hijo ({type(e).__name__}: {e}) — se usará el respaldo de pypdf.")
        return None
    finally:
        if proceso is not None and proceso.is_alive():
            proceso.terminate()
        if archivo_temporal and os.path.exists(archivo_temporal):
            os.remove(archivo_temporal)
