import re
from langdetect import detect, LangDetectException

MIN_CARACTERES_CHUNK_VALIDO = 60
MIN_CARACTERES_PAGINA_CON_CONTENIDO = 25  # menos que esto se considera "sin texto real"
MARCADOR_PAGINA_VACIA = "[Página {n} sin texto extraíble — posible imagen, portada o página en blanco. Si la descripción de imágenes de Docling está activa y detectó algo aquí, debería aparecer en su lugar.]"
MARCADOR_PAGINA = "\x00PAGINA:{n}\x00"
PATRON_MARCADOR = re.compile(r"\x00PAGINA:(\d+)\x00")
PATRON_FILA_TABLA_MD = re.compile(r"^\s*\|.*\|\s*$")
PATRON_GUION_CORTE = re.compile(r"(?<=\w)\s?-\s*$")
PATRON_ENCABEZADO_MD = re.compile(r"^\s*#{1,6}\s")
PATRON_MARCADOR_PAGINA_VACIA = re.compile(r"sin texto extraíble")


def marcar_paginas_vacias(paginas: list[str]) -> list[str]:
    """
    Antes de trocear: cualquier página con muy poco o ningún texto real
    (portadas, páginas en blanco, imágenes sin describir) se reemplaza
    por un marcador EXPLÍCITO en vez de dejarse casi vacía.

    Por qué importa: una página casi vacía, dejada tal cual, se fusiona
    en silencio con sus vecinas durante el troceo (para alcanzar un
    tamaño de fragmento válido) — y eso puede borrar una frontera real
    entre secciones sin dejar rastro (el caso real de hoy: las páginas
    145-146, casi vacías, hicieron que un fragmento se extendiera desde
    plena sección 6.2 hasta adentro de Anexo 1). Con el marcador
    explícito, la página sigue "existiendo" con contenido real — se
    puede clasificar aparte (ver _clasificar_tipo_contenido) y el hueco
    queda documentado, no oculto.

    Si Docling con descripción de imágenes ya puso una descripción real
    ahí, esta función no la toca — solo actúa sobre páginas
    genuinamente vacías o casi vacías.
    """
    resultado = []
    for i, texto in enumerate(paginas):
        if len(texto.strip()) < MIN_CARACTERES_PAGINA_CON_CONTENIDO:
            # Con salto de línea al final — sin esto, el marcador se pega
            # directo al texto de la página siguiente (que normalmente sí
            # trae su propio salto final), fusionando ambas páginas en
            # una sola "unidad" indivisible durante el troceo y dejando
            # pasar por alto cualquier frontera de sección justo ahí.
            resultado.append(MARCADOR_PAGINA_VACIA.format(n=i + 1) + "\n")
        else:
            resultado.append(texto)
    return resultado


def _clasificar_tipo_contenido(texto: str) -> str:
    """
    Clasifica un fragmento ya armado como 'tabla', 'titulo', 'vacia_o_imagen'
    o 'texto' — funciona igual sin importar qué motor extrajo el texto
    (Docling o pypdf), porque se basa en el patrón del texto final, no
    en metadatos internos de ningún motor. Con Docling las tablas llegan
    ya en formato de tabla Markdown real, así que esta detección es más
    certera que antes (con pypdf puro, una tabla de PDF nunca llegaba a
    verse como tabla Markdown en absoluto).
    """
    if PATRON_MARCADOR_PAGINA_VACIA.search(texto) and len(texto.strip()) < 500:
        return "vacia_o_imagen"

    lineas = [l for l in texto.split("\n") if l.strip()]
    if not lineas:
        return "texto"

    lineas_tabla = sum(1 for l in lineas if PATRON_FILA_TABLA_MD.match(l))
    if lineas_tabla / len(lineas) >= 0.6:  # la mayoría de las líneas son filas de tabla
        return "tabla"

    if len(lineas) <= 2 and PATRON_ENCABEZADO_MD.match(lineas[0]):
        return "titulo"

    return "texto"


def _detectar_idioma(texto: str) -> str | None:
    """Detecta el idioma de un fragmento — no cambia nada del troceo,
    solo etiqueta cada fragmento para poder alertar si un documento
    resulta mezclado en idiomas (ver calidad_ingesta.py). Con texto muy
    corto o ambiguo, langdetect puede fallar — se trata como
    'desconocido' en vez de romper el troceo del documento."""
    try:
        return detect(texto)
    except LangDetectException:
        return None


def _unir_palabras_cortadas(lineas: list[str]) -> list[str]:
    """
    Une líneas donde una palabra quedó partida por guion de separación
    silábica al final de línea — común en texto extraído de PDF, ej.
    "...mediante el fo-" seguido de "mento de la productividad...". Sin
    esto, el corte por tamaño puede caer justo entre esas dos líneas y
    partir la palabra (y a veces la página citada) entre dos fragmentos
    distintos.

    Maneja guiones ENCADENADOS (dos o más palabras cortadas seguidas,
    ej. "...los pro-" / "cedimientos...cohe-" / "rente...") — sigue
    uniendo mientras el resultado siga terminando en guion, en vez de
    unir solo un par y dejar el segundo corte sin resolver.
    """
    resultado = []
    i = 0
    while i < len(lineas):
        linea = lineas[i]
        j = i
        while PATRON_GUION_CORTE.search(linea.rstrip()) and j + 1 < len(lineas):
            j += 1
            linea = PATRON_GUION_CORTE.sub("", linea.rstrip()) + lineas[j].lstrip()
        resultado.append(linea)
        i = j + 1
    return resultado


def _agrupar_en_unidades(lineas: list[str]) -> list[str]:
    """
    Agrupa líneas consecutivas de una tabla Markdown (formato con pipes
    "| celda | celda |") en una sola unidad indivisible, para que el
    troceo nunca corte una tabla a la mitad — una tabla partida entre dos
    fragmentos es fácil de malinterpretar (encabezados en un fragmento,
    datos en otro, sin saber a qué columna corresponden).

    NOTA DE ALCANCE: esto detecta tablas en sintaxis Markdown real (las
    que trae el contenido de Firecrawl/web). Las tablas dentro de PDFs
    extraídas con pypdf pierden su estructura y se convierten en texto
    plano sin pipes — ahí esta detección no aplica; para preservar esas
    haría falta extracción de tablas con pdfplumber, un cambio aparte.
    """
    lineas = _unir_palabras_cortadas(lineas)
    unidades = []
    buffer_tabla = []

    for linea in lineas:
        if PATRON_FILA_TABLA_MD.match(linea):
            buffer_tabla.append(linea)
        else:
            if buffer_tabla:
                unidades.append("\n".join(buffer_tabla))
                buffer_tabla = []
            unidades.append(linea)

    if buffer_tabla:
        unidades.append("\n".join(buffer_tabla))

    return unidades


def crear_chunks_markdown(texto: str, max_palabras: int = 350) -> list[str]:
    """
    Trocea por estructura de encabezados Markdown. Usa CARACTERES (no
    palabras) para decidir el límite de tamaño.

    Sin información de página — úsalo para contenido sin páginas reales
    (artículos web de Firecrawl). Para PDFs, usa crear_chunks_con_paginas.
    """
    chunks_con_pagina = crear_chunks_con_paginas([texto], max_palabras)
    return [c["texto"] for c in chunks_con_pagina]


def crear_chunks_con_paginas(paginas: list[str], max_palabras: int = 350, paginas_frontera: set[int] | None = None) -> list[dict]:
    """
    Igual que crear_chunks_markdown, pero recibe el texto YA separado por
    página (una entrada de la lista = una página del PDF) y devuelve cada
    fragmento con su página de origen real, capturada en la ingesta — no
    adivinada después por el modelo a partir de un pie de página embebido
    en el texto.

    DISEÑO: la página se seguía antes escaneando qué marcadores "quedaron
    dentro" de cada fragmento ya armado — pero si un corte por tamaño cae
    justo en el punto exacto donde el marcador de una página ya fue
    consumido por el fragmento ANTERIOR (sin que haya llegado todavía el
    marcador de la página siguiente), el fragmento nuevo perdía el rastro
    de en qué página iba, aunque su contenido siguiera siendo de la misma
    página. Ahora se seguimiento CONTINUO de la página mientras se arma
    el documento — cada fragmento nuevo hereda la página en la que
    íbamos en ese momento, sin importar si el marcador cae dentro de él
    o quedó en el fragmento anterior.

    paginas_frontera: páginas donde EMPIEZA una sección del índice — se
    fuerza un corte de fragmento ahí, sin importar si cabría dentro del
    límite de caracteres. Mismo principio que ya usamos para proteger
    tablas: un fragmento nunca debe mezclar contenido de dos secciones
    distintas, sin importar qué tan poco texto tenga la página que
    causaría la fusión (el caso real: páginas casi vacías entre "6.2
    Indicadores de seguimiento" y "Anexo 1" hacían que un fragmento se
    extendiera de una sección a la otra).

    Devuelve: [{"texto": str, "pagina_inicio": int, "pagina_fin": int}, ...]
    Los números de página son 1-indexados, según el orden de la lista.
    """
    if not paginas:
        return []

    paginas_frontera = paginas_frontera or set()
    texto_con_marcadores = "".join(
        MARCADOR_PAGINA.format(n=i + 1) + pagina for i, pagina in enumerate(paginas)
    )

    max_caracteres = max_palabras * 6  # ~6 caracteres promedio por palabra en español
    secciones = re.split(r"(?=\n#{1,3}\s)", texto_con_marcadores)
    chunks_brutos = []  # [{"texto_crudo": str, "pagina_inicio": int, "pagina_fin": int}, ...]

    pagina_actual = 1  # seguimiento continuo, cruza fronteras de fragmento y de sección

    for seccion in secciones:
        contenido = seccion.strip()
        if not contenido:
            continue

        lineas = contenido.split("\n")
        titulo = lineas[0] if lineas[0].startswith("#") else ""
        cuerpo = lineas[1:] if titulo else lineas
        unidades = _agrupar_en_unidades(cuerpo)

        chunk_actual = (titulo + "\n") if titulo else ""
        pagina_inicio_chunk = pagina_actual
        pagina_fin_chunk = pagina_actual

        for unidad in unidades:
            marcadores_unidad = [int(n) for n in PATRON_MARCADOR.findall(unidad)]
            unidad_con_salto = unidad + "\n"
            cabe_en_chunk_actual = len(chunk_actual) + len(unidad_con_salto) <= max_caracteres

            # Frontera de sección: esta unidad introduce una página que
            # es el INICIO de una sección del índice, y ya veníamos de
            # una página distinta — se corta AQUÍ sin importar el
            # límite de caracteres, para nunca mezclar dos secciones.
            cruza_frontera_de_seccion = any(
                m in paginas_frontera and m != pagina_actual for m in marcadores_unidad
            )

            if (cruza_frontera_de_seccion or not cabe_en_chunk_actual) and chunk_actual.strip():
                chunks_brutos.append({
                    "texto_crudo": chunk_actual.strip(),
                    "pagina_inicio": pagina_inicio_chunk,
                    "pagina_fin": pagina_fin_chunk,
                })
                chunk_actual = (titulo + "\n" if titulo else "") + unidad_con_salto
                if cruza_frontera_de_seccion:
                    # El corte fue POR la frontera de sección que esta
                    # misma unidad trae — el fragmento nuevo debe
                    # arrancar en la página frontera real (ej. 147, el
                    # inicio real de Anexo 1), no en la página vieja de
                    # antes del corte. Si no, el fragmento queda con la
                    # página de origen equivocada, aunque el corte en sí
                    # haya caído en el lugar correcto — el mismo tipo de
                    # bug de "un paso atrás" que ya corregimos hoy para
                    # los cortes por tamaño.
                    pagina_inicio_chunk = min(m for m in marcadores_unidad if m in paginas_frontera)
                else:
                    # Corte normal por tamaño: el fragmento nuevo arranca
                    # en la página en la que íbamos justo antes de esta
                    # unidad — no en una que aparezca más adelante dentro
                    # de la unidad misma.
                    pagina_inicio_chunk = pagina_actual
                pagina_fin_chunk = pagina_inicio_chunk
            else:
                chunk_actual += unidad_con_salto

            if marcadores_unidad:
                pagina_actual = max(marcadores_unidad)
                pagina_fin_chunk = pagina_actual
            # Si la unidad es una tabla más grande que max_caracteres ella
            # sola, se deja intacta igual (mejor un fragmento grande con
            # la tabla completa que una tabla partida a la mitad).

        if chunk_actual.strip():
            chunks_brutos.append({
                "texto_crudo": chunk_actual.strip(),
                "pagina_inicio": pagina_inicio_chunk,
                "pagina_fin": pagina_fin_chunk,
            })

    # Limpieza final: quitar los marcadores del texto (no deben llegar al
    # embedding ni a lo que lee el modelo) y descartar fragmentos vacíos
    # o demasiado cortos para ser útiles.
    resultado = []
    for chunk in chunks_brutos:
        texto_limpio = PATRON_MARCADOR.sub("", chunk["texto_crudo"]).strip()
        if not texto_limpio or len(texto_limpio) < MIN_CARACTERES_CHUNK_VALIDO:
            continue
        resultado.append({
            "texto": texto_limpio,
            "pagina_inicio": chunk["pagina_inicio"],
            "pagina_fin": chunk["pagina_fin"],
            "tipo_contenido": _clasificar_tipo_contenido(texto_limpio),
            "idioma": _detectar_idioma(texto_limpio),
        })

    if not resultado and paginas:
        texto_plano = "\n\n".join(paginas).strip()
        if texto_plano:
            resultado = [{
                "texto": texto_plano, "pagina_inicio": 1, "pagina_fin": len(paginas),
                "tipo_contenido": _clasificar_tipo_contenido(texto_plano),
                "idioma": _detectar_idioma(texto_plano),
            }]

    return resultado
