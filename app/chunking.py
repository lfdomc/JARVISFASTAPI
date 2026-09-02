import re

MIN_CARACTERES_CHUNK_VALIDO = 60
MARCADOR_PAGINA = "\x00PAGINA:{n}\x00"
PATRON_MARCADOR = re.compile(r"\x00PAGINA:(\d+)\x00")
PATRON_FILA_TABLA_MD = re.compile(r"^\s*\|.*\|\s*$")
PATRON_GUION_CORTE = re.compile(r"(?<=\w)-\s*$")


def _unir_palabras_cortadas(lineas: list[str]) -> list[str]:
    """
    Une líneas donde una palabra quedó partida por guion de separación
    silábica al final de línea — común en texto extraído de PDF, ej.
    "...mediante el fo-" seguido de "mento de la productividad...". Sin
    esto, el corte por tamaño puede caer justo entre esas dos líneas y
    partir la palabra (y a veces la página citada) entre dos fragmentos
    distintos.
    """
    resultado = []
    i = 0
    while i < len(lineas):
        linea = lineas[i]
        if PATRON_GUION_CORTE.search(linea.rstrip()) and i + 1 < len(lineas):
            siguiente = lineas[i + 1]
            linea = PATRON_GUION_CORTE.sub("", linea.rstrip()) + siguiente.lstrip()
            resultado.append(linea)
            i += 2
        else:
            resultado.append(linea)
            i += 1
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


def crear_chunks_con_paginas(paginas: list[str], max_palabras: int = 350) -> list[dict]:
    """
    Igual que crear_chunks_markdown, pero recibe el texto YA separado por
    página (una entrada de la lista = una página del PDF) y devuelve cada
    fragmento con su página de origen real, capturada en la ingesta — no
    adivinada después por el modelo a partir de un pie de página embebido
    en el texto. Esta es la corrección de raíz al problema de "el modelo
    cita la página equivocada": si la página nunca se pierde como dato
    estructurado, no hay nada que adivinar.

    Devuelve: [{"texto": str, "pagina_inicio": int, "pagina_fin": int}, ...]
    Los números de página son 1-indexados, según el orden de la lista.
    """
    if not paginas:
        return []

    texto_con_marcadores = "".join(
        MARCADOR_PAGINA.format(n=i + 1) + pagina for i, pagina in enumerate(paginas)
    )

    max_caracteres = max_palabras * 6  # ~6 caracteres promedio por palabra en español
    secciones = re.split(r"(?=\n#{1,3}\s)", texto_con_marcadores)
    chunks_brutos = []

    for seccion in secciones:
        contenido = seccion.strip()
        if not contenido:
            continue

        if len(contenido) <= max_caracteres:
            chunks_brutos.append(contenido)
            continue

        lineas = contenido.split("\n")
        titulo = lineas[0] if lineas[0].startswith("#") else ""
        cuerpo = lineas[1:] if titulo else lineas
        unidades = _agrupar_en_unidades(cuerpo)

        chunk_actual = (titulo + "\n") if titulo else ""

        for unidad in unidades:
            unidad_con_salto = unidad + "\n"
            cabe_en_chunk_actual = len(chunk_actual) + len(unidad_con_salto) <= max_caracteres

            if not cabe_en_chunk_actual and chunk_actual.strip():
                chunks_brutos.append(chunk_actual.strip())
                chunk_actual = (titulo + "\n" if titulo else "") + unidad_con_salto
            else:
                chunk_actual += unidad_con_salto
            # Si la unidad es una tabla más grande que max_caracteres ella
            # sola, se deja intacta igual (mejor un fragmento grande con
            # la tabla completa que una tabla partida a la mitad).

        if chunk_actual.strip():
            chunks_brutos.append(chunk_actual.strip())

    # Para cada chunk bruto: extraer qué páginas cubre (de sus marcadores)
    # y luego quitar los marcadores del texto final (no deben llegar al
    # embedding ni a lo que lee el modelo).
    resultado = []
    pagina_previa = 1
    for chunk in chunks_brutos:
        numeros_pagina = [int(n) for n in PATRON_MARCADOR.findall(chunk)]
        texto_limpio = PATRON_MARCADOR.sub("", chunk).strip()

        if not texto_limpio or len(texto_limpio) < MIN_CARACTERES_CHUNK_VALIDO:
            continue

        if numeros_pagina:
            pagina_inicio = min(numeros_pagina)
            pagina_fin = max(numeros_pagina)
            pagina_previa = pagina_fin
        else:
            # Chunk sin su propio marcador (siguió fluyendo de la sección
            # anterior tras el split) — hereda la última página vista.
            pagina_inicio = pagina_fin = pagina_previa

        resultado.append({"texto": texto_limpio, "pagina_inicio": pagina_inicio, "pagina_fin": pagina_fin})

    if not resultado and paginas:
        texto_plano = "\n\n".join(paginas).strip()
        if texto_plano:
            resultado = [{"texto": texto_plano, "pagina_inicio": 1, "pagina_fin": len(paginas)}]

    return resultado
