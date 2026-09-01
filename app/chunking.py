import re

MIN_CARACTERES_CHUNK_VALIDO = 60
MARCADOR_PAGINA = "\x00PAGINA:{n}\x00"
PATRON_MARCADOR = re.compile(r"\x00PAGINA:(\d+)\x00")


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
        chunk_actual = (titulo + "\n") if titulo else ""

        for linea in lineas[1:]:
            linea_con_salto = linea + "\n"
            if len(chunk_actual) + len(linea_con_salto) > max_caracteres and chunk_actual.strip():
                chunks_brutos.append(chunk_actual.strip())
                chunk_actual = (titulo + "\n" if titulo else "") + linea_con_salto
            else:
                chunk_actual += linea_con_salto

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
