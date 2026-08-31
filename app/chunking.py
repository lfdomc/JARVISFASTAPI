import re

MIN_CARACTERES_CHUNK_VALIDO = 60


def crear_chunks_markdown(texto: str, max_palabras: int = 350) -> list[str]:
    """
    Trocea por estructura de encabezados Markdown. Usa CARACTERES (no
    palabras) para decidir el límite de tamaño — el bug que encontramos
    hoy en la versión de Apps Script era contar por split(espacios), que
    subestima drásticamente texto sin espacios internos (tablas de
    contenido con puntos suspensivos, celdas de tablas pegadas).
    """
    if not texto or not texto.strip():
        return []

    max_caracteres = max_palabras * 6  # ~6 caracteres promedio por palabra en español
    secciones = re.split(r"(?=\n#{1,3}\s)", texto)
    chunks_finales = []

    for seccion in secciones:
        contenido = seccion.strip()
        if not contenido:
            continue

        if len(contenido) <= max_caracteres:
            chunks_finales.append(contenido)
            continue

        lineas = contenido.split("\n")
        titulo = lineas[0] if lineas[0].startswith("#") else ""
        chunk_actual = (titulo + "\n") if titulo else ""

        for linea in lineas[1:]:
            linea_con_salto = linea + "\n"
            if len(chunk_actual) + len(linea_con_salto) > max_caracteres and chunk_actual.strip():
                chunks_finales.append(chunk_actual.strip())
                chunk_actual = (titulo + "\n" if titulo else "") + linea_con_salto
            else:
                chunk_actual += linea_con_salto

        if chunk_actual.strip():
            chunks_finales.append(chunk_actual.strip())

    # Descarta chunks huérfanos (solo encabezado, sin contenido real)
    chunks_finales = [c for c in chunks_finales if len(c) >= MIN_CARACTERES_CHUNK_VALIDO]

    return chunks_finales if chunks_finales else [texto]
