"""
Detección y extracción del índice (tabla de contenidos) de un documento.

Muchos documentos formales (planes, manuales, normativas) traen un índice
al principio que ya mapea cada capítulo/sección a su página real. Si lo
detectamos y lo estructuramos, le da al modelo un "mapa" del documento
completo antes de responder — útil para preguntas de análisis global
("analiza todo el documento") y como una capa extra de verificación:
si una cita de página cae fuera de las secciones conocidas, es una señal
más de alerta.
"""
import re

PATRON_ENTRADA = re.compile(r'^\s*(\d+(?:\.\d+){0,3})\.?\s*(.+?)[\s\.]{1,}(\d{1,4})\s*$')
PATRON_ANEXO = re.compile(r'^\s*(anexo\s*\d+|ap[eé]ndice\s*\d+)\s+(\d{1,4})\s*$', re.IGNORECASE)
PATRON_ENCABEZADO_INDICE = re.compile(
    r'^\s*(contenido|[ií]ndice|tabla de contenido[s]?|table of contents)\s*$', re.IGNORECASE
)

MIN_ENTRADAS_PARA_CONSIDERARLO_INDICE = 6
PAGINAS_MAX_A_REVISAR = 15  # el índice casi siempre está al principio


def extraer_indice_documento(paginas: list[str]) -> dict | None:
    """
    Busca un índice en las primeras páginas del documento. Si encuentra
    suficientes entradas con el patrón "número + título + página", lo
    considera un índice real y lo estructura.

    Devuelve None si no se detecta un índice (no todos los documentos
    lo tienen — no es un error, simplemente no aplica).

    Devuelve: {
        "markdown": str,           # el índice como outline en Markdown
        "entradas": [              # estructurado, por si se quiere usar programáticamente
            {"nivel": int, "numero": str, "titulo": str, "pagina": int}, ...
        ],
        "pagina_encontrado": int,  # en qué página del PDF estaba el índice
    }
    """
    limite = min(len(paginas), PAGINAS_MAX_A_REVISAR)

    for num_pagina in range(limite):
        texto_pagina = paginas[num_pagina]
        if not texto_pagina or not texto_pagina.strip():
            continue

        lineas = texto_pagina.split("\n")
        tiene_encabezado = any(PATRON_ENCABEZADO_INDICE.match(l.strip()) for l in lineas[:5])

        entradas = _parsear_entradas(texto_pagina)

        # También revisa la página siguiente por si el índice sigue ahí
        # (son comunes los índices de 2 páginas)
        if num_pagina + 1 < len(paginas):
            entradas += _parsear_entradas(paginas[num_pagina + 1])

        if len(entradas) >= MIN_ENTRADAS_PARA_CONSIDERARLO_INDICE or (tiene_encabezado and len(entradas) >= 3):
            return {
                "markdown": _construir_markdown(entradas),
                "entradas": entradas,
                "pagina_encontrado": num_pagina + 1,
            }

    return None


def _parsear_entradas(texto_pagina: str) -> list[dict]:
    entradas = []
    for linea in texto_pagina.split("\n"):
        linea = linea.strip()
        if not linea:
            continue

        m = PATRON_ENTRADA.match(linea)
        if m:
            numero, titulo, pagina = m.groups()
            nivel = numero.count(".") + 1
            titulo_limpio = titulo.strip(" .")
            if titulo_limpio:  # descarta líneas donde el "título" quedó vacío
                entradas.append({
                    "nivel": nivel, "numero": numero, "titulo": titulo_limpio, "pagina": int(pagina)
                })
            continue

        m2 = PATRON_ANEXO.match(linea)
        if m2:
            entradas.append({
                "nivel": 1, "numero": None, "titulo": m2.group(1).strip(), "pagina": int(m2.group(2))
            })

    return entradas


def _construir_markdown(entradas: list[dict]) -> str:
    lineas = ["# Índice del documento\n"]
    for e in entradas:
        sangria = "  " * (e["nivel"] - 1)
        prefijo = f"{e['numero']} " if e["numero"] else ""
        lineas.append(f"{sangria}- {prefijo}{e['titulo']} — pág. {e['pagina']}")
    return "\n".join(lineas)


def encontrar_seccion_para_pagina(entradas: list[dict], pagina: int) -> str | None:
    """
    Dado el índice ya parseado y una página, devuelve el título de la
    sección más específica (más profunda) a la que pertenece esa página
    — construido a partir de los rangos de página entre una entrada del
    índice y la siguiente.
    """
    if not entradas or not pagina:
        return None

    entradas_ordenadas = sorted(entradas, key=lambda e: e["pagina"])
    mejor_candidata = None

    for i, entrada in enumerate(entradas_ordenadas):
        pagina_inicio_seccion = entrada["pagina"]
        pagina_fin_seccion = (
            entradas_ordenadas[i + 1]["pagina"] - 1
            if i + 1 < len(entradas_ordenadas)
            else float("inf")
        )
        if pagina_inicio_seccion <= pagina <= pagina_fin_seccion:
            # Se queda con la coincidencia de nivel más profundo (más específica)
            if mejor_candidata is None or entrada["nivel"] >= mejor_candidata["nivel"]:
                mejor_candidata = entrada

    if not mejor_candidata:
        return None

    prefijo = f"{mejor_candidata['numero']} " if mejor_candidata["numero"] else ""
    return f"{prefijo}{mejor_candidata['titulo']}"
