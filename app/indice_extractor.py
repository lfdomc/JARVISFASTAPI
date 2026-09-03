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
# Documentos legales/normativos suelen poner la PALABRA antes del número
# ("Artículo 5. Definiciones. 12", "TÍTULO I. Disposiciones. 5") — al
# revés del formato "número primero" de planes/manuales ("5.5.4
# Encadenamientos... 128"). Soporta número romano o decimal después de
# la palabra, y también sirve para "Sección", "Apartado", "Cláusula",
# "Inciso" — términos comunes en distintos tipos de documento. Incluye
# también los equivalentes en inglés (Article, Chapter, Title, Section,
# Appendix, Clause, Item, Paragraph) — respaldo para cuando Docling cae
# a pypdf y el documento no está en español (con Docling funcionando
# completo, la detección de encabezados por estructura ya cubre esto
# sin depender del idioma).
PATRON_ENTRADA_PALABRA_PRIMERO = re.compile(
    r'^\s*(t[ií]tulo|art[ií]culo|cap[ií]tulo|secci[oó]n|apartado|cl[aá]usula|inciso|anexo|'
    r'title|article|chapter|section|appendix|clause|item|paragraph)\s+'
    r'([IVXLCDM]+|\d+(?:\.\d+)*)\.?\s*(.+?)[\s\.]{1,}(\d{1,4})\s*$',
    re.IGNORECASE
)
PATRON_ANEXO = re.compile(r'^\s*(anexo\s*\d+|ap[eé]ndice\s*\d+|appendix\s*\d+)\s+(\d{1,4})\s*$', re.IGNORECASE)
PATRON_ENCABEZADO_INDICE = re.compile(
    r'^\s*(contenido|[ií]ndice|tabla de contenido[s]?|table of contents|contents)\s*$', re.IGNORECASE
)

MIN_ENTRADAS_PARA_CONSIDERARLO_INDICE = 6
PAGINAS_MAX_A_REVISAR = 15  # el índice casi siempre está al principio
VENTANA_CORRECCION_PAGINA = 3  # cuántas páginas alrededor de la reportada se revisan


def corregir_paginas_indice(entradas: list[dict], paginas: list[str]) -> list[dict]:
    """
    El índice IMPRESO de un documento puede tener errores propios del
    documento original — caso real encontrado hoy: el índice dice
    "Anexo 2 → página 150", pero el contenido real de "ANEXO 2" empieza
    en la página 149. Confiar ciegamente en el número impreso para
    forzar cortes de fragmento (ver chunking.py, paginas_frontera)
    propaga ese error del documento original a nuestros propios datos.

    Aquí se verifica cada entrada contra el contenido REAL: si el
    título no aparece literal en la página que dice el índice, se busca
    en un rango pequeño de páginas cercanas (antes y después) y se
    corrige si se encuentra ahí. Si no se encuentra en ningún lado
    cercano, se deja el número original tal cual — mejor no corregir
    que adivinar mal.
    """
    corregidas = []
    for entrada in entradas:
        pagina_declarada = entrada.get("pagina")
        texto_buscar = (entrada.get("titulo") or "").strip()
        if not pagina_declarada or not texto_buscar or len(texto_buscar) < 4:
            corregidas.append(entrada)
            continue

        idx_declarada = pagina_declarada - 1  # a 0-indexado
        ya_esta_bien = (
            0 <= idx_declarada < len(paginas)
            and texto_buscar.lower() in (paginas[idx_declarada] or "").lower()
        )
        if ya_esta_bien:
            corregidas.append(entrada)
            continue

        pagina_corregida = None
        for delta in range(1, VENTANA_CORRECCION_PAGINA + 1):
            for candidata in (pagina_declarada - delta, pagina_declarada + delta):
                idx = candidata - 1
                if 0 <= idx < len(paginas) and texto_buscar.lower() in (paginas[idx] or "").lower():
                    pagina_corregida = candidata
                    break
            if pagina_corregida:
                break

        if pagina_corregida and pagina_corregida != pagina_declarada:
            nueva_entrada = dict(entrada)
            nueva_entrada["pagina"] = pagina_corregida
            corregidas.append(nueva_entrada)
        else:
            corregidas.append(entrada)

    return corregidas


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
        "pagina_encontrado": int,  # en qué página del PDF EMPIEZA el índice
        "pagina_fin_indice": int,  # en qué página del PDF TERMINA el índice
    }
    Estas dos últimas son la ubicación FÍSICA del índice en el PDF — no
    confundir con las páginas que el índice describe en su contenido.
    Sirven para poder excluir el texto del índice de la ingesta como
    fragmento buscable normal (ver documents.py) — si el índice queda
    indexado igual que cualquier otro contenido, una búsqueda por el
    nombre de una sección puede encontrar la ENTRADA del índice (que
    menciona esa sección) y citar la página del índice mismo, en vez de
    la página real donde está el contenido — justo el tipo de error que
    esto evita de raíz.
    """
    limite = min(len(paginas), PAGINAS_MAX_A_REVISAR)

    for num_pagina in range(limite):
        texto_pagina = paginas[num_pagina]
        if not texto_pagina or not texto_pagina.strip():
            continue

        lineas = texto_pagina.split("\n")
        tiene_encabezado = any(PATRON_ENCABEZADO_INDICE.match(l.strip()) for l in lineas[:5])

        entradas = _parsear_entradas(texto_pagina)
        pagina_fin_indice = num_pagina + 1

        # Sigue revisando páginas siguientes MIENTRAS continúen aportando
        # entradas de índice — algunos índices ocupan más de 2 páginas.
        siguiente = num_pagina + 1
        while siguiente < len(paginas):
            entradas_siguiente = _parsear_entradas(paginas[siguiente])
            if len(entradas_siguiente) < 2:  # ya no parece índice, se acabó
                break
            entradas += entradas_siguiente
            pagina_fin_indice = siguiente + 1
            siguiente += 1

        if len(entradas) >= MIN_ENTRADAS_PARA_CONSIDERARLO_INDICE or (tiene_encabezado and len(entradas) >= 3):
            entradas = corregir_paginas_indice(entradas, paginas)
            return {
                "markdown": _construir_markdown(entradas),
                "entradas": entradas,
                "pagina_encontrado": num_pagina + 1,
                "pagina_fin_indice": pagina_fin_indice,
                "tipo_documento_probable": clasificar_tipo_documento(entradas),
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

        m_palabra = PATRON_ENTRADA_PALABRA_PRIMERO.match(linea)
        if m_palabra:
            prefijo, numero, titulo, pagina = m_palabra.groups()
            # Nivel por conteo de puntos si es numeración decimal (3.1);
            # con números romanos (TÍTULO I) no hay forma clara de saber
            # la profundidad solo por el número, se asume nivel 1.
            nivel = numero.count(".") + 1 if "." in numero else 1
            titulo_limpio = titulo.strip(" .")
            if titulo_limpio:
                entradas.append({
                    "nivel": nivel,
                    "numero": f"{prefijo.capitalize()} {numero}",  # ej. "Artículo 3.1"
                    "titulo": titulo_limpio,
                    "pagina": int(pagina),
                })
                continue
            # BUG REAL ENCONTRADO HOY: líneas tipo "ANEXO 1        147"
            # (sin título real entre el número y la página) hacían
            # coincidir este patrón general PRIMERO, con un "título"
            # que quedaba vacío tras limpiar — y como el código hacía
            # `continue` de inmediato, la línea se descartaba en
            # silencio sin darle oportunidad a PATRON_ANEXO (el patrón
            # específico para este formato exacto, sin título) más
            # abajo. Ahora, si el título queda vacío, se sigue probando
            # con los patrones siguientes en vez de descartar de una vez.

        m2 = PATRON_ANEXO.match(linea)
        if m2:
            entradas.append({
                "nivel": 1, "numero": None, "titulo": m2.group(1).strip(), "pagina": int(m2.group(2))
            })

    return entradas


# Palabras que sugieren cada tipo de documento — no es ciencia exacta,
# solo una pista útil: "Artículo/Cláusula/Inciso" son casi siempre
# normativos/legales; "Eje/Capítulo/numeración decimal" son más comunes
# en planes, manuales y documentos técnicos.
PALABRAS_TIPO_LEGAL = {"artículo", "cláusula", "inciso", "título", "article", "clause", "item", "title"}
PALABRAS_TIPO_TECNICO = {"eje", "capítulo", "sección", "apartado", "anexo", "chapter", "section", "appendix", "paragraph"}


def clasificar_tipo_documento(entradas: list[dict]) -> str | None:
    """
    Estima el tipo de documento a partir de qué palabras predominan en
    las entradas de su índice — una primera señal automática, útil para
    sugerir la categoría o para calibrar reglas específicas por tipo de
    documento más adelante. No es definitivo, es una pista basada en
    patrones de formato, no en el contenido real.
    """
    if not entradas:
        return None

    conteo_legal = 0
    conteo_tecnico = 0
    conteo_numerico_puro = 0  # sin palabra prefijo, ej. "5.5.4 Encadenamientos..."

    for e in entradas:
        numero = (e.get("numero") or "").lower()
        primera_palabra = numero.split(" ")[0] if " " in numero else None
        if primera_palabra in PALABRAS_TIPO_LEGAL:
            conteo_legal += 1
        elif primera_palabra in PALABRAS_TIPO_TECNICO:
            conteo_tecnico += 1
        elif numero and numero[0].isdigit():
            conteo_numerico_puro += 1

    total = len(entradas)
    if conteo_legal / total >= 0.4:
        return "legal_normativo"
    if (conteo_tecnico + conteo_numerico_puro) / total >= 0.4:
        return "tecnico_planificacion"
    return "indeterminado"


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
