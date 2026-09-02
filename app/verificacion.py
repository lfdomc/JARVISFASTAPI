"""
Blindaje contra alucinaciones — capa 3 de 3.

Versión estricta: no basta con que una cita "exista en algún fragmento" ni
que una página "esté entre las recuperadas" — cada cita textual se cruza
con el fragmento EXACTO donde aparece, y se exige que la página (y la
sección, si el fragmento la tiene) que el modelo escribió cerca de esa
cita coincidan con ese fragmento específico. Esto atrapa el caso más
sutil que encontramos: una frase real, pero atribuida a la página de un
fragmento vecino en vez de la suya propia.

Sigue siendo determinístico — comparación de texto, no otro LLM.
"""
import re
import unicodedata

PATRON_COMILLAS = re.compile(r'["“”«»]([^"“”«»]{15,})["“”«»]')
PATRON_PAGINA = re.compile(r'p[aá]g(?:s|ina[s]?)?\.?\s*(\d+)(?:\s*[-–—]\s*(\d+))?', re.IGNORECASE)
VENTANA_BUSQUEDA_PAGINA = 200  # caracteres después de la cita donde se busca su "(pág. N)"

# Números "clave" — porcentajes y cantidades grandes — son exactamente el
# tipo de dato que se escapaba del verificador cuando el modelo parafrasea
# sin comillas (así fue como "70%" y "10%" quedaron mal atribuidos sin que
# el verificador anterior lo detectara). Se verifican existan en ALGÚN
# fragmento, sin exigir el cruce a un fragmento único como con las citas
# textuales — es una verificación más simple, pero atrapa la fabricación
# total de un número que ningún fragmento respalda en absoluto.
PATRON_PORCENTAJE = re.compile(r'\d+(?:[.,]\d+)?\s*%')
PATRON_CANTIDAD_GRANDE = re.compile(r'\d+(?:[.,]\d+)?\s*(?:mil millones|millones)\b', re.IGNORECASE)


def _normalizar(texto: str) -> str:
    """Minúsculas, sin acentos, espacios colapsados — para comparar sin
    que un espacio de más o un acento distinto genere un falso positivo."""
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    texto = texto.lower()
    return re.sub(r'\s+', ' ', texto).strip()


def _buscar_fragmento_de_la_cita(cita_normalizada: str, fragmentos: list[dict]) -> dict | None:
    """Encuentra el fragmento EXACTO que contiene esta cita literal (no
    solo "algún" fragmento del conjunto general)."""
    for f in fragmentos:
        if cita_normalizada in _normalizar(f.get("contenido_chunk", "")):
            return f
    return None


def _extraer_numeros_clave(texto: str) -> list[tuple[str, str]]:
    """Devuelve [(texto_completo, solo_digitos), ...]."""
    encontrados = []
    for patron in (PATRON_PORCENTAJE, PATRON_CANTIDAD_GRANDE):
        for m in patron.finditer(texto):
            texto_num = m.group(0).strip()
            digitos = re.match(r'[\d.,]+', texto_num).group(0)
            encontrados.append((texto_num, digitos))
    return encontrados


def _compacto(texto: str) -> str:
    """Quita todos los espacios — para comparar '70%' contra '70 %' o
    '4,9 mil millones' contra '4,9mil millones' sin que el espaciado
    genere un falso positivo."""
    return re.sub(r'\s+', '', _normalizar(texto))


def _numero_respaldado(digitos: str, texto_fragmentos_compacto: str) -> bool:
    """
    Busca el número dentro de una ventana corta de texto, no pegado
    exactamente — en español es común escribir rangos compartiendo el
    símbolo una sola vez (ej. "entre el 70 y el 80%"), así que "70" real
    puede no tener el "%" pegado justo al lado en el fragmento original,
    aunque el dato sí sea legítimo.
    """
    patron_ventana = re.compile(re.escape(_compacto(digitos)) + r'.{0,20}?(%|millones|milmillones)')
    return bool(patron_ventana.search(texto_fragmentos_compacto))


def verificar_respuesta(respuesta: str, fragmentos: list[dict]) -> dict:
    """
    Devuelve:
    {
      "ok": bool,
      "citas_no_verificadas": [str, ...],
      "citas_pagina_incorrecta": [
          {"cita": str, "pagina_citada": int, "pagina_real": str, "seccion_real": str|None}, ...
      ],
      "paginas_no_verificadas": [int, ...],
    }
    """
    texto_fragmentos_normalizado = _normalizar(
        "\n".join(f.get("contenido_chunk", "") for f in fragmentos)
    )

    paginas_disponibles: set[int] = set()
    for f in fragmentos:
        metadata = f.get("metadata") or {}
        p_ini = f.get("pagina_inicio") or metadata.get("pagina_inicio")
        p_fin = f.get("pagina_fin") or metadata.get("pagina_fin")
        if p_ini and p_fin:
            paginas_disponibles.update(range(int(p_ini), int(p_fin) + 1))

    citas_no_verificadas = []
    citas_pagina_incorrecta = []

    for m in PATRON_COMILLAS.finditer(respuesta):
        cita = m.group(1)
        cita_norm = _normalizar(cita)

        if cita_norm not in texto_fragmentos_normalizado:
            citas_no_verificadas.append(cita.strip())
            continue

        fragmento_origen = _buscar_fragmento_de_la_cita(cita_norm, fragmentos)
        if not fragmento_origen:
            continue

        metadata = fragmento_origen.get("metadata") or {}
        p_ini = fragmento_origen.get("pagina_inicio") or metadata.get("pagina_inicio")
        p_fin = fragmento_origen.get("pagina_fin") or metadata.get("pagina_fin")
        if not p_ini:
            continue

        ventana = respuesta[m.end():m.end() + VENTANA_BUSQUEDA_PAGINA]
        m_pagina = PATRON_PAGINA.search(ventana)
        if not m_pagina:
            continue

        pagina_citada = int(m_pagina.group(1))
        p_fin_real = p_fin or p_ini
        if not (int(p_ini) <= pagina_citada <= int(p_fin_real)):
            pagina_str = f"pág. {p_ini}" if p_ini == p_fin_real else f"págs. {p_ini}-{p_fin_real}"
            citas_pagina_incorrecta.append({
                "cita": cita.strip(),
                "pagina_citada": pagina_citada,
                "pagina_real": pagina_str,
                "seccion_real": fragmento_origen.get("seccion"),
            })

    paginas_no_verificadas = []
    if paginas_disponibles:
        for p_ini_str, p_fin_str in PATRON_PAGINA.findall(respuesta):
            for p in filter(None, [p_ini_str, p_fin_str]):
                p_num = int(p)
                if p_num not in paginas_disponibles and p_num not in paginas_no_verificadas:
                    paginas_no_verificadas.append(p_num)

    # Números clave (porcentajes, cantidades grandes) que no aparecen en
    # NINGÚN fragmento — la señal de fabricación total de un dato numérico,
    # con o sin comillas alrededor.
    numeros_no_verificados = []
    fragmentos_compacto = _compacto(
        "\n".join(f.get("contenido_chunk", "") for f in fragmentos)
    )
    for numero_texto, digitos in _extraer_numeros_clave(respuesta):
        if not _numero_respaldado(digitos, fragmentos_compacto) and numero_texto not in numeros_no_verificados:
            numeros_no_verificados.append(numero_texto)

    ok = (
        not citas_no_verificadas and not citas_pagina_incorrecta
        and not paginas_no_verificadas and not numeros_no_verificados
    )
    return {
        "ok": ok,
        "citas_no_verificadas": citas_no_verificadas,
        "citas_pagina_incorrecta": citas_pagina_incorrecta,
        "paginas_no_verificadas": paginas_no_verificadas,
        "numeros_no_verificados": numeros_no_verificados,
    }


def construir_instruccion_correctiva(resultado: dict) -> str:
    """Mensaje de corrección específico — le señala al modelo exactamente
    qué falló y cuál es el dato correcto."""
    partes = ["Tu respuesta anterior tiene problemas de verificación que debes corregir:"]

    if resultado["citas_no_verificadas"]:
        lista = "; ".join(f'"{c}"' for c in resultado["citas_no_verificadas"])
        partes.append(
            f"- Estas frases entre comillas NO se encontraron literalmente en los fragmentos: {lista}. "
            "Quítales las comillas y usa tus propias palabras, o elimínalas si no son esenciales."
        )

    if resultado["citas_pagina_incorrecta"]:
        for item in resultado["citas_pagina_incorrecta"]:
            seccion_txt = f", sección \"{item['seccion_real']}\"" if item["seccion_real"] else ""
            partes.append(
                f"- La cita \"{item['cita']}\" la atribuiste a la página {item['pagina_citada']}, pero "
                f"esa cita en realidad viene de {item['pagina_real']}{seccion_txt}. Corrige el número de "
                "página (y la sección, si la mencionas) exactamente a ese valor."
            )

    if resultado["paginas_no_verificadas"]:
        lista = ", ".join(str(p) for p in resultado["paginas_no_verificadas"])
        partes.append(
            f"- Citaste la(s) página(s) {lista}, pero ningún fragmento recuperado corresponde a esa "
            "página. Corrige el número de página al que realmente aparece en la etiqueta \"Página "
            "verificada\" del fragmento que respalda ese dato, o quita el número de página si no puedes "
            "verificarlo."
        )

    if resultado.get("numeros_no_verificados"):
        lista = ", ".join(resultado["numeros_no_verificados"])
        partes.append(
            f"- Estos números NO se encontraron en ningún fragmento recuperado: {lista}. Revisa si los "
            "escribiste mal, si los confundiste con un dato distinto, o si los inventaste — corrígelos "
            "usando el número exacto del fragmento correspondiente, o quítalos si no puedes verificarlos."
        )

    partes.append("Genera la respuesta corregida completa, manteniendo el resto igual.")
    return "\n".join(partes)
