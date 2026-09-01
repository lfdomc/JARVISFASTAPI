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

    ok = not citas_no_verificadas and not citas_pagina_incorrecta and not paginas_no_verificadas
    return {
        "ok": ok,
        "citas_no_verificadas": citas_no_verificadas,
        "citas_pagina_incorrecta": citas_pagina_incorrecta,
        "paginas_no_verificadas": paginas_no_verificadas,
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

    partes.append("Genera la respuesta corregida completa, manteniendo el resto igual.")
    return "\n".join(partes)
