"""
Blindaje contra alucinaciones — capa 3 de 3.

Las capas 1 y 2 (metadata de página estructurada + instrucciones de
prompt) reducen el problema, pero un LLM puede seguir sin cumplir una
instrucción. Esta capa no depende de que el modelo se porte bien: revisa
la respuesta YA GENERADA contra el texto real de los fragmentos, con
comparación de cadenas (determinística, sin costo de otra llamada a IA
salvo cuando de verdad hace falta corregir).

Qué detecta:
1. Comillas fabricadas — texto entre comillas en la respuesta que NO
   aparece literalmente en ningún fragmento recuperado.
2. Páginas mal atribuidas — un "(pág. N)" en la respuesta donde N no
   corresponde al rango de página de ningún fragmento recuperado.

Qué NO hace: no intenta "entender" si una interpretación es razonable —
eso requeriría otro LLM, con su propio riesgo de alucinar sobre la
alucinación. Esta capa es deliberadamente literal y mecánica.
"""
import re
import unicodedata

PATRON_COMILLAS = re.compile(r'["“”«»]([^"“”«»]{15,})["“”«»]')
PATRON_PAGINA = re.compile(r'p[aá]g(?:s|ina[s]?)?\.?\s*(\d+)(?:\s*[-–—]\s*(\d+))?', re.IGNORECASE)


def _normalizar(texto: str) -> str:
    """Minúsculas, sin acentos, espacios colapsados — para comparar sin
    que un espacio de más o un acento distinto genere un falso positivo."""
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    texto = texto.lower()
    return re.sub(r'\s+', ' ', texto).strip()


def verificar_respuesta(respuesta: str, fragmentos: list[dict]) -> dict:
    """
    Devuelve:
    {
      "ok": bool,
      "citas_no_verificadas": [str, ...],       # texto entre comillas que no se encontró literal
      "paginas_no_verificadas": [int, ...],      # números de página citados sin fragmento que los respalde
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
    for cita in PATRON_COMILLAS.findall(respuesta):
        if _normalizar(cita) not in texto_fragmentos_normalizado:
            citas_no_verificadas.append(cita.strip())

    paginas_no_verificadas = []
    if paginas_disponibles:  # solo aplica si al menos algún fragmento trae página estructurada
        for p_ini_str, p_fin_str in PATRON_PAGINA.findall(respuesta):
            paginas_citadas = [int(p_ini_str)]
            if p_fin_str:
                paginas_citadas.append(int(p_fin_str))
            for p in paginas_citadas:
                if p not in paginas_disponibles and p not in paginas_no_verificadas:
                    paginas_no_verificadas.append(p)

    ok = not citas_no_verificadas and not paginas_no_verificadas
    return {
        "ok": ok,
        "citas_no_verificadas": citas_no_verificadas,
        "paginas_no_verificadas": paginas_no_verificadas,
    }


def construir_instruccion_correctiva(resultado: dict) -> str:
    """Mensaje de corrección específico para reenviar al modelo cuando la
    verificación falla — le señala EXACTAMENTE qué falló, para que no
    tenga que adivinar qué corregir."""
    partes = ["Tu respuesta anterior tiene problemas de verificación que debes corregir:"]

    if resultado["citas_no_verificadas"]:
        lista = "; ".join(f'"{c}"' for c in resultado["citas_no_verificadas"])
        partes.append(
            f"- Estas frases entre comillas NO se encontraron literalmente en los fragmentos: {lista}. "
            "Quítales las comillas y usa tus propias palabras, o elimínalas si no son esenciales."
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
