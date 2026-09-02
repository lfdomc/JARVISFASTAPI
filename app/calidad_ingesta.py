"""
Autochequeo de calidad de ingesta.

Todo lo que verificamos A MANO durante la sesión de hoy (palabras
cortadas por guion, huecos de página, fragmentos del índice colados,
tamaños de fragmento anómalos) se vuelve código que corre AUTOMÁTICAMENTE
en cada documento que se sube — sin importar de qué tipo sea. No
reemplaza probar con documentos reales de distintos tipos (eso sigue
haciendo falta para calibrar cosas como el detector de índice), pero sí
elimina la necesidad de ir a revisar a mano cada vez: si algo estructural
salió mal, queda registrado con evidencia, listo para revisar.

Deliberadamente NO usa IA — son chequeos estructurales, deterministas,
sin costo de tokens ni de otra llamada al modelo.
"""
import re

PATRON_TERMINA_EN_GUION = re.compile(r"\w-\s*$")

# Un fragmento "normal" debería rondar el tamaño configurado (350
# palabras ~ 2100 caracteres). Fuera de este rango con frecuencia
# delata un problema de troceo, no necesariamente un error, pero vale
# la pena señalarlo para revisión humana.
CARACTERES_FRAGMENTO_SOSPECHOSAMENTE_CORTO = 80
CARACTERES_FRAGMENTO_SOSPECHOSAMENTE_LARGO = 4000


def auditar_calidad_ingesta(chunks: list[dict], total_paginas: int) -> dict:
    """
    Revisa la lista de fragmentos YA trocedos (antes de guardarlos) contra
    invariantes estructurales que deberían cumplirse sin importar el tipo
    de documento.

    Devuelve un reporte:
    {
        "ok": bool,                          # True si no se encontró nada sospechoso
        "fragmentos_palabra_cortada": [...],  # índices de fragmentos con palabra cortada
        "paginas_sin_cobertura": [...],       # páginas del documento sin ningún fragmento
        "fragmentos_tamano_anomalo": [...],   # fragmentos muy cortos o muy largos
        "total_fragmentos": int,
    }
    """
    fragmentos_palabra_cortada = []
    fragmentos_tamano_anomalo = []
    paginas_cubiertas = set()

    for i, chunk in enumerate(chunks):
        texto = chunk.get("texto", "")

        if PATRON_TERMINA_EN_GUION.search(texto.rstrip()):
            fragmentos_palabra_cortada.append({
                "indice": i, "pagina_inicio": chunk.get("pagina_inicio"),
                "fin_del_texto": texto[-60:],
            })

        largo = len(texto)
        if largo < CARACTERES_FRAGMENTO_SOSPECHOSAMENTE_CORTO or largo > CARACTERES_FRAGMENTO_SOSPECHOSAMENTE_LARGO:
            fragmentos_tamano_anomalo.append({
                "indice": i, "pagina_inicio": chunk.get("pagina_inicio"), "caracteres": largo,
            })

        p_ini, p_fin = chunk.get("pagina_inicio"), chunk.get("pagina_fin")
        if p_ini and p_fin:
            paginas_cubiertas.update(range(p_ini, p_fin + 1))

    paginas_sin_cobertura = [p for p in range(1, total_paginas + 1) if p not in paginas_cubiertas]

    ok = not fragmentos_palabra_cortada and not paginas_sin_cobertura

    return {
        "ok": ok,
        "fragmentos_palabra_cortada": fragmentos_palabra_cortada,
        "paginas_sin_cobertura": paginas_sin_cobertura,
        "fragmentos_tamano_anomalo": fragmentos_tamano_anomalo,
        "total_fragmentos": len(chunks),
    }


def resumen_legible(reporte: dict) -> str:
    """Versión corta para logs — una línea, fácil de escanear en Railway."""
    if reporte["ok"] and not reporte["fragmentos_tamano_anomalo"]:
        return f"✅ Calidad de ingesta OK ({reporte['total_fragmentos']} fragmentos, sin anomalías)."

    partes = []
    if reporte["fragmentos_palabra_cortada"]:
        partes.append(f"{len(reporte['fragmentos_palabra_cortada'])} fragmento(s) con palabra cortada")
    if reporte["paginas_sin_cobertura"]:
        partes.append(f"{len(reporte['paginas_sin_cobertura'])} página(s) sin cobertura: {reporte['paginas_sin_cobertura'][:10]}")
    if reporte["fragmentos_tamano_anomalo"]:
        partes.append(f"{len(reporte['fragmentos_tamano_anomalo'])} fragmento(s) de tamaño atípico")

    return "⚠️ Calidad de ingesta con hallazgos: " + "; ".join(partes)


def validar_indice_contra_chunks(entradas_indice: list[dict], chunks: list[dict]) -> dict:
    """
    Usa el ÍNDICE del documento como su propia respuesta correcta: cada
    entrada ("3.5 Empleo turístico, pág. 50") es un caso de prueba que el
    documento mismo nos da gratis — sin IA, sin datos externos. Por cada
    entrada, revisa si de verdad quedó un fragmento guardado que cubra
    esa página. Si el troceo se equivocó de página en algún tramo (como
    pasó hoy con "Meta 4a"), esto lo atrapa automáticamente, con
    cualquier documento que tenga índice — no hace falta que alguien
    pregunte por esa sección para descubrirlo.

    Devuelve:
    {
        "ok": bool,
        "entradas_sin_cobertura": [           # el índice dice X está en la
            {"numero": str, "titulo": str,     # página N, pero ningún
             "pagina_esperada": int},          # fragmento cubre esa página
            ...
        ],
        "total_entradas_verificadas": int,
    }
    """
    if not entradas_indice or not chunks:
        return {"ok": True, "entradas_sin_cobertura": [], "total_entradas_verificadas": 0}

    paginas_cubiertas = set()
    for c in chunks:
        p_ini, p_fin = c.get("pagina_inicio"), c.get("pagina_fin")
        if p_ini and p_fin:
            paginas_cubiertas.update(range(p_ini, p_fin + 1))

    entradas_sin_cobertura = []
    for entrada in entradas_indice:
        pagina_esperada = entrada.get("pagina")
        if pagina_esperada and pagina_esperada not in paginas_cubiertas:
            entradas_sin_cobertura.append({
                "numero": entrada.get("numero"),
                "titulo": entrada.get("titulo"),
                "pagina_esperada": pagina_esperada,
            })

    return {
        "ok": not entradas_sin_cobertura,
        "entradas_sin_cobertura": entradas_sin_cobertura,
        "total_entradas_verificadas": len(entradas_indice),
    }


def resumen_legible_validacion_indice(reporte: dict) -> str:
    if reporte["ok"]:
        return f"✅ Índice validado contra fragmentos: {reporte['total_entradas_verificadas']}/{reporte['total_entradas_verificadas']} páginas cubiertas."
    faltantes = ", ".join(f"\"{e['numero']} {e['titulo']}\" (pág. {e['pagina_esperada']})" for e in reporte["entradas_sin_cobertura"][:5])
    return (
        f"⚠️ El índice menciona {len(reporte['entradas_sin_cobertura'])}/{reporte['total_entradas_verificadas']} "
        f"entradas cuya página no quedó cubierta por ningún fragmento: {faltantes}"
    )
