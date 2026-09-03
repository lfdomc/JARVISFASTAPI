"""
Pruebas de las funciones puras de main.py — el ensamblado de citas
agrupadas, la detección de categoría/subcategoría mencionada, y el
filtro de alta confianza. Ninguna tenía prueba formal — se verificaron
a mano en el sandbox durante el desarrollo del día, lo cual no protege
contra regresiones futuras.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import (
    _ensamblar_respuesta_estructurada,
    _parsear_json_respuesta,
    _filtrar_por_alta_confianza,
    _normalizar_para_cache,
)
from app.models import AfirmacionEstructurada


MAPA_FRAGMENTOS_PRUEBA = {
    "F1": {"pagina_inicio": 104, "pagina_fin": 105, "seccion": "5.2 Modelo de turismo", "nombre_documento": "Plan de turismo"},
    "F2": {"pagina_inicio": 103, "pagina_fin": 104, "seccion": "5.2 Modelo de turismo", "nombre_documento": "Plan de turismo"},
    "F3": {"pagina_inicio": 50, "pagina_fin": 51, "seccion": "3.5 Empleo turístico", "nombre_documento": "Plan de turismo"},
}


class TestEnsambladoDeCitasAgrupadas:
    """El caso real reportado hoy: 4 afirmaciones seguidas citando la
    misma página repetían la cita 4 veces — debe agruparse en 1 sola."""

    def test_citas_iguales_seguidas_se_agrupan(self):
        items = [
            AfirmacionEstructurada(texto="Punto uno.", fragmentos=[1]),
            AfirmacionEstructurada(texto="Punto dos.", fragmentos=[1]),
            AfirmacionEstructurada(texto="Punto tres.", fragmentos=[1]),
        ]
        resultado = _ensamblar_respuesta_estructurada(items, MAPA_FRAGMENTOS_PRUEBA)
        assert resultado.count("págs. 104-105") == 1
        assert "Punto uno." in resultado
        assert "Punto tres." in resultado

    def test_citas_distintas_no_se_agrupan(self):
        items = [
            AfirmacionEstructurada(texto="Dato de la página 104.", fragmentos=[1]),
            AfirmacionEstructurada(texto="Dato de la página 50.", fragmentos=[3]),
        ]
        resultado = _ensamblar_respuesta_estructurada(items, MAPA_FRAGMENTOS_PRUEBA)
        assert "págs. 104-105" in resultado
        assert "págs. 50-51" in resultado

    def test_citas_no_consecutivas_no_se_agrupan_de_mas(self):
        """Si A-B-A (misma cita pero con una distinta en medio), NO deben
        fusionarse A y A — solo se agrupan corridas consecutivas."""
        items = [
            AfirmacionEstructurada(texto="Primero.", fragmentos=[1]),
            AfirmacionEstructurada(texto="Intermedio.", fragmentos=[3]),
            AfirmacionEstructurada(texto="Tercero.", fragmentos=[1]),
        ]
        resultado = _ensamblar_respuesta_estructurada(items, MAPA_FRAGMENTOS_PRUEBA)
        assert resultado.count("págs. 104-105") == 2  # NO se agrupan, están separadas por otra cita

    def test_afirmacion_sin_fragmentos_no_lleva_cita(self):
        items = [AfirmacionEstructurada(texto="Interpretación propia del asistente.", fragmentos=[])]
        resultado = _ensamblar_respuesta_estructurada(items, MAPA_FRAGMENTOS_PRUEBA)
        assert resultado == "Interpretación propia del asistente."

    def test_marcador_residual_se_limpia(self):
        items = [AfirmacionEstructurada(texto="Dato con marcador suelto [F1, F3]", fragmentos=[1])]
        resultado = _ensamblar_respuesta_estructurada(items, MAPA_FRAGMENTOS_PRUEBA)
        assert "[F1" not in resultado

    def test_misma_seccion_paginas_distintas_se_consolida(self):
        """El caso real reportado: 3 fragmentos de la sección 5.2, cada
        uno con un sub-rango de página distinto (102-103, 104-105,
        101-102) — debe consolidarse en UNA cita con el rango real
        101-105, calculado de los datos reales, no adivinado."""
        mapa = {
            "F1": {"pagina_inicio": 102, "pagina_fin": 103, "seccion": "5.2 Modelo de turismo", "nombre_documento": "Plan"},
            "F2": {"pagina_inicio": 104, "pagina_fin": 105, "seccion": "5.2 Modelo de turismo", "nombre_documento": "Plan"},
            "F3": {"pagina_inicio": 101, "pagina_fin": 102, "seccion": "5.2 Modelo de turismo", "nombre_documento": "Plan"},
        }
        items = [
            AfirmacionEstructurada(texto="Punto A.", fragmentos=[3]),
            AfirmacionEstructurada(texto="Punto B.", fragmentos=[1]),
            AfirmacionEstructurada(texto="Punto C.", fragmentos=[2]),
        ]
        resultado = _ensamblar_respuesta_estructurada(items, mapa)
        assert "págs. 101-105" in resultado
        assert resultado.count("Modelo de turismo") == 1  # una sola cita, no tres

    def test_secciones_distintas_no_se_fusionan_por_error(self):
        """Aunque estén seguidas, fragmentos de secciones DISTINTAS no
        deben mezclarse en un solo rango de página falso."""
        mapa = {
            "F1": {"pagina_inicio": 50, "pagina_fin": 51, "seccion": "3.5 Empleo turístico", "nombre_documento": "Plan"},
            "F2": {"pagina_inicio": 104, "pagina_fin": 105, "seccion": "5.2 Modelo de turismo", "nombre_documento": "Plan"},
        }
        items = [
            AfirmacionEstructurada(texto="Dato de empleo.", fragmentos=[1]),
            AfirmacionEstructurada(texto="Dato del modelo.", fragmentos=[2]),
        ]
        resultado = _ensamblar_respuesta_estructurada(items, mapa)
        assert "págs. 50-51" in resultado
        assert "págs. 104-105" in resultado
        # Nunca debe fusionar 51 con 104 en un rango falso 50-105
        assert "págs. 50-105" not in resultado




class TestParsearJsonRespuesta:
    def test_json_limpio(self):
        items = _parsear_json_respuesta('[{"texto": "dato", "fragmentos": [1]}]')
        assert items is not None
        assert len(items) == 1
        assert items[0].texto == "dato"

    def test_json_envuelto_en_markdown(self):
        items = _parsear_json_respuesta('```json\n[{"texto": "dato", "fragmentos": [1]}]\n```')
        assert items is not None
        assert items[0].texto == "dato"

    def test_json_malformado_devuelve_none(self):
        assert _parsear_json_respuesta("esto no es json") is None

    def test_no_es_lista_devuelve_none(self):
        assert _parsear_json_respuesta('{"texto": "dato"}') is None


class TestNormalizarParaCache:
    def test_quita_puntuacion_y_normaliza_espacios(self):
        """OJO: esta función NO quita tildes (esa es _quitar_acentos, una
        función distinta) — solo puntuación y espacios repetidos."""
        a = _normalizar_para_cache("¿Cuál es la meta de   turismo?")
        b = _normalizar_para_cache("cuál es la meta de turismo")
        assert a == b == "cuál es la meta de turismo"


class TestFiltroAltaConfianza:
    def test_no_recorta_con_pocos_fragmentos(self):
        """El mínimo para activar el filtro es 6 — con menos, no debe tocar nada."""
        fragmentos = [{"similitud_coseno": 0.5 + i * 0.05} for i in range(4)]
        resultado = _filtrar_por_alta_confianza(fragmentos)
        assert len(resultado) == len(fragmentos)

    def test_recorta_cuando_hay_un_match_muy_claro(self):
        fragmentos = [{"similitud_coseno": 0.97}] + [{"similitud_coseno": 0.4 + i * 0.02} for i in range(7)]
        resultado = _filtrar_por_alta_confianza(fragmentos)
        assert len(resultado) < len(fragmentos)
        assert resultado[0]["similitud_coseno"] == 0.97


class TestMencionDeSeccionConNumeroSimple:
    """El bug real de hoy: 'anexo 1'/'anexo 2' (un solo dígito) nunca
    activaban la búsqueda directa por sección — el filtro de longitud
    mínima bloqueaba TODO número corto por igual, incluso cuando se
    busca con la frase completa (no ambigua). Esto hacía que el sistema
    dependiera solo de búsqueda semántica, que a veces confundía
    'Anexo 2' con contenido de otra sección."""

    def _calcular_termino(self, texto):
        import re
        patron = re.compile(r"\b(secci[oó]n|cap[ií]tulo|eje(?:\s+estrat[ée]gico)?|anexo|art[ií]culo|t[ií]tulo|apartado|cl[aá]usula|inciso)\s+(\d+(?:\.\d+){0,3}|[IVXLCDM]+)\b", re.IGNORECASE)
        m = patron.search(texto)
        if not m:
            return None
        numero = m.group(2)
        con_punto = "." in numero
        if con_punto and len(numero) < 2:
            return None
        return numero if con_punto else f"{m.group(1)} {numero}"

    def test_anexo_con_un_solo_digito_no_se_bloquea(self):
        assert self._calcular_termino("en que pagina esta el anexo 2") == "anexo 2"
        assert self._calcular_termino("en que pagina esta el anexo 1") == "anexo 1"

    def test_seccion_con_punto_sigue_buscandose_sola(self):
        """No debe romperse el caso que ya funcionaba bien: la columna
        seccion guarda '3.5 Empleo turístico', SIN la palabra 'sección'
        — buscar la frase completa ahí no encontraría nada."""
        assert self._calcular_termino("explícame la sección 3.5") == "3.5"

    def test_numero_romano_de_un_caracter_ya_no_se_bloquea(self):
        """Antes se bloqueaba por longitud — ahora se busca como frase
        completa ('capítulo I'), que no es ambigua aunque el número en
        sí tenga un solo carácter."""
        assert self._calcular_termino("¿qué dice el capítulo I?") == "capítulo I"
