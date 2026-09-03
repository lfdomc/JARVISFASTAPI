"""
Pruebas de calidad_ingesta.py — el autochequeo estructural y la
detección de idioma mezclado, que hoy solo se habían probado a mano en
el sandbox durante el desarrollo.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.calidad_ingesta import auditar_calidad_ingesta, resumen_legible


def _chunk(texto="Texto de relleno suficientemente largo para pasar el mínimo de caracteres exigido.", pagina_inicio=1, pagina_fin=1, idioma=None):
    return {"texto": texto, "pagina_inicio": pagina_inicio, "pagina_fin": pagina_fin, "idioma": idioma}


class TestDeteccionDeIdiomaMezclado:
    def test_documento_uniforme_no_alerta(self):
        chunks = [_chunk(pagina_inicio=i, pagina_fin=i, idioma="es") for i in range(1, 11)]
        reporte = auditar_calidad_ingesta(chunks, 10)
        assert reporte["idioma_mezclado"] is False
        assert reporte["idioma_dominante"] == "es"

    def test_documento_mezclado_70_30_alerta(self):
        """El caso real: 70% español / 30% inglés — por debajo del 85%
        de umbral, debe alertar."""
        chunks = (
            [_chunk(pagina_inicio=i, pagina_fin=i, idioma="es") for i in range(1, 8)] +
            [_chunk(pagina_inicio=i, pagina_fin=i, idioma="en") for i in range(8, 11)]
        )
        reporte = auditar_calidad_ingesta(chunks, 10)
        assert reporte["idioma_mezclado"] is True
        assert reporte["idioma_dominante"] == "es"
        assert "idioma mezclado" in resumen_legible(reporte)

    def test_documento_95_5_no_alerta(self):
        """95/5 está por encima del umbral del 85% — no debería alertar,
        una mención aislada en otro idioma no amerita aviso."""
        chunks = (
            [_chunk(pagina_inicio=i, pagina_fin=i, idioma="es") for i in range(1, 20)] +
            [_chunk(pagina_inicio=20, pagina_fin=20, idioma="en")]
        )
        reporte = auditar_calidad_ingesta(chunks, 20)
        assert reporte["idioma_mezclado"] is False

    def test_sin_idioma_detectado_no_falla(self):
        """Si ningún fragmento tiene idioma detectado (ej. langdetect
        falló en todos), no debe reventar ni marcar mezcla falsa."""
        chunks = [_chunk(pagina_inicio=i, pagina_fin=i, idioma=None) for i in range(1, 5)]
        reporte = auditar_calidad_ingesta(chunks, 4)
        assert reporte["idioma_mezclado"] is False
        assert reporte["idioma_dominante"] is None


class TestAutochequeoEstructural:
    def test_palabra_cortada_se_detecta(self):
        chunks = [_chunk(texto="Este es un texto que termina en una palabra que quedó cortada por un guion-")]
        reporte = auditar_calidad_ingesta(chunks, 1)
        assert reporte["ok"] is False
        assert len(reporte["fragmentos_palabra_cortada"]) == 1

    def test_hueco_de_pagina_se_detecta(self):
        chunks = [_chunk(pagina_inicio=1, pagina_fin=1), _chunk(pagina_inicio=3, pagina_fin=3)]  # falta la 2
        reporte = auditar_calidad_ingesta(chunks, 3)
        assert reporte["ok"] is False
        assert 2 in reporte["paginas_sin_cobertura"]

    def test_documento_limpio_pasa(self):
        chunks = [_chunk(pagina_inicio=i, pagina_fin=i) for i in range(1, 6)]
        reporte = auditar_calidad_ingesta(chunks, 5)
        assert reporte["ok"] is True
        assert reporte["paginas_sin_cobertura"] == []
