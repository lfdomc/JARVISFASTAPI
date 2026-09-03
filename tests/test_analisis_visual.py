"""
Pruebas de analisis_visual.py — el reemplazo del umbral ciego de
caracteres. Usa Gemini simulado (mock) para no depender de una clave
real ni de internet; lo que se prueba es la LÓGICA de combinación de
texto, no la llamada real a la API.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from unittest.mock import patch, AsyncMock
import app.analisis_visual as av


def _correr(coro):
    return asyncio.run(coro)


class TestPreservacionDeTextoReal:
    """El hallazgo real de hoy: el umbral anterior DESCARTABA texto
    corto pero real (títulos, divisores de sección) — esto nunca debe
    volver a pasar."""

    def test_titulo_corto_real_se_preserva(self):
        with patch.object(av, "_renderizar_pagina_como_png", return_value=b"fake_png"), \
             patch.object(av, "_analizar_imagen_con_gemini", new=AsyncMock(return_value="Es un divisor de sección con el título 'Capítulo 3'.")):
            resultado = _correr(av.analizar_paginas_con_poco_texto(b"fake_pdf", ["Capítulo 3"]))
        assert "Capítulo 3" in resultado[0]  # el texto original NUNCA desaparece

    def test_pagina_vacia_confirmada_no_pierde_informacion(self):
        with patch.object(av, "_renderizar_pagina_como_png", return_value=b"fake_png"), \
             patch.object(av, "_analizar_imagen_con_gemini", new=AsyncMock(return_value="Página en blanco.")):
            resultado = _correr(av.analizar_paginas_con_poco_texto(b"fake_pdf", [""]))
        assert "en blanco" in resultado[0].lower()

    def test_pie_de_pagina_enganoso_se_detecta(self):
        """El caso real que el umbral anterior se perdía: una página
        que es enteramente imagen, pero tiene un pie de página repetido
        de ~40 caracteres — no se marcaba antes, ahora sí se analiza."""
        texto_pie_de_pagina = "Plan nacional de turismo de Costa Rica.147"
        with patch.object(av, "_renderizar_pagina_como_png", return_value=b"fake_png"), \
             patch.object(av, "_analizar_imagen_con_gemini", new=AsyncMock(return_value="Muestra un gráfico de barras con llegadas turísticas 2018-2027.")):
            resultado = _correr(av.analizar_paginas_con_poco_texto(b"fake_pdf", [texto_pie_de_pagina]))
        assert texto_pie_de_pagina in resultado[0]  # el pie de página no se pierde
        assert "gráfico de barras" in resultado[0]  # y se agrega la descripción real


class TestManejoDeFallos:
    def test_analisis_fallido_deja_texto_original(self):
        """Si Gemini falla (sin clave, error de red, lo que sea), la
        página se queda con su texto original — nunca rompe la ingesta."""
        with patch.object(av, "_renderizar_pagina_como_png", return_value=b"fake_png"), \
             patch.object(av, "_analizar_imagen_con_gemini", new=AsyncMock(return_value=None)):
            resultado = _correr(av.analizar_paginas_con_poco_texto(b"fake_pdf", ["Texto corto original"]))
        assert resultado[0] == "Texto corto original"

    def test_renderizado_fallido_no_rompe(self):
        with patch.object(av, "_renderizar_pagina_como_png", return_value=None):
            resultado = _correr(av.analizar_paginas_con_poco_texto(b"fake_pdf", ["Texto que no se pudo renderizar"]))
        assert resultado[0] == "Texto que no se pudo renderizar"


class TestUmbral:
    def test_paginas_con_suficiente_texto_no_se_analizan(self):
        texto_largo = "Este es un párrafo con contenido real y sustancial. " * 10  # > 150 caracteres
        with patch.object(av, "_renderizar_pagina_como_png") as mock_render:
            resultado = _correr(av.analizar_paginas_con_poco_texto(b"fake_pdf", [texto_largo]))
        mock_render.assert_not_called()  # nunca debió intentar renderizar esta página
        assert resultado[0] == texto_largo
