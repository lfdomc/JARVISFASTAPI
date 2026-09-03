"""
Pruebas de models.py — el punto exacto donde tuvimos un bug real hoy
("fragmento" singular vs "fragmentos" lista en la respuesta de Gemini).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pydantic import ValidationError
from app.models import AfirmacionEstructurada, FragmentoChunk, EntradaIndice


class TestAfirmacionEstructurada:
    def test_caso_normal(self):
        a = AfirmacionEstructurada.model_validate({"texto": "dato", "fragmentos": [1, 3]})
        assert a.fragmentos == [1, 3]

    def test_entero_suelto_se_normaliza_a_lista(self):
        """El caso real: Gemini a veces manda 'fragmentos': 3 en vez de [3]."""
        a = AfirmacionEstructurada.model_validate({"texto": "dato", "fragmentos": 3})
        assert a.fragmentos == [3]

    def test_sin_fragmentos_queda_lista_vacia(self):
        a = AfirmacionEstructurada.model_validate({"texto": "interpretación propia"})
        assert a.fragmentos == []

    def test_sin_texto_falla_la_validacion(self):
        with pytest.raises(ValidationError):
            AfirmacionEstructurada.model_validate({"fragmentos": [1]})


class TestFragmentoChunk:
    def test_caso_normal(self):
        f = FragmentoChunk.model_validate({"texto": "contenido", "pagina_inicio": 1, "pagina_fin": 2})
        assert f.pagina_inicio == 1

    def test_pagina_no_numerica_falla(self):
        with pytest.raises(ValidationError):
            FragmentoChunk.model_validate({"texto": "contenido", "pagina_inicio": "uno", "pagina_fin": 2})


class TestEntradaIndice:
    def test_caso_normal(self):
        e = EntradaIndice.model_validate({"nivel": 1, "numero": "5.5.4", "titulo": "Encadenamientos", "pagina": 128})
        assert e.pagina == 128

    def test_numero_puede_ser_nulo(self):
        """Para entradas de anexo, que no siempre traen número."""
        e = EntradaIndice.model_validate({"nivel": 1, "numero": None, "titulo": "Anexo 1", "pagina": 45})
        assert e.numero is None
