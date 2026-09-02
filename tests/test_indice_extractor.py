"""
Pruebas de indice_extractor.py — los dos formatos reales que probamos
hoy (número-primero, tipo plan de turismo; palabra-primero, tipo
documento legal) y la validación cruzada contra fragmentos.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.indice_extractor import _parsear_entradas, clasificar_tipo_documento
from app.calidad_ingesta import validar_indice_contra_chunks


class TestFormatoNumeroPrimero:
    """Formato del plan de turismo: '5.5.4 Encadenamientos productivos. 128'"""

    def test_entrada_simple(self):
        entradas = _parsear_entradas("5.5.4 Encadenamientos productivos.        128")
        assert len(entradas) == 1
        assert entradas[0]["numero"] == "5.5.4"
        assert entradas[0]["titulo"] == "Encadenamientos productivos"
        assert entradas[0]["pagina"] == 128
        assert entradas[0]["nivel"] == 3

    def test_multiples_entradas(self):
        texto = "1. Contexto internacional. 11\n1.1 Impacto económico. 11\n1.2 Aumento de riesgos. 15"
        entradas = _parsear_entradas(texto)
        assert len(entradas) == 3
        assert entradas[0]["nivel"] == 1
        assert entradas[1]["nivel"] == 2


class TestFormatoPalabraPrimero:
    """Formato legal/normativo: 'Artículo 5. Definiciones. 12'"""

    def test_articulo(self):
        entradas = _parsear_entradas("Artículo 1. Objeto y ámbito de aplicación. 5")
        assert len(entradas) == 1
        assert entradas[0]["numero"] == "Artículo 1"
        assert entradas[0]["titulo"] == "Objeto y ámbito de aplicación"
        assert entradas[0]["pagina"] == 5

    def test_titulo_con_numero_romano(self):
        entradas = _parsear_entradas("TÍTULO I. Disposiciones generales. 5")
        assert len(entradas) == 1
        assert entradas[0]["numero"] == "Título I"

    def test_articulo_con_subnivel_decimal(self):
        entradas = _parsear_entradas("Artículo 3.1 Documentación necesaria. 11")
        assert entradas[0]["nivel"] == 2  # tiene un punto -> subnivel

    def test_documento_legal_completo(self):
        texto = "\n".join([
            "TÍTULO I. Disposiciones generales. 5",
            "Artículo 1. Objeto y ámbito de aplicación. 5",
            "Artículo 2. Definiciones. 7",
            "CAPÍTULO 2. Procedimiento. 10",
            "Artículo 3. Requisitos. 10",
        ])
        entradas = _parsear_entradas(texto)
        assert len(entradas) == 5


class TestClasificacionTipoDocumento:
    def test_documento_legal_se_clasifica_como_legal(self):
        entradas = _parsear_entradas("\n".join([
            "Artículo 1. Objeto. 5", "Artículo 2. Definiciones. 7",
            "Artículo 3. Alcance. 9", "Artículo 4. Requisitos. 11",
        ]))
        assert clasificar_tipo_documento(entradas) == "legal_normativo"

    def test_documento_tecnico_se_clasifica_como_tecnico(self):
        entradas = _parsear_entradas("\n".join([
            "1. Contexto. 11", "1.1 Impacto. 11",
            "5.5.4 Encadenamientos productivos. 128", "5.5.6 Alineamiento. 132",
        ]))
        assert clasificar_tipo_documento(entradas) == "tecnico_planificacion"


class TestValidacionIndiceContraFragmentos:
    """La validación que usa el índice como su propia respuesta correcta
    — sin IA, sin datos externos."""

    def test_todo_cubierto_pasa(self):
        entradas = [{"numero": "3.5", "titulo": "Empleo turístico", "pagina": 50}]
        chunks = [{"texto": "...", "pagina_inicio": 49, "pagina_fin": 51}]
        reporte = validar_indice_contra_chunks(entradas, chunks)
        assert reporte["ok"] is True
        assert reporte["entradas_sin_cobertura"] == []

    def test_pagina_faltante_se_detecta(self):
        """El caso real: Meta 4a decía página 128, pero el fragmento
        real quedó tageado como 129 solamente."""
        entradas = [{"numero": "5.5.4", "titulo": "Encadenamientos productivos", "pagina": 128}]
        chunks = [{"texto": "...", "pagina_inicio": 129, "pagina_fin": 129}]  # falta la 128
        reporte = validar_indice_contra_chunks(entradas, chunks)
        assert reporte["ok"] is False
        assert len(reporte["entradas_sin_cobertura"]) == 1
        assert reporte["entradas_sin_cobertura"][0]["pagina_esperada"] == 128
