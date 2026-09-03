"""
Pruebas de indice_extractor.py — los dos formatos reales que probamos
hoy (número-primero, tipo plan de turismo; palabra-primero, tipo
documento legal) y la validación cruzada contra fragmentos.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.indice_extractor import _parsear_entradas, clasificar_tipo_documento, corregir_paginas_indice
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


class TestFormatoPalabraPrimeroIngles:
    """Respaldo para cuando Docling cae a pypdf y el documento no está
    en español."""

    def test_documento_legal_ingles(self):
        texto = "\n".join([
            "Article 1. Scope and purpose. 5",
            "Article 2. Definitions. 7",
            "Chapter 2. Procedure. 10",
        ])
        entradas = _parsear_entradas(texto)
        assert len(entradas) == 3
        assert entradas[0]["numero"] == "Article 1"

    def test_appendix_ingles(self):
        entradas = _parsear_entradas("Appendix 1 45")
        assert len(entradas) == 1
        assert entradas[0]["pagina"] == 45

    def test_anexo_sin_titulo_no_se_pierde(self):
        """El bug real de hoy: 'ANEXO 1            147' (sin texto de
        título entre el número y la página) hacía que el patrón general
        de palabra-primero lo atrapara con un título vacío, y la entrada
        se descartaba en silencio sin darle oportunidad al patrón
        específico de anexo — perdiendo por completo la entrada."""
        entradas = _parsear_entradas("ANEXO 1            147")
        assert len(entradas) == 1
        assert entradas[0]["pagina"] == 147
        assert entradas[0]["titulo"] == "ANEXO 1"

    def test_clasificacion_legal_en_ingles(self):
        entradas = _parsear_entradas("\n".join([
            "Article 1. Scope. 5", "Article 2. Definitions. 7", "Article 3. Requirements. 10",
        ]))
        assert clasificar_tipo_documento(entradas) == "legal_normativo"


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


class TestCorreccionDePaginasIndice:
    """El caso real de hoy: el índice IMPRESO decía 'Anexo 2 → página
    150', pero el contenido real de 'ANEXO 2' empieza en la página 149
    — un error del documento original, no nuestro. Confiar ciegamente
    en el número impreso propagaba ese error a nuestros propios cortes
    de fragmento."""

    def test_pagina_correcta_no_se_toca(self):
        entradas = [{"nivel": 1, "numero": None, "titulo": "ANEXO 1", "pagina": 3}]
        paginas = ["", "", "ANEXO 1: contenido real aquí mismo"]  # está exactamente en la 3
        corregidas = corregir_paginas_indice(entradas, paginas)
        assert corregidas[0]["pagina"] == 3

    def test_pagina_incorrecta_se_corrige_con_la_real(self):
        """El caso real: índice dice página 4, pero el título real está en la 3."""
        entradas = [{"nivel": 1, "numero": None, "titulo": "ANEXO 2", "pagina": 4}]
        paginas = ["", "", "ANEXO 2: contenido real que sí empieza aquí", "otra cosa en la 4"]
        corregidas = corregir_paginas_indice(entradas, paginas)
        assert corregidas[0]["pagina"] == 3  # corregido a donde está de verdad

    def test_sin_coincidencia_cercana_no_se_toca(self):
        """Si no se encuentra en ningún lado cercano, mejor no adivinar
        — se deja el número original del índice tal cual."""
        entradas = [{"nivel": 1, "numero": None, "titulo": "ANEXO 3", "pagina": 4}]
        paginas = ["", "", "", "", "", "", "", "", "", ""]  # ANEXO 3 no aparece en ningún lado
        corregidas = corregir_paginas_indice(entradas, paginas)
        assert corregidas[0]["pagina"] == 4  # no se toca

    def test_titulos_muy_cortos_se_ignoran(self):
        """Títulos de menos de 4 caracteres no se usan para buscar —
        demasiado ambiguos, alto riesgo de falso positivo."""
        entradas = [{"nivel": 1, "numero": "1", "titulo": "A", "pagina": 5}]
        corregidas = corregir_paginas_indice(entradas, ["A"] * 10)
        assert corregidas[0]["pagina"] == 5  # no se toca, título muy corto


class TestFormatoMarkdownDeDocling:
    """Docling exporta el índice como Markdown, no como texto plano de
    pypdf — estos casos confirman que los mismos patrones de siempre lo
    reconocen, sin necesitar una segunda extracción de respaldo."""

    def test_fila_de_tabla_markdown(self):
        entradas = _parsear_entradas("| 5.5.4 Encadenamientos productivos | 128 |")
        assert len(entradas) == 1
        assert entradas[0]["numero"] == "5.5.4"
        assert entradas[0]["pagina"] == 128

    def test_separador_de_tabla_se_descarta(self):
        assert _parsear_entradas("|---|---|") == []

    def test_encabezado_markdown_se_reconoce(self):
        entradas = _parsear_entradas("## 5.5.4 Encadenamientos productivos. 128")
        assert len(entradas) == 1
        assert entradas[0]["pagina"] == 128

    def test_anexo_en_tabla_markdown(self):
        """El caso real que motivó este arreglo: Anexo 2 en formato tabla."""
        entradas = _parsear_entradas("| ANEXO 2 | 149 |")
        assert len(entradas) == 1
        assert entradas[0]["titulo"] == "ANEXO 2"
        assert entradas[0]["pagina"] == 149

    def test_texto_plano_de_pypdf_sigue_funcionando(self):
        """Prueba de regresión — el formato de siempre no debe romperse."""
        entradas = _parsear_entradas("5.5.4 Encadenamientos productivos.        128")
        assert len(entradas) == 1
        assert entradas[0]["pagina"] == 128
