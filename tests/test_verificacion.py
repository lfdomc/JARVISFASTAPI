"""
Pruebas de verificacion.py — reproduce los casos reales de alucinación
que encontramos hoy: la frase "amigable, cultivada y feliz" (fabricada),
números fuera de comillas ("4,9 mil millones" inventado), y el rango
compartido en español ("entre el 70 y el 80%") que causaba un falso
positivo.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.verificacion import verificar_respuesta


FRAGMENTOS_PRUEBA = [
    {
        "contenido_chunk": "Mantener la proporción de empresas pequeñas y medianas de hospedaje entre el 70 y el 80% respecto de la oferta nacional.",
        "pagina_inicio": 143, "pagina_fin": 143, "seccion": "6.2 Indicadores de seguimiento",
    },
    {
        "contenido_chunk": "Generar 4,9 mil millones de dólares en el ingreso de divisas por concepto de turismo al 2027.",
        "pagina_inicio": 140, "pagina_fin": 140, "seccion": "6.2 Indicadores de seguimiento",
    },
]


class TestVerificacionDeCitas:
    def test_cita_real_no_se_marca(self):
        respuesta = 'El plan busca captar "4,9 mil millones de dólares en el ingreso de divisas por concepto de turismo al 2027".'
        resultado = verificar_respuesta(respuesta, FRAGMENTOS_PRUEBA)
        assert resultado["citas_no_verificadas"] == []

    def test_cita_fabricada_se_detecta(self):
        """El caso real de hoy: 'amigable, cultivada y feliz' no existía
        en ningún fragmento."""
        respuesta = 'La sociedad se caracteriza por ser "amigable, cultivada y feliz" según el plan.'
        fragmentos_sin_esa_frase = [{
            "contenido_chunk": "El país busca posicionarse como un destino atractivo.",
            "pagina_inicio": 1, "pagina_fin": 1, "seccion": None,
        }]
        resultado = verificar_respuesta(respuesta, fragmentos_sin_esa_frase)
        assert len(resultado["citas_no_verificadas"]) == 1


class TestVerificacionDeNumeros:
    def test_rango_compartido_no_da_falso_positivo(self):
        """El bug real que encontramos: '70 y el 80%' en el documento
        comparte un solo símbolo % — pero '70%' y '80%' escritos por
        separado en la respuesta SÍ deben verificarse como reales."""
        respuesta = "El plan busca mantener entre el 70% y el 80% de PYMES en hospedaje."
        resultado = verificar_respuesta(respuesta, FRAGMENTOS_PRUEBA)
        assert resultado["numeros_no_verificados"] == []

    def test_numero_fabricado_se_detecta(self):
        respuesta = "Se busca un crecimiento del 15% en asegurados."
        resultado = verificar_respuesta(respuesta, FRAGMENTOS_PRUEBA)
        assert "15%" in resultado["numeros_no_verificados"]

    def test_numero_real_sin_comillas_no_se_marca(self):
        """El punto de la ronda 3 de hoy: los números deben verificarse
        aunque NO estén entre comillas (el modelo a veces parafrasea)."""
        respuesta = "El plan busca generar 4,9 mil millones de dólares en divisas."
        resultado = verificar_respuesta(respuesta, FRAGMENTOS_PRUEBA)
        assert resultado["numeros_no_verificados"] == []


class TestVerificacionDePaginaPorCita:
    def test_pagina_correcta_no_se_marca(self):
        respuesta = 'Se busca "generar 4,9 mil millones de dólares en el ingreso de divisas por concepto de turismo al 2027" (pág. 140).'
        resultado = verificar_respuesta(respuesta, FRAGMENTOS_PRUEBA)
        assert resultado["citas_pagina_incorrecta"] == []

    def test_pagina_incorrecta_para_cita_real_se_detecta(self):
        """Cita real, pero atribuida a la página de OTRO fragmento."""
        respuesta = 'Se busca "generar 4,9 mil millones de dólares en el ingreso de divisas por concepto de turismo al 2027" (pág. 143).'
        resultado = verificar_respuesta(respuesta, FRAGMENTOS_PRUEBA)
        assert len(resultado["citas_pagina_incorrecta"]) == 1
        assert resultado["citas_pagina_incorrecta"][0]["pagina_real"] == "pág. 140"


class TestNumeracionInternaDelDocumentoNoSeConfundeConCita:
    """Bug real encontrado hoy: un manual (CLIA 1000) imprime su propia
    numeración interna de capítulo (ej. 'página 1-42', muy distinta a la
    página real del PDF). El modelo la menciona correctamente como texto
    descriptivo — pero el verificador la confundía con una cita real
    nuestra, ya que buscaba 'página N' en TODO el texto, no solo dentro
    de paréntesis (donde SIEMPRE van nuestras citas reales)."""

    def test_numero_de_pagina_en_prosa_sin_parentesis_se_ignora(self):
        fragmentos = [{"contenido_chunk": "contenido real de la página", "pagina_inicio": 79, "pagina_fin": 80, "seccion": None}]
        respuesta = 'Se encuentra en la sección 1.5.2, en la página 1-42 (págs. 79-80).'
        resultado = verificar_respuesta(respuesta, fragmentos)
        assert resultado["paginas_no_verificadas"] == []

    def test_cita_real_entre_parentesis_sigue_verificandose(self):
        """El arreglo no debe volverse permisivo — una cita real (entre
        paréntesis) que de verdad no corresponde a ningún fragmento debe
        seguir marcándose."""
        fragmentos = [{"contenido_chunk": "contenido real", "pagina_inicio": 79, "pagina_fin": 80, "seccion": None}]
        respuesta = "Esto se encuentra en la página (pág. 999)."
        resultado = verificar_respuesta(respuesta, fragmentos)
        assert 999 in resultado["paginas_no_verificadas"]


class TestExigirMarcadorDeCita:
    """El caso real de hoy: la instrucción decía 'SI necesitas citar' —
    dejaba la decisión al modelo, y respondía la misma pregunta unas
    veces con cita y otras sin ninguna, sin la línea de Fuentes al
    final. Ahora es obligatorio, con esta red de seguridad a nivel de
    código para cuando el modelo la ignore de todas formas."""

    FRAGMENTOS_CMD = [{"contenido_chunk": "El CMD 800 es un analizador automático de química clínica", "pagina_inicio": 5, "pagina_fin": 5, "seccion": "CMD 800 F1"}]

    def test_respuesta_sustancial_sin_ningun_marcador_falla(self):
        respuesta = "Señor, el CMD 800 es un analizador automático de química clínica, diseñado para laboratorios de mediana complejidad, capaz de realizar pruebas fotométricas."
        resultado = verificar_respuesta(respuesta, self.FRAGMENTOS_CMD, exigir_marcador_de_cita=True)
        assert resultado["ok"] is False
        assert resultado["sin_ninguna_cita"] is True

    def test_respuesta_con_marcador_pasa(self):
        respuesta = "Señor, el CMD 800 es un analizador automático de química clínica [F1]."
        resultado = verificar_respuesta(respuesta, self.FRAGMENTOS_CMD, exigir_marcador_de_cita=True)
        assert resultado["ok"] is True

    def test_modo_profundo_no_exige_marcador(self):
        """Modo profundo ya viene con citas reales sustituidas por
        construcción — no debe exigir el marcador [F<n>], que ahí ni
        siquiera aplica."""
        respuesta = "Señor, el CMD 800 es un analizador automático de química clínica (pág. 5)."
        resultado = verificar_respuesta(respuesta, self.FRAGMENTOS_CMD, exigir_marcador_de_cita=False)
        assert resultado["ok"] is True

    def test_saludo_corto_no_exige_cita(self):
        resultado = verificar_respuesta("Buenos días, señor.", self.FRAGMENTOS_CMD, exigir_marcador_de_cita=True)
        assert resultado["ok"] is True

    def test_sin_fragmentos_no_exige_cita(self):
        resultado = verificar_respuesta("Señor, no tengo información sobre eso en la base de conocimiento.", [], exigir_marcador_de_cita=True)
        assert resultado["ok"] is True
