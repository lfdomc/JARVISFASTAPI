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
