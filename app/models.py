"""
Modelos Pydantic para las estructuras de datos internas más propensas a
error — no es una conversión exhaustiva de todo el sistema a Pydantic,
es deliberadamente enfocada en los puntos donde YA tuvimos bugs reales
por diccionarios sueltos sin validar (ej. el campo "fragmento" singular
vs "fragmentos" lista en el esquema de respuesta de Gemini).
"""
from pydantic import BaseModel, Field, field_validator


class AfirmacionEstructurada(BaseModel):
    """Un elemento del arreglo JSON que devuelve Gemini en modo profundo.
    Si Gemini alguna vez devuelve una forma distinta a la esperada (el
    tipo de bug real que tuvimos hoy), esto lo rechaza de inmediato con
    un error claro, en vez de fallar en silencio más adelante al
    ensamblar la respuesta."""
    texto: str
    fragmentos: list[int] = Field(default_factory=list)

    @field_validator("fragmentos", mode="before")
    @classmethod
    def _aceptar_entero_suelto(cls, v):
        # Defensivo: si Gemini manda un entero suelto en vez de una lista
        # (ej. "fragmentos": 3 en vez de [3]), se normaliza en vez de
        # fallar la validación completa.
        if isinstance(v, int):
            return [v]
        return v or []


class FragmentoChunk(BaseModel):
    """Un fragmento ya trocedo, con su página real capturada en la
    ingesta — el contrato entre chunking.py y el resto del sistema."""
    texto: str
    pagina_inicio: int
    pagina_fin: int


class EntradaIndice(BaseModel):
    """Una entrada del índice/tabla de contenidos detectado."""
    nivel: int
    numero: str | None = None
    titulo: str
    pagina: int


class ReporteCalidadIngesta(BaseModel):
    """Resultado del autochequeo estructural de un documento recién
    subido — mismo contrato que se guarda en documentos_gdp.reporte_calidad_ingesta."""
    ok: bool
    fragmentos_palabra_cortada: list[dict] = Field(default_factory=list)
    paginas_sin_cobertura: list[int] = Field(default_factory=list)
    fragmentos_tamano_anomalo: list[dict] = Field(default_factory=list)
    total_fragmentos: int
    validacion_indice: dict | None = None
