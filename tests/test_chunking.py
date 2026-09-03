"""
Pruebas de chunking.py — cada caso aquí es un bug REAL que encontramos
hoy con el plan de turismo, convertido en prueba automatizada. Si
alguien vuelve a tocar chunking.py y reintroduce alguno de estos
problemas, esta suite lo atrapa antes de llegar a producción — hoy tuvimos
que descubrirlos a mano, tres veces, con el mismo archivo.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.chunking import crear_chunks_con_paginas, _unir_palabras_cortadas, marcar_paginas_vacias


def _marcar_paginas(paginas: list[str]) -> str:
    """Reconstruye el mismo formato de marcadores que usa chunking.py
    internamente, para pruebas unitarias que no dependen del PDF real."""
    from app.chunking import MARCADOR_PAGINA
    return "".join(MARCADOR_PAGINA.format(n=i + 1) + p for i, p in enumerate(paginas))


class TestGuionesEncadenados:
    """El bug real: 'pro-' / 'cedimientos...cohe-' / 'rente...' — dos
    palabras cortadas SEGUIDAS. La primera versión del arreglo solo unía
    un par a la vez y dejaba el segundo corte sin resolver."""

    def test_una_sola_palabra_cortada(self):
        lineas = ["el turismo mediante el fo-", "mento de la productividad"]
        resultado = _unir_palabras_cortadas(lineas)
        assert len(resultado) == 1
        assert "fomento" in resultado[0]
        assert "fo-" not in resultado[0]

    def test_dos_palabras_cortadas_seguidas(self):
        """El caso real que se nos escapó la primera vez."""
        lineas = [
            "garanticen que los pro-",
            "cedimientos y requisitos de inmigración puedan gestionarse de manera oportuna, fiable y cohe-",
            "rente en todos los sistemas",
        ]
        resultado = _unir_palabras_cortadas(lineas)
        assert len(resultado) == 1
        texto = resultado[0]
        assert "procedimientos" in texto
        assert "coherente" in texto
        assert "pro-" not in texto
        assert "cohe-" not in texto

    def test_guion_con_espacio_antes(self):
        """El caso 'algu -' con espacio — un patrón ligeramente distinto
        al guion pegado directo."""
        lineas = ["están abiertos al turismo, algu -", "nos sin restricciones"]
        resultado = _unir_palabras_cortadas(lineas)
        assert "algunos" in resultado[0]

    def test_no_afecta_lineas_normales(self):
        lineas = ["Esta es una línea normal.", "Y esta es otra línea normal."]
        resultado = _unir_palabras_cortadas(lineas)
        assert resultado == lineas


class TestSeguimientoContinuoDePagina:
    """El bug real: un corte normal por tamaño caía justo donde el
    marcador de una página ya había sido consumido por el fragmento
    ANTERIOR, y el fragmento nuevo perdía el rastro de en qué página
    seguía, aunque su contenido siguiera siendo la misma página."""

    def test_fragmento_no_pierde_pagina_en_corte_normal(self):
        # Página 1 larga (fuerza un corte por tamaño en medio de la
        # página), página 2 corta
        pagina_1 = "Primera oración. " * 200  # suficientemente larga para forzar más de un fragmento
        pagina_2 = "Contenido de la página 2."
        chunks = crear_chunks_con_paginas([pagina_1, pagina_2], max_palabras=50)

        # Ningún fragmento debería tener pagina_inicio > pagina_fin
        for c in chunks:
            assert c["pagina_inicio"] <= c["pagina_fin"]

        # Los fragmentos que vienen de la página 1 deben decir página 1,
        # incluso si el corte cayó a la mitad
        fragmentos_pagina_1 = [c for c in chunks if c["pagina_inicio"] == 1]
        assert len(fragmentos_pagina_1) >= 1

    def test_todas_las_paginas_quedan_cubiertas(self):
        paginas = [f"Contenido de la página {i}. " * 30 for i in range(1, 11)]
        chunks = crear_chunks_con_paginas(paginas, max_palabras=100)

        paginas_cubiertas = set()
        for c in chunks:
            paginas_cubiertas.update(range(c["pagina_inicio"], c["pagina_fin"] + 1))

        assert paginas_cubiertas == set(range(1, 11))

    def test_ningun_fragmento_termina_en_guion(self):
        """Prueba de regresión general — combina hyphens + troceo real."""
        pagina_con_guion = (
            "Texto de relleno. " * 50 +
            "el proceso de gestión requiere de procedimientos ade-\ncuados para su correcta implementación. " +
            "Más texto de relleno. " * 50
        )
        chunks = crear_chunks_con_paginas([pagina_con_guion], max_palabras=30)
        for c in chunks:
            assert not c["texto"].rstrip().endswith("-"), f"Fragmento termina en guion: {c['texto'][-50:]!r}"


class TestFronterasDeSeccion:
    """El caso real: páginas 145-146 casi vacías hacían que un fragmento
    se extendiera desde plena sección 6.2 hasta adentro de Anexo 1 —
    ahora cada entrada del índice es un corte obligatorio."""

    def test_fragmento_no_cruza_frontera_de_seccion(self):
        paginas = [f"Contenido de la sección A, página {i}. " * 10 + "\n" for i in range(1, 4)]
        paginas += [f"Contenido de la sección B, página {i}. " * 10 + "\n" for i in range(4, 7)]
        # Frontera: la sección B empieza en la página 4
        chunks = crear_chunks_con_paginas(paginas, max_palabras=200, paginas_frontera={4})
        for c in chunks:
            assert not (c["pagina_inicio"] < 4 and c["pagina_fin"] >= 4), \
                f"Fragmento {c['pagina_inicio']}-{c['pagina_fin']} cruza la frontera de la página 4"

    def test_fragmento_nuevo_arranca_en_la_pagina_frontera_real(self):
        """El bug específico de hoy: el corte caía en el lugar correcto,
        pero el fragmento nuevo se etiquetaba con la página VIEJA (antes
        de la frontera), no con la página real donde empieza — mismo
        patrón del bug de 'un paso atrás' que ya se había corregido para
        cortes por tamaño, pero no se había corregido para cortes por
        frontera de sección."""
        paginas = ["Página 1 con algo de contenido normal aquí de sobra.\n"]
        paginas += ["Página 2, casi vacía\n"]  # muy poco contenido
        paginas += ["ANEXO 1: contenido real de la nueva sección que empieza aquí mismo.\n"] * 3
        chunks = crear_chunks_con_paginas(paginas, max_palabras=200, paginas_frontera={3})
        fragmentos_de_anexo = [c for c in chunks if "ANEXO 1" in c["texto"]]
        assert len(fragmentos_de_anexo) >= 1
        assert fragmentos_de_anexo[0]["pagina_inicio"] == 3  # no 2, la página vieja

    def test_sin_fronteras_se_comporta_igual_que_antes(self):
        """Sin paginas_frontera (o vacío), el comportamiento debe ser
        idéntico al troceo normal por tamaño — no debe forzar cortes
        de la nada."""
        paginas = [f"Contenido de relleno normal en la página {i}. " * 10 for i in range(1, 6)]
        chunks_sin_frontera = crear_chunks_con_paginas(paginas, max_palabras=200)
        chunks_frontera_vacia = crear_chunks_con_paginas(paginas, max_palabras=200, paginas_frontera=set())
        assert len(chunks_sin_frontera) == len(chunks_frontera_vacia)


class TestMarcadoDePaginasVacias:
    def test_pagina_vacia_se_marca(self):
        paginas = ["", "Contenido real y sustancial de la página dos, con bastante texto útil."]
        marcadas = marcar_paginas_vacias(paginas)
        assert "sin texto extraíble" in marcadas[0]
        assert marcadas[1] == paginas[1]  # la página con contenido real no se toca

    def test_pagina_con_poco_texto_se_marca(self):
        paginas = ["Anexos"]  # 6 caracteres, como el caso real de hoy
        marcadas = marcar_paginas_vacias(paginas)
        assert "sin texto extraíble" in marcadas[0]

    def test_marcador_termina_en_salto_de_linea(self):
        """El bug real: sin el salto de línea final, el marcador se pega
        al texto de la página siguiente, fusionando sus marcadores de
        página en una sola unidad indivisible."""
        marcadas = marcar_paginas_vacias([""])
        assert marcadas[0].endswith("\n")

    def test_chunk_vacio_se_clasifica_como_tal(self):
        from app.chunking import _clasificar_tipo_contenido
        marcadas = marcar_paginas_vacias([""])
        assert _clasificar_tipo_contenido(marcadas[0]) == "vacia_o_imagen"


class TestPreservacionDeTablas:
    def test_tabla_markdown_no_se_parte(self):
        tabla = "\n".join([
            "| Columna A | Columna B |",
            "|---|---|",
            "| dato 1 | dato 2 |",
            "| dato 3 | dato 4 |",
        ])
        pagina = "Texto antes. " * 5 + "\n" + tabla + "\n" + " Texto después. " * 5
        chunks = crear_chunks_con_paginas([pagina], max_palabras=15)  # límite chico a propósito

        # La tabla completa debe aparecer entera en UN solo fragmento
        fragmentos_con_tabla = [c for c in chunks if "Columna A" in c["texto"]]
        assert len(fragmentos_con_tabla) == 1
        assert "dato 4" in fragmentos_con_tabla[0]["texto"]


class TestClasificacionTipoContenido:
    """Distinguir tabla/texto/título — funciona igual sin importar qué
    motor extrajo el texto, porque se basa en el patrón final."""

    def test_tabla_se_reconoce(self):
        from app.chunking import _clasificar_tipo_contenido
        tabla = "| Columna A | Columna B |\n|---|---|\n| dato 1 | dato 2 |"
        assert _clasificar_tipo_contenido(tabla) == "tabla"

    def test_texto_normal_se_reconoce(self):
        from app.chunking import _clasificar_tipo_contenido
        texto = "Este es un párrafo normal con varias oraciones sobre turismo."
        assert _clasificar_tipo_contenido(texto) == "texto"

    def test_titulo_se_reconoce(self):
        from app.chunking import _clasificar_tipo_contenido
        assert _clasificar_tipo_contenido("# 5.5.4 Encadenamientos productivos") == "titulo"

    def test_chunks_incluyen_tipo_contenido(self):
        """El campo debe venir en cada fragmento que arma crear_chunks_con_paginas."""
        chunks = crear_chunks_con_paginas(["Texto de una página normal con suficiente contenido para pasar el mínimo."])
        assert all("tipo_contenido" in c for c in chunks)


class TestDeteccionDeIdioma:
    def test_espanol_se_detecta(self):
        from app.chunking import _detectar_idioma
        assert _detectar_idioma("El turismo en Costa Rica ha crecido significativamente en los últimos años.") == "es"

    def test_ingles_se_detecta(self):
        from app.chunking import _detectar_idioma
        assert _detectar_idioma("Tourism in Costa Rica has grown significantly over the last several years.") == "en"

    def test_chunks_incluyen_idioma(self):
        chunks = crear_chunks_con_paginas(["Texto de una página normal con suficiente contenido para pasar el mínimo."])
        assert all("idioma" in c for c in chunks)
