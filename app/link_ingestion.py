import asyncio
import hashlib
import logging
from app import firecrawl_client, gemini_client, ingestion, telegram_client, logging_utils

logger = logging.getLogger("link_ingestion")


async def iniciar_guardado_link_web(url: str, categoria: str, chat_id: int) -> dict:
    """
    Fase 1 (rápida, corre dentro de la respuesta al webhook): scrapea,
    revisa duplicados, y crea el documento maestro en estado PROCESSING.
    La fase pesada (chunking + embeddings) se agenda aparte y no bloquea
    esta respuesta.
    """
    resultado = await firecrawl_client.scrapear(url)
    if not resultado:
        return {"exito": False, "mensaje": "No se pudo extraer contenido del link (Firecrawl falló o el sitio no es accesible)."}

    hash_contenido = hashlib.sha256(resultado["markdown"].encode("utf-8")).hexdigest()
    dup = await ingestion.buscar_documento_duplicado(hash_contenido)
    if dup:
        return {"exito": False, "mensaje": f"Este contenido ya está guardado como \"{dup['nombre_archivo']}\"."}

    nombre_completo = f"{resultado['titulo']} — {url}"
    documento_id = await ingestion.crear_documento_maestro(
        nombre_completo, resultado["markdown"], categoria, "text/markdown",
        creado_por="telegram_link", estado="PROCESSING"
    )
    if not documento_id:
        return {"exito": False, "mensaje": "No se pudo crear el documento maestro en Supabase."}

    # Chequeo de coherencia — rápido, se hace ya, para incluirlo en el
    # mensaje final sin tener que esperar a que termine todo el chunking.
    advertencia = None
    try:
        muestra = (resultado["titulo"] + "\n" + resultado["markdown"])[:1500]
        emb_muestra = await gemini_client.generar_embedding(muestra)
        if emb_muestra:
            similitud, comparados = await ingestion.verificar_coherencia_categoria(emb_muestra, categoria)
            if comparados > 0 and similitud < ingestion.UMBRAL_COHERENCIA_CATEGORIA:
                advertencia = f"\n\n🤔 Este artículo no se parece mucho a lo que ya tienes en '{categoria}' ({round(similitud*100)}% de coherencia)."
    except Exception as e:
        logger.warning(f"No se pudo verificar coherencia: {e}")

    # Agenda la fase pesada — se ejecuta en el mismo proceso, sin bloquear
    # la respuesta del webhook a Telegram.
    asyncio.create_task(
        _procesar_y_notificar(documento_id, resultado["markdown"], categoria, nombre_completo, resultado["titulo"], chat_id, advertencia)
    )

    return {
        "exito": True,
        "en_segundo_plano": True,
        "titulo": resultado["titulo"],
        "categoria": categoria,
    }


async def _procesar_y_notificar(documento_id: str, markdown: str, categoria: str, nombre: str, titulo: str, chat_id: int, advertencia: str | None):
    try:
        guardados, descartados = await ingestion.procesar_fragmentos_en_segundo_plano(documento_id, markdown, categoria, nombre)

        if guardados == 0:
            await telegram_client.enviar_mensaje(chat_id, f"❌ \"{titulo}\" se guardó, pero ningún fragmento pasó el filtro de relevancia, señor.")
            return

        linea_descartados = f"\n🚫 Descartados por baja relevancia: {descartados}" if descartados else ""
        await telegram_client.enviar_mensaje(
            chat_id,
            f"✅ Indexado: *{titulo}*\n📌 Categoría: `{categoria}`\n🧩 Fragmentos guardados: {guardados}{linea_descartados}{advertencia or ''}"
        )
    except Exception as e:
        await logging_utils.registrar_error("LINK_PROCESAR_NOTIFICAR", str(e), documento_id)
        await telegram_client.enviar_mensaje(chat_id, f"❌ Hubo un problema técnico procesando \"{titulo}\", señor.")
