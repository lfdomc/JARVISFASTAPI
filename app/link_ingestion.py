import logging
from app import firecrawl_client, gemini_client, ingestion, chunking, logging_utils

logger = logging.getLogger("link_ingestion")

LIMITE_CHUNKS_INMEDIATO = 80


async def guardar_link_web(url: str, categoria: str, telegram_id: str) -> dict:
    resultado = await firecrawl_client.scrapear(url)
    if not resultado:
        return {"exito": False, "mensaje": "No se pudo extraer contenido del link (Firecrawl falló o el sitio no es accesible)."}

    nombre_completo = f"{resultado['titulo']} — {url}"

    dup = await ingestion.buscar_documento_duplicado(
        __import__("hashlib").sha256(resultado["markdown"].encode("utf-8")).hexdigest()
    )
    if dup:
        return {"exito": False, "mensaje": f"Este contenido ya está guardado como \"{dup['nombre_archivo']}\"."}

    chunks = chunking.crear_chunks_markdown(resultado["markdown"])
    if len(chunks) > LIMITE_CHUNKS_INMEDIATO:
        logger.warning(f"Artículo con {len(chunks)} fragmentos, excede el límite inmediato — se procesa igual (sin cola en esta versión).")

    documento_id = await ingestion.crear_documento_maestro(nombre_completo, resultado["markdown"], categoria, "text/markdown", creado_por="telegram_link")
    if not documento_id:
        return {"exito": False, "mensaje": "No se pudo crear el documento maestro en Supabase."}

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

    vector_referencia = await gemini_client.generar_embedding(f"{categoria}: {resultado['titulo']}")

    guardados = 0
    descartados = 0
    for i, chunk in enumerate(chunks):
        embedding = await gemini_client.generar_embedding(chunk)
        if not embedding:
            continue
        if vector_referencia and ingestion.similitud_coseno(vector_referencia, embedding) < ingestion.UMBRAL_RELEVANCIA_FRAGMENTO:
            descartados += 1
            continue
        metadata = {"origen": url, "chunk_index": i + 1, "total_chunks": len(chunks)}
        if await ingestion.guardar_fragmento(chunk, categoria, embedding, documento_id, metadata):
            guardados += 1

    if guardados == 0 and len(chunks) > 0:
        return {"exito": False, "mensaje": "El artículo se guardó, pero ningún fragmento pasó el filtro de relevancia."}

    return {
        "exito": True,
        "titulo": resultado["titulo"],
        "categoria": categoria,
        "fragmentos": guardados,
        "descartados": descartados,
        "advertencia": advertencia,
    }
