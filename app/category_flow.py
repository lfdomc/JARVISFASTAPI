import httpx
from app.config import settings
from app import state, telegram_client, gemini_client, ingestion


def _headers() -> dict:
    return {
        "apikey": settings.SUPABASE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def _base() -> str:
    return settings.SUPABASE_URL.rstrip("/")


async def _categorias_frecuentes(limite: int = 6) -> list[str]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{_base()}/rest/v1/fragmentos_vectoriales_gdp?select=categoria&limit=5000", headers=_headers())
        if resp.status_code != 200:
            return []
        conteo: dict[str, int] = {}
        for fila in resp.json():
            cat = fila.get("categoria")
            if cat:
                conteo[cat] = conteo.get(cat, 0) + 1
        return [c for c, _ in sorted(conteo.items(), key=lambda x: x[1], reverse=True)[:limite]]


async def pedir_categoria_interactiva(chat_id: int, tipo: str, datos: dict, categoria_sugerida: str | None = None):
    token = state.guardar_categoria_pendiente({"tipo": tipo, **datos})
    categorias = await _categorias_frecuentes()

    filas = []
    if categoria_sugerida:
        filas.append([{"text": f"⭐ {categoria_sugerida} (sugerida)", "callback_data": f"catsel:{token}:{categoria_sugerida}"}])

    fila_actual = []
    for cat in categorias:
        if cat == categoria_sugerida:
            continue
        fila_actual.append({"text": cat, "callback_data": f"catsel:{token}:{cat}"})
        if len(fila_actual) == 2:
            filas.append(fila_actual)
            fila_actual = []
    if fila_actual:
        filas.append(fila_actual)

    filas.append([{"text": "➕ Nueva categoría", "callback_data": f"catnew:{token}"}])
    filas.append([{"text": "❌ Cancelar", "callback_data": f"catcancel:{token}"}])

    etiqueta = "este artículo" if tipo == "link" else "esta nota"
    await telegram_client.enviar_mensaje_con_botones(chat_id, f"📂 ¿En qué categoría guardo {etiqueta}, señor?", filas)


async def finalizar_guardado_con_categoria(token: str, categoria_elegida: str, chat_id: int, telegram_id: str):
    datos = state.obtener_categoria_pendiente(token)
    if not datos:
        await telegram_client.enviar_mensaje(chat_id, "⚠️ La solicitud ha expirado, señor. Intente de nuevo.")
        return

    categoria_final = categoria_elegida.strip().upper().replace(" ", "_") or "JARVIS_NOTA"

    if datos["tipo"] == "link":
        from app import link_ingestion
        await telegram_client.enviar_mensaje(chat_id, "🔗 Leyendo el artículo, señor — le aviso apenas termine de indexarlo.")
        resultado = await link_ingestion.iniciar_guardado_link_web(datos["url"], categoria_final, chat_id)
        if not resultado["exito"]:
            await telegram_client.enviar_mensaje(chat_id, "❌ " + resultado.get("mensaje", "No se pudo procesar el link."))

    elif datos["tipo"] == "nota":
        await _procesar_confirmacion_nota(chat_id, datos["texto"], categoria_final, telegram_id, datos.get("nombre_completo", ""))


async def _procesar_confirmacion_nota(chat_id: int, texto: str, categoria: str, telegram_id: str, nombre_completo: str):
    """Chequea duplicados y muestra el paso final de confirmación (✅/❌)."""
    advertencia_dup = ""
    try:
        embedding = await gemini_client.generar_embedding(texto)
        if embedding:
            similar = await ingestion.buscar_nota_similar(embedding)
            if similar:
                porcentaje = round(similar.get("similarity", 0) * 100)
                extracto = str(similar.get("contenido_chunk", ""))[:150]
                advertencia_dup = f"\n\n⚠️ *Posible duplicado ({porcentaje}% similar)* en categoría `{similar.get('categoria')}`:\n\"_{extracto}..._\""
    except Exception:
        pass

    token = state.guardar_categoria_pendiente({"tipo": "confirmar_nota", "texto": texto, "categoria": categoria, "telegram_id": telegram_id})
    teclado = [[
        {"text": "✅ Confirmar y Guardar", "callback_data": f"save_yes:{token}"},
        {"text": "❌ Cancelar", "callback_data": f"save_no:{token}"},
    ]]
    await telegram_client.enviar_mensaje_con_botones(
        chat_id,
        f"Señor, ¿confirma que proceda a registrar esta información?\n\n📌 *Categoría:* `{categoria}`\n👤 *Operador:* `{nombre_completo} ({telegram_id})`\n📝 *Texto:* \"{texto}\"{advertencia_dup}",
        teclado
    )


async def manejar_callback_query(callback_query: dict):
    chat_id = callback_query["message"]["chat"]["id"]
    callback_query_id = callback_query["id"]
    data = callback_query.get("data", "")
    partes = data.split(":")
    accion = partes[0]
    token = partes[1] if len(partes) > 1 else ""

    await telegram_client.responder_callback(callback_query_id, "Procesando...")

    if accion == "catsel":
        categoria_elegida = ":".join(partes[2:])
        from_user = callback_query.get("from", {})
        telegram_id = str(from_user.get("id", chat_id))
        await finalizar_guardado_con_categoria(token, categoria_elegida, chat_id, telegram_id)

    elif accion == "catnew":
        state.marcar_esperando_categoria_nueva(str(chat_id), token)
        await telegram_client.enviar_mensaje(chat_id, "✏️ Escribe el nombre de la nueva categoría, señor:")

    elif accion == "catcancel":
        state.obtener_categoria_pendiente(token)
        await telegram_client.enviar_mensaje(chat_id, "❌ Operación cancelada.")

    elif accion == "save_yes":
        datos = state.obtener_categoria_pendiente(token)
        if not datos:
            await telegram_client.enviar_mensaje(chat_id, "⚠️ La orden ha expirado, señor.")
            return
        embedding = await gemini_client.generar_embedding(datos["texto"])
        if not embedding:
            await telegram_client.enviar_mensaje(chat_id, "❌ No se pudo generar el embedding.")
            return
        from app import supabase_client
        exito = await supabase_client.guardar_nota_personal(datos["texto"], datos["categoria"], embedding, datos["telegram_id"])
        if exito:
            await telegram_client.enviar_mensaje(chat_id, f"✅ Hecho, señor. Información guardada en `{datos['categoria']}`.")
        else:
            await telegram_client.enviar_mensaje(chat_id, "❌ Hubo un fallo técnico al guardar.")

    elif accion == "save_no":
        state.obtener_categoria_pendiente(token)
        await telegram_client.enviar_mensaje(chat_id, "❌ Operación cancelada.")
