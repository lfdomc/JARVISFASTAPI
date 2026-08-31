# JARVIS API (FastAPI)

## Probar localmente

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # y llena tus valores reales
uvicorn app.main:app --reload
```

Para probar el webhook con Telegram real desde tu computadora, usa un túnel
público (ngrok o Cloudflare Tunnel) apuntando a `http://localhost:8000`.

## Desplegar en Railway

1. Sube este proyecto a un repositorio de GitHub (debe incluir el `Procfile`).
2. En [railway.com](https://railway.com): **New Project** -> **Deploy from GitHub repo** -> selecciona el repositorio.
3. Railway detecta Python automáticamente y usa el `Procfile` para saber
   cómo arrancar — no necesitas configurar build/start command a mano.
4. Ve a la pestaña **Variables** del servicio y agrega todas las de
   `.env.example` con tus valores reales (incluye tantas `GEMINI_KEY_N`
   como cuentas tengas).
5. Ve a **Settings** -> **Networking** -> **Generate Domain**. Railway te
   da una URL pública tipo `https://tu-servicio.up.railway.app`.
6. Cada vez que hagas `git push`, Railway despliega automáticamente.

## Configurar el webhook de Telegram apuntando a Railway

Una sola vez, corre esto (reemplaza los valores):

```bash
curl -X POST "https://api.telegram.org/bot<TU_TOKEN>/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://tu-servicio.up.railway.app/webhook/telegram",
    "secret_token": "el-mismo-valor-de-TELEGRAM_WEBHOOK_SECRET"
  }'
```

Confirma que quedó bien configurado:
```bash
curl "https://api.telegram.org/bot<TU_TOKEN>/getWebhookInfo"
```


## Qué falta para tener paridad completa con la versión de Apps Script

Esto es un punto de partida funcional (responde preguntas con RAG, guarda
notas), no una réplica 1:1 todavía. Pendiente de portar cuando quieras:

- [ ] Selector interactivo de categorías con botones (hoy solo `[CATEGORIA]` explícita)
- [ ] Detección y chequeo de duplicados antes de guardar
- [ ] Ingesta de artículos web vía Firecrawl
- [ ] Filtro de relevancia por fragmento
- [ ] Caché de preguntas frecuentes
- [ ] Idempotencia de `update_id` (evitar procesar el mismo mensaje dos veces)
