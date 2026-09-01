"""
Configuración central del backend. Todas las credenciales viven en
variables de entorno (nunca en el código) — en Render se configuran en
Dashboard -> tu servicio -> Environment.
"""
import os


class Settings:
    # Supabase
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")  # DEBE ser la clave service_role, nunca anon — ver .env.example

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_WEBHOOK_SECRET: str = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")

    # Firecrawl (para la ingesta de links web, se conecta después)
    FIRECRAWL_API_KEY: str = os.environ.get("FIRECRAWL_API_KEY", "")

    # Clave compartida para proteger el dashboard — sin esto, cualquiera
    # con la URL de Vercel podría ver/subir/borrar tus documentos.
    DASHBOARD_API_KEY: str = os.environ.get("DASHBOARD_API_KEY", "")

    # Pool de claves de Gemini — mismo patrón que en Apps Script: cualquier
    # variable de entorno que empiece con GEMINI_KEY_ se suma al pool,
    # rotando automáticamente si una se queda sin cuota (429).
    @staticmethod
    def obtener_pool_claves_gemini() -> list[str]:
        """
        CORREGIDO: si el valor de una variable tiene comas (ej. copiaste
        directo el valor de GEMINI_KEYS_POOL de Apps Script, que junta
        varias claves en un solo texto), se divide en claves individuales.
        Antes se mandaba el texto completo como si fuera UNA sola clave,
        causando 401 (Google la rechaza por no ser una clave real).
        """
        claves = []
        for nombre, valor in os.environ.items():
            if nombre.upper().startswith("GEMINI_KEY_") and valor.strip():
                for pedazo in valor.split(","):
                    pedazo_limpio = pedazo.strip()
                    if pedazo_limpio:
                        claves.append(pedazo_limpio)
        # Elimina duplicados preservando el orden
        vistas = set()
        unicas = []
        for c in claves:
            if c not in vistas:
                vistas.add(c)
                unicas.append(c)
        return unicas


settings = Settings()
