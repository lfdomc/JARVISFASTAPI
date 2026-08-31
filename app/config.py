"""
Configuración central del backend. Todas las credenciales viven en
variables de entorno (nunca en el código) — en Render se configuran en
Dashboard -> tu servicio -> Environment.
"""
import os


class Settings:
    # Supabase
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")  # clave publicable (anon)

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_WEBHOOK_SECRET: str = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")

    # Firecrawl (para la ingesta de links web, se conecta después)
    FIRECRAWL_API_KEY: str = os.environ.get("FIRECRAWL_API_KEY", "")

    # Pool de claves de Gemini — mismo patrón que en Apps Script: cualquier
    # variable de entorno que empiece con GEMINI_KEY_ se suma al pool,
    # rotando automáticamente si una se queda sin cuota (429).
    @staticmethod
    def obtener_pool_claves_gemini() -> list[str]:
        claves = []
        for nombre, valor in os.environ.items():
            if nombre.upper().startswith("GEMINI_KEY_") and valor.strip():
                claves.append(valor.strip())
        return claves


settings = Settings()
