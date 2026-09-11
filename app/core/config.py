# Файл: app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Указываем переменную и её тип. Pydantic сам заберет её из .env
    GEMINI_API_KEY: str

    # Дополнительные настройки (название проекта, порты и т.д.)
    PROJECT_NAME: str = "RAG Evaluate API"

    # Указываем pydantic читать .env файл из корня проекта
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Создаем синглтон настроек для импорта в другие модули
settings = Settings()
