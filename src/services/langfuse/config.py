from pydantic_settings import BaseSettings, SettingsConfigDict


class LangFuseConfig(BaseSettings):

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True
    langfuse_flush_interval: int = 1
    langfuse_prompt_label: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def is_configured_instance(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)
