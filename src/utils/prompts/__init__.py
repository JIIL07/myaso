from src.config.settings import settings


def get_langfuse_label() -> str:
    return settings.langfuse.langfuse_prompt_label or "production"