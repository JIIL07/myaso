from src.config.settings import settings


def get_langfuse_label() -> str:
    label = settings.langfuse.langfuse_prompt_label
    return label or "production"

