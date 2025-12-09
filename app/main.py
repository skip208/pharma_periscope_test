import logging

from fastapi import FastAPI

from app.config import public_settings, setup_logging

logger = setup_logging()
app = FastAPI(title="LOTR RAG Bot")

logger.info("Application starting")
logger.info("Loaded settings: %s", public_settings())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

