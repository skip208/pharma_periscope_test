import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.config import public_settings, setup_logging

logger = setup_logging()
app = FastAPI(title="LOTR RAG Bot")

logger.info("Application starting")
logger.info("Loaded settings: %s", public_settings())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error", extra={"path": request.url.path})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(api_router)

