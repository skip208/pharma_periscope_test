from fastapi import FastAPI

app = FastAPI(title="LOTR RAG Bot")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

