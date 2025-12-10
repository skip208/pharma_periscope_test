# LOTR RAG сервис

## Быстрый старт
- Требования: Python 3.11+, локальный Chroma (persist в `data/vector_store`), данные корпуса в `data/corpus`.
- Установка: `pip install -r requirements.txt`
- Переменные окружения (пример `.env`):
  - `OPENAI_API_KEY=<ключ>`
  - `LLM_MODEL_NAME=gpt-4.1-mini`
  - `EMBEDDING_MODEL_NAME=text-embedding-3-small`
  - `VECTOR_STORE_BACKEND=chroma`
  - `VECTOR_STORE_PATH=./data/vector_store`
  - `CORPUS_DIR=./data/corpus`
  - `RELEVANCE_THRESHOLD=0.78`, `MIN_GOOD_CHUNKS=2`, `MAX_CONTEXT_CHUNKS=5`
  - `CHUNK_SIZE_CHARS=1000`, `CHUNK_OVERLAP_CHARS=200`
  - `ADMIN_TOKEN=<секрет для /admin/reindex>`
  - `APP_HOST=0.0.0.0`, `APP_PORT=8000`

## Индексация
- Корпус: 3 файла, каждый содержит 2 книги (итого 6 book_part).
- Запуск полного reindex: `python -m scripts.reindex_corpus`
- Inspect индекса: `python -m scripts.list_book_parts` (части) и `python -m scripts.inspect_index --limit 5`
- Поиск по индексу: `python -m scripts.search_query --query "..." --top-k 5`

## Docker
- Сборка: `docker build -t lotr-rag .`
- Запуск: `docker run -p 8000:8000 --env-file .env -v $(pwd)/data/vector_store:/app/data/vector_store -v $(pwd)/data/corpus:/app/data/corpus:ro lotr-rag`
- Через compose: `docker compose up --build` (использует `docker-compose.yml`, читает переменные из текущего окружения/.env).
- Том `data/vector_store` монтируется для сохранения персистентного индекса между рестартами.

## Запуск API
- `python -m uvicorn app.main:app --reload`
- Эндпоинты:
  - `GET /health` — проверка.
  - `POST /admin/reindex` — пересборка корпуса (заголовок `X-Admin-Token`).
  - `POST /api/v1/ask` — вопрос к RAG (см. модели в `app/models/schemas.py`).

## Архитектура (кратко)
- Конфиг: `app/config.py` (Pydantic Settings).
- Векторка: `app/vector_store/chroma_store.py`, фабрика `get_vector_store()`.
- Индексация: `app/indexing/parser.py` (парсинг + book_part 1–6), `chunker.py` (чанки с overlap), `pipeline.py` (батчевые эмбеддинги и upsert).
- RAG: `app/rag/pipeline.py` — retrieve → guardrails по порогу → формирование system/user сообщений → вызов LLM → разбор JSON.
- Контекст: для процитированных чанков берутся соседние (левый/правый) из той же главы, чтобы расширить ответ.
- CLI: `scripts/reindex_corpus.py`, `scripts/search_query.py`, `scripts/inspect_index.py`, `scripts/list_book_parts.py`.

## Описание пайплайна ответа
1) Нормализация вопроса.  
2) Эмбеддинг вопроса и поиск в Chroma (top_k = `MAX_CONTEXT_CHUNKS` с запасом).  
3) Фильтр по `RELEVANCE_THRESHOLD` и `MIN_GOOD_CHUNKS`; при недостатке — отказ.  
4) Формирование prompt (см. `_build_messages` в `app/rag/pipeline.py`) с требованием JSON-формата.  
5) В ответ добавляются соседние чанки к процитированным (расширенный контекст).  
6) Парсинг JSON, маппинг источников, возврат `AskResponse`.
