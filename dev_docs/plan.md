
⸻

Этап 0. Подготовка проекта

Цель: завести репозиторий и базовый каркас сервиса.

Задачи:
	1.	Создать репозиторий lotr-rag-bot.
	2.	Добавить базовую структуру директорий (как в ТЗ):
	•	app/ (main, api, config, …)
	•	data/corpus/, data/vector_store/
	•	scripts/
	3.	Создать requirements.txt и зафиксировать зависимости:
	•	fastapi, uvicorn, pydantic, python-dotenv
	•	openai, chromadb, tqdm и т.д.
	4.	Настроить .env.example по образцу из ТЗ.
	5.	Реализовать минимальный app/main.py с FastAPI и /health.

Результат: проект поднимается командой uvicorn app.main:app --reload, /health возвращает {"status": "ok"}.

⸻

Этап 1. Конфигурация и инфраструктура

Цель: единый конфиг, удобное управление параметрами.

Задачи:
	1.	Реализовать app/config.py:
	•	Чтение всех параметров из env (см. раздел 8).
	•	Класс Settings (pydantic), глобальный settings.
	2.	Подключить Settings во всех нужных модулях:
	•	VectorStore, LLM, Embeddings, Indexing.
	3.	Добавить базовый логгер (можно через стандартный logging):
	•	Формат [%(asctime)s] [%(levelname)s] %(name)s - %(message)s.
	•	Логировать старт приложения и конфиг (без секретов).

Результат: все константы берутся из settings, секреты не захардкожены.

⸻

Этап 2. Векторное хранилище

Цель: абстракция над векторным индексом и реализация на Chroma.

Задачи:
	1.	Реализовать интерфейс VectorStore (app/vector_store/base.py):
	•	clear()
	•	upsert_documents(documents: List[Dict])
	•	search(query_embedding: list, top_k: int) -> List[(doc, score)]
	2.	Реализовать ChromaVectorStore (app/vector_store/chroma_store.py):
	•	Инициализация chromadb.Client с persist_directory=settings.vector_store_path.
	•	Создание коллекции lotr_corpus.
	•	Реализация clear, upsert_documents, search.
	3.	Сделать фабрику get_vector_store() (например в app/vector_store/__init__.py) на будущее, чтобы легко подменить backend.

Результат: можно руками вставить пару тестовых документов в Chroma и убедиться, что search возвращает их с адекватным score.

⸻

Этап 3. Клиенты Embeddings и LLM

Цель: единая обёртка над OpenAI.

3.1. EmbeddingsClient

Задачи:
	1.	Реализовать app/embeddings/client.py:
	•	Инициализация клиента OpenAI с settings.openai_api_key.
	•	Методы:
	•	embed_texts(texts: List[str]) -> List[List[float]]
	•	embed_text(text: str) -> List[float]
	2.	Заложить возможность батчинга (минимум простой вариант).

3.2. LLMClient

Задачи:
	1.	Реализовать app/llm/client.py:
	•	Инициализация OpenAI с settings.openai_api_key.
	•	Параметры model, temperature из конфига.
	•	Метод chat(messages: list, response_format: dict | None) -> str:
	•	Делает chat.completions.create.
	•	Возвращает message.content.

Результат: простейший скрипт-тест может сделать запрос к LLM и к embeddings.

⸻

Этап 4. Индексация корпуса

Цель: пройти полный путь от сырых текстов до заполненного векторного индекса.

4.1. Парсер книг (parser.py)

Задачи:
	1.	Реализовать загрузку файлов (load_book_files()):
	•	Читает .txt из settings.corpus_dir.
	2.	Реализовать маппинг файл → мета книги (map_file_to_book_info):
	•	book, book_id, source_file.
	3.	Реализовать выделение глав:
	•	Разбивать текст по регулярке ^ГЛАВА\s+\d+ (или выбранный формат).
	•	На выходе структура:

[
  {
    "book": "The Fellowship of the Ring",
    "book_id": "fellowship",
    "chapters": [
      {
        "chapter_index": 1,
        "chapter_title": "ГЛАВА 1. ...",
        "text": "сплошной текст главы"
      },
      ...
    ]
  },
  ...
]



4.2. Очистка и нормализация (parser.py / отдельная утилита)

Задачи:
	1.	Функция clean_text(text: str) -> str:
	•	Удалить лишние пустые строки, дублирующиеся пробелы.
	•	Нормализовать кавычки/тире по желанию.

4.3. Чанкинг (chunker.py)

Задачи:
	1.	Реализовать функцию chunk_chapter_text(text: str, chapter_index: int, book_info: dict) -> List[DocumentChunk]:
	•	Настройки chunk_size_chars, chunk_overlap_chars из settings.
	•	Для каждого чанка:
	•	id (например, "{book_id}_ch{chapter_index}_{chunk_index:04d}").
	•	text — фрагмент.
	•	metadata: book, book_id, chapter_title, chapter_index, chunk_index, position, source_file.

4.4. Индексационный пайплайн (pipeline.py)

Задачи:
	1.	Реализовать reindex_corpus(vector_store: VectorStore, embeddings_client: EmbeddingsClient) -> int:
	•	vector_store.clear().
	•	Загрузить все книги.
	•	Для каждой главы сделать чанкинг.
	•	Собрать все чанки в список.
	•	Пройти по чанкам батчами:
	•	Считать эмбеддинги.
	•	Вызвать vector_store.upsert_documents().
	•	Вернуть количество проиндексированных чанков.
	2.	Логировать шаги: сколько книг, глав, чанков, время.

Результат: запуск индексации наполняет Chroma; можно вручную проверить количество документов в коллекции.

⸻

Этап 5. CLI и админ-эндпоинт для переиндексации

Цель: удобный запуск переиндексации.

5.1. CLI (scripts/reindex_corpus.py)

Задачи:
	1.	Скрипт, который:
	•	инициализирует settings;
	•	создаёт ChromaVectorStore и EmbeddingsClient;
	•	вызывает reindex_corpus();
	•	печатает в консоль статистику.

5.2. HTTP-эндпоинт /admin/reindex

Задачи:
	1.	В app/models/schemas.py реализовать ReindexRequest.
	2.	В app/api/routes.py:
	•	POST /admin/reindex:
	•	Проверка X-Admin-Token == settings.admin_token.
	•	Вызов reindex_corpus() (синхронно для простоты).
	•	Возврат { "status": "completed", "indexed_chunks": N }.

Результат: можно запустить индексацию через CLI или HTTP.

⸻

Этап 6. RAG-пайплайн

Цель: реализовать логику ответа на вопрос по корпусу.

6.1. Модельки запрос/ответ API (schemas.py)

Задачи:
	1.	Реализовать AskRequest, Citation, ContextChunk, AskResponse (как в описании).

6.2. Нормализация вопроса

Задачи:
	1.	В app/rag/pipeline.py реализовать функцию normalize_question(text: str) -> str:
	•	trim;
	•	удаление лишних \n и т.п.

6.3. Поиск релевантных чанков

Задачи:
	1.	В app/rag/pipeline.py реализовать функцию retrieve_relevant_chunks(question: str, vector_store, embeddings_client) -> List[(chunk, score)]:
	•	Вызвать normalize_question.
	•	Считать embedding вопроса.
	•	Вызвать vector_store.search с top_k = settings.max_context_chunks (или чуть больше).
	•	Вернуть список (doc, score).
	2.	Имплементировать логику порога:
	•	Вычислить max_score.
	•	Отфильтровать чанки по score >= settings.relevance_threshold.

6.4. Guardrails до LLM

Задачи:
	1.	В high-level функции answer_question(...):
	•	Если max_score < threshold или кол-во чанков >= threshold < min_good_chunks:
	•	сразу вернуть AskResponse с отказом:
	•	can_answer = False
	•	answer_short = answer_full = стандартная фраза
	•	citations = [], context_chunks = [].

6.5. Формирование промпта

Задачи:
	1.	В answer_question(...) сформировать:
	•	system_message (как в спецификации).
	•	user_message, включающий:
	•	оригинальный вопрос;
	•	список фрагментов [Фрагмент N] с книгой, главой, позицией и текстом.
	•	Добавить в user_message JSON-формат ответа, который мы ждём.

6.6. Вызов LLM и post-guardrails

Задачи:
	1.	Вызвать LLMClient.chat c:
	•	messages=[system_message, user_message].
	•	(Опционально) response_format={"type": "json_object"}.
	2.	Спарсить ответ как JSON:
	•	Валидация полей: answer_short, answer_full, sources, can_answer.
	3.	Если парсинг не удался или поля неправильные:
	•	fallback к отказу (как в guardrails).
	4.	Если can_answer == false:
	•	вернуть отказ.
	5.	Если can_answer == true:
	•	маппинг sources из LLM → наши Citation:
	•	сверить book, chapter, position с реальными метаданными, если возможно (минимум — просто пробросить).
	•	собрать context_chunks из найденных чанков.

Результат: есть функция answer_question(question: str) -> AskResponse.

⸻

Этап 7. HTTP API для пользователя

Цель: «обернуть» RAG-пайплайн в FastAPI.

Задачи:
	1.	В app/api/routes.py:
	•	Создать APIRouter().
	•	POST /api/v1/ask:
	•	Принимает AskRequest.
	•	Проверяет, что вопрос не пустой (иначе HTTPException(400)).
	•	Инициализирует vector_store, embeddings_client, llm_client (или использует DI/синглтоны).
	•	Вызывает answer_question().
	•	Возвращает AskResponse.
	2.	Подключить роутер в app/main.py.

Результат: можно отправить запрос в /api/v1/ask и получить структурированный ответ/отказ.

⸻

Этап 8. Нефункционал: логирование, ошибки, markdown

Цель: привести сервис к минимальному продоподобному состоянию.

Задачи:
	1.	Логирование:
	•	Логировать:
	•	входящие запросы /api/v1/ask (коротко: id запроса, длина вопроса).
	•	результаты RETRIEVAL (id чанков + score).
	•	ошибки при запросе к OpenAI/Chroma.
	2.	Обработка ошибок:
	•	Глобальный обработчик исключений (опционально).
	•	Явные HTTPException для 400 (пустой вопрос, невалидный запрос).
	3.	Markdown:
	•	В промпте попросить LLM использовать markdown (заголовки, списки, цитаты).
	•	Убедиться, что answer_short и answer_full возвращаются как строки без экранирования.

⸻

Этап 9. Тесты

Цель: минимально закрыть критичные части логикой тестов.

Задачи:
	1.	Юнит-тесты:
	•	Чанкинг:
	•	вход: текст главы → выход: правильное количество чанков, корректные position и chunk_index.
	•	Нормализация вопроса.
	•	Guardrails:
	•	при низком score → отказ.
	2.	Интеграционные:
	•	Смок-тест RAG-пайплайна с маленьким тестовым корпусом (2–3 коротких фрагмента).

⸻

