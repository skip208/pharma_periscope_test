from app.vector_store import get_vector_store
from app.vector_store.base import DocumentChunk
vs = get_vector_store()
vs.clear()
docs = [
    DocumentChunk(
        id="test1",
        text="Фродо несёт Кольцо в Мордор",
        metadata={"book": "fellowship", "chapter": 1, "chunk_index": 1, "position": 0},
        embedding=[0.1, 0.2, 0.3],  # длина должна совпадать с моделью, пока фиктивно
    )
]
vs.upsert_documents(docs)
res = vs.search([0.1, 0.2, 0.3], top_k=1)
print(res)