lotr-rag-bot/
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ config.py
│  ├─ api/
│  │   ├─ __init__.py
│  │   └─ routes.py
│  ├─ models/
│  │   ├─ __init__.py
│  │   └─ schemas.py
│  ├─ vector_store/
│  │   ├─ __init__.py
│  │   ├─ base.py
│  │   └─ chroma_store.py
│  ├─ embeddings/
│  │   ├─ __init__.py
│  │   └─ client.py
│  ├─ llm/
│  │   ├─ __init__.py
│  │   └─ client.py
│  ├─ indexing/
│  │   ├─ __init__.py
│  │   ├─ parser.py
│  │   ├─ chunker.py
│  │   └─ pipeline.py
│  └─ rag/
│      ├─ __init__.py
│      └─ pipeline.py
├─ scripts/
│  ├─ __init__.py
│  └─ reindex_corpus.py
├─ data/
│  ├─ corpus/
│  │   ├─ lotr_fellowship.txt
│  │   ├─ lotr_two_towers.txt
│  │   └─ lotr_return_of_king.txt
│  └─ vector_store/          # для Chroma persistent
├─ .env.example
├─ requirements.txt
└─ README.md