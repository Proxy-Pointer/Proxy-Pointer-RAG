# Proxy-Pointer

**Structural RAG for Document Analysis** — A hierarchical, pointer-based RAG pipeline that retrieves full document sections using structural tree navigation instead of blind vector similarity.

---

## How It Works

```
PDF ─→ LlamaParse ─→ .md ─→ PageIndex ─→ tree.json ─→ LLM noise filter ─→ chunks ─→ FAISS (1536-dim)
                                                                                          │
                                                                   query ─→ vector search (k=200)
                                                                                          │
                                                                               dedup by node_id
                                                                                          │
                                                                          LLM re-ranker (top 5)
                                                                                          │
                                                                     load full sections from .md
                                                                                          │
                                                                          LLM synthesizer ─→ answer
```

Instead of retrieving small, context-less chunks, Proxy-Pointer:

1. **Builds a structural tree** of each document (like a table of contents)
2. **Filters noise** (TOC, abbreviations, foreword, etc.) using an LLM
3. **Indexes structural pointers** — each chunk carries metadata about its position in the document hierarchy
4. **Re-ranks by structure** — an LLM re-ranker selects the most relevant sections by their hierarchical path, not just embedding similarity
5. **Loads full sections** — the synthesizer sees complete document sections, not truncated 2000-char chunks

---

## 5-Minute Quickstart

A sample document (SADU Spring 2024 Full Report) is included with a pre-built tree, so you can start querying immediately.

### 1. Clone (with PageIndex submodule)

```bash
git clone --recursive https://github.com/youruser/Proxy-Pointer.git
cd Proxy-Pointer
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -r vendor/PageIndex/requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
# Edit .env → add your GOOGLE_API_KEY
```

### 5. Build the index

```bash
python -m src.indexing.build_index --fresh
```

This uses the pre-shipped SADU tree and markdown to build the FAISS index.

### 6. Start querying

```bash
python -m src.agent.rag_bot
```

Try a query like:
```
User >> What are the main themes discussed in the SADU report?
```

---

## Adding Your Own Documents

### Option A: You already have Markdown files

1. Place `.md` files in `data/documents/`
2. Run the indexer (it will auto-build trees via PageIndex):
   ```bash
   python -m src.indexing.build_index --fresh
   ```

### Option B: Start from PDFs

1. Add `LLAMA_CLOUD_API_KEY` to your `.env` file
2. Place PDFs in `data/pdf/`
3. Extract to markdown:
   ```bash
   python -m src.extraction.extract_pdf_to_md
   ```
4. Build the index:
   ```bash
   python -m src.indexing.build_index --fresh
   ```

---

## Project Structure

```
Proxy-Pointer/
├── src/
│   ├── config.py                  # Centralized configuration
│   ├── extraction/
│   │   └── extract_pdf_to_md.py   # PDF → Markdown (LlamaParse)
│   ├── indexing/
│   │   └── build_index.py         # Tree building + noise filter + FAISS indexing
│   └── agent/
│       └── rag_bot.py             # Interactive RAG bot
├── data/
│   ├── pdf/                       # Source PDFs
│   ├── documents/                 # Extracted Markdown files
│   ├── trees/                     # Structure tree JSONs
│   └── index/                     # Generated FAISS index (gitignored)
├── vendor/
│   └── PageIndex/                 # VectifyAI/PageIndex (git submodule)
└── docs/
    └── architecture.md            # Architecture deep-dive
```

---

## Configuration

All configuration is centralized in `src/config.py`. Override via environment variables:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `LLAMA_CLOUD_API_KEY` | (optional) | LlamaParse API key for PDF extraction |
| `PP_DATA_DIR` | `data/documents/` | Markdown source directory |
| `PP_TREES_DIR` | `data/trees/` | Structure tree directory |
| `PP_INDEX_DIR` | `data/index/` | FAISS index directory |
| `PP_PAGEINDEX_DIR` | `vendor/PageIndex/` | PageIndex installation |

---

## Design Decisions

- **1536-dim embeddings**: Half of Gemini's default 3072. Faster indexing, smaller FAISS files, minimal accuracy loss for structural retrieval.
- **LLM noise filter**: Replaces brittle regex/hardcoded title matching. Catches variations like "Note of Thanks" → Acknowledgments.
- **Structural re-ranker**: Ranks by hierarchical breadcrumb path, not embedding similarity. A query about "cash flow" correctly prioritizes `Financial Statements > Cash Flows` over a paragraph that mentions "cash flow" in passing.
- **Full section loading**: The synthesizer sees the complete document section (including tables), not a truncated chunk.

---

## Dependencies

- [PageIndex](https://github.com/VectifyAI/PageIndex) — Hierarchical tree structure generation
- [Gemini](https://ai.google.dev/) — Embeddings, noise filter, re-ranker, synthesis
- [LangChain](https://github.com/langchain-ai/langchain) + [FAISS](https://github.com/facebookresearch/faiss) — Vector indexing
- [LlamaParse](https://cloud.llamaindex.ai/) — PDF to Markdown extraction (optional)

---

## License

MIT — see [LICENSE](LICENSE).
