"""
Proxy-Pointer: Structural RAG Bot

Interactive RAG bot that:
  1. Vector search (k=200) for broad recall
  2. Deduplicates by node_id
  3. LLM re-ranker selects top 5 by hierarchical path relevance
  4. Loads full document sections from source .md files
  5. LLM synthesizer generates grounded answers

Usage:
    python -m src.agent.rag_bot
"""
import os
import re
import sys
import json

# Add project root to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR, INDEX_DIR, EMBEDDING_MODEL, EMBEDDING_DIMS, SYNTH_MODEL

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


# ── Custom Embedding Wrapper ────────────────────────────────────────────
class GeminiEmbeddings(Embeddings):
    """LangChain-compatible wrapper for Gemini embeddings at configurable dims."""

    def __init__(self, model=EMBEDDING_MODEL, dimensionality=EMBEDDING_DIMS):
        self.model = model
        self.dimensionality = dimensionality

    def embed_documents(self, texts):
        result = genai.embed_content(
            model=self.model,
            content=texts,
            output_dimensionality=self.dimensionality
        )
        return result['embedding']

    def embed_query(self, text):
        result = genai.embed_content(
            model=self.model,
            content=text,
            output_dimensionality=self.dimensionality
        )
        return result['embedding']


class ProxyPointerRAG:
    def __init__(self, index_path=None, data_dir=None):
        self.data_dir = str(data_dir or DATA_DIR)
        index_path = str(index_path or INDEX_DIR)

        # 1. Load Gemini Embeddings
        print(f"Loading {EMBEDDING_MODEL} @ {EMBEDDING_DIMS} dims...")
        self.embeddings = GeminiEmbeddings()

        # 2. Load FAISS Index
        print(f"Loading index from {index_path}...")
        self.vector_db = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # 3. Initialize synthesis model
        self.model = genai.GenerativeModel(SYNTH_MODEL)

    def retrieve_unique_nodes(self, query, k_search=200, k_final=5):
        """Stage 1: Broad vector recall → Stage 2: LLM re-ranking."""

        # Stage 1: Broad Recall
        docs = self.vector_db.similarity_search(query, k=k_search)

        candidates = []
        unique_map = {}
        for doc in docs:
            node_id = doc.metadata.get("node_id")
            if node_id not in unique_map:
                doc_id = doc.metadata.get("doc_id", "UNK")
                internal_crumb = doc.metadata.get("breadcrumb", "Unknown Path")
                global_crumb = f"{doc_id} > {internal_crumb}"

                info = {
                    "node_id": node_id,
                    "global_breadcrumb": global_crumb,
                    "doc_id": doc_id,
                    "start_line": int(doc.metadata.get("start_line", 0)),
                    "end_line": int(doc.metadata.get("end_line", 0)),
                    "content": doc.page_content,
                }
                candidates.append(info)
                unique_map[node_id] = info

        # Stage 2: LLM Re-Ranker
        candidates_text = ""
        for i, h in enumerate(candidates[:50]):
            candidates_text += (
                f"{i}. [{h['global_breadcrumb']}] (ID: {h['node_id']})\n"
            )

        prompt = f"""You are a structural re-ranker. 
Your goal is to find the Top 10 most relevant Document IDs based on their HIERARCHICAL PATH relative to the user's query.

User Query: "{query}"

CANDIDATE HIERARCHIES (ID | Full Path):
{candidates_text}

RANKING RULES:
1. Highly Specific Matches (e.g. if query is about 'Chapter 2 questions', a path like 'Chapter 2 > Intro > Questions' is Rank 1).
2. If specific matches are not found, include similar, partial matches.
3. If the query is not pointing to any specific chapter or section, look for the most relevant Contextual Matches (e.g. if query is about 'India growth', a path like 'Chapter 1 > Outlook > Country outlooks' is very strong).
4. Structural Priority: Prioritize exact structural anchors (Box 1.1, Figure B1.1) if the query mentions them.
5. Each ID must appear ONLY ONCE. Do not repeat any ID.
6. Output ONLY a comma-separated list of the Top 10 unique IDs. No text, no explanation.

Output Example: 0098, 0096, 0101, 0012, 0033
"""
        try:
            response = self.model.generate_content(prompt).text.strip()
            clean_text = re.sub(r"[^0-9, ]", "", response)
            ranked_ids = [
                rid.strip() for rid in clean_text.split(",") if rid.strip()
            ]

            final_pointers = []
            seen = set()
            for rid in ranked_ids:
                if rid in unique_map and rid not in seen:
                    final_pointers.append(unique_map[rid])
                    seen.add(rid)
                if len(final_pointers) >= k_final:
                    break

            if final_pointers:
                return final_pointers
        except Exception as e:
            print(f"WARNING: LLM Ranker Failed ({e}). Falling back to top 5.")

        # Fallback to top-k unique by similarity
        return candidates[:k_final]

    def chat(self, query):
        """Orchestrate Retrieval and Synthesis."""
        pointers = self.retrieve_unique_nodes(query)

        print("\n" + "=" * 100)
        print(f"Final Context Selection (Top {len(pointers)} Unique Nodes):")
        for p in pointers:
            print(f"  -> Node {p['node_id']:<6} | {p['global_breadcrumb']}")
        print("=" * 100 + "\n")

        context = []
        for p in pointers:
            md_path = os.path.join(self.data_dir, f"{p['doc_id']}.md")
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    text = "".join(lines[p["start_line"] : p["end_line"]])
                    context.append(
                        f"### REFERENCE: {p['global_breadcrumb']} "
                        f"(ID: {p['node_id']})\n{text}"
                    )
            else:
                # Fallback to vector DB chunk content if .md file is missing
                context.append(
                    f"### REFERENCE: {p['global_breadcrumb']} "
                    f"(ID: {p['node_id']})\n{p['content']}"
                )

        synth_prompt = (
            f"Query: {query}\n\nContext:\n"
            + "\n\n".join(context)
            + "\n\nAnswer the query concisely citing sources."
        )
        response = self.model.generate_content(synth_prompt)
        return response.text


if __name__ == "__main__":
    bot = ProxyPointerRAG()

    print("\nProxy-Pointer RAG Bot ready. Type 'exit' to quit.\n")
    while True:
        user_in = input("User >> ")
        if user_in.lower() in ["exit", "quit"]:
            break
        print(bot.chat(user_in))
