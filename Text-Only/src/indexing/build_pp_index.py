"""
Proxy-Pointer: Build Index Pipeline

Combined pipeline that:
  Step 0: Builds skeleton trees for new .md files (pure-Python, no external deps)
  Step 1: LLM-based noise filtering (removes TOC, abbreviations, etc.)
  Step 2: Chunks and embeds document sections (1536-dim Gemini embeddings)
  Step 3: Builds/updates FAISS vector index

Usage:
    python -m src.indexing.build_index           # incremental (default)
    python -m src.indexing.build_index --fresh    # rebuild from scratch
"""
import os
import sys
import json
import logging
import argparse

# Add project root to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (
    DATA_DIR, TREES_DIR, INDEX_DIR,
    EMBEDDING_MODEL, EMBEDDING_DIMS, NOISE_FILTER_MODEL
)
from src.indexing.build_skeleton_trees import build_skeleton_trees

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ── Custom Embedding Wrapper ────────────────────────────────────────────
class GeminiEmbeddings(Embeddings):
    """LangChain-compatible wrapper for Gemini embeddings at configurable dims."""

    def __init__(self, model=EMBEDDING_MODEL, dimensionality=EMBEDDING_DIMS):
        self.model = model
        self.dimensionality = dimensionality

    def embed_documents(self, texts):
        """Embed a list of documents."""
        result = genai.embed_content(
            model=self.model,
            content=texts,
            output_dimensionality=self.dimensionality
        )
        return result['embedding']

    def embed_query(self, text):
        """Embed a single query."""
        result = genai.embed_content(
            model=self.model,
            content=text,
            output_dimensionality=self.dimensionality
        )
        return result['embedding']


# ── Noise Filter ────────────────────────────────────────────────────────
def get_noise_node_ids(doc_name, structure):
    """Send the tree to an LLM and return a set of noise node_ids."""
    tree_json = json.dumps(structure, indent=2, ensure_ascii=False)

    prompt = f"""You are a document-structure analyst. I will give you the
structural tree of a document called "{doc_name}" as JSON.

Your task: Identify every node whose title matches one of these
noise categories:
  1. Table of contents (e.g. Contents, Summary of Contents, Index of Sections)
  2. Abbreviations or acronym lists (e.g. Abbreviations, Abbreviations (continued), Glossary)
  3. Acknowledgments (e.g. Acknowledgements, Note of Thanks, Credits)
  4. Foreword (e.g. Preface, Introductory Remarks)
  5. Executive Summary (e.g. Overview Summary, Key Highlights)
  6. References (e.g. Bibliography, Works Cited, Sources)

Only flag nodes that clearly fall into one of the above 6 categories.
Do NOT flag anything else.

── DOCUMENT TREE ──
{tree_json}

── RESPONSE FORMAT ──
Return ONLY a valid JSON object:
{{{{
  "noise_nodes": [
    {{{{"node_id": "XXXX", "title": "...", "category": "which of the 6 above"}}}}
  ]
}}}}

No markdown fencing, no extra text.
"""

    model = genai.GenerativeModel(NOISE_FILTER_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=2048,
        )
    )

    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()

    result = json.loads(text)
    noise_ids = set()
    for entry in result.get("noise_nodes", []):
        nid = entry.get("node_id")
        if nid:
            noise_ids.add(nid)
            logging.info(
                f"  [NOISE] {nid}  {entry.get('title', '')}  "
                f"— {entry.get('category', '')}"
            )

    return noise_ids


# ── Main Build Pipeline ────────────────────────────────────────────────
def build_proxy_index(incremental=True):
    trees_dir = str(TREES_DIR)
    data_dir = str(DATA_DIR)
    save_path = str(INDEX_DIR)

    # Step 0: Build skeleton trees for any new .md files
    logging.info("\n" + "=" * 60)
    logging.info("STEP 0: Building skeleton trees for new documents...")
    logging.info("=" * 60)
    build_skeleton_trees(data_dir, trees_dir)

    # Step 1: Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    # Step 2: Initialize Gemini Embeddings
    embeddings = GeminiEmbeddings()
    logging.info(
        f"Embedding model: {embeddings.model} @ {embeddings.dimensionality} dims"
    )

    existing_docs = set()
    vector_db = None
    if incremental and os.path.exists(save_path):
        try:
            vector_db = FAISS.load_local(
                save_path, embeddings, allow_dangerous_deserialization=True
            )
            # Find which documents are already in the database
            for doc in vector_db.docstore._dict.values():
                if "doc_id" in doc.metadata:
                    existing_docs.add(doc.metadata["doc_id"])
            logging.info(f"Loaded existing index with {len(existing_docs)} completely indexed document(s).")
        except Exception as e:
            logging.warning(f"Could not load existing index: {e}. Building fresh.")
            vector_db = None

    all_chunks = []

    # Process all tree files in the trees folder
    tree_files = sorted([
        f for f in os.listdir(trees_dir)
        if f.endswith("_structure.json") and os.path.isfile(os.path.join(trees_dir, f))
    ])
    logging.info(f"Found {len(tree_files)} tree(s): {', '.join(tree_files)}")

    for file in tree_files:
        tree_path = os.path.join(trees_dir, file)
        if not os.path.exists(tree_path):
            logging.error(f"Tree file {tree_path} not found.")
            continue

        with open(tree_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)

        doc_id = tree_data.get("doc_name", file.replace("_structure.json", ""))
        md_file = os.path.join(data_dir, f"{doc_id}.md")

        if not os.path.exists(md_file):
            logging.error(f"Markdown file {md_file} not found.")
            continue

        if doc_id in existing_docs:
            logging.info(f"  [SKIP] {doc_id}: Already completely indexed in FAISS.")
            continue

        with open(md_file, "r", encoding="utf-8") as f:
            md_lines = f.readlines()

        if not tree_data.get("structure"):
            logging.warning(
                f"  [SKIP] {doc_id}: No structure found (headerless document)."
            )
            continue

        logging.info(f"Processing: {doc_id}...")

        # LLM-based noise filter
        noise_node_ids = get_noise_node_ids(doc_id, tree_data["structure"])
        logging.info(f"  Noise nodes excluded: {len(noise_node_ids)}")

        def process_node(node_list, parent_end=None, breadcrumb=""):
            if parent_end is None:
                parent_end = len(md_lines)

            for i, node in enumerate(node_list):
                node_id = node.get("node_id")
                title = node.get("title", "")

                if node_id in noise_node_ids:
                    continue

                current_crumb = (
                    f"{breadcrumb} > {title}" if breadcrumb else title
                )
                start_idx = node["line_num"] - 1

                if i + 1 < len(node_list):
                    end_idx = node_list[i + 1]["line_num"] - 1
                else:
                    end_idx = parent_end

                node_end = end_idx
                if "nodes" in node and node["nodes"]:
                    first_child_line = node["nodes"][0]["line_num"] - 1
                    end_idx = min(end_idx, first_child_line)

                section_text = "".join(md_lines[start_idx:end_idx]).strip()

                if len(section_text) >= 100:
                    chunks = text_splitter.split_text(section_text)
                    for chunk in chunks:
                        enriched_content = f"[{current_crumb}]\n{chunk}"
                        doc = Document(
                            page_content=enriched_content,
                            metadata={
                                "doc_id": doc_id,
                                "node_id": node_id,
                                "title": title,
                                "breadcrumb": current_crumb,
                                "start_line": start_idx,
                                "end_line": node_end,
                            },
                        )
                        all_chunks.append(doc)

                if "nodes" in node and node["nodes"]:
                    process_node(node["nodes"], node_end, current_crumb)

        if "structure" in tree_data:
            process_node(tree_data["structure"])

    if not all_chunks:
        logging.warning("No new chunks generated.")
        return

    logging.info(f"\nAdding {len(all_chunks)} chunks to index...")

    if vector_db is not None:
        vector_db.add_documents(all_chunks)
    else:
        vector_db = FAISS.from_documents(all_chunks, embeddings)

    os.makedirs(save_path, exist_ok=True)
    vector_db.save_local(save_path)
    logging.info(f"Index successfully saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Proxy-Pointer FAISS index")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Rebuild index from scratch (default: incremental)"
    )
    args = parser.parse_args()
    build_proxy_index(incremental=not args.fresh)
