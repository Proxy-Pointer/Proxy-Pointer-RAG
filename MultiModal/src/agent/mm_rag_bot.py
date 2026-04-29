"""
Multimodal Proxy-Pointer: Structural RAG Bot with Image Selection

Interactive RAG bot that:
  1. Vector search (k=200) for broad recall
  2. Deduplicates by (doc_id, node_id)
  3. LLM re-ranker selects top 5 by hierarchical path + semantic snippets
  4. Loads full section text from Markdown via line-slicing
  5. Extracts image anchors from skeleton tree JSON
  6. LLM synthesizer generates grounded answers + selects relevant images
  7. Optional vision filter validates image relevance

Usage:
    python -m src.agent.mm_rag_bot
"""
import os
import re
import sys
import json
import PIL.Image as Image
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from src.config import DATASET_DIR, TREES_DIR, INDEX_DIR, EMBEDDING_MODEL, EMBEDDING_DIMS, SYNTH_MODEL, VISION_FILTER
from src.indexing.md_tree_builder import get_md_path_for_doc


# ── Custom Embedding Wrapper ────────────────────────────────────────────
class GeminiEmbeddings(Embeddings):
    """LangChain-compatible wrapper for Gemini embeddings at configurable dims."""

    def __init__(self, model=EMBEDDING_MODEL, dimensionality=EMBEDDING_DIMS):
        self.model = model
        self.dimensionality = dimensionality

    def embed_documents(self, texts):
        res = genai.embed_content(model=self.model, content=texts, output_dimensionality=self.dimensionality)
        return res['embedding']

    def embed_query(self, text):
        res = genai.embed_content(model=self.model, content=text, output_dimensionality=self.dimensionality)
        return res['embedding']


class MultimodalProxyPointerRAG:
    def __init__(self, index_path=None, trees_dir=None, dataset_dir=None):
        self.dataset_dir = str(dataset_dir or DATASET_DIR)
        self.trees_dir = str(trees_dir or TREES_DIR)
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

        # 3. Initialize models
        self.llm = genai.GenerativeModel(SYNTH_MODEL)

    # ── RAG Pipeline Features ────────────────────────────────────────────────
    VISION_FILTER = VISION_FILTER  # Controlled from config.py

    def retrieve_unique_nodes(self, query, k_search=200, k_final=5):
        """Stage 1: Broad vector recall → Stage 2: LLM structural re-ranking."""
        
        # Stage 1: Broad Recall
        docs = self.vector_db.similarity_search(query, k=k_search)

        candidates = []
        seen_nodes = set()   # (doc_id, node_id)
        for doc in docs:
            node_id = doc.metadata.get("node_id")
            doc_id = doc.metadata.get("doc_id", "UNK")
            dedup_key = (doc_id, node_id)
            if dedup_key not in seen_nodes:
                seen_nodes.add(dedup_key)
                info = {
                    "node_id": node_id,
                    "doc_id": doc_id,
                    "breadcrumb": doc.metadata.get("breadcrumb", "Unknown Path"),
                    "snippet": doc.page_content[:150].replace("\n", " "),
                    "full_metadata": doc.metadata
                }
                candidates.append(info)

        # Stage 2: LLM Re-Ranker (Anchor-Aware + Semantic Snippets)
        index_map = {str(i): c for i, c in enumerate(candidates[:50])}
        candidates_text = ""
        for i, c in enumerate(candidates[:50]):
            candidates_text += f"{i}. [{c['doc_id']} > {c['breadcrumb']}] | Preview: {c['snippet']}...\n"

        prompt = f"""You are a structural & semantic re-ranker for technical research papers.
Your goal is to find the Top {k_final} most relevant sections based on their HIERARCHICAL PATH and the content snippets provided.

User Query: "{query}"

CANDIDATES (INDEX | Path | Snippet):
{candidates_text}

RANKING RULES:
1. ANCHOR AWARENESS: If the query mentions a specific anchor like 'Figure 5' or 'Table I', prioritize sections that physically contain that reference.
2. TECHNICAL SPECIFICITY: Prioritize technical deep-dives (e.g. 'Methodology', 'Experiments') over generic introductions.
3. CONTEXTUAL RELEVANCE: Match the query's technical terms to the content snippets.
4. Each INDEX must appear ONLY ONCE.
5. Output ONLY a comma-separated list of the Top {k_final} unique numeric indices. No text.

Output Example: 4, 12, 0, 9, 2
"""
        try:
            response = self.llm.generate_content(prompt).text.strip()
            clean_text = re.sub(r"[^0-9, ]", "", response)
            ranked_ids = [rid.strip() for rid in clean_text.split(",") if rid.strip()]

            final_pointers = []
            seen = set()
            for rid in ranked_ids:
                if rid in index_map and rid not in seen:
                    final_pointers.append(index_map[rid])
                    seen.add(rid)
                if len(final_pointers) >= k_final:
                    break

            if final_pointers:
                return final_pointers
        except Exception as e:
            print(f"WARNING: LLM Ranker Failed ({e}). Falling back to top {k_final}.")

        return candidates[:k_final]

    def chat(self, query):
        """Orchestrate Retrieval, Multimodal Synthesis, and UI Parsing."""
        pointers = self.retrieve_unique_nodes(query)

        context_blocks = []
        found_images = []

        for p in pointers:
            # 1. Load full markdown text
            md_path = get_md_path_for_doc(self.dataset_dir, p['doc_id'])
            if md_path:
                with open(md_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    section_text = "".join(lines[p['full_metadata']["start_line"] : p['full_metadata']["end_line"]])
            else:
                section_text = f"(Full text missing) {p['snippet']}"

            context_blocks.append(f"### REFERENCE: {p['doc_id']} > {p['breadcrumb']}\n{section_text}")

            # 2. Extract specific image anchors from tree JSON
            tree_path = os.path.join(self.trees_dir, f"{p['doc_id']}_structure.json")
            if os.path.exists(tree_path):
                with open(tree_path, "r", encoding="utf-8") as f:
                    tree_data = json.load(f)
                
                def _find_node(node_list, target_id):
                    for node in node_list:
                        if node.get("node_id") == target_id:
                            return node
                        if node.get("nodes"):
                            found = _find_node(node["nodes"], target_id)
                            if found: return found
                    return None
                
                target_node = _find_node(tree_data.get("structure", []), p['node_id'])
                if target_node and target_node.get("figures"):
                    doc_folder = os.path.join(self.dataset_dir, p['doc_id'])
                    for fig in target_node["figures"]:
                        full_img_path = os.path.join(doc_folder, fig["filename"]).replace("\\", "/")
                        rel_path = os.path.join(p['doc_id'], fig["filename"]).replace("\\", "/")
                        
                        clean_doc_id = str(p['doc_id']).strip().rstrip('-').strip()
                        clean_fig_label = str(fig.get('label', 'Figure')).strip().lstrip('-').strip()

                        found_images.append({
                            "label": f"{clean_doc_id} - {clean_fig_label}",
                            "relative_path": rel_path,
                            "full_path": full_img_path,
                            "exists": os.path.exists(full_img_path)
                        })

        # 3. Vision-Aware Synthesis
        synth_prompt = f"""You are an advanced Multimodal RAG Assistant. 

Query: "{query}"

Context:
{chr(10).join(context_blocks)}

INSTRUCTIONAL RULES:
1. Answer the query concisely using ONLY the provided context. 
2. If the context contains a table or figure anchor that answers the query, explicitly mention its ID (e.g. Figure 5).
3. Do NOT reference internal node IDs (e.g. 'node: 0045') or breadcrumb segments in the body of the answer.
4. IMAGE SELECTION: If a figure or table mentioned in the context is highly relevant, list its filename and a SHORT caption (including Figure/Table number) in brackets: [SHOW: filename | short caption]. Provide ONE image per bracket. Limit to TOP 6 most relevant.

Output format:
[Answer Text]

[SHOW: figure1.png | Figure 1: Short caption text]
"""
        generation_config = genai.GenerationConfig(temperature=0.0)
        response = self.llm.generate_content(synth_prompt, generation_config=generation_config)
        answer_text = response.text

        # 4. Extract filenames and labels, apply Vision Filter
        requested_images = []
        img_matches = re.findall(r"\[SHOW:\s*([^\|\]]+)(?:\|\s*([^\]]+))?\]", answer_text, re.I)
        requested_filenames = []
        llm_labels = {}
        for fname, label in img_matches:
            clean_fname = os.path.basename(fname.strip()).lower()
            requested_filenames.append(clean_fname)
            if label:
                llm_labels[clean_fname] = label.strip()
            
        for clean_fname in requested_filenames:
            for meta in found_images:
                meta_fname = os.path.basename(meta["full_path"]).strip().lower()
                # Match if filename is identical or if LLM path is a subset of full path
                if clean_fname == meta_fname or clean_fname in meta["full_path"].lower():
                    if meta not in requested_images:
                        # Use LLM description if available, otherwise keep original
                        if clean_fname in llm_labels:
                            doc_prefix = meta["label"].split(" - ")[0]
                            meta["label"] = f"{doc_prefix} - {llm_labels[clean_fname]}"
                        requested_images.append(meta)
                    break

        # --- STAGE 4: Final Vision Filter (Optional) ---
        final_verified_images = []
        if requested_images and self.VISION_FILTER:
            try:
                # Filter only existing images
                valid_images = [img for img in requested_images if img["exists"]]
                if valid_images:
                    # Prepare Batch Vision Prompt
                    vision_prompt = f"""You are a MultiModal Verification Agent.
User Query: "{query}"
Proposed Answer: "{answer_text}"

Below are {len(valid_images)} images the text-model selected. 
For each image, determine if it is relevant evidence for the query and answer.
Return a simple comma-separated list of ONLY the indices (e.g., 0, 2) of the images that are RELEVANT. 
If none are relevant, return 'NONE'.
"""
                    # Attach all images to the prompt
                    content_list = [vision_prompt]
                    for img in valid_images:
                        content_list.append(Image.open(img["full_path"]))
                    
                    v_res = self.llm.generate_content(content_list)
                    res_text = v_res.text.upper()
                    
                    if "NONE" not in res_text:
                        # Extract indices
                        indices = re.findall(r"\d+", res_text)
                        for idx_str in indices:
                            idx = int(idx_str)
                            if 0 <= idx < len(valid_images):
                                final_verified_images.append(valid_images[idx])
                    else:
                        final_verified_images = [] # None verified
                else:
                    final_verified_images = []
            except Exception as e:
                # Fallback: include all requested if batch fails
                final_verified_images = requested_images
        else:
            # Skip verification pass for benchmark speed
            final_verified_images = requested_images

        # Deduplicate paths for clean citation
        unique_paths = []
        seen_paths = set()
        for p in pointers:
            path_str = f"{p['doc_id']} > {p['breadcrumb']}"
            if path_str not in seen_paths:
                unique_paths.append(path_str)
                seen_paths.add(path_str)

        return {
            "text": answer_text,
            "images": final_verified_images,
            "paths": unique_paths
        }
