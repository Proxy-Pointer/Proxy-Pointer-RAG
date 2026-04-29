import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.mm_rag_bot import MultimodalProxyPointerRAG
from src.config import INDEX_DIR, TREES_DIR, DATASET_DIR, RESULTS_DIR

# Ensure results dir exists
RESULTS_DIR = Path(RESULTS_DIR)
RESULTS_DIR.mkdir(exist_ok=True)

def run_suite():
    # 1. Load Queries from results dir
    query_file = RESULTS_DIR / "test_queries.json"
    if not query_file.exists():
        print(f"Error: {query_file} not found.")
        return
        
    with open(query_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        queries = data.get("test_queries", [])

    # 2. Init Bot
    print(f"Initializing Multimodal RAG Bot (Index: {INDEX_DIR})...")
    bot = MultimodalProxyPointerRAG(INDEX_DIR, TREES_DIR, DATASET_DIR)
    
    results_log = []
    log_path = RESULTS_DIR / "test_log.json"
    
    print(f"\nStarting Test Suite ({len(queries)} queries)...\n")
    
    for i, q in enumerate(queries):
        qid = q.get("id")
        text_query = q.get("query")
        category = q.get("category")
        
        print(f"[{i+1}/{len(queries)}] Running Query: {text_query[:60]}...")
        
        start_time = time.time()
        try:
            response = bot.chat(text_query)
            elapsed = time.time() - start_time
            
            # Format results for logging
            result_entry = {
                "id": qid,
                "category": category,
                "query": text_query,
                "response": response["text"],
                "sources": response.get("paths", []),
                "images_found": [
                    {
                        "label": img["label"],
                        "path": img["full_path"],
                        "exists": img["exists"]
                    } for img in response.get("images", [])
                ],
                "time_seconds": round(elapsed, 2)
            }
            results_log.append(result_entry)
            
            # Save incrementally after each query
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({"run_summary": {"total": len(queries), "completed": i+1}, "results": results_log}, f, indent=2)
                
            print(f"  -> SUCCESS (Took {elapsed:.1f}s, found {len(response.get('images', []))} images)")
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results_log.append({"id": qid, "query": text_query, "error": str(e)})

    print(f"\nTest Suite Complete! Report saved to: {log_path}")

if __name__ == "__main__":
    run_suite()
