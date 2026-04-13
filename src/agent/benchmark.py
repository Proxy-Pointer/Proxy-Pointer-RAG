import os
import sys
import io
import time
import pandas as pd
import google.generativeai as genai

# Add project root to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR, INDEX_DIR, RESULTS_DIR, SYNTH_MODEL
from pp_rag_bot import ProxyPointerRAG

def evaluate_response_llm(eval_model, question, ground_truth, bot_response):
    """Uses LLM-as-a-judge to evaluate the bot response against the ground truth."""
    prompt = f"""You are an expert financial auditor benchmarking an AI assistant.
Compare the BOt RESPONSE against the GROUND TRUTH for the following QUESTION.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
BOT RESPONSE: {bot_response}

Your task is to yield a structured evaluation. Determine if the Bot Response is correct.
Output EXACTLY two lines in the following format:
SCORE: <icon>
NOTES: <your brief 1-2 sentence explanation>

For the <icon>, use exactly one of the following:
🟢 - Perfect match, encompasses, or explicitly improves upon the Ground Truth.
🟡 - Partial match; correct logic but minor data extraction variance or missing context.
🔴 - Fail / Hallucination / Contradicts reality.
"""
    try:
        result = eval_model.generate_content(prompt).text.strip()
        lines = result.split("\n")
        score = "🟡"
        notes = "Error parsing evaluation."
        
        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                extracted = line.replace("SCORE:", "").strip()
                if "🟢" in extracted: score = "🟢"
                elif "🔴" in extracted: score = "🔴"
                elif "🟡" in extracted: score = "🟡"
            elif line.startswith("NOTES:"):
                notes = line.replace("NOTES:", "").strip()
        return score, notes
    except Exception as e:
        return "🔴", f"LLM Judge failed: {str(e)}"

def run_benchmark(excel_path):
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} does not exist.")
        sys.exit(1)

    print(f"Loading dataset: {excel_path}...")
    df = pd.read_excel(excel_path)

    # Autodetect column names
    q_cols = [c for c in df.columns if c.lower() in ["question", "questions"]]
    a_cols = [c for c in df.columns if c.lower() in ["answer", "answers", "ground truth", "ground_truth"]]
    
    if not q_cols or not a_cols:
        print("Error: Could not find Question and Answer columns in the Excel file.")
        print("Found columns:", df.columns.tolist())
        sys.exit(1)
        
    q_col = q_cols[0]
    a_col = a_cols[0]
    print(f"Mapped columns: Question -> '{q_col}', Ground Truth -> '{a_col}'")

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(excel_path))[0]
    timestamp = str(int(time.time()))
    log_file = os.path.join(RESULTS_DIR, f"{base_name}_benchmark_{timestamp}.log")
    scorecard_file = os.path.join(RESULTS_DIR, f"{base_name}_scorecard_{timestamp}.md")

    # Setup the RAG bot
    idx_path = str(INDEX_DIR)
    data_path = str(DATA_DIR)
    print(f"Initializing ProxyPointerRAG with index: {idx_path}")
    bot = ProxyPointerRAG(idx_path, data_path)
    
    # Initialize the Judge
    eval_model = genai.GenerativeModel(SYNTH_MODEL)

    scorecard_data = []
    total_questions = len(df)
    
    print(f"Starting evaluation for {total_questions} questions...")

    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write(f"=== PROXY-POINTER AUTOMATED BENCHMARK ===\n")
        f_log.write(f"Dataset: {excel_path}\n\n")

        for index, row in df.iterrows():
            q = str(row[q_col])
            gt = str(row[a_col])
            subject = q[:40] + "..." if len(q) > 40 else q
            
            f_log.write("=" * 80 + "\n")
            f_log.write(f"USER QUERY: {q}\n")
            f_log.write("-" * 80 + "\n")

            # Capture stdout from bot to extract nodes
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()
            
            try:
                # Ask the bot
                answer = bot.chat(q)
                
                # Restore stdout Output
                sys.stdout = old_stdout
                output_text = mystdout.getvalue()
                
                f_log.write("--- BOT INTERNAL LOG ---\n")
                f_log.write(output_text.strip() + "\n")
                f_log.write("\n--- BOT SYNTHESIZED RESPONSE ---\n")
                f_log.write(f"{answer}\n\n")
                f_log.write("--- GROUND TRUTH ---\n")
                f_log.write(f"{gt}\n\n")
                
                # Evaluate using LLM judge
                score, notes = evaluate_response_llm(eval_model, q, gt, answer)
                
                scorecard_data.append({
                    "Q#": f"Q{index+1}",
                    "Query Subject": subject.replace("\n", " "),
                    "Ground Truth": (gt[:50] + "...").replace("\n", " "),
                    "Bot Output": (answer[:50] + "...").replace("\n", " "),
                    "Score": score,
                    "Notes": notes.replace("\n", " ")
                })
                
                f_log.write(f"--- JUDGE EVALUATION ---\n")
                f_log.write(f"SCORE: {score}\n")
                f_log.write(f"NOTES: {notes}\n\n")
                
            except Exception as e:
                sys.stdout = old_stdout
                f_log.write(f"ERROR processing query: {e}\n\n")
                scorecard_data.append({
                    "Q#": f"Q{index+1}",
                    "Query Subject": subject.replace("\n", " "),
                    "Ground Truth": (gt[:50] + "...").replace("\n", " "),
                    "Bot Output": "ERROR",
                    "Score": "🔴",
                    "Notes": f"Exception thrown: {str(e)}"
                })
            
            print(f"Processed: {q[:50]}... [{score}]")

    # Generate Scorecard Markdown
    with open(scorecard_file, "w", encoding="utf-8") as f_md:
        f_md.write(f"### Proxy-Pointer Automated Benchmark Scorecard ({base_name})\n\n")
        f_md.write("**Key:**\n")
        f_md.write("🟢 **Green:** Matches, encompasses, or explicitly improves upon the Ground Truth.\n")
        f_md.write("🟡 **Yellow:** Partial match; correct logic but minor data extraction variance.\n")
        f_md.write("🔴 **Red:** Fail / Hallucination / Contradicts reality.\n\n")
        
        f_md.write("| Q# | Query Subject | Ground Truth Summary | Bot Output Summary | Score | Notes |\n")
        f_md.write("| :--- | :--- | :--- | :--- | :---: | :--- |\n")
        
        green_count = 0
        yellow_count = 0
        red_count = 0
        for data in scorecard_data:
            if "🟢" in data["Score"]: green_count += 1
            elif "🟡" in data["Score"]: yellow_count += 1
            elif "🔴" in data["Score"]: red_count += 1
            row_str = f"| **{data['Q#']}** | {data['Query Subject']} | {data['Ground Truth']} | {data['Bot Output']} | {data['Score']} | {data['Notes']} |\n"
            f_md.write(row_str)
            
        f_md.write(f"\n**Final Score:** {green_count} 🟢 | {yellow_count} 🟡 | {red_count} 🔴\n")

    print(f"\nEvaluation complete.")
    print(f"Log saved to: {log_file}")
    print(f"Scorecard saved to: {scorecard_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <path_to_excel_file>")
        sys.exit(1)
        
    excel_file = sys.argv[1]
    run_benchmark(excel_file)
