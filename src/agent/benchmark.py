import os
import sys

# Force UTF-8 encoding for Windows console emoji support
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import io
import time
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

# Add project root to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR, INDEX_DIR, RESULTS_DIR, SYNTH_MODEL
from pp_rag_bot import ProxyPointerRAG

def retry_api_call(func, *args, max_retries=5, initial_delay=5, **kwargs):
    """Executes a function with exponential backoff on 429/ResourceExhausted errors."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "resource" in error_str or "quota" in error_str:
                if attempt == max_retries - 1:
                    raise e
                print(f"\n[429 Quota Error] Backing off for {delay} seconds before retry...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e

def evaluate_response_llm(eval_model, question, ground_truth, bot_response):
    """Uses LLM-as-a-judge to evaluate the bot response against the ground truth."""
    prompt = f"""You are an expert financial auditor benchmarking an AI assistant.
Compare the BOT RESPONSE against the GROUND TRUTH for the following QUESTION.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
BOT RESPONSE: {bot_response}

Your task is to yield a structured evaluation. Determine if the Bot Response is fundamentally correct.

EVALUATION GUIDELINES:
1. STRICT RULE: DO NOT dock points for confusing "percent" vs "percentage points". You MUST treat them as identical for this evaluation. This is not a strict audit report. If the numerical value is correct, score it 🟢 regardless of whether the suffix is "%", "percent", or "percentage points".
2. STRICT RULES FOR SCORING:
- Ignore minor rounding differences (e.g., 7.09% vs. 7.40%, or 1.22x vs 1.3x).
- Ignore pedantic language differences (e.g., "percentage points" vs "percent" or "absolute increase" terminology) as long as the underlying math and logical conclusion are correct.
- If the bot correctly computes a difference from a negative value to a positive value (e.g., -10 to +20 is an absolute increase of 30), DO NOT penalize it for "misrepresenting directionality."
- DO NOT hallucinate alternative financial figures from your own pre-training data. If the bot cites specific numbers from its retrieved context (e.g., "$200 million"), you MUST accept those numbers as factually retrieved. Judge ONLY whether the bot's reasoning using those numbers aligns with the essence of the Ground Truth.
- If the bot provides a correct, multi-step calculation that arrives at the GT but includes extra information, score it 🟢.
3. If the user asks for "an alternative" approach, and the BOT provides a mathematically sound, valid alternative that differs from the specific example in the GROUND TRUTH, you MUST score it 🟢. 
4. Extra contextual depth added by the bot does not penalize the score.

Output EXACTLY two lines in the following format:
SCORE: <icon>
NOTES: <your brief 1-2 sentence explanation>

For the <icon>, use exactly one of the following:
🟢 - Fundamentally correct. Matches the core facts, encompasses the truth, OR provides a mathematically valid alternative when requested.
🟡 - Partial match. Conceptually on the right track but misses a key data point or has a minor quantitative error.
🔴 - Fail. Hallucination, completely wrong data, or contradicts reality.
"""
    try:
        def _call_eval():
            return eval_model.generate_content(prompt, generation_config={"temperature": 0.0})
        result = retry_api_call(_call_eval).text.strip()
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
    
    qno_col = next((c for c in df.columns if c.lower() in ["qno", "sno", "q#", "id"]), None)
    company_col = next((c for c in df.columns if c.lower() in ["company", "ticker"]), None)
    
    # Pre-filter empty rows
    valid_mask = df[q_col].notna() & (df[q_col].astype(str).str.strip().str.lower() != "nan") & (df[q_col].astype(str).str.strip() != "")
    df_filtered = df[valid_mask].copy()
    total_questions = len(df_filtered)
    
    print(f"Starting evaluation for {total_questions} questions...")

    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write(f"=== PROXY-POINTER AUTOMATED BENCHMARK ===\n")
        f_log.write(f"Dataset: {excel_path}\n\n")

        def clean_md(s):
            return str(s).replace("|", "-").replace("\n", " ").replace("\r", "").strip()

        def trunc(s, max_len):
            s = clean_md(s)
            return s if len(s) <= max_len else s[:max_len-3] + "..."

        for i, (orig_index, row) in enumerate(df_filtered.iterrows()):
            q = str(row[q_col]).strip()
            gt = str(row[a_col]).strip()
            subject = q[:40] + "..." if len(q) > 40 else q
            
            # Format qno cleanly to avoid '1.0' from Pandas
            q_val = row[qno_col] if qno_col and not pd.isna(row[qno_col]) else (i + 1)
            if isinstance(q_val, float) and q_val.is_integer():
                q_val = int(q_val)
            q_val_str = str(q_val).strip()
            
            q_label = q_val_str if q_val_str.lower().startswith('q') else f"Q{q_val_str}"
            company_label = f" [{str(row[company_col]).strip()}]" if company_col and not pd.isna(row[company_col]) and str(row[company_col]).strip().lower() != "nan" else ""
            display_q = f"{q_label}{company_label}"
            
            f_log.write("=" * 80 + "\n")
            f_log.write(f"{display_q} USER QUERY: {q}\n")
            f_log.write("-" * 80 + "\n")

            # Capture stdout from bot to extract nodes
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()
            score = "🔴"
            
            try:
                # Ask the bot (using our retry helper)
                answer = retry_api_call(bot.chat, q)
                
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
                    "Q#": f"**{trunc(display_q, 25)}**",
                    "Query Subject": trunc(q, 35),
                    "Ground Truth": trunc(gt, 40),
                    "Bot Output": trunc(answer, 40),
                    "Score": clean_md(score),
                    "Notes": clean_md(notes)
                })
                
                f_log.write(f"--- JUDGE EVALUATION ---\n")
                f_log.write(f"SCORE: {score}\n")
                f_log.write(f"NOTES: {notes}\n\n")
                
            except Exception as e:
                sys.stdout = old_stdout
                f_log.write(f"ERROR processing query: {e}\n\n")
                scorecard_data.append({
                    "Q#": f"**{trunc(display_q, 25)}**",
                    "Query Subject": trunc(q, 35),
                    "Ground Truth": trunc(gt, 40),
                    "Bot Output": "ERROR",
                    "Score": "🔴",
                    "Notes": clean_md(f"Exception thrown: {str(e)}")
                })
            
            print(f"Processed {display_q}: {q[:50]}... [{score}]")

    # Generate Scorecard Markdown
    with open(scorecard_file, "w", encoding="utf-8") as f_md:
        f_md.write(f"### Proxy-Pointer Automated Benchmark Scorecard ({base_name})\n\n")
        f_md.write("**Key:**\n")
        f_md.write("🟢 **Green:** Matches, encompasses, or explicitly improves upon the Ground Truth.\n")
        f_md.write("🟡 **Yellow:** Partial match; correct logic but minor data extraction variance.\n")
        f_md.write("🔴 **Red:** Fail / Hallucination / Contradicts reality.\n\n")
        
        # Calculate col widths based on max lengths
        c1 = max(len("Q#"), max([len(d["Q#"]) for d in scorecard_data] + [0]))
        c2 = max(len("Query Subject"), max([len(d["Query Subject"]) for d in scorecard_data] + [0]))
        c3 = max(len("Ground Truth Summary"), max([len(d["Ground Truth"]) for d in scorecard_data] + [0]))
        c4 = max(len("Bot Output Summary"), max([len(d["Bot Output"]) for d in scorecard_data] + [0]))
        c5 = max(len("Score"), max([len(d["Score"]) for d in scorecard_data] + [0]))

        h1 = "Q#".ljust(c1)
        h2 = "Query Subject".ljust(c2)
        h3 = "Ground Truth Summary".ljust(c3)
        h4 = "Bot Output Summary".ljust(c4)
        h5 = "Score".ljust(c5)

        f_md.write(f"| {h1} | {h2} | {h3} | {h4} | {h5} | Notes |\n")
        f_md.write(f"| {'-'*c1} | {'-'*c2} | {'-'*c3} | {'-'*c4} | {'-'*c5} | :--- |\n")
        
        green_count = 0
        yellow_count = 0
        red_count = 0
        for data in scorecard_data:
            if "🟢" in data["Score"]: green_count += 1
            elif "🟡" in data["Score"]: yellow_count += 1
            elif "🔴" in data["Score"]: red_count += 1
            
            r1 = data["Q#"].ljust(c1)
            r2 = data["Query Subject"].ljust(c2)
            r3 = data["Ground Truth"].ljust(c3)
            r4 = data["Bot Output"].ljust(c4)
            r5 = data["Score"].ljust(c5)
            
            row_str = f"| {r1} | {r2} | {r3} | {r4} | {r5} | {data['Notes']} |\n"
            f_md.write(row_str)
            
        f_md.write(f"\n**Final Score:** {green_count} 🟢 | {yellow_count} 🟡 | {red_count} 🔴\n")

    print(f"\nEvaluation complete.")
    print(f"Log saved to: {log_file}")
    print(f"Scorecard saved to: {scorecard_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <path_to_excel_file>")
        sys.exit(1)
        
    excel_file = " ".join(sys.argv[1:])
    run_benchmark(excel_file)
