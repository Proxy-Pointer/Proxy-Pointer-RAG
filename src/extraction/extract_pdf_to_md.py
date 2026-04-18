"""
Proxy-Pointer: PDF to Markdown Extraction via LlamaParse

Converts PDF documents to structured Markdown files.
Reads from data/pdf/, writes to data/documents/.
Skips PDFs that already have a corresponding .md file.

Usage:
    python -m src.extraction.extract_pdf_to_md              # batch convert all
    python -m src.extraction.extract_pdf_to_md --file x.pdf # single file
"""
import os
import sys
import logging
import argparse

# Add project root to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PDF_DIR, DATA_DIR, LLAMA_PARSE_TIER

logging.basicConfig(level=logging.INFO, format="%(message)s")


def extract_pdf(pdf_path, output_dir):
    """Convert a single PDF to Markdown using LlamaParse."""
    try:
        from llama_cloud import LlamaCloud
    except ImportError:
        logging.error("llama-cloud not installed. Run: pip install llama-cloud")
        sys.exit(1)

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        logging.error("LLAMA_CLOUD_API_KEY not set in .env")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.md")

    if os.path.exists(output_path):
        logging.info(f"  [SKIP] {base_name}.md already exists.")
        return

    logging.info(f"  Extracting: {os.path.basename(pdf_path)}...")

    try:
        client = LlamaCloud(api_key=api_key)

        # Upload, parse, and wait for result in one call
        with open(pdf_path, "rb") as f:
            result = client.parsing.parse(
                upload_file=f,
                tier=LLAMA_PARSE_TIER,
                version="latest",
                expand=["markdown"],
            )

        # Result.markdown.pages is a list of page objects with .markdown text
        pages = result.markdown.pages
        full_md = "\n\n".join(page.markdown for page in pages if page.markdown)

        # Write markdown output
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(full_md)

        logging.info(f"  -> Saved: {output_path} ({len(pages)} pages, {len(full_md)} chars)")

    except Exception as e:
        logging.error(f"  -> Failed: {e}")
        logging.info("  Check https://docs.cloud.llamaindex.ai/ for API updates.")


def batch_extract(pdf_dir, output_dir):
    """Convert all PDFs in pdf_dir to Markdown."""
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logging.warning(f"No PDF files found in {pdf_dir}")
        return

    logging.info(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")
    for file in sorted(pdf_files):
        extract_pdf(os.path.join(pdf_dir, file), output_dir)

    logging.info("Extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDFs to Markdown via LlamaParse")
    parser.add_argument("--file", help="Single PDF file to extract (full path)")
    args = parser.parse_args()

    if args.file:
        extract_pdf(args.file, str(DATA_DIR))
    else:
        batch_extract(str(PDF_DIR), str(DATA_DIR))
