import os
import json
import logging
import zipfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult

logging.basicConfig(level=logging.INFO)
# Load credentials from root .env
load_dotenv(Path(__file__).parent.parent.parent / '.env')

def get_credentials():
    client_id = os.environ.get("ADOBE_CLIENT_ID")
    client_secret = os.environ.get("ADOBE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Adobe credentials missing from .env")
    return ServicePrincipalCredentials(
        client_id=client_id,
        client_secret=client_secret
    )

def extract_pdf_to_md(pdf_path: str, output_dir: str):
    pdf_name = Path(pdf_path).stem
    paper_dir = os.path.join(output_dir, pdf_name)
    os.makedirs(paper_dir, exist_ok=True)
    
    zip_path = os.path.join(paper_dir, "extract.zip")
    md_path = os.path.join(paper_dir, f"{pdf_name}.md")
    
    # If already extracted, skip Adobe API call
    if not os.path.exists(zip_path):
        logging.info(f"Extracting {pdf_name} using Adobe SDK...")
        credentials = get_credentials()
        pdf_services = PDFServices(credentials=credentials)
        
        with open(pdf_path, 'rb') as f:
            input_stream = f.read()
        
        input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)
        
        # Configure extraction params for text, tables, and figures
        extract_pdf_params = ExtractPDFParams(
            elements_to_extract=[ExtractElementType.TEXT],
            elements_to_extract_renditions=[ExtractRenditionsElementType.TABLES, ExtractRenditionsElementType.FIGURES]
        )
        
        extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)
        location = pdf_services.submit(extract_pdf_job)
        pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)
        
        result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
        stream_asset: StreamAsset = pdf_services.get_content(result_asset)
        
        with open(zip_path, "wb") as f:
            f.write(stream_asset.get_input_stream())
    
    logging.info(f"Processing ZIP file for {pdf_name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract images from zip into a subfolder
        for member in zip_ref.namelist():
            if member.startswith('figures/') or member.startswith('tables/'):
                zip_ref.extract(member, paper_dir)
                
        # Read the structural JSON
        with zip_ref.open("structuredData.json") as json_file:
            data = json.load(json_file)
            
    markdown_lines = []
    
    for element in data.get("elements", []):
        path = element.get("Path", "")
        text = element.get("Text", "").strip()
        
        # Check if there are associated file renditions (images/tables)
        if "filePaths" in element and len(element["filePaths"]) > 0:
            for file_path in element["filePaths"]:
                # The file might be in figures/ or tables/ folder
                # We format it as a markdown image reference
                markdown_lines.append(f"\n![{text}]({file_path})\n")
            continue
            
        if not text:
            continue
            
        if "/H1" in path:
            markdown_lines.append(f"\n# {text}\n")
        elif "/Title" in path:
            markdown_lines.append(f"\n# {text}\n")
        elif "/H2" in path:
            markdown_lines.append(f"\n## {text}\n")
        elif "/H3" in path:
            markdown_lines.append(f"\n### {text}\n")
        elif "/H4" in path:
            markdown_lines.append(f"\n#### {text}\n")
        elif "/P" in path:
            markdown_lines.append(f"\n{text}\n")
        elif "/LBody" in path or "/LI" in path:
            markdown_lines.append(f"- {text}")
        else:
            markdown_lines.append(f"{text}")
            
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(markdown_lines))
        
    logging.info(f"Markdown and images generated in {paper_dir}")

def process_all_pdfs():
    base_dir = Path(__file__).parent.parent.parent
    pdf_dir = base_dir / "data" / "pdf"
    output_dir = base_dir / "data" / "extracted_papers"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        extract_pdf_to_md(str(pdf_file), str(output_dir))

if __name__ == "__main__":
    process_all_pdfs()
