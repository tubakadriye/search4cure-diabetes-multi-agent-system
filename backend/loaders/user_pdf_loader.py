import os
import fitz  # PyMuPDF
import requests
from tempfile import NamedTemporaryFile

def load_user_pdfs_from_folder(folder: str = "user_uploads"):
    pdfs = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            try:
                pdf = fitz.open(path)
                pdfs.append({"pdf": pdf, "title": filename, "url": f"file://{path}"})
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return pdfs

def load_user_pdf_from_url(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            pdf = fitz.open(tmp_path)
            return [{"pdf": pdf, "title": os.path.basename(tmp_path), "url": url}]
    except Exception as e:
        print(f"Error loading PDF from URL {url}: {e}")
    return []
