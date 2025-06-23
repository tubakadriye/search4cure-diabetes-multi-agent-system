import requests
from langchain_community.document_loaders import PubMedLoader
import time
import xml.etree.ElementTree as ET
import fitz  # pymupdf
from io import BytesIO

class PubMedPDFLoader:
    def __init__(self, query, max_docs=300):
        self.query = query
        self.max_docs = max_docs
        self.loader = PubMedLoader(query=query, load_max_docs=max_docs)

    def fetch_metadata(self, uid):
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": uid, "retmode": "xml"}
        r = requests.get(url, params=params)
        if r.status_code != 200:
            print(f"Failed to fetch metadata for UID {uid}")
            return None

        root = ET.fromstring(r.text)
        title = f"uid_{uid}"  # default

        # Find DOI
        doi = None
        for article_id in root.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = article_id.text
        
        # Find PMCID
        pmcid = None
        for article_id in root.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "pmc":
                pmcid = article_id.text
        return {"doi": doi, "pmcid": pmcid}

    def get_pdf_url(self, metadata):
        pmcid = metadata.get("pmcid")
        if pmcid:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        #TODO: extend here to try DOI-based URL if needed
        return None

    def download_pdfs(self):
        docs = self.loader.load()
        pdf_docs_with_meta = []

        for doc in docs:
            uid = doc.metadata.get("uid")        
            if not uid:
                continue
            #title = doc.metadata.get("Title", f"uid_{uid}").strip().replace(" ", "_").replace("/", "_")
            raw_title = doc.metadata.get("Title")
            print("raw_title", raw_title)
            if isinstance(raw_title, str):
                title = raw_title.strip().replace(" ", "_").replace("/", "_")
            elif isinstance(raw_title, dict) and "text" in raw_title:
                title = raw_title["text"].strip().replace(" ", "_").replace("/", "_")
            else:
                title = f"uid_{uid}"

            metadata = self.fetch_metadata(uid)
            if not metadata:
                continue
            pdf_url = self.get_pdf_url(metadata)
            if not pdf_url:
                print(f"No PDF URL for UID {uid}, skipping.")
                continue

            print(f"PDF URL for UID {uid}: {pdf_url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer": "https://www.ncbi.nlm.nih.gov/"
            }
            r = requests.get(pdf_url, headers=headers)
            content_type = r.headers.get("Content-Type", "")
            print(f"Status code: {r.status_code} | Content-Type: {content_type} | Content-Length: {len(r.content)}")

            if r.status_code == 200 and content_type.startswith("application/pdf"):
                try:
                    pdf_stream = BytesIO(r.content)
                    pdf = fitz.open(stream=pdf_stream, filetype="pdf")
                    print(f"Successfully opened PDF for UID {uid}, appending.")
                    pdf_docs_with_meta.append({
                        "pdf": pdf,
                        "title": title,
                        "url": pdf_url,
                        "uid": uid
                    })
                except Exception as e:
                    print(f"fitz.open failed for UID {uid}: {e}")
            else:
                print(f"Failed to retrieve valid PDF for UID {uid}.")
            time.sleep(1) 

        print(f"\nTotal PDFs collected: {len(pdf_docs_with_meta)}") 
        return pdf_docs_with_meta