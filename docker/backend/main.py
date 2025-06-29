#streamlit run main.py
import os
from google.adk import configure_dev_ui
from agent import root_agent
from agent_runner import call_agent


# Optionally enable serving the ADK UI via Streamlit
serve_developer_ui = True

# if serve_developer_ui:
#     configure_dev_ui(agent=root_agent)
import socket
ON_CLOUD = 'google' in socket.gethostname()

from tqdm import tqdm
from db.mongo_utils import insert_df_to_mongodb
from embeddings.gemini_text_embedding import get_gemini_embedding
import streamlit as st
from loaders.user_pdf_loader import load_user_pdfs_from_folder, load_user_pdf_from_url
from loaders.arxiv_loader import ArxivPDFLoader
from loaders.pubmed_loader import PubMedPDFLoader
from multimodal.pdf_processing import process_pdfs_and_upload_images
import os
from PIL import Image
import io
from embeddings.image_embedding_pipeline import embed_docs_with_clip
from utils.attribute_combiner import combine_all_attributes
from utils.csv_processing import process_and_upload_csv
from utils.gcs_utils import upload_image_to_gcs
from embeddings.clip import get_clip_embedding
from db.mongodb_client import mongodb_client
from db.index_utils import create_vector_index, create_multivector_index
from multimodal.pdf_processing import process_and_embed_docs
import pandas as pd
import datetime

from utils.general_helpers import print_dataframe_info

DB_NAME = "diabetes_data"
response_collection = mongodb_client[DB_NAME]["responses"]

# --- Setup ---
st.set_page_config(page_title="Search4Cure.AI: Diabetes", layout="wide")
st.title("🔬 Search4Cure.AI: Diabetes")
st.markdown("**Search4Cure.AI: Diabetes** is a multimodal research assistant designed to help you explore, analyze, and embed diabetes-related scientific documents, images, and datasets using AI-powered search and visualization.")


# Setup MongoDB
db = mongodb_client["diabetes_data"]
pdf_collection = db["docs_multimodal"]


# --- Session States ---
for key in ["user_pdfs", "embedded_docs", "search_results"]:
    st.session_state.setdefault(key, [])

# --- Human-in-the-Loop HITL Session Keys ---
for key in ["agent_raw_response", "agent_approved_text", "agent_review_mode"]:
    st.session_state.setdefault(key, "" if "response" in key else False)

# --- Sidebar Upload ---
with st.sidebar:
    st.header("Upload PDFs, CSVs, or Images")

    # PDFs from file
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        user_upload_dir = "user_uploads"
        os.makedirs(user_upload_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            path = os.path.join(user_upload_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
        st.success(f"Saved {len(uploaded_files)} PDF(s).")
        #user_pdfs = load_user_pdfs_from_folder(user_upload_dir)
        # Load PDFs from folder and append to session state
        st.session_state.user_pdfs += load_user_pdfs_from_folder(user_upload_dir)

    # PDFs from URL
    pdf_url = st.text_input("Enter PDF URL:")
    if pdf_url:
        #user_pdfs = load_user_pdf_from_url(pdf_url)
        # Append loaded PDF(s) from URL to session state list
        st.session_state.user_pdfs += load_user_pdf_from_url(pdf_url)

    # --- CSV Upload ---
    uploaded_csv = st.file_uploader("📊 Upload CSV", type=["csv", "xlsx", "xls", "json"])
    csv_name = uploaded_csv.name.rsplit('.', 1)[0] if uploaded_csv else ""
    st.text_input("CSV Name:", value=csv_name)

    # --- Optional Arxiv & PubMed fetching ---
    add_research = st.checkbox("🔍 Redo Diabetes research from Arxiv & PubMed")

    #all_pdfs = user_pdfs

    if add_research:
        if st.button("📥 Load Arxiv & PubMed PDFs"):
            with st.spinner("Loading Arxiv and PubMed PDFs..."):
                arxiv_pdfs = ArxivPDFLoader(query="Diabetes").download_pdfs()
                pubmed_pdfs = PubMedPDFLoader(query="Diabetes").download_pdfs()
                #all_pdfs += arxiv_pdfs + pubmed_pdfs
                st.session_state.user_pdfs += arxiv_pdfs + pubmed_pdfs
                st.success(f"Loaded {len(arxiv_pdfs) + len(pubmed_pdfs)} research PDFs.")

    # Upload image
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    default_name = uploaded_image.name.rsplit('.', 1)[0] if uploaded_image else ""
    custom_image_name = st.text_input("Image name:", default_name)

    # -- Process all button ---
    if st.button("📥 Process All"):
        if not st.session_state.user_pdfs and not uploaded_image:
            st.warning("Please upload PDFs or an image first.")
        else:
            with st.spinner("Embedding and saving..."):
                embedded_docs = process_and_embed_docs(
                    pdfs=st.session_state.user_pdfs,
                    uploaded_image=uploaded_image,
                    image_name=custom_image_name,
                    get_embedding=get_clip_embedding ## for image embedding
                )
                st.session_state.embedded_docs = embedded_docs
                if embedded_docs:
                    # Now embed documents
                    #embedded_docs = embed_docs_with_clip(docs, get_clip_embedding)                   
                    # Insert into MongoDB
                    pdf_collection.insert_many(embedded_docs)                 

                if uploaded_csv:
                    process_and_upload_csv(
                        uploaded_csv,
                        datasets_col=db["datasets"],
                        data_col=db["records_embeddings"],
                        embedding_fn=get_gemini_embedding
                    )
 


                st.success(f"✅ Processed, uploaded, and embedded {len(embedded_docs)} items.")
                # Example: show first 3 embedded docs metadata or images
                for doc in embedded_docs[:3]:
                    st.image(doc["image"], caption=f'{doc["pdf_title"]} - Page {doc["page_number"]}', width=150)


# --- Search Interface  ---

st.markdown("<h2 style='text-align:center'>🔍 Search Query</h2>", unsafe_allow_html=True)
query = st.text_input("", placeholder="Enter your search query here...", key="search_query", max_chars=200)

search_button = st.button("Search")

if search_button:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Using Diabetes Research Asisstant Agent to answer..."):
            agent_output = call_agent(query)
            st.success("Agent response received!")
            st.markdown(agent_output)
        st.session_state.agent_raw_response = agent_output
        st.session_state.agent_review_mode = True  # activate HITL mode

# --- Human-in-the-Loop Review Interface ---
if st.session_state.agent_review_mode:
    st.markdown("### 🤖 Suggested Answer by Agent")
             
    st.session_state.agent_approved_text = st.text_area(
        "Edit or approve the agent's response:",
        value=st.session_state.agent_raw_response,
        height=200,
        key="editable_response"
    )
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve"):
            st.session_state.agent_review_mode = False
            st.success("✅ Approved Response:")
            st.markdown(f"**{st.session_state.agent_approved_text}**")
            # Here you can save the approved response to a database/log
            response_collection.insert_one({"query": query, "response": st.session_state.agent_approved_text})


    with col2:
        if st.button("🔄 Regenerate Agent Response"):
            with st.spinner("Regenerating answer..."):
                new_agent_response = call_agent(query)
                new_response_text = new_agent_response.get("output", str(new_agent_response))
                st.session_state.agent_approved_text = new_response_text
                st.experimental_rerun()


    # Expert input section
    st.markdown("### 📝 Expert Input")
    expert_input = st.text_area(
        "Expert can edit or write their own answer below:",
        value=st.session_state.agent_approved_text,
        height=200,     
        key="expert_response"
        )

    if st.button("✅ Submit Expert Answer"):
        st.session_state.agent_approved_text = expert_input
        st.session_state.agent_review_mode = False
        st.success("✅ Expert-approved Response:")
        st.markdown(f"**{st.session_state.agent_approved_text}**")
        response_collection.insert_one({"query": query, "expert_input": st.session_state.agent_approved_text})





            

