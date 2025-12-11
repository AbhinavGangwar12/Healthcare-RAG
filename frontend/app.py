import streamlit as st
import requests

# Backend API endpoints
API_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_URL}/upload"
QUERY_ENDPOINT = f"{API_URL}/query"

st.set_page_config(page_title="Healthcare RAG Chat", layout="wide")
st.title("Healthcare Document QA")

# PDF upload section
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.sidebar.write(f"Uploading: {uploaded_file.name}")
        try:
            # Read file content
            file_bytes = uploaded_file.read()
            # Reset file pointer after read
            uploaded_file.seek(0)
            # Build multipart file
            files = {"file": (uploaded_file.name, file_bytes, "application/pdf")}
            response = requests.post(UPLOAD_ENDPOINT, files=files)
            if response.status_code == 200:
                st.sidebar.success(f"Uploaded: {uploaded_file.name}")
            else:
                st.sidebar.error(f"Failed: {uploaded_file.name} — {response.text}")
        except Exception as e:
            st.sidebar.error(f"Error uploading {uploaded_file.name}: {e}")

# User question input
st.header("Ask a Question")
question = st.text_input("Enter your question about the uploaded documents:")
if st.button("Submit Question"):
    if not question:
        st.warning("Please enter a question.")
    else:
        try:
            res = requests.post(QUERY_ENDPOINT, json={"query": question})
            data = res.json()
            answer = data.get("answer", "")
            citations = data.get("citations", [])
            confidence = data.get("confidence", 0.0)
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Citations")
            for c in citations:
                st.markdown(f"- **{c['source']}** (page {c['page']}, chunk {c['chunk']}), confidence: {c['confidence']:.2f}")
            st.subheader("Confidence")
            st.write(f"{confidence:.2f}")
        except Exception as e:
            st.error(f"Query failed: {e}")
