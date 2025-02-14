import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"  # Adjust if deployed elsewhere

def upload_files():
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Upload") and uploaded_files:
        for file in uploaded_files:
            files = {"file": (file.name, file.read(), "application/pdf")}
            response = requests.post(f"{API_BASE_URL}/documents/upload/", files=files)
            if response.status_code == 200:
                st.success(f"{file.name} uploaded successfully!")
            else:
                st.error(response.json().get("detail", "Upload failed"))

def list_documents():
    st.header("Uploaded Documents")
    response = requests.get(f"{API_BASE_URL}/documents/list/")
    
    if response.status_code == 200:
        documents = response.json().get("documents", [])
        if not documents:
            st.info("No documents uploaded yet.")
        else:
            for doc in documents:
                with st.expander(doc["filename"]):
                    st.write(f"**Pages:** {doc.get('page_count', 'N/A')}")
                    st.write(f"**Summary:** {doc.get('summary', 'N/A')}")
                    st.write(f"**Chunks:** {doc.get('chunk_count', 0)}")

                    chunk_input = st.number_input(
                        "Enter chunk number to view:",
                        min_value=1,
                        max_value=doc.get("chunk_count", 1),
                        step=1,
                        key=f"chunk_{doc['filename']}"
                    )
                    
                    if st.button(f"View Chunk: {doc['filename']}", key=f"view_{doc['filename']}"):
                        chunk_response = requests.get(
                            f"{API_BASE_URL}/documents/get_chunk/{doc['filename']}/{chunk_input}"
                        )
                        if chunk_response.status_code == 200:
                            st.write(f"Chunk {chunk_input}: {chunk_response.json()['content']}")
                        else:
                            st.error("Failed to retrieve chunk.")

                    if st.button(f"Delete: {doc['filename']}", key=f"delete_{doc['filename']}"):
                        delete_response = requests.delete(f"{API_BASE_URL}/documents/delete/{doc['filename']}")
                        if delete_response.status_code == 200:
                            st.success(f"{doc['filename']} deleted successfully!")
                        else:
                            st.error("Failed to delete document.")
    else:
        st.error("Failed to retrieve document list.")

def ask_questions():
    st.header("Ask Questions")
    response = requests.get(f"{API_BASE_URL}/documents/list/")
    
    if response.status_code == 200:
        documents = response.json().get("documents", [])
        if documents:
            doc_names = [doc["filename"] for doc in documents]
            selected_doc = st.selectbox("Select a document", doc_names)
            question = st.text_input("Enter your question")

            if st.button("Ask"):
                qa_response = requests.post(
                    f"{API_BASE_URL}/documents/ask/",
                    json={"document": selected_doc, "question": question}
                )
                if qa_response.status_code == 200:
                    st.write(f"**Answer:** {qa_response.json()['answer']}")
                else:
                    st.error("Failed to get answer.")
        else:
            st.info("No documents available to query.")
    else:
        st.error("Failed to load documents.")

st.title("Intelligent Document QA System")

upload_files()
list_documents()
ask_questions()
