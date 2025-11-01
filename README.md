# 🧠 Multimodal RAG System: Contextual AI for Diverse Data

## Live Demo

[![Gradio Deployment](https://img.shields.io/badge/Test_Live_Demo-Gradio_Hugging_Face_Space-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/VimalrajS04/gradio_interface)

---

## Overview

This project implements a **Multimodal Retrieval-Augmented Generation (RAG) system** designed to process, store, and query diverse data types, including **plain text, images, and complex PDFs** with mixed content. It leverages **OCR, a Vision Language Model (VLM), vector search, and a powerful LLM** (Groq) to provide accurate, context-aware answers with source attribution.

The system is exposed via a **Flask REST API**, enabling seamless integration for document ingestion and querying.

---

## 🚀 Key Features and Technical Specifications

| Category | Feature | Implementation Details |
| :--- | :--- | :--- |
| **Data Ingestion** | Multimodal Support | Processes `.txt`, `.md`, `.jpg`, `.jpeg`, `.png`, and `.pdf` files. |
| | Intelligent Image Handling | Uses **`doctr` (OCR)** for extracting text from images/scans and a **Vision Model (`llama-4-scout-17b`)** for summarizing non-textual visuals (charts, diagrams). |
| | Advanced PDF Processing | Extracts text, performs visual analysis on each page, and processes embedded images (using OCR or VLM) within PDFs. |
| **Storage & Retrieval** | Vector Database | **Qdrant** is used for efficient vector storage and retrieval. |
| | Embeddings | **`sentence-transformers/all-MiniLM-L6-v2`** is used for generating dense vector embeddings. |
| | Metadata | Stores **`source` (filename), `file_type`, and `upload_timestamp`** for attribution. |
| **Query & RAG** | Core LLM | **Groq's `llama-3.3-70b-versatile`** for rapid and accurate answer generation. |
| | Cross-Modal Queries | The retrieval logic includes a **reranking mechanism** to boost results from "Visual Analysis" for queries containing keywords like 'chart', 'graph', or 'diagram'. |
| **Backend** | API | Built with **Flask**, providing dedicated endpoints for document ingestion and querying. |

---

## 📐 Architecture Overview

The system follows a classic RAG pattern, enhanced for multimodality:

1.  **Ingestion Pipeline:**
    * Files are uploaded via the `/upload` endpoint (or via the Gradio interface).
    * The file type determines the processing path.
    * **PDFs** are iterated page-by-page, extracting text and processing visuals/embedded images.
    * **Images** are first attempted with OCR; if the text is insufficient, the VLM is used for a detailed summary.
    * Processed content is wrapped into LangChain `Document` objects.
    * Documents are embedded and stored in the **Qdrant** vector store.
2.  **Query Pipeline:**
    * A user query is sent to the `/query` endpoint (or via the Gradio chat interface).
    * The query is embedded.
    * **Qdrant** performs a similarity search to retrieve the top *k* most relevant document chunks.
    * A **reranking step** is applied to prioritize visual-based context for visual queries.
    * The retrieved context and the user query are passed to the **Groq LLM** with a strict system prompt.
    * The LLM generates a final answer, and the API returns the answer along with the source attribution and relevance scores.



---

## 🛠️ Setup Instructions

### Prerequisites

* Python 3.8+
* A **Groq API Key** for LLM and VLM access.
* Access to a **Qdrant Vector Database** (cloud or local).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPO_URL>
    cd multimodal-rag-system
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Note: doctr might require specific system dependencies (e.g., libgl1-mesa-glx on Linux)
    ```

3.  **Set Environment Variables:**
    Create a `.env` file (or export) with your credentials and configuration. The provided Python script uses:

    ```bash
    export GROQ_API_KEY="your-groq-api-key"
    export QDRANT_API_KEY="your-qdrant-api-key"
    # The QDRANT_URL is hardcoded in the script for simplicity:
    # QDRANT_URL = "[https://bdf142ef-7e2a-433b-87a0-301ff301e3af.us-east4-0.gcp.cloud.qdrant.io:6333](https://bdf142ef-7e2a-433b-87a0-301ff301e3af.us-east4-0.gcp.cloud.qdrant.io:6333)" 
    ```

4.  **Prepare Sample Data:**
    Create a `data/` folder in the root directory and place your sample text, image, and PDF files inside. This folder will be processed automatically on the first run.

### Running the API

The application automatically builds the initial Qdrant store from the `data/` folder on startup.

```bash
python your_main_file_name.py
