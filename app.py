from doctr.models import ocr_predictor
from PIL import Image
import numpy as np
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
from groq import Groq
import base64
from io import BytesIO
import fitz  # PyMuPDF
import time
import shutil

# Flask imports
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile

# Qdrant imports
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

# -------------------------------
# Configuration
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
ocr_model = ocr_predictor(pretrained=True).to(device)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_MODEL = "llama-3.3-70b-versatile"

QDRANT_URL = "https://bdf142ef-7e2a-433b-87a0-301ff303e3af.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "multimodal_rag_store"

# -------------------------------
# Helper Functions
# -------------------------------
def has_substantial_text(text, min_words=10):
    words = text.split()
    return len(words) >= min_words

def analyze_image_with_vision(img_path=None, img_bytes=None, pil_image=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            if pil_image:
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_data = buffered.getvalue()
                img_format = "png"
            elif img_path:
                with open(img_path, "rb") as img_file:
                    img_data = img_file.read()
                img_format = img_path.lower().split('.')[-1]
            elif img_bytes:
                img_data = img_bytes
                img_format = "png"
            else:
                return ""
            
            base64_image = base64.b64encode(img_data).decode('utf-8')
            if img_format == 'jpg':
                img_format = 'jpeg'
            
            vision_prompt = """Analyze this image carefully and provide a detailed description:
1. IDENTIFY THE TYPE: Is this a chart, graph, table, diagram, photograph, or text document?
2. IF IT'S A CHART/GRAPH/TABLE:
   - Specify the exact type (bar chart, pie chart, line graph, scatter plot, table, etc.)
   - List ALL categories/labels shown
   - Describe the data values and trends
   - Mention axis labels, title, legend if present
   - Highlight key insights or patterns
3. IF IT'S A PHOTOGRAPH/DIAGRAM:
   - Describe what you see in detail
   - Identify key objects, people, or concepts
   - Note any text visible in the image
4. IF IT'S A TEXT DOCUMENT:
   - Summarize the main content and structure
Provide a comprehensive description suitable for semantic search. Be specific and detailed."""

            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{img_format};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                model=VISION_MODEL,
                temperature=0.2,
                max_tokens=1500,
            )
            summary = chat_completion.choices[0].message.content
            if summary and len(summary.strip()) > 30:
                return summary
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return ""
    return ""

def extract_text_from_image(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        result = ocr_model([image_np])
        text = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join([word.value for word in line.words])
                    text.append(line_text)
        ocr_text = "\n".join(text)
        
        if has_substantial_text(ocr_text, min_words=10):
            print(f"📄 {os.path.basename(img_path)}: Using OCR")
            return ocr_text
        else:
            print(f"🖼️  {os.path.basename(img_path)}: Using Vision Model")
            vision_summary = analyze_image_with_vision(img_path=img_path)
            return vision_summary if vision_summary else ocr_text
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"📝 {os.path.basename(file_path)}: Extracted text")
        return text
    except Exception as e:
        print(f"❌ Error reading text file {file_path}: {e}")
        return ""

def extract_content_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        all_content = []
        
        for page_num, page in enumerate(doc, 1):
            page_content = []
            text = page.get_text()
            
            if text.strip():
                page_content.append(f"[Page {page_num} - Text Content]\n{text}")
            
            try:
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                page_image = Image.open(BytesIO(img_data)).convert("RGB")
                
                vision_analysis = analyze_image_with_vision(pil_image=page_image)
                if vision_analysis and len(vision_analysis.strip()) > 30:
                    page_content.append(f"[Page {page_num} - Visual Analysis]\n{vision_analysis}")
            except Exception as e:
                print(f"❌ Error rendering page {page_num}: {e}")
            
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list, 1):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    image_np = np.array(image)
                    result = ocr_model([image_np])
                    ocr_text = []
                    for ocr_page in result.pages:
                        for block in ocr_page.blocks:
                            for line in block.lines:
                                line_text = " ".join([word.value for word in line.words])
                                ocr_text.append(line_text)
                    extracted_text = "\n".join(ocr_text)
                    
                    if has_substantial_text(extracted_text, min_words=10):
                        page_content.append(f"[Page {page_num} - Embedded Image {img_index} OCR]\n{extracted_text}")
                    else:
                        vision_summary = analyze_image_with_vision(img_bytes=image_bytes)
                        if vision_summary:
                            page_content.append(f"[Page {page_num} - Embedded Image {img_index} Analysis]\n{vision_summary}")
                except Exception as e:
                    print(f"❌ Error processing embedded image {img_index}: {e}")
                    continue
            
            if page_content:
                combined_page = "\n\n---SECTION BREAK---\n\n".join(page_content)
                all_content.append(combined_page)
        
        doc.close()
        final_content = "\n\n---PAGE BREAK---\n\n".join(all_content)
        return final_content
    except Exception as e:
        print(f"❌ Error processing PDF {pdf_path}: {e}")
        return ""

def create_documents_from_folder(folder_path):
    docs = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            full_path = os.path.join(root, filename)
            file_ext = filename.lower().split('.')[-1]
            
            text = ""
            if file_ext in ["jpg", "jpeg", "png"]:
                text = extract_text_from_image(full_path)
            elif file_ext in ["txt", "md"]:
                text = extract_text_from_txt(full_path)
            elif file_ext == "pdf":
                text = extract_content_from_pdf(full_path)
            else:
                continue
            
            if text.strip():
                relative_path = os.path.relpath(full_path, folder_path)
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": relative_path,
                        "filename": filename,
                        "file_type": file_ext,
                        "upload_timestamp": os.path.getmtime(full_path)
                    }
                )
                docs.append(doc)
                print(f"✅ Added {filename}")
    return docs

def build_or_update_qdrant_store(folder_path):
    print("\n🔄 Building Qdrant collection...")
    docs = create_documents_from_folder(folder_path)
    if not docs:
        print("⚠️  No valid documents found!")
        return None
    
    try:
        vector_store = Qdrant.from_documents(
            docs,
            embedding_model,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            force_recreate=True
        )
        print(f"✅ Created collection with {len(docs)} documents")
        return vector_store
    except Exception as e:
        print(f"❌ Error with Qdrant: {e}")
        return None

def query_qdrant_store(query_text, k=3):
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=20)
        vector_store = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embedding_model
        )
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        return []
    
    initial_k = k * 3
    results = vector_store.similarity_search_with_score(query_text, k=initial_k)
    
    visual_query_keywords = ['chart', 'graph', 'bar', 'pie', 'plot', 'diagram', 'table', 'visual', 'visualization']
    is_visual_query = any(keyword in query_text.lower() for keyword in visual_query_keywords)
    
    if is_visual_query:
        reranked_results = []
        for doc, score in results:
            boost = 0.0
            if "Visual Analysis]" in doc.page_content:
                boost += 0.5
            adjusted_score = score - boost
            reranked_results.append((doc, adjusted_score))
        reranked_results.sort(key=lambda x: x[1])
        results = reranked_results[:k]
    else:
        results = results[:k]
    
    retrieved_docs = []
    for doc, score in results:
        retrieved_docs.append({
            "source": doc.metadata['source'],
            "content": doc.page_content,
            "score": float(score),
            "metadata": doc.metadata
        })
    return retrieved_docs

def answer_question_with_llm(query_text, retrieved_docs, max_tokens=1000):
    if not retrieved_docs:
        return "❌ No relevant documents found."
    
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc['source']
        content = doc['content']
        
        max_content_length = 2500
        if len(content) > max_content_length:
            content = content[:max_content_length] + "...[truncated]"
        
        context_parts.append(f"--- Document {i} ---\nSource: {source}\n\n{content}\n")
    
    context = "\n".join(context_parts)
    
    system_prompt = """You are a concise AI assistant. Answer the user's question *only* using the provided documents.
- Be brief and to the point.
- If the answer is not in the documents, state 'That information is not available in the documents.'"""
    
    user_prompt = f"""DOCUMENTS:
{context}
QUESTION: {query_text}
ANSWER:"""
    
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_rag_response(query_text, k=3):
    print(f"\n❓ Query: {query_text}")
    retrieved_docs = query_qdrant_store(query_text, k=k)
    
    if not retrieved_docs:
        return {
            "answer": "❌ No relevant documents found.",
            "sources": []
        }
    
    answer = answer_question_with_llm(query_text, retrieved_docs)
    sources_list = [{"source": doc['source'], "score": doc['score']} for doc in retrieved_docs]
    
    return {
        "answer": answer,
        "sources": sources_list
    }

def process_single_file(file_path, filename):
    file_ext = filename.lower().split('.')[-1]
    text = ""
    
    if file_ext in ["jpg", "jpeg", "png"]:
        text = extract_text_from_image(file_path)
    elif file_ext in ["txt", "md"]:
        text = extract_text_from_txt(file_path)
    elif file_ext == "pdf":
        text = extract_content_from_pdf(file_path)
    else:
        return None
    
    if text.strip():
        doc = Document(
            page_content=text,
            metadata={
                "source": filename,
                "filename": filename,
                "file_type": file_ext,
                "upload_timestamp": time.time()
            }
        )
        print(f"✅ Processed {filename}")
        return doc
    return None

def add_documents_to_qdrant(docs):
    if not docs:
        return
    
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        vector_store = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embedding_model
        )
        vector_store.add_documents(docs)
        print(f"✅ Added {len(docs)} documents to Qdrant")
    except Exception as e:
        print(f"❌ Error adding to Qdrant: {e}")
        raise

# -------------------------------
# Flask App - API ONLY
# -------------------------------
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "🧠 Multimodal RAG API",
        "endpoints": {
            "/query": "POST - Query documents",
            "/upload": "POST - Upload files",
            "/health": "GET - Health check"
        }
    })

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query = data.get('query', '')
    k = data.get('k', 3)
    
    try:
        response_data = get_rag_response(query, k)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    processed_files = []
    failed_files = []
    docs_to_add = []
    
    for file in files:
        if file.filename == '':
            continue
        
        try:
            filename = secure_filename(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            
            doc = process_single_file(tmp_path, filename)
            
            if doc:
                docs_to_add.append(doc)
                processed_files.append(filename)
            else:
                failed_files.append(filename)
            
            os.unlink(tmp_path)
        except Exception as e:
            print(f"❌ Error: {e}")
            failed_files.append(file.filename)
    
    if docs_to_add:
        try:
            add_documents_to_qdrant(docs_to_add)
        except Exception as e:
            return jsonify({"error": f"Failed to add to database: {str(e)}"}), 500
    
    return jsonify({
        "message": f"Processed {len(processed_files)} files",
        "processed_files": processed_files,
        "failed_files": failed_files
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "API is running"})

# -------------------------------
# Initialize and Run
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Multimodal RAG API...")
    
    # Build initial database if data folder exists
    folder = "data"
    if os.path.exists(folder):
        print(f"\n📂 Found '{folder}' folder, building database...")
        build_or_update_qdrant_store(folder)
    
    print("\n✅ Flask API starting on http://0.0.0.0:7860")
    print("   Endpoints:")
    print("   - GET  / (Home/Docs)")
    print("   - POST /query")
    print("   - POST /upload")
    print("   - GET  /health\n")
    
    app.run(host='0.0.0.0', port=7860, debug=False)
