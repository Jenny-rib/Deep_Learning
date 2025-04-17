import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import os
import pickle

# === 严格 Prompt 模板 ===
PROMPT = """
### ROLE ###
You are a sustainability expert AI assistant that ONLY responds based on verified information.

### INSTRUCTION STEPS ###
1. STRICTLY ANALYZE the user's question type:
   - Factual question (what/where/when/who)
   - Explanatory question (why/how)
   - Significance question (importance/impact)

2. CONTEXT PROCESSING:
   Read the provided context EXACTLY as written. 
   Identify SPECIFIC sentences that DIRECTLY relate to the question.

3. RESPONSE GENERATION RULES:
   [REQUIRED] Answer MUST be grounded in explicit context statements
   [REQUIRED] For "why" questions: 
       - Only explain reasons EXPLICITLY stated in context
       - If no causation mentioned, use predefined response
   [REQUIRED] When information is:
       a) Fully available → Concise 1-3 sentence answer
       b) Partially related → "I cannot find..." response
       c) Missing → "I cannot find..." response
   [PROHIBITED] Never:
       - Assume unstated connections
       - Combine information from different sections
       - Use examples not in context

### FORMAT CONTROL ###
- No markdown of any kind
- Avoid transitional phrases ("However", "Additionally")
- Use bullet points ONLY when listing explicit items from context

### CONTEXT ###
{context}

### QUESTION ###
{question}

### SAFETY PROTOCOL ###
If uncertain about ANY part of the response, immediately fallback to: 
"I cannot find relevant information in the provided documents."

### RESPONSE ###
"""


def build_prompt(context, question):
    sanitized_context = " ".join(context.strip().splitlines())
    sanitized_question = question.strip().replace("\n", " ")
    max_context_length = 3000
    truncated_context = (sanitized_context[:max_context_length] + "...") if len(sanitized_context) > max_context_length else sanitized_context
    return PROMPT.format(context=truncated_context, question=sanitized_question)


CACHE_FILE = "cached_chunks.pkl"


@st.cache_resource
def init_components():
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    milvus_client = MilvusClient(uri="http://localhost:19530")
    return embedding_model, tokenizer, model, milvus_client


def load_cached_chunks():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return None

def save_chunks(text_lines):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(text_lines, f)


def process_document():
    loader = PyPDFLoader("WHITEPAPER_Future_of_Sustainability_2025.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return [chunk.page_content for chunk in chunks]


def build_vector_db(text_lines, embedding_model, milvus_client):
    collection_name = "rag_collection"
    if collection_name in milvus_client.list_collections():
        milvus_client.drop_collection(collection_name)

    vector_dim = embedding_model.encode(["test"])[0].shape[0]

    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=vector_dim,
        metric_type="IP",
        consistency_level="Strong"
    )

    batch_size = 100
    for i in range(0, len(text_lines), batch_size):
        batch = text_lines[i:i+batch_size]
        vectors = embedding_model.encode([f"passage: {text}" for text in batch])
        data = [{
            "id": i + j,
            "vector": vectors[j].tolist(),
            "text": text
        } for j, text in enumerate(batch)]
        milvus_client.insert(collection_name, data)

    return len(text_lines)


def retrieve_context(query, milvus_client, embedding_model, top_k=15):
    query_vector = embedding_model.encode([f"query: {query}"])[0]
    results = milvus_client.search(
        collection_name="rag_collection",
        data=[query_vector.tolist()],
        limit=top_k,
        output_fields=["text"]
    )
    return "\n".join([hit["entity"]["text"] for hit in results[0]])


def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit page setting
def main():
    st.title("📘 Sustainability Whitepaper Q&A")
    st.markdown("Ask questions about the Future of Sustainability 2025 whitepaper.")

    embedding_model, tokenizer, model, milvus_client = init_components()

    if "processed" not in st.session_state:
        cached = load_cached_chunks()
        if cached:
            st.session_state.text_lines = cached
            st.session_state.processed = True
            st.success(f"Loaded {len(cached)} cached chunks.")
        else:
            with st.status("Processing document..."):
                st.write("Loading and splitting PDF...")
                text_lines = process_document()

                st.write("Building vector database...")
                count = build_vector_db(text_lines, embedding_model, milvus_client)

                save_chunks(text_lines)
                st.session_state.text_lines = text_lines
                st.session_state.processed = True
                st.success(f"Processed and cached {count} chunks.")


    if st.button("🔁 Rebuild vector DB (for dev/debug)"):
        if "text_lines" in st.session_state:
            build_vector_db(st.session_state.text_lines, embedding_model, milvus_client)
            st.success("Rebuilt vector DB!")

    question = st.text_input("Enter your question:", placeholder="e.g. What is a circular economy?")

    if question:
        with st.spinner("Generating answer..."):
            start = time.time()
            context = retrieve_context(question, milvus_client, embedding_model)
            prompt = build_prompt(context, question)
            answer = generate_answer(prompt, tokenizer, model)
            latency = time.time() - start

        st.subheader("Answer")
        st.info(answer)

        with st.expander("Debug info"):
            st.caption(f"Response time: {latency:.2f}s")
            st.text_area("Retrieved Context", context, height=180)
            st.text_area("Full Prompt", prompt, height=180)

if __name__ == "__main__":
    main()
