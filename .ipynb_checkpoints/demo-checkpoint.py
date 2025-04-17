import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# 初始化全局组件
@st.cache_resource
def init_components():
    # 初始化模型
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    answer_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    
    # 初始化Milvus客户端
    milvus_client = MilvusClient(uri="http://localhost:19530")
    
    return embedding_model, tokenizer, answer_model, milvus_client

# 文档处理管道
def process_document():
    loader = PyPDFLoader("WHITEPAPER_Future_of_Sustainability_2025.pdf")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    return [chunk.page_content for chunk in chunks]

# 构建向量数据库
def build_vector_db(text_lines, embedding_model, milvus_client):
    collection_name = "rag_collection"
    
    if collection_name in milvus_client.list_collections():
        milvus_client.drop_collection(collection_name)
    
    # 获取向量维度
    vector_dim = embedding_model.encode(["test"]).shape[1]
    
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=vector_dim,
        metric_type="IP",
        consistency_level="Strong"
    )
    
    # 批量插入数据
    batch_size = 100
    for i in range(0, len(text_lines), batch_size):
        batch = text_lines[i:i+batch_size]
        vectors = embedding_model.encode(batch)
        query_vector = embedding_model.encode([query])[0]

        
        data = [{
            "id": i+j,
            "vector": vectors[j].tolist(),
            "text": text
        } for j, text in enumerate(batch)]
        
        milvus_client.insert(collection_name, data)
    
    return len(text_lines)

# 检索上下文
def retrieve_context(query, milvus_client, embedding_model, top_k=5):
    query_vector = embedding_model.encode([f"query: {query}"])[0]
    search_res = milvus_client.search(
        collection_name="rag_collection",
        data=[query_vector.tolist()],
        limit=top_k,
        output_fields=["text"]
    )
    return "\n".join([hit["entity"]["text"] for hit in search_res[0]])

# 生成回答
PROMPT = """
You are an AI assistant specialized in sustainability. Your task is to answer the user's question using only the information in the <context> below.

Instructions:
1. Use only the facts that are explicitly stated in the context. Do not invent or assume anything.
2. If the question asks "what is", "why", or "how", and the context contains even partial explanations, answer as clearly and concisely as possible.
3. If the context includes examples or descriptions that help answer the question, summarize them clearly.
4. If the context is unrelated or clearly insufficient, say: "I cannot find relevant information in the provided context."
5. Limit your answer to 1–3 sentences. Avoid markdown or any formatting.
6. For definition questions (e.g., "What is..."), summarize the concept clearly in your own words using facts from the context.

<context>
{context}
</context>

<question>
{question}
</question>

<answer>

"""


def build_prompt(context: str, question: str) -> str:
    return PROMPT.format(context=context.strip(), question=question.strip())

def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=4
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit界面
def main():
    st.title("📚 Sustainability Whitepaper Q&A")
    st.markdown("You can ask questions about the Future of Sustainability 2025 whitepaper by using this web page")
    
    # 初始化组件
    with st.spinner("Initializing system..."):
        embedding_model, tokenizer, answer_model, milvus_client = init_components()
    
    # 文档处理
    if "processed" not in st.session_state:
        with st.status("Processing document..."):
            st.write("Loading PDF...")
            text_lines = process_document()
            
            st.write("Building vector database...")
            count = build_vector_db(text_lines, embedding_model, milvus_client)
            st.session_state.processed = True
            st.success(f"Processed {count} document chunks")
    
    # 问答界面
    question = st.text_input("Enter your question:", placeholder="What are the key sustainability trends?")
    
    if question:
        with st.spinner("Searching knowledge base..."):
            start_time = time.time()
            
            # 检索上下文
            context = retrieve_context(question, milvus_client, embedding_model)
            
            # 构建prompt
            prompt = PROMPT.format(context=context, question=question)
            
            # 生成回答
            answer = generate_answer(prompt, tokenizer, answer_model)
            
            latency = time.time() - start_time
        
        # 显示结果
        st.subheader("Answer")
        st.info(answer)
        
        with st.expander("Debug Info"):
            st.caption(f"Response latency: {latency:.2f}s")
            st.text_area("Retrieved Context", context, height=200)
            st.text_area("Full Prompt", prompt, height=200)

if __name__ == "__main__":
    main()