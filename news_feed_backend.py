# ============================================================
# 1. IMPORTS
# ============================================================

from typing import List, Dict, Any
from rag_components import EmbeddingManager,RAGRetriever,VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


# ============================================================
# 2. CONFIGURATION
# ============================================================

CHROMA_DIR = "./chroma_db"


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

embedding_manager = EmbeddingManager(
    "BAAI/bge-large-en-v1.5"
)


# ============================================================
# 4. LOAD EXISTING VECTOR DATABASE
# ============================================================

vectorstore = VectorStore(
    collection_name="langchain",
    persist_directory=CHROMA_DIR
)


# ============================================================
# 5. CREATE RETRIEVER
# ============================================================

rag_retriever = RAGRetriever(
    vector_store=vectorstore,
    embedding_manager=embedding_manager
)


# %%
rag_retriever.retrieve("What is todays news?")

# %%
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print(api_key is not None)

# %%


# %%
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# %%
def extract_response_text(response):
    return "\n".join(
        block["text"]
        for block in response.content
        if block.get("type") == "text"
    )

# %%
def ask_rag(query):

    # 1. Retrieve relevant documents
    results = rag_retriever.retrieve(query,top_k=5)

    # 2. Combine retrieved documents
    context = "\n\n".join(result["content"] for result in results)
        # 3. Create prompt
    prompt = f"""
You are a financial news assistant.

Answer the question using ONLY the information
provided in the context.

If the answer is not present in the context,
say that you don't have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

    # 4. Ask Gemini
    response = llm.invoke(prompt)

    return extract_response_text(response)

# %%
answer = ask_rag("What is today's news?")
print(answer)

# %%


# %%



