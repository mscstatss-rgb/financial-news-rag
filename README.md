# Financial News RAG Assistant

A Retrieval-Augmented Generation (RAG) system for querying financial news using semantic search and an LLM.

## Overview

This project builds a RAG pipeline that collects financial news articles, processes and chunks the content, generates vector embeddings using BGE-large, stores them in ChromaDB, and retrieves relevant news context to generate answers using Gemini.

## Architecture

```text
News RSS Feed
      ↓
Article Extraction
      ↓
Document Chunking
      ↓
BGE-large Embeddings
      ↓
ChromaDB
      ↓
Semantic Retrieval
      ↓
Gemini LLM
      ↓
Context-grounded Answer
```

## Features

- Financial news ingestion from RSS feeds
- Article text extraction and preprocessing
- Recursive text chunking
- BGE-large-en-v1.5 embeddings
- ChromaDB vector storage
- Semantic similarity search
- Top-K document retrieval
- Context-grounded responses using Gemini
- Streamlit interface

## Tech Stack

- Python
- LangChain
- Sentence Transformers
- BAAI/bge-large-en-v1.5
- ChromaDB
- Google Gemini
- Streamlit
- RSS / Feedparser

## Project Structure

```text
financial-news-rag/
│
├── app.py
├── news_main_backend.py
├── load_database.py
├── rag_components.py
├── requirements.txt
├── .env
└── chroma_db/

```

### Installation

```text
git clone <your-repository-url>
cd financial-news-rag

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables

Create a .env file:
```text
GEMINI_API_KEY=your_api_key
```
Running the Project

### First build the vector database:
```text
python load_database.py
```
### Then run the application:
```text
streamlit run app.py
```
