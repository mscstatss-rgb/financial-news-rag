import feedparser, requests, trafilatura
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_components import EmbeddingManager, VectorStore
from langchain_chroma import Chroma
import os
import shutil
# %%
rss_feeds_cnbc=[{'Main_title':"Finance",'rss':"https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"},{'Main_title':"Economy",'rss':"https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"},{'Main_title':"Business",'rss':"https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147"}]

# %%
documents = []

for feed_info in rss_feeds_cnbc:

    main_title = feed_info["Main_title"]
    rss_url = feed_info["rss"]

    feed = feedparser.parse(rss_url)
    print(f"Processing: {feed.feed.title}")

    for article in feed.entries:

        url = article.link

        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )

        text = trafilatura.extract(response.text)

        if text:
            doc = Document(
                page_content=text,
                metadata={
                    "Main_title": main_title,
                    "title": article.title,
                    "link": url
                }
            )

            documents.append(doc)

# %%

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n","\n"," ",""]
    )
    split_docs=text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata:{split_docs[0].metadata}")

    return split_docs

# %%
chunks=split_documents(documents,chunk_size=400)

# %%


embeddingmanager=EmbeddingManager()
# %%


CHROMA_DIR = "./chroma_db"

if os.path.exists(CHROMA_DIR):
    print("Deleting existing Chroma database...")
    shutil.rmtree(CHROMA_DIR)

# Create new database
print("Creating new Chroma database...")

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddingmanager
)

vectorstore.add_documents(chunks)

print("New Chroma database created.")
print("Documents:", vectorstore._collection.count())