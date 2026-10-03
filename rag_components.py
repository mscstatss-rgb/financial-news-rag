# %%
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import List,Dict,Any
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List

class EmbeddingManager(Embeddings):
    def __init__(self,model_name: str="BAAI/bge-large-en-v1.5"):
        self.model_name=model_name
        self.model=None
        self._load_model()
        
    def _load_model(self):
        try:
            self.model=SentenceTransformer(self.model_name)
            print(f"Model loaded successfully.Embedding dim:{self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            raise
    
    def generate_embeddings(self,texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts ---")
        embeddings=self.model.encode(texts,show_progress_bar=True)
        print(f"Generated embeddings with shape:{embeddings.shape}")
        return embeddings
    # Required by LangChain
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.generate_embeddings(texts)
        return embeddings.tolist()

    # Required by LangChain
    def embed_query(self, text: str) -> List[float]:
        embedding = self.generate_embeddings([text])[0]
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model_get_sentence_embedding_dimension()

class VectorStore():
    def __init__(self,collection_name: str="pdf_documents",persist_directory: str="../data/vector_store"):
        self.collection_name=collection_name
        self.persist_directory=persist_directory
        self.clint=None
        self.collection=None
        self._initialize_store()
    
    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client=chromadb.PersistentClient(path=self.persist_directory)

            self.collection=self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=({'discription':"PDF document embedding for RAG"})
            )
            print(f"vector store initialized Collection:{self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            

        except Exception as e:
            print(f"Error")
            raise
    
    def add_documents(self,documents: List[Any],embeddings:np.array):
        if len(documents) !=len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding{len(documents)}")

        ids=[]
        metadatas=[]
        documents_text=[]
        embeddings_list=[]

        for i,(doc,embedding) in enumerate(zip(documents, embeddings)):
            doc_id=f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata=dict(doc.metadata)
            metadata['doc_index']=i
            metadata['content_length']=len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)}")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print("Error")
            raise
class RAGRetriever:

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:

        print(f"Retrieve Query: {query}")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")


        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
        

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:

                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

                print(f"Retrieved: {len(retrieved_docs)}")

            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Retrieval error: {e}")
            return []