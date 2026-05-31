import os
import requests
import chromadb
import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.notion import NotionPageReader
from llama_index.core.node_parser import SentenceSplitter

def get_api_key(key_name):
    """로컬(.env)과 클라우드(st.secrets) 환경을 모두 지원하여 API 키를 가져옵니다."""
    if key_name in st.secrets:
        return st.secrets[key_name]
    return os.environ.get(key_name)

def initialize_settings():
    gemini_key = get_api_key("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")
    
    Settings.llm = Gemini(model="models/gemini-1.5-pro", api_key=gemini_key)
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=gemini_key)
    Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

def get_all_accessible_notion_ids(notion_token):
    url = "https://api.notion.com/v1/search"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    page_ids = []
    has_more = True
    next_cursor = None
    
    while has_more:
        payload = {"filter": {"value": "page", "property": "object"}}
        if next_cursor: 
            payload["start_cursor"] = next_cursor
            
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        for item in data.get("results", []): 
            page_ids.append(item["id"])
            
        has_more = data.get("has_more", False)
        next_cursor = data.get("next_cursor")
        
    return page_ids

def get_or_build_notion_index(db_path="./chroma_db", collection_name="club_docs"):
    initialize_settings()
    
    # ChromaDB 클라이언트 및 컬렉션 로드
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 1. DB에 데이터가 남아있다면 빠르게 로드 (캐싱 히트)
    if chroma_collection.count() > 0:
        print("기존 ChromaDB에서 벡터 인덱스를 로드합니다.")
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, 
            storage_context=storage_context
        )
    
    # 2. DB가 비어있거나 날아갔다면 노션에서 자동 복구 (캐싱 미스)
    print("ChromaDB가 비어있습니다. 노션에서 최신 데이터를 가져와 빌드합니다...")
    notion_token = get_api_key("NOTION_TOKEN")
    if not notion_token:
        raise ValueError("NOTION_TOKEN이 설정되지 않았습니다.")
        
    page_ids = get_all_accessible_notion_ids(notion_token)
    
    if not page_ids:
        print("접근 가능한 노션 페이지가 없습니다. 권한을 확인하세요.")
        return None
        
    reader = NotionPageReader(integration_token=notion_token)
    documents = reader.load_data(page_ids=page_ids)
    
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context
    )
    print("ChromaDB에 노션 데이터 인덱싱이 완료되었습니다.")
    
    return index