import os
import requests
import chromadb
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.notion import NotionPageReader
from llama_index.core.node_parser import SentenceSplitter

def initialize_settings():
    """Gemini LLM 및 임베딩 모델 설정을 초기화합니다."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
    
    # 텍스트 생성용 모델 설정
    Settings.llm = Gemini(
        model="models/gemini-2.5-flash", 
        api_key=api_key
    )
    # 텍스트 임베딩용 모델 설정
    Settings.embed_model = GeminiEmbedding(
        model_name="gemini-embedding-001", 
        api_key=api_key
    )
    # 청크 조절
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
    
    print("노션 서버에서 하위 페이지들을 탐색 중입니다...")
    
    # 페이지가 아주 많을 경우를 대비해 반복문(Pagination)으로 모두 긁어옵니다.
    while has_more:
        # 데이터베이스(표)가 아닌 '일반 페이지'만 필터링해서 가져옴
        payload = {"filter": {"value": "page", "property": "object"}}
        if next_cursor:
            payload["start_cursor"] = next_cursor
            
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        
        # 탐색된 페이지 ID를 리스트에 추가
        for item in data.get("results", []):
            page_ids.append(item["id"])
            
        has_more = data.get("has_more", False)
        next_cursor = data.get("next_cursor")
        
    print(f"총 {len(page_ids)}개의 노션 페이지 ID를 자동으로 찾았습니다!")
    return page_ids

def get_or_build_notion_index(db_path="./chroma_db", collection_name="club_notion_docs"):
    initialize_settings()
    
    # 1. ChromaDB 클라이언트 및 컬렉션 생성/로드
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    
    # 2. LlamaIndex에 ChromaDB 연결
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 3. 데이터가 이미 존재하는지 확인 (컬렉션에 저장된 문서 개수로 판단)
    if chroma_collection.count() > 0:
        print("기존 ChromaDB에서 벡터 인덱스를 로드합니다...")
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, 
            storage_context=storage_context
        )
    
    # 4. 데이터가 없으면 노션에서 새로 가져와서 임베딩
    print("ChromaDB가 비어있습니다. Notion API로 데이터를 수집하여 임베딩합니다...")
    notion_token = os.environ.get("NOTION_TOKEN")
    page_ids = get_all_accessible_notion_ids(notion_token)
    
    reader = NotionPageReader(integration_token=notion_token)
    documents = reader.load_data(page_ids=page_ids)
    
    # ChromaDB를 저장소로 지정하여 인덱스 생성
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context
    )
    print("ChromaDB에 새로운 노션 데이터 인덱싱이 완료되었습니다.")
    
    return index