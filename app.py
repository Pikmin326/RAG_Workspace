import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from utils.index_builder import get_or_build_notion_index

# 로컬 테스트용 환경변수 로드 (Streamlit Cloud에서는 무시됨)
load_dotenv()

st.set_page_config(page_title="RAG 어시스턴트", page_icon="📚", layout="centered")
st.title("📚 노션 연동 RAG 어시스턴트")

# --- 데이터 동기화 사이드바 ---
with st.sidebar:
    st.header("⚙️ 시스템 관리")
    st.info("💡 클라우드 서버가 재부팅되면 노션 최신 데이터를 자동으로 가져옵니다. 수동 갱신이 필요할 때만 아래 버튼을 누르세요.")
    
    if st.button("🔄 노션 데이터 수동 동기화"):
        with st.spinner("기존 데이터를 삭제하고 노션에서 최신 정보를 가져오는 중입니다..."):
            # 1. 휘발성 ChromaDB 폴더 강제 삭제
            db_path = "./chroma_db"
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            
            # 2. Streamlit 메모리 캐시 초기화
            st.cache_resource.clear()
            
            # 3. 앱 새로고침 (이때 코드가 DB가 없음을 감지하고 노션 API를 호출함)
            st.rerun()

# --- 시스템 로드 ---
@st.cache_resource
def load_system():
    return get_or_build_notion_index()

index = load_system()
if index:
    query_engine = index.as_query_engine(similarity_top_k=3)
else:
    st.error("인덱스를 불러오지 못했습니다. 노션 API 설정 및 권한을 확인해 주세요.")
    st.stop()

# --- 채팅 인터페이스 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요 (예: 이번 정기 공연 예산안 요약해 줘)"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("노션 지식 베이스를 검색 중입니다..."):
            response = query_engine.query(prompt)
            st.markdown(response.response)
            
    st.session_state.messages.append({"role": "assistant", "content": response.response})