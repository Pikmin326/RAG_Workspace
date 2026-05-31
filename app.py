import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from utils.index_builder import get_or_build_notion_index

# 환경변수 로드 (.env)
load_dotenv()

# 웹 페이지 기본 설정
st.set_page_config(page_title="통합 RAG 어시스턴트", page_icon="📚")
st.title("📚 통합 RAG AI 어시스턴트")

with st.sidebar: # 동기화 버튼
    st.header("⚙️ 시스템 설정")
    if st.button("🔄 노션 데이터 최신화 (동기화)"):
        with st.spinner("기존 데이터를 지우고 노션에서 최신 데이터를 가져오는 중입니다..."):
            # 1. 기존에 저장된 벡터 DB 폴더 강제 삭제
            if os.path.exists("./storage"):
                shutil.rmtree("./storage")
            
            # 2. Streamlit이 기억하고 있던 메모리 캐시 비우기
            st.cache_resource.clear()
            
            # 3. 앱 강제 새로고침 (이때 최신 데이터를 다시 불러오게 됨)
            st.rerun()

# 인덱스를 한 번만 로드하도록 캐싱 (속도 향상)
@st.cache_resource
def load_system():
    # 기존 SimpleDirectoryReader 대신 최종 작성한 노션 빌더 함수 호출
    return get_or_build_notion_index()

index = load_system()
# 검색 상위 3개의 맥락을 참조하도록 쿼리 엔진 설정
query_engine = index.as_query_engine(similarity_top_k=3)

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 채팅 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력창
if prompt := st.chat_input("질문을 입력하세요 (예: 이번 정기 공연 예산안 요약해 줘)"):
    
    # 사용자 메시지 화면에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답 생성 및 화면에 추가
    with st.chat_message("assistant"):
        with st.spinner("문서 저장소를 검색하고 답변을 작성 중입니다..."):
            response = query_engine.query(prompt)
            st.markdown(response.response)
            
            # AI가 참고한 문서 출처(파일명) 표기
            if response.source_nodes:
                with st.expander("참고한 원본 문서 확인"):
                    for node in response.source_nodes:
                        file_name = node.metadata.get('file_name', '알 수 없는 파일')
                        st.write(f"- {file_name}")
                        
    # AI 응답을 세션에 저장
    st.session_state.messages.append({"role": "assistant", "content": response.response})