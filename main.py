import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 문서 불러오기, 자르기
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 임베딩, 벡터 DB 저장
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 문서 불러오기
loader = PyPDFLoader("rag.pdf")
docs = loader.load()
# print(f"원본 문서 페이지 수: {len(docs)}")

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # 한 조각당 약 500자씩 자름
    chunk_overlap=50 # 앞 뒤 조각이 50자씩 겹치게 자름 (문맥 끊김 방지)
)
splits = text_splitter.split_documents(docs)

# 좀 더 정교하게 청크 자르는 방법 공부할 것

# print(f"쪼개진 텍스트 조각(Chunk) 개수: {len(splits)}")
# print(f"첫 번째 조각 내용 미리보기:\n{splits[0].page_content}")

# 임베딩 모델 (한국어 성능이 뛰어난 무료 오픈 소스 모델)
print("임베딩 모델 로딩 중...")
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 조각들을 임베딩하여 FAISS (meta에서 만든 초고속 벡터 검색 라이브러리) 벡터 DB에 저장
print("문서를 벡터 DB에 저장 중...")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
print("저장 완료!")

# 4. 검색(Retriever) 테스트
# 가장 질문과 유사한 텍스트 조각 3개(k=3)를 찾아오도록 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

query = "RAG의 정의"
search_results = retriever.invoke(query)

# 검색 결과 출력
print(f"\n--- '{query}'에 대한 검색 결과 ---")
for i, res in enumerate(search_results):
    print(f"\n[문서 조각 {i+1}]")
    print(res.page_content)

# # .env의 API키 불러오기
# load_dotenv() 

# # groq 전용 클래스로 모델 초기화
# llm = ChatGroq(model_name="groq/compound-mini")

# st.title('Ask Anything')

# # 주제 설정 (웹에서 입력)
# content = st.text_input('검색 주제 작성')

# # 버튼 눌렀을 경우 프롬포트 전달
# if st.button('프롬포트 전달'):
#     with st.spinner('Wait for it. . .'):

#         # API 호출 중 에러가 양이 방대해서 서버가 터지는 문제가 발생하여, 프롬포트를 '요약해줘' 로 변경
#         result = llm.invoke(content + "에 대해 요약해서 알려줘")
#         st.write(result.content)