# ===========================================================================================
# 패키지&모듈 인지 class인지 비교하는 방법 (파이썬 개발자들은 아래처럼 약속된대로 이름을 짓는다)
# 1. 소문자 + 언더바 -> 패키지, 모듈, 메소드/변수
# ex) langchain_community(패키지), document_loaders(모듈), loader.load() (메소드)
# 
# 2. 단어 첫 글자마다 대문자 -> class
# ex) ChatOpenAI, RecursiveCharacterTextSplitter
# ===========================================================================================


import streamlit as st                # 파이썬으로 간편하게 UI제작 프레임워크
from dotenv import load_dotenv        # API키 보안 정보를 환경변수로 안전하게 불러옴
from langchain_groq import ChatGroq   # Groq API를 렝체인에서 쉽게 쓰도록 연결하는 클래스

# langchain_community -> 최상위 패키지
# document_loaders -> 모듈
# PyPDFLoader -> class
from langchain_community.document_loaders import PyPDFLoader         # pyPDFLoader: PDF문서에서 글자를 추출해 렝체인이 읽을 수 있도록 문서 객체로 듦
from langchain_text_splitters import RecursiveCharacterTextSplitter  # RecursiveCharacterTextSplitter: 긴 글을 문맥이 깨지지 않게 일정 크기(청크)로 자르는 텍스트 분할기
from langchain_huggingface import HuggingFaceEmbeddings              # HuggingFaceEmbeddings: 텍스트를 AI가 이해할 수 있는 벡터로 바꿔주는 임베딩 모델
from langchain_community.vectorstores import FAISS                   # FAISS: Meta가 만든 초고속 벡터 DB로, 유저 질문과 가장 유사한 문서 조각을 검색

from langchain_core.prompts import ChatPromptTemplate                        # ChatPromptTemplate: AI에게 전달할 지시사항과 질문 양식(프롬프트)을 만드는 틀
from langchain_core.runnables import RunnableParallel, RunnablePassthrough   # RunnableParallel / RunnablePassthrough: LCEL 체인 안에서 데이터를 동시에 처리하거나(Parallel) 입력값을 그대로 통과(Passthrough)시켜 주는 파이프라인 제어 도구
from langchain_core.output_parsers import StrOutputParser                    # StrOutputParser: LLM의 복잡한 원본 결과물에서 순수한 글자(String) 답변만 뽑아내는 정제기

load_dotenv()

# 데코레이터: 새로고침할 때마다 PDF를 다시 읽는 것은 바효율적이므로, 함수의 결과물을 메모리에 한 번만 저장해두고, 다음번에는 함수를 다시 실행하지 말고 저장된 결과물을 그대로 재사용하라는 의미
@st.cache_resource                
def init_vector_db():
    
    # PyPDFLoader는 클래스이다. "rag.pdf"라는 재료를 전달하여 실제로 작동할 수 있는 'rag.pdf' 전용 로더기계 (객체)를 실물로 만든다
    loader = PyPDFLoader("rag.pdf") 
    # .load는 loader의 객체, 즉 PyPDFLoader 클래스의 메서드이다 (PDF의 텍스트를 읽어와 파이썬이 다룰 수 있는 리스트 형태로 만든다)
    docs = loader.load()

    # <문서를 청크 단위로 자름>
    # RecursiveCharacterTextSplitter클래스를 통해 text_splitter객체를 만들고, 인자값으로 chunk_size=500, chunk_overlap=50을 전달
    # overlap은 앞 청크와 뒤 청크 사이에 겹치는 부분을 만들어 문맥이 잘리는 것을 방지한다
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # text_splitter 객체를 사용하여 실제로 문서(docs)를 문맥에 맞게 자르는 동작(.split_documents(docs))을 실행
    # splits는 문맥에 맞게 잘린 텍스트가 들어간 리스트이다 (출처정보(metadata)도 포함)
    splits = text_splitter.split_documents(docs)

    # <청크를 벡터 단위로 임베딩>
    # HuggingFaceEmbeddings클래스로 embeddings객체 생성, "jhgan/ko-sroberta-multitask"는 Hugging Face Hub라는 AI모델 공유 플렛폼에 등록된 모델이다. 
    # 첫번째 실행시, HuggingFaceEmbeddings 클래스가 허깅페이스 서버로 접속하여 해당 모델파일을 자동으로 다운로드하여 ~/.cache 폴더에 캐시로 저장하기 때문에 로컬에서 저장된 캐시를 바로 불러온다
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    # FAISS: Class, .from_documents(): 클래스 메소드 
    # 데이터를 받아와 알아서 임베딩하고 DB객체까지 한 번에 만들 수 있다
    # splits를 가져와 embeddings로 임베딩한 결과를 vectorstore에 저장
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore

#  문서 조각(List[Document])을 단일 텍스트로 합성해 주는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

st.title('🔍 RAG는 무엇일까?')

vectorstore = init_vector_db()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model_name="groq/compound-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """당신은 문서 기반 질의응답을 수행하는 친절한 AI 어시스턴트이다.
아래 제공된 '검색된 문서 내용'만을 바탕으로 질문에 답변할 것.
만약 검색된 내용에 답변이 없다면, "제공된 문서에서는 해당 내용을 찾을 수 없습니다."라고 솔직하게 말할 것.
절대 지어서 답변하지 말 것.

'검색된 문서 내용'
{context}

'질문'
{input}

답변:"""
)

# 답변 생성을 위한 서브 체인
chain_answer = (
    RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | prompt
    | llm
    | StrOutputParser()
)

# 원문 문서(context)와 답변(answer)을 동시에 딕셔너리로 리턴하는 LCEL 체인 조립
rag_chain = RunnableParallel({
    "context": (lambda x: x["input"]) | retriever,
    "input": lambda x: x["input"]
}).assign(answer=chain_answer)

content = st.text_input('나무위키의 검색증강생성 문서를 바탕으로 답변합니다.')

if st.button('질문하기'):
    if content:
        with st.spinner('문서를 검색하고 답변을 생성하는 중입니다...'):
            try:
                response = rag_chain.invoke({"input": content})
                
                # response["answer"]와 response["context"]를 안전하게 추출
                st.write(response["answer"])
                
                with st.expander("📚 참고한 문서 원문 보기"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**[조각 {i+1}]**\n{doc.page_content}\n---")
                        
            except Exception as e:
                st.error(f"⚠️ 오류가 발생했습니다: {e}")
    else:
        st.warning('질문을 입력해주세요!')