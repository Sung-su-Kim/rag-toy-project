import streamlit as st                # 파이썬으로 간편하게 UI제작 프레임워크
from dotenv import load_dotenv        # API키 보안 정보를 환경변수로 안전하게 불러옴
from langchain_groq import ChatGroq   # Groq API를 렝체인에서 쉽게 쓰도록 연결하는 클래스

from langchain_community.document_loaders import PyPDFLoader         # pyPDFLoader: PDF문서에서 글자를 추출해 렝체인이 읽을 수 있도록 문서 객체로 듦
from langchain_text_splitters import RecursiveCharacterTextSplitter  # RecursiveCharacterTextSplitter: 긴 글을 문맥이 깨지지 않게 일정 크기(청크)로 자르는 텍스트 분할기
from langchain_huggingface import HuggingFaceEmbeddings              # HuggingFaceEmbeddings: 텍스트를 AI가 이해할 수 있는 벡터로 바꿔주는 임베딩 모델
from langchain_community.vectorstores import FAISS                   # FAISS: Meta가 만든 초고속 벡터 DB로, 유저 질문과 가장 유사한 문서 조각을 검색

from langchain_core.prompts import ChatPromptTemplate                        # ChatPromptTemplate: AI에게 전달할 지시사항과 질문 양식(프롬프트)을 만드는 틀
from langchain_core.runnables import RunnableParallel, RunnablePassthrough   # RunnableParallel / RunnablePassthrough: LCEL 체인 안에서 데이터를 동시에 처리하거나(Parallel) 입력값을 그대로 통과(Passthrough)시켜 주는 파이프라인 제어 도구
from langchain_core.output_parsers import StrOutputParser                    # StrOutputParser: LLM의 복잡한 원본 결과물에서 순수한 글자(String) 답변만 뽑아내는 정제기

load_dotenv()

@st.cache_resource
def init_vector_db():
    loader = PyPDFLoader("rag.pdf")
    docs = loader.load()

    # 문서를 청크 단위로 자름
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 청크를 벡터 단위로 임베딩
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
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

# 2. 답변 생성을 위한 서브 체인
chain_answer = (
    RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | prompt
    | llm
    | StrOutputParser()
)

# 3. 원문 문서(context)와 답변(answer)을 동시에 딕셔너리로 리턴하는 LCEL 체인 조립
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