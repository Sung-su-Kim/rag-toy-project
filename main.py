from dotenv import load_dotenv
from langchain_groq import ChatGroq

# .env의 API키 불러오기
load_dotenv() 

# groq 전용 클래스로 모델 초기화
llm = ChatGroq(model_name="groq/compound-mini")

# 질문 전달
result = llm.invoke("넌 누구니")
print(result.content)