import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# .env의 API키 불러오기
load_dotenv() 

# groq 전용 클래스로 모델 초기화
llm = ChatGroq(model_name="groq/compound-mini")

st.title('Ask Anything')

# 주제 설정 (웹에서 입력)
content = st.text_input('검색 주제 작성')

# 버튼 눌렀을 경우 프롬포트 전달
if st.button('프롬포트 전달'):
    with st.spinner('Wait for it. . .'):

        # API 호출 중 에러가 양이 방대해서 서버가 터지는 문제가 발생하여, 프롬포트를 '요약해줘' 로 변경
        result = llm.invoke(content + "에 대해 요약해서 알려줘")
        st.write(result.content)