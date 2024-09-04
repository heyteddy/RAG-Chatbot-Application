#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from decouple import config
from langchain.memory import ConversationBufferWindowMemory
import os

# Key 입력
os.environ["OPENAI_API_KEY"] = "sk-"

# PromptTemplate 객체를 생성하여 AI 모델에 전달할 프롬프트를 템플릿화
# chat_history와 question 변수를 입력으로 받아,
# 대화 이력을 반영한 질문에 대한 AI의 응답을 생성

prompt = PromptTemplate(
    input_variables = ["chat_history", "question"],
    template = """You are a AI assistant. You are currently having a conversation with a human.
    Answer the question.
    
    chat_history: {chat_history},
    human: {question}
    AI: """
)

# LLM 모델 생성
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo') # 창의성 0, 모델명 gpt-4

# 이전 내용 k(4)개 기억하기
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4)

# LLMChain 객체를 생성
llm_chain = LLMChain(llm=llm, memory=memory, prompt=prompt)

# 제목 지정
st.title("ChatGPT AI Assistant")

# 세션에서 메시지를 확인하고 존재하지 않는 경우 생성
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요. 저는 AI Assistant입니다."}]

# 모든 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자가 텍스트 입력 필드에 메시지를 입력하고, Enter 키를 눌러 메시지를 제출
user_prompt = st.chat_input()

# 질문했던 내용과 답변을 추가(append)
if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

# 마지막 메시지[-1]가 assistant로 받은게 아니면 새로운 답변 생성
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = llm_chain.predict(question=user_prompt) # 사용자가 질문한 것에 답변 해줘
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)


# In[ ]:




