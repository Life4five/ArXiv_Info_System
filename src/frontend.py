# frontend.py
import streamlit as st
import requests
from typing import Optional

st.set_page_config(page_title="Science RAG", layout="wide")
st.title("🧬 Science RAG Assistant")

# Настройки
BACKEND_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 600  # секунды

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Sources (Top-K)", 1, 10, 3)
    st.divider()
    st.caption("📡 Backend: " + BACKEND_URL)
    
    # Проверка доступности backend
    try:
        health_check = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if health_check.status_code == 200:
            st.success("✅ Backend Online")
        else:
            st.error("❌ Backend Error")
    except:
        st.error("❌ Backend Offline")

# Инициализация истории сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Обработка нового запроса
if prompt := st.chat_input("Ask about scientific papers..."):
    # Валидация ввода
    if len(prompt.strip()) == 0:
        st.warning("Please enter a question")
        st.stop()
    
    if len(prompt) > 1000:
        st.warning("Question is too long (max 1000 characters)")
        st.stop()
    
    # Добавление вопроса пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генерация ответа
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/ask", 
                    json={"text": prompt, "top_k": top_k},
                    timeout=REQUEST_TIMEOUT
                )
                
                if res.status_code == 200:
                    data = res.json()
                    
                    # 1. Вывод ответа
                    st.markdown(data["answer"])
                    
                    # 2. Вывод источников (Context Check)
                    if data.get("sources"):
                        with st.expander(f"Used Sources ({len(data['sources'])})"):
                            for i, source in enumerate(data["sources"]):
                                st.markdown(f"**{i+1}.**")
                                st.caption(source['text'])
                                st.divider()
                    
                    # 3. Метаданные
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"⏱️ {data['process_time']}s")
                    with col2:
                        st.caption(f"📄 {len(data.get('sources', []))} sources")
                    
                    # Сохранение в историю
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": data["answer"]
                    })
                
                elif res.status_code == 503:
                    st.error("⏳ System is still loading. Please wait...")
                else:
                    st.error(f"❌ Error {res.status_code}: {res.text}")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timeout. Try a simpler question.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to backend. Is it running?")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.caption("Please try again or contact support.")
