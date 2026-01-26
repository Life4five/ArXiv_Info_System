# frontend.py
import streamlit as st
import requests

st.set_page_config(page_title="Science RAG", layout="wide")
st.title("🧬 Science RAG Assistant")

with st.sidebar:
    top_k = st.slider("Sources (Top-K)", 1, 10, 3)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about scientific papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    "http://localhost:8000/ask", 
                    json={"text": prompt, "top_k": top_k}
                )
                if res.status_code == 200:
                    data = res.json()
                    st.markdown(data["answer"])
                    st.caption(f"⏱️ {data['process_time']}s")
                    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection failed. Is backend running? {e}")