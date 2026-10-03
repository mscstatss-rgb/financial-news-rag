import streamlit as st
from news_feed_backend import ask_rag

st.title("📰 Financial News Assistant")

query = st.chat_input("Ask about financial news...")

if query:

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer = ask_rag(query)

        st.write(answer)