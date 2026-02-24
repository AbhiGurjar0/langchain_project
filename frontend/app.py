import streamlit as st
import requests

st.set_page_config(page_title="Titanic Chat Agent", page_icon="🚢")

st.title("🚢 Titanic Chat Agent")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input at bottom (ChatGPT style)
if prompt := st.chat_input("Ask about Titanic dataset..."):

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://localhost:8000/chat",
                json={"question": prompt}
            )

            answer = response.json()["answer"]["response"]

            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )