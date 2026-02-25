import streamlit as st
import requests
import os

st.title("🚢 Titanic Chat Agent")

BACKEND_URL = os.getenv("BACKEND_URL")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if msg["type"] == "chart":

            st.image(msg["content"])

        else:

            st.markdown(msg["content"])


# Chat input
if prompt := st.chat_input("Ask about Titanic dataset..."):

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "type": "text"}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = requests.post(
                    f"{BACKEND_URL}/chat", json={"question": prompt}, timeout=60
                )

                response = response.json()

                # Chart response
                if response["type"] == "chart":

                    chart_url = response["chart_url"]

                    st.image(chart_url)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": chart_url, "type": "chart"}
                    )

                # Text response
                else:

                    answer = response["answer"]

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "type": "text"}
                    )

            except Exception as e:

                st.error(f"Backend error: {str(e)}")
