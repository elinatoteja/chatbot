from openai import OpenAI
import streamlit as st
from rag import QA_MODEL, streaming_question_answering, get_similar_context

FIRST_MESSAGE = {"role": "assistant", "content": "Hello there! How can I help you?"}
IMAGE_ADDRESS = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.com%2Ffree-photos-vectors%2Fspecial-needs-logo&psig=AOvVaw2bBSDvvDZ1Lq7c0aOhkpon&ust=1740098545736000&source=images&cd=vfe&opi=89978449&ved=0CBYQjRxqFwoTCNiU0bGC0YsDFQAAAAAdAAAAABAE"

# set the title
st.title("Child Supporter 👩‍🍼")
# set the image
st.image(IMAGE_ADDRESS, caption = 'Child Supporter')
st.subheader("Chat with Us 🤖")
# set openai key
client = OpenAI(api_key= st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = QA_MODEL

if "messages" not in st.session_state:
    st.session_state.messages = [FIRST_MESSAGE]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        pinecone_context = get_similar_context(prompt)
        response = st.write_stream(streaming_question_answering(prompt, pinecone_context))
    st.session_state.messages.append({"role": "assistant", "content": response})
