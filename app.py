from openai import OpenAI
import streamlit as st
from rag import QA_MODEL, streaming_question_answering, get_similar_context

FIRST_MESSAGE = {"role": "assistant", "content": "Hello there! How can I help you?"}
IMAGE_ADDRESS = "https://img.freepik.com/free-vector/autism-ribbon-campaign_24877-82161.jpg?t=st=1740012450~exp=1740016050~hmac=7508370bc994bfae10416fe222c2c05e607ff5aa2aad44227e7204cd7ca833c7&w=1380"

# set the title
st.title("Supporter Chatbot")
# set the image
st.image(IMAGE_ADDRESS, caption = 'Child Supporter')
st.subheader("Chat with Us")
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
