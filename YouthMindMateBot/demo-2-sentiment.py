import streamlit as st
#from langchain_ollama import ChatOllama
#from langchain_ollama import OllamaLLM
from langchain.chat_models import ChatOllama
#from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# set up streamlit UI
st.set_page_config(layout="wide")


# side-bar for user settings
st.sidebar.header("Settings")
model_map = {"model-1": "llama3.2", "model-2": "deepseek-r1:1.5b"}
model_options = ["llama3.2", "deepseek-r1:1.5b"]
model_keyoptions = ["model-1", "model-2"]
MODEL = st.sidebar.selectbox("Choose a Model", model_keyoptions, index=0)
MAX_HISTORY = st.sidebar.number_input("Max History", min_value=1, max_value=10, value=2, step=1)
CONTEXT_SIZE = st.sidebar.number_input("Context Size", min_value=1024, max_value=16384, value=8192, step=1024)


#manage memory
def clear_memory():
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "prev_context_size" not in st.session_state or st.session_state.prev_context_size != CONTEXT_SIZE:
    clear_memory()
    st.session_state.prev_context_size = CONTEXT_SIZE

# initiate chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# set up langchain AI model
llm = ChatOllama(model=model_map[MODEL], streaming=True)

text = "I love programming in Python! It's such a versatile language."
sentiment_prompt = (
        "Analyze the sentiment of the following text. "
        "Respond with only one word: positive, negative, or neutral.\n"
        f"Text: {text}\nSentiment:"
    )
messages = [
    (
        "system",
        "You are an expert for sentiment analysis.",
    ),
    ("human", sentiment_prompt),
]
result = llm.invoke(messages)

#result = llm([HumanMessag(content=sentiment_prompt)])
if hasattr(result, "content"):
    print(result.content.strip().lower())
elif isinstance(result, str):
    print(result.strip().lower())
else:
    print("unknown")

# prompt_template = PromptTemplate(
#     input_variables=["history", "human_input"],
#     template="{history}\nUser: {human_input}\nAssistant:"
# )

# define the prompt structure
# chain = LLMChain(llm=llm, prompt=prompt_template, memory=st.session_state.memory)
# #display chat history
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])


# # handle user input
# def trim_memory():
#     while len(st.session_state.chat_history) > MAX_HISTORY * 2:
#         st.session_state.chat_history.pop(0)
#         if st.session_state.chat_history:
#             st.session_state.chat_history.pop(0)

# if prompt := st.chat_input("Say something"):
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.chat_history.append({"role": "user", "content": prompt})
#     trim_memory()
#     with st.chat_message("assistant"):
#         response_container = st.empty()
#         full_response = ""
#         for chunk in chain.stream({"human_input": prompt}):
#             if isinstance(chunk, dict) and "text" in chunk:
#                 text_chunk = chunk["text"]
#                 full_response += text_chunk
#                 response_container.markdown(full_response)
#     st.session_state.chat_history.append({"role": "assistant", "content": full_response})
#     trim_memory()


