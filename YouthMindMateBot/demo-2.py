import json 
import random
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

st.subheader("Youth Personal Assistant for stress reliever", divider="red", anchor=False)

#manage memory
def clear_memory():
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# handle user input
def trim_memory():
    while len(st.session_state.chat_history) > MAX_HISTORY * 2:
        st.session_state.chat_history.pop(0)
        if st.session_state.chat_history:
            st.session_state.chat_history.pop(0)

import json

def load_data(filepath):
    """
    Load a list from a JSON file.
    Returns the list if successful, otherwise returns an empty list.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []


def sentiment_analysis(llm, text):
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
    
    if hasattr(result, "content"):
        return result.content.strip().lower()
    elif isinstance(result, str):
        return result.strip().lower()
    else:
        return "unknown"


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

prompt_template = PromptTemplate(
    input_variables=["history", "human_input"],
    template="{history}\nUser: {human_input}\nAssistant:"
)

# define the prompt structure
chain = LLMChain(llm=llm, prompt=prompt_template, memory=st.session_state.memory)
#display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

idx = 0
jokes_file=r"C:\\Users\\twang\Work\Demo\\knowledgebase\\jokes.json"
music_file=r"C:\\Users\\twang\Work\Demo\\knowledgebase\\relax_music.json"
meditation_file=r"C:\\Users\\twang\Work\Demo\\knowledgebase\\meditation.json"

jokes = load_data(jokes_file)
musics = load_data(music_file)
meditations = load_data(meditation_file)

negative_responses = [
    "I'm sorry to hear that. Would you like to hear a joke or some relaxing music?",
    "That sounds tough. How about a joke or some relaxing music to lighten the mood?",
    "I understand. Would you like to hear a joke or listen to some relaxing music?",
]

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    if idx == 0:
        idx += 1
        sentiment = sentiment_analysis(llm=llm, text=prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": f"Sentiment: {sentiment}"})
        with st.chat_message("assistant"):
            st.markdown(f"Sentiment: {sentiment}")
        print(f"Sentiment: {sentiment}")

    trim_memory()

    if sentiment in ["negative"]:
        selector = random.randint(0, 10)
        if selector >=0 and selector < 1 and jokes:
            joke = jokes.pop(0)          
            with st.chat_message("assistant"):
                st.markdown(f"Here's a joke for you: {joke['body']}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Here's a joke for you: {joke['body']}"})
        elif selector >= 1 and selector < 2 and musics:
            song = musics.pop(0)        
            with st.chat_message("assistant"):
                st.markdown(f"How about some relaxing music ? {song['title']} with link: {song['link']}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"How about some relaxing music ? {song['title']} with link: {song['link']}"})
        elif selector >=2 and selector <= 4  and meditations:
            meditation = meditations.pop(0)
            st.session_state.chat_history.append({"role": "assistant", "content": f"How about a short meditation? {meditation['instruction']}"})
            with st.chat_message("assistant"):
                st.markdown(f"How about a short meditation? {meditation['instruction']}")
        else:
            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""
                for chunk in chain.stream({"human_input": prompt}):
                    if isinstance(chunk, dict) and "text" in chunk:
                        text_chunk = chunk["text"]
                        full_response += text_chunk
                        response_container.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        trim_memory()
        
    else:
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            for chunk in chain.stream({"human_input": prompt}):
                if isinstance(chunk, dict) and "text" in chunk:
                    text_chunk = chunk["text"]
                    full_response += text_chunk
                    response_container.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    trim_memory()

    idx = 0


