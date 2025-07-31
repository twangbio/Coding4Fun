import json
import streamlit as st
#from langchain_ollama import ChatOllama
#from langchain_ollama import OllamaLLM
from langchain.chat_models import ChatOllama
#from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


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


jokes_file=r"C:\\Users\\twang\Work\Demo\\knowledgebase\\jokes.json"
music_file=r"C:\\Users\\twang\Work\Demo\\knowledgebase\\relax_music.json"

jokes = load_data(jokes_file)
music = load_data(music_file)

joke = jokes.pop(0) if jokes else None
print(joke['body'])
