from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

import tiktoken
import openai

import os
from dotenv import load_dotenv
from utils import read_text_from_txt

# load environment variables
load_dotenv()
TEXTBOOK_TXT_PATH = os.getenv('TEXTBOOK_TXT_PATH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Hyperparameters
CHUNK_SIZE = 400
CHUNK_OVERLAP = 20

# initialize tokenizer
tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding('cl100k_base')

# create length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# split text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = tiktoken_len,
        separators=["\n\n", "\n", " ", ""])
    
    return text_splitter.split_text(text)

# build text embeddings
def build_embeddings(text):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=OPENAI_API_KEY
    )