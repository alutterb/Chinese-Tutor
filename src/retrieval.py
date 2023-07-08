from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

import tiktoken
import openai 

import pinecone
from tqdm.auto import tqdm # smart progress bar
from uuid import uuid4 # unique identifiers for indexing

import os
from dotenv import load_dotenv
from utils import read_text_from_txt

# load environment variables
load_dotenv()
TEXTBOOK_TXT_PATH = os.getenv('TEXTBOOK_TXT_PATH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
# Hyperparameters
CHUNK_SIZE = 400 # # tokens per chunk
CHUNK_OVERLAP = 20 # # tokens overlap

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
def build_embeddings(model):
    embeddings = OpenAIEmbeddings(
        model=model,
        openai_api_key=OPENAI_API_KEY
    )
    return embeddings

# initialize pinecone vector db and set up indexing
def init_pinecone(embd_len, index_name = 'langchain-retrieval-augmentation'):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=embd_len
        )
    index = pinecone.GRPCIndex(index_name)
    return index

# index text
def index_text():
    BATCH_LIMIT = 1000
    texts = []
    metadatas = []