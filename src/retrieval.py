from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone

from utils import dict_slice

import tiktoken

import pinecone
from tqdm.auto import tqdm # smart progress bar
from uuid import uuid4 # unique identifiers for indexing

class RetrievalAugmentationQA:
    def __init__(self, index_name, openai_key, pinecone_key, pinecone_env, data) -> None:
        self.index_name = index_name
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.pinecone_env = pinecone_env
        self.data = data

        # setup OpenAI embeddings
        print("Creating OpenAI Embeddings model...")
        self.embed = OpenAIEmbeddings(model='text-embedding-ada-002')
        self.res = self.embed.embed_documents(self.data['TEXT'])
        # setup Pinecone index
        print("Creating Pinecone index...")
        pinecone.init(api_key=self.pinecone_key, environment=self.pinecone_env)
        if self.index_name not in pinecone.list_indexes():
            # create new index
            pinecone.create_index(name=self.index_name, 
                                  metric='cosine', 
                                  dimension=len(self.res[0])
                                )

        self.index = pinecone.GRPCIndex(self.index_name)
    
    # index all text in data
    def add_to_index(self, batch_limit=100):
        texts = []
        metadatas = []
        text_splitter = self._split_text()

        for i, _ in enumerate(tqdm(self.data['LESSON'])): # using a column here to get length of dict, then we look at each row in the dictionary
            record = dict_slice(self.data, i)
            # acquire metadata from record
            metadata = {
                'PAGE': record['PAGE'],
                'LESSON': record['LESSON']
            }

            # create chunks from record text
            record_texts = text_splitter.split_text(record['TEXT'])
            # create medadata dict for each chunk
            record_metadatas=[{
                "chunk":j, "text": text, **metadata
            } for j, text in enumerate(record_texts)]
            # append to current batch
            texts.extend(record_texts)
            metadatas.extend(record_metadatas)
            # if batch limit reached, add text
            if len(texts) >= batch_limit:
                ids = [str(uuid4()) for _ in range(len(texts))]
                embeds = self.embed.embed_documents(texts)
                self.index.upsert(vectors=zip(ids, embeds, metadatas))
                # reset batch
                texts = []
                metadatas = []

        # add remaining texts
        if len(texts) > 0:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = self.embed.embed_documents(texts)
            self.index.upsert(vectors=zip(ids, embeds, metadatas))
        print("All text successfully added to index.")

    # create a vector store and query
    def query(self, prompt):
        # initialize vector store
        text_field = "text"
        # switch back to normal index for langchain
        index = pinecone.Index(self.index_name)
        vector_store = Pinecone(index,
                                self.embed.embed_query,
                                text_field)
        # setup completion llm
        llm = ChatOpenAI(
            openai_api_key=self.openai_key,
            model_name='gpt-3.5-turbo',
            temperature=0.0
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )

        return qa.run(prompt)
    
    # helper function to acquire tokenizer length
    def _tiktoken_length(self, text):
        tokenizer = tiktoken.get_encoding('cl100k_base')
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    # helper function to split text using langchain text splitter
    def _split_text(self, chunk_size=400, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._tiktoken_length,
            separators=['\n\n','\n',' ', '']
        )

        return text_splitter
    
