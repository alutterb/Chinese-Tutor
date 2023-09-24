from retrieval import RetrievalAugmentationQA
from dotenv import load_dotenv
import os
import pickle

# load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
# load pickle file
with open('Data/Textbook.pkl', 'rb') as handle:
    lesson_text_dict = pickle.load(handle)


if __name__ == "__main__":
    ra = RetrievalAugmentationQA(index_name='chinese-tutor',
                                 openai_key=OPENAI_API_KEY, 
                                 pinecone_key=PINECONE_API_KEY,
                                 pinecone_env=PINECONE_ENVIRONMENT,
                                 data = lesson_text_dict)
    ra.add_to_index()
    
    while True:
        user_query = input("Enter a question (type q to quit): ") 
        if user_query.lower() == 'q':
            break
            
        response = ra.query(user_query)
        print(response)