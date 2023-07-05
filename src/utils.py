import os
import pdfplumber
from dotenv import load_dotenv
import openai

# load environment variables
load_dotenv()
# Hyperparameters
MAX_TOKENS = 2000

# =======================  TEXT EXTRACTION ===================== #
'''
Converts the textbook pdf into images, and then uses an ocr scanner
to extract the text
'''

# Paths to pdf input and text output
TEXTBOOK_PDF_PATH = os.getenv('TEXTBOOK_PDF_PATH')
TEXTBOOK_TXT_PATH = os.getenv('TEXTBOOK_TXT_PATH')

# load openai api key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Truncates raw text based on MAX_TOKENS size and returns chunks of text 
# where each chunk is less than MAX_TOKENS
def truncate_text(raw_text):
    text_len = len(raw_text)
    remaining = text_len - MAX_TOKENS
    if remaining <= 0:
        return [raw_text]
    pointer = 0
    chunks = []
    while remaining > 0:
        lower = pointer
        upper = min(text_len, pointer + MAX_TOKENS)
        chunk = raw_text[lower:upper]
        chunks.append(chunk)
        new_text_len = len(raw_text)
        remaining = text_len - new_text_len
        pointer += MAX_TOKENS
    return chunks

# Call openai api to correct spelling, grammar, punctuation, and add tone markers to
# to pinyin 
def correct_text(text):
    system_intel = '''
    You are a chinese and linguistic expert. Fix the following broken text into a cohesive sentence. 
    Be sure to also add tones to pinyin with missing tones:\n'''
    corrected_text = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role" : "system", "content" : system_intel},
            {"role" : "user", "content" : text}
        ],
        temperature=0,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    return corrected_text

# Converts pdf to text
def extract_and_process_text_from_pdf(pdf_path):
    all_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # extract raw text from pdf
            raw_text = page.extract_text()
            # break up input into chunks up to MAX_TOKENS
            raw_chunks = truncate_text(raw_text)
            # loop through chunks and apply openai api to correct text
            for chunk in raw_chunks:
                corrected_text = correct_text(chunk)
                # acquire text from openai result
                corrected_text = corrected_text['choices'][0]['message']['content']
                all_text += corrected_text
    return all_text

# Do everything
def write_pdf_to_text():
    try:
        if not os.path.exists(TEXTBOOK_PDF_PATH):
            print(f"File not found: {TEXTBOOK_PDF_PATH}")
            return
        # extract text from pdf and process it
        print("Extracting and correcting text...")
        all_text = extract_and_process_text_from_pdf(TEXTBOOK_PDF_PATH)

        # save all text to a txt file.
        with open(TEXTBOOK_TXT_PATH, 'w', encoding='utf-8') as f:
            f.write(all_text)

        print(f"Text successfully written to: {TEXTBOOK_TXT_PATH}")

    except FileNotFoundError:
        print(f"File not found: {TEXTBOOK_PDF_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
# ========================================================================

    
    

    
    
