import os
import pdfplumber
import pickle
import json
from dotenv import load_dotenv
import openai
from prompts import SYSTEM_INTEL
import time

# load environment variables
load_dotenv()
# Hyperparameters
MAX_TOKENS = 400
MAX_RETRIES = 5
# Sections for textbook
with open(os.getenv('LESSONS_JSON_PATH')) as f:
    LESSONS = json.load(f)

# =======================  TEXT EXTRACTION ===================== #

# Paths to pdf input and text output
TEXTBOOK_PDF_PATH = os.getenv('TEXTBOOK_PDF_PATH')
TEXTBOOK_DF_PATH = os.getenv('TEXTBOOK_DF_PATH')

# load openai api key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Truncates raw text based on MAX_TOKENS size and returns chunks of text 
# where each chunk is less than MAX_TOKENS
def truncate_text(raw_text):
    text_len = len(raw_text)
    if text_len <= MAX_TOKENS:
        return [raw_text]
    chunks = [raw_text[i:i+MAX_TOKENS] for i in range(0, text_len, MAX_TOKENS)]
    return chunks


# Call openai api to correct spelling, grammar, punctuation, and add tone markers to
# to pinyin with missing tones.
def correct_text(text):
    system_intel = SYSTEM_INTEL
    max_attempts = MAX_RETRIES
    for attempt in range(1, max_attempts + 1):
        try:
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
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            if attempt < max_attempts:  # Don't delay if it's the last attempt
                print(f"An error occurred: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"An error occurred: {e}. No more retries.")
                raise


# Converts pdf to text
def extract_and_process_text_from_pdf(pdf_path):
    text_dict = {}
    lesson_list = []
    page_list = []
    text_list = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"Processing page {i}...")
            # record page number
            page_list.append(i)
            # identify which lesson this page belongs to
            for lesson in LESSONS:
                if i >= lesson['START'] and i <= lesson['END']:
                    lesson_list.append(lesson['LESSON'])
                    break
                
            # extract raw text from pdf
            raw_text = page.extract_text()
            # break up input into chunks up to MAX_TOKENS
            raw_chunks = truncate_text(raw_text)
            # loop through chunks and apply openai api to correct text
            all_text = ''
            for chunk in raw_chunks:
                corrected_text = correct_text(chunk)
                # acquire text from openai result
                corrected_text = corrected_text['choices'][0]['message']['content']
                all_text += corrected_text
            text_list.append(all_text)
    print("Text from pdf successfully extracted and corrected.")
    text_dict['PAGE'] = page_list
    text_dict['LESSON'] = lesson_list
    text_dict['TEXT'] = text_list
    return text_dict

# Do everything
def write_pdf_to_pickle():
    try:
        if not os.path.exists(TEXTBOOK_PDF_PATH):
            print(f"File not found: {TEXTBOOK_PDF_PATH}")
            return
        # extract text from pdf and process it
        print("Extracting and correcting text...")
        text_df = extract_and_process_text_from_pdf(TEXTBOOK_PDF_PATH)

        # save all text to a pickle file.
        print("Saving to pickle file...")
        with open(TEXTBOOK_DF_PATH, 'wb') as f:
            pickle.dump(text_df, f)

        print(f"Text successfully written to: {TEXTBOOK_DF_PATH}")

    except FileNotFoundError:
        print(f"File not found: {TEXTBOOK_PDF_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
# ========================================================================
write_pdf_to_pickle()

# ======================== Helper Functions ==============================
    
    

    
    
