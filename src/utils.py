import os
import pdfplumber
import pickle
import json
from dotenv import load_dotenv
import openai
from prompts import SYSTEM_INTEL
import time

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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
                model="gpt-4",
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

def process_page(page, lesson_range):
    i = page.page_number - 1
    print(f"Processing page {i}...")

    # Determine lesson
    for lesson in lesson_range:
        if i >= lesson['START'] and i <= lesson['END']:
            lesson_name = lesson['LESSON']
            break
    else:
        lesson_name = None

    # Extract and process text
    raw_text = page.extract_text()
    raw_chunks = truncate_text(raw_text)
    all_text = ''
    for chunk in raw_chunks:
        corrected_text = correct_text(chunk)
        corrected_text = corrected_text['choices'][0]['message']['content']
        all_text += corrected_text

    return i, lesson_name, all_text

def extract_and_process_text_from_pdf(pdf_path):
    text_dict = {'PAGE': [], 'LESSON': [], 'TEXT': []}

    with pdfplumber.open(pdf_path) as pdf:
        with ThreadPoolExecutor() as executor:
            # Map the process_page function to all pages
            futures = [executor.submit(process_page, page, LESSONS) for page in pdf.pages]

            for future in concurrent.futures.as_completed(futures):
                page_num, lesson_name, text = future.result()
                text_dict['PAGE'].append(page_num)
                text_dict['LESSON'].append(lesson_name)
                text_dict['TEXT'].append(text)

    print("Text from pdf successfully extracted and corrected.")
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
#write_pdf_to_pickle()   


# ============================================ MISC ======================
# returns the ith row of a dictionary
def dict_slice(dict, i):
    return {k: dict[k][i] for k in dict.keys()}

    

    
    

    
    
