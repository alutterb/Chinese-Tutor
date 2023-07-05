import os
import pdfplumber
from dotenv import load_dotenv
import pygtrie
import re

load_dotenv()

# ======================= ENGLISH DICTIONARY AND PINYIN DATASET ============
PINYIN_PATH = os.getenv('PINYIN_PATH')

# Load pinyin dataset
def load_pinyin_set():
    pinyin = set()
    with open(PINYIN_PATH) as file:
        for word in file:
            word = word.strip()
            pinyin.add(word)
    return pinyin

pinyin_set = load_pinyin_set()

# =======================  TEXT EXTRACTION ===================== #
'''
Converts the textbook pdf into images, and then uses an ocr scanner
to extract the text
'''

TEXTBOOK_PDF_PATH = os.getenv('TEXTBOOK_PDF_PATH')
TEXTBOOK_TXT_PATH = os.getenv('TEXTBOOK_TXT_PATH')

# Check if a string contains pinyin characters
def contains_pinyin(word):
    return bool(re.match(r'[a-zA-Z]*[āēīōūǖĀĒĪŌŪǕáéíóúǘÁÉÍÓÚǗǎěǐǒǔǚǍĚǏǑǓǙàèìòùǜÀÈÌÒÙǛ][a-zA-Z]*', word))

# Create trie structure based on pinyin data
def load_pinyin_trie():
    trie = pygtrie.CharTrie()
    for pinyin in pinyin_set:
        trie[pinyin] = pinyin
    return trie

# Checks word in trie with longest matching prefix with the supplied pinyin word
def pinyin_match_trie(pinyin, correct_pinyins_trie):
    match = correct_pinyins_trie.longest_prefix(pinyin)
    if match:
        return match.value
    else:
        return pinyin


# Proccesses pinyin in each page, attempts to fix missing tones from extracted text
def process_pinyin(input_text, correct_pinyins_trie):
    words = input_text.split()
    processed_words = []
    for word in words:
        if contains_pinyin(word): 
            word = pinyin_match_trie(word, correct_pinyins_trie)
        processed_words.append(word)
    return ' '.join(processed_words)

# Converts pdf to text
def extract_text_from_pdf(pdf_path):
    all_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            all_text += text
    return all_text

# Do everything
def extract_and_process_text_from_pdf():
    correct_pinyins = load_pinyin_trie()
    try:
        if not os.path.exists(TEXTBOOK_PDF_PATH):
            print(f"File not found: {TEXTBOOK_PDF_PATH}")
            return
        raw_text = extract_text_from_pdf(TEXTBOOK_PDF_PATH)
        # clean pinyin from raw text
        all_text = process_pinyin(raw_text, correct_pinyins)
        # save all text to a txt file.
        with open(TEXTBOOK_TXT_PATH, 'w', encoding='utf-8') as f:
            f.write(all_text)

        print(f"Text successfully written to: {TEXTBOOK_TXT_PATH}")

    except FileNotFoundError:
        print(f"File not found: {TEXTBOOK_PDF_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

extract_and_process_text_from_pdf()
# ========================================================================

    
    

    
    
