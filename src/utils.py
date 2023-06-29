import os
import pytesseract
from PIL import Image
import pdf2image
from dotenv import load_dotenv
import pygtrie
import re

load_dotenv()

TEXTBOOK_PDF_PATH = os.getenv('TEXTBOOK_PDF_PATH')
TEXTBOOK_TXT_PATH = os.getenv('TEXTBOOK_TXT_PATH')
PINYIN_PATH = os.getenv('PINYIN_PATH')

def is_chinese_char(char):
    """Check if char is Chinese character."""
    return '\u4e00' <= char <= '\u9fff'

def is_pinyin_word(word):
    """Check if word is pinyin word by checking if it contains only latin characters and tones."""
    return bool(re.match("^[a-zāēīōūǖáéíóúǘǎěǐǒǔǚàèìòùǜ]*$", word))

def load_pinyin_trie():
    trie = pygtrie.CharTrie()
    with open(PINYIN_PATH) as file:
        for word in file:
            word = word.strip()
            trie[word] = word
    return trie

def pinyin_match_trie(pinyin, correct_pinyins_trie):
    # checks word in trie with longest matching prefix with the supplied pinyin word
    match = correct_pinyins_trie.longest_prefix(pinyin)
    if match:
        return match.value
    else:
        return pinyin

def process_text(text, correct_pinyins):
    words = text.split()
    processed_words = []
    for word in words:
        if all(is_chinese_char(char) for char in word):
            processed_words.append(word) # ignore chinese characters for now
        elif is_pinyin_word(word): 
            corrected_pinyin = pinyin_match_trie(word, correct_pinyins)
            processed_words.append(corrected_pinyin)
        else:
            processed_words.append(word)
    return ' '.join(processed_words)


def extract_and_process_text_from_pdf():
    correct_pinyins = load_pinyin_trie()
    try:
        if not os.path.exists(TEXTBOOK_PDF_PATH):
            print(f"File not found: {TEXTBOOK_PDF_PATH}")
            return
        
        print("Converting images..")
        images = pdf2image.convert_from_path(TEXTBOOK_PDF_PATH)
        all_text = ''
        print("processing text...")
        for image in images:
            text = pytesseract.image_to_string(image)
            #processed_text = process_text(text, correct_pinyins)
            all_text += text + '\n'
        
        # save all text to a txt file.
        with open(TEXTBOOK_TXT_PATH, 'w', encoding='utf-8') as f:
            f.write(all_text)

        print(f"Text successfully written to: {TEXTBOOK_TXT_PATH}")

    except FileNotFoundError:
        print(f"File not found: {TEXTBOOK_PDF_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

extract_and_process_text_from_pdf()
