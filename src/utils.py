import os
import jieba
from io import StringIO
from dotenv import load_dotenv
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
import textdistance

load_dotenv()

TEXTBOOK_PDF_PATH = os.getenv('TEXTBOOK_PDF_PATH')
TEXTBOOK_TXT_PATH = os.getenv('TEXTBOOK_TXT_PATH')
PINYIN_PATH = os.getenv('PINYIN_PATH')

def load_pinyin():
    with open(PINYIN_PATH) as file:
        words = file.readlines()
    words = [word.strip() for word in words]
    return words

def pinyin_match(pinyin, correct_pinyins):
    lev = textdistance.Levenshtein()
    closest_pinyin = min(correct_pinyins, key=lambda correct_pinyin: lev.distance(pinyin, correct_pinyin))
    return closest_pinyin

def create_text_from_pdf():
    output_string = StringIO()
    correct_pinyins = load_pinyin()

    try:
        with open(TEXTBOOK_PDF_PATH, "rb") as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            laparams = LAParams(char_margin=1.0, line_margin=0.5, detect_vertical=True)
            device = TextConverter(rsrcmgr, output_string, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)

        text = output_string.getvalue()
        seg_list = jieba.cut(text, cut_all=False)
        seg_list = [pinyin_match(pinyin, correct_pinyins=correct_pinyins) for pinyin in seg_list]
        seg_text = " ".join(seg_list)

        with open(TEXTBOOK_TXT_PATH, 'w', encoding='utf-8') as f:
            f.write(seg_text)

        print(f"Text successfully written to: {TEXTBOOK_TXT_PATH}")

    except FileNotFoundError:
        print(f"File not found: {TEXTBOOK_PDF_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

create_text_from_pdf()
