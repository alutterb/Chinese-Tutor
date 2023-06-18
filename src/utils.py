import pdftotext
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from io import StringIO
from dotenv import load_dotenv
import os
import jieba

load_dotenv()

TEXTBOOK_PDF_PATH = os.getenv('TEXTBOOK_PDF_PATH')
TEXTBOOK_TXT_PATH = os.getenv('TEXTBOOK_TXT_PATH')

def create_text_from_pdf():
    output_string = StringIO()
    try:
        # Check if the PDF file exists
        if not os.path.exists(TEXTBOOK_PDF_PATH):
            print(f"File not found: {TEXTBOOK_PDF_PATH}")
            return

        # Parse
        with open(TEXTBOOK_PDF_PATH, "rb") as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)

        # retrieve processed text
        text = output_string.getvalue()
        # segment text
        seg_list = jieba.cut(text, cut_all=False)
        seg_text = " ".join(seg_list)
        
        # save all text to a txt file.
        with open(TEXTBOOK_TXT_PATH, 'w', encoding='utf-8') as f:
            f.write(seg_text)

        print(f"Text successfully written to: {TEXTBOOK_TXT_PATH}")

    except FileNotFoundError:
        print(f"File not found: {TEXTBOOK_PDF_PATH}")
    except pdftotext.Error as e:
        print(f"Error while converting PDF to text: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

create_text_from_pdf()