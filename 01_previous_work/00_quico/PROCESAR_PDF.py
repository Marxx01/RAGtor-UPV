
#PROCEADOR DE LOS PDF'S SOLO EN UN IDIOMA

from langchain.text_splitter import CharacterTextSplitter
from langdetect import detect, DetectorFactory
import os
import re
import PyPDF2

def read_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            # Extraer el texto y manejar la codificación
            page_text = page.extract_text()
            if page_text:
                # Reemplazar caracteres no deseados
                page_text = page_text.replace('\xa0', ' ')
                text += page_text
    return text

def process_text(text):
    splitter = CharacterTextSplitter()
    chunks = splitter.split_text(text)
    return chunks

def process_pdfs_from_directory(directory):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            texts = []
            for filename in os.listdir(dir_path):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(dir_path, filename)
                    text = read_pdf(file_path)
                    processed_text = process_text(text)
                    texts.append(processed_text)
            # Guardar los textos en un archivo .txt
            with open(os.path.join(dir_path, f'{dir_name}.txt'), 'w', encoding='utf-8') as txt_file:
                for chunk in texts:
                    txt_file.write('\n'.join(chunk) + '\n')

# Ejemplo de uso
directory = input('Introduce el directorio: ')
process_pdfs_from_directory(directory)