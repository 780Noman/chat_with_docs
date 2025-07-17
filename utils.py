import PyPDF2
import os
import docx2txt
import google.generativeai as genai
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
import spacy
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document

MODEL_NAME = 'models/gemini-2.0-flash-lite'

def extract_text(file):
    if isinstance(file, str):
        # Handle file path
        if file.lower().endswith('.pdf'):
            return extract_text_from_pdf(file)
        elif file.lower().endswith('.docx'):
            return docx2txt.process(file)
        elif file.lower().endswith('.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return None
    else:
        # Handle uploaded file object
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == '.pdf':
            return extract_text_from_pdf(file)
        elif file_extension == '.docx':
            return docx2txt.process(file)
        elif file_extension == '.txt':
            return file.getvalue().decode("utf-8")
        else:
            return None

# Defining a function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    # Accepts both file path and file-like object
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



def extract_text_from_url(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to load
        
        page_source = driver.page_source
        driver.quit()
        
        soup = BeautifulSoup(page_source, "html.parser")
        # Remove script and style elements to clean up the text
        for script in soup(["script", "style"]):
            script.extract()
            
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        return f"Error fetching or parsing URL with Selenium: {e}"

# Function to generate a summary from text using Gemini
def get_summary_from_gemini(text):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""You are an expert in summarizing text. Please provide a concise summary of the following text:
    {text} Summary:"""
    response = model.generate_content(prompt)
    return response.text.strip()

def analyze_text_with_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    entities = {
        "Person": [],
        "Organization": [],
        "Location": [],
        "Date": [],
    }

    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text not in entities["Person"]:
            entities["Person"].append(ent.text)
        elif ent.label_ == "ORG" and ent.text not in entities["Organization"]:
            entities["Organization"].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"] and ent.text not in entities["Location"]:
            entities["Location"].append(ent.text)
        elif ent.label_ == "DATE" and ent.text not in entities["Date"]:
            entities["Date"].append(ent.text)

    # Keyword Extraction
    keywords = [token.text for token in doc if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "PROPN"]]
    top_keywords = [item[0] for item in Counter(keywords).most_common(10)]

    return entities, top_keywords

def get_text_chunks(text, filename):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vector_store

