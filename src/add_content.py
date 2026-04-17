import os
import sys
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

DATA_PATH = "data/mediwaste_info.txt"

def append_to_info(text, source_name):
    if not text.strip():
        print(f"Warning: No valid text to append from {source_name}")
        return

    with open(DATA_PATH, 'a', encoding='utf-8') as f:
        f.write(f"\n\n--- Content from {source_name} ---\n\n")
        f.write(text)
    
    print(f"Successfully appended content from {source_name} to {DATA_PATH}.")
    print("Remember to run ingest.py to update the AI's knowledge base!")

def scrape_url(url):
    print(f"Scraping URL: {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator='\n')
        
        # Clean up empty lines
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        append_to_info(text, url)
    except Exception as e:
        print(f"Error scraping {url}: {e}")

def extract_pdf(pdf_path):
    print(f"Extracting PDF: {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' does not exist.")
        return

    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        append_to_info(text, os.path.basename(pdf_path))
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python add_content.py url <http://example.com>")
        print("  python add_content.py pdf <path_to_file.pdf>")
        sys.exit(1)
        
    command = sys.argv[1].lower()
    source = sys.argv[2]
    
    if command == "url":
        scrape_url(source)
    elif command == "pdf":
        extract_pdf(source)
    else:
        print(f"Unknown command: {command}. Use 'url' or 'pdf'.")
