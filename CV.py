import os
import fitz  # PyMuPDF
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Define the root folder path where your PDFs are located
root_folder_path = r"C:\Users\hp\Downloads\archive (3)\data\data"

# Load the CSV file containing job descriptions
csv_file_path = r"C:\Users\hp\Downloads\training_data.csv" 
df = pd.read_csv(csv_file_path)

# Initialize the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Initialize lists to store results
top_candidates = []

# Function to extract text from a PDF file using PyMuPDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    pdf_document = fitz.open(pdf_file_path)
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()
    return text

# Function to extract key details from extracted text
def extract_key_details(text):
    details = {
        'Category': None,
        'Skills': [],
        'Education': []
    }

    # Implement your logic here to extract Category, Skills, and Education
    # For example, use regular expressions or other techniques

    return details

# Traverse through department folders and process PDFs
for department_folder in os.listdir(root_folder_path):
    department_folder_path = os.path.join(root_folder_path, department_folder)
    
    # Check if it's a directory (to skip any non-directory files)
    if os.path.isdir(department_folder_path):
        for pdf_file_name in os.listdir(department_folder_path):
            if pdf_file_name.endswith('.pdf'):
                pdf_file_path = os.path.join(department_folder_path, pdf_file_name)
                text = extract_text_from_pdf(pdf_file_path)
                key_details = extract_key_details(text)

                # Loop through job descriptions
                for i, description in enumerate(df['job_description']):
                    # Tokenize the job description
                    job_description_tokens = tokenizer(description, padding=True, truncation=True, return_tensors="pt")

                    # Extract embeddings for the job description
                    job_description_embeddings = model(**job_description_tokens).last_hidden_state.mean(dim=1)

                    # Tokenize the CV details
                    cv_tokens = tokenizer(key_details['Skills'], padding=True, truncation=True, return_tensors="pt")

                    # Extract embeddings for the CV details
                    cv_embeddings = model(**cv_tokens).last_hidden_state.mean(dim=1)

                    # Calculate cosine similarity between job description and CV embeddings
                    similarity_score = cosine_similarity(job_description_embeddings, cv_embeddings)[0][0]

                    # Store the result
                    top_candidates.append({
                        'Job Description': description,
                        'Similarity Score': similarity_score,
                        'CV Details': key_details
                    })

# Sort the candidates by similarity score (descending order)
top_candidates = sorted(top_candidates, key=lambda x: x['Similarity Score'], reverse=True)

# Print the top 5 candidates for each job description
for i, candidate in enumerate(top_candidates):
    if i % 5 == 0:
        print(f"Job Description: {candidate['Job Description']}")
    print(f"Top Candidate {i % 5 + 1}: Similarity Score = {candidate['Similarity Score']:.4f}")
    print(f"   CV Details: {candidate['CV Details']}")
    print()
