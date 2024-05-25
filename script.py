
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import PyPDF2

def load_data_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, np.newaxis] * np.linalg.norm(b, axis=1))

def get_gemini_response(file_path_pdf, user_query):
    load_dotenv(override=True)

    # Step 1: Create an embedding model object
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Step 2: Load and process the PDF data
    pdf_text = load_data_from_pdf(file_path_pdf)
    embedded_data = model.encode([pdf_text])

    # Step 3: Embed the user query
    queries = [user_query]
    embedded_queries = model.encode(queries)

    # Step 4: Configure Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    print("API Key:", api_key)
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 1000,
    }

    Gemini = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
    )

    # Step 5: Get the result
    for i, query_vec in enumerate(embedded_queries):
        # Compute similarities
        similarities = cosine_similarity(query_vec[np.newaxis, :], embedded_data)
        
        # Get top 3 indices based on similarities
        top_indices = np.argsort(similarities[0])[::-1][:3]
        top_docs = [pdf_text for _ in range(len(top_indices))]
        
        # Create the augmented prompt
        augmented_prompt = f"You are an expert question answering system designed to help people learn more about the University of Texas. I'll give you a question and context based on UT history and you'll return the answer. Use an easy Texan drawl accent when answering questions. Keep the responses on the shorter side especially if including texas jokes or something like that. Query: {queries[i]} Contexts: {top_docs[0]}"
        
        # Generate the model output
        model_output = Gemini.generate_content(augmented_prompt)
        
        # Return the output
        return model_output.text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/hello', methods=['POST'])
def hello():
    first_name = request.form['first_name']
    file_path_pdf = "University_of_Texas_at_Austin.pdf"
    message = get_gemini_response(file_path_pdf, first_name)
    return jsonify(message=message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)

