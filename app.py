import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load and extract text from the PDF
def load_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Split text into overlapping sections (context window)
def split_text_with_context(text, window_size=500, overlap=100):
    words = text.split()
    sections = []
    for i in range(0, len(words), window_size - overlap):
        section = " ".join(words[i:i + window_size])
        if len(section) > 20:
            sections.append(section.strip())
    return sections

# Function to get the best answer
def get_best_answer(sections, query):
    section_embeddings = model.encode(sections, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, section_embeddings).squeeze(0)
    best_score_index = scores.argmax().item()
    best_score = scores[best_score_index].item()
    best_section = sections[best_score_index]
    return best_score, best_section

# Streamlit app UI
st.title("PDF Question-Answering System")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
query = st.text_area("Enter your question")

if uploaded_file and query:
    # Process the PDF and get the answer when 'Get Answer' button is clicked
    document_text = load_pdf(uploaded_file)
    sections = split_text_with_context(document_text)
    
    # Show loading indicator while processing
    with st.spinner("Finding the best answer..."):
        best_score, best_answer = get_best_answer(sections, query)
    
    # Display the best answer
    st.subheader("Best Answer")
    st.write(best_answer)
    st.write(f"**Relevance Score:** {best_score:.4f}")
