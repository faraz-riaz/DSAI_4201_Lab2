import streamlit as st
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Set page configuration
st.set_page_config(
    page_title="Document Retrieval System",
    page_icon="📚",
    layout="wide"
)

# Add title and description
st.title("📚 Document Retrieval System")
st.markdown("""
This app uses Word2Vec and Reuters corpus to find relevant documents based on your query.
Enter a search query below to find similar documents from the Reuters dataset.
""")

@st.cache_resource
def load_data():
    """Load the pickled corpus and model"""
    try:
        corpus_sentences = pickle.load(open('corpus.pkl', 'rb'))
        model = Word2Vec.load('word2vec.model')
        return corpus_sentences, model
    except FileNotFoundError:
        st.error("Required model files not found. Please ensure corpus.pkl and word2vec.model exist.")
        return None, None

def get_document_embedding(tokens, word2vec_model):
    word_embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    return np.mean(word_embeddings, axis=0) if word_embeddings else None

def retrieve_documents(query, word2vec_model, corpus_sentences, top_n=5):
    query_tokens = [word for word in nltk.word_tokenize(query.lower()) 
                   if word.isalnum() and word not in stopwords.words('english')]
    
    query_embedding = get_document_embedding(query_tokens, word2vec_model)
    if query_embedding is None:
        return []
    
    similarities = []
    with st.spinner("Searching documents..."):
        for doc_idx, doc_tokens in enumerate(corpus_sentences):
            doc_embedding = get_document_embedding(doc_tokens, word2vec_model)
            if doc_embedding is not None:
                similarity = cosine_similarity(query_embedding.reshape(1, -1), 
                                            doc_embedding.reshape(1, -1))[0][0]
                similarities.append((doc_idx, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Initialize NLTK
@st.cache_resource
def initialize_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')

# Load data and initialize NLTK
initialize_nltk()
corpus_sentences, model = load_data()

if corpus_sentences is not None and model is not None:
    # Create the search interface
    st.subheader("Search Documents")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter your search query:", "Artificial Intelligence")
    with col2:
        top_n = st.slider("Number of results:", min_value=1, max_value=10, value=5)

    if st.button("Search") or query:
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            results = retrieve_documents(query, model, corpus_sentences, top_n)
            
            if not results:
                st.warning("No relevant documents found. Try a different query.")
            else:
                st.subheader(f"Top {len(results)} Documents")
                
                for doc_idx, similarity in results:
                    with st.expander(f"Document {doc_idx} (Similarity: {similarity:.4f})"):
                        st.markdown("**Content:**")
                        st.write(' '.join(corpus_sentences[doc_idx][:50]) + "...")

    # Add footer
    st.markdown("---")
    st.markdown("Built with Streamlit and Word2Vec using Reuters corpus") 