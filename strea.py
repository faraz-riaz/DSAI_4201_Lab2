import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

# Process corpus
corpus_sentences = pickle.load(open('corpus.pkl', 'rb'))

# Train Word2Vec model
model = Word2Vec.load('word2vec.model')

# Document retrieval functions
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
    for doc_idx, doc_tokens in enumerate(corpus_sentences):
        doc_embedding = get_document_embedding(doc_tokens, word2vec_model)
        if doc_embedding is not None:
            similarity = cosine_similarity(query_embedding.reshape(1, -1), 
                                        doc_embedding.reshape(1, -1))[0][0]
            similarities.append((doc_idx, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Example query
query = "Artificial Intelligence"
results = retrieve_documents(query, model, corpus_sentences)

print(f"\nTop {len(results)} documents for query: '{query}'\n")
for doc_idx, similarity in results:
    print(f"Document {doc_idx}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Content: {' '.join(corpus_sentences[doc_idx][:50])}...")
    print()