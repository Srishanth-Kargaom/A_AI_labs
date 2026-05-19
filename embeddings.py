import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ssl_fix  # noqa — needed even here as sentence-transformers downloads model weights

'''
Sentence Embeddings and Semantic Search

An embedding converts text into a fixed-size vector of numbers such that
semantically similar pieces of text end up close to each other in vector
space. This is the foundation of modern search, RAG, and recommendation
systems.

SentenceTransformer models are pre-trained to produce these vectors.
"all-MiniLM-L6-v2" is a popular lightweight choice — fast and accurate
enough for most retrieval tasks.

Cosine similarity measures the angle between two vectors:
  - Score of 1.0  -> identical direction (very similar meaning)
  - Score of 0.0  -> orthogonal (unrelated)
  - Score near -1 -> opposite meaning

The workflow here mimics a minimal vector search engine:
  1. Embed all documents upfront (this would be your vector database)
  2. Embed the query at search time
  3. Rank documents by similarity to the query
  4. Return the best match
'''

from sentence_transformers import SentenceTransformer

# Knowledge base — Bollywood movie descriptions
documents = [
    "Dangal is a biographical sports drama about wrestler Mahavir Singh Phogat training his daughters to become world-class wrestlers.",
    "3 Idiots is a comedy-drama following three engineering students challenging the rigid education system in India.",
    "Lagaan is a period drama where villagers challenge British colonisers to a cricket match to avoid heavy taxes.",
    "Queen is a coming-of-age story about a woman who takes her honeymoon trip alone after her fiance cancels their wedding.",
    "Dil Chahta Hai follows three best friends navigating love, friendship, and growing up after college.",
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode all documents into vectors
doc_embeddings = model.encode(documents)

# Semantic query — searching by meaning, not keywords
query = "A film about a father pushing his children to achieve in sports"
query_embedding = model.encode([query])

# Compute similarity scores
scores = model.similarity(doc_embeddings, query_embedding)
print("Raw similarity scores:")
print(scores)
print()

# Find the best matching document
best_index, best_score = sorted(
    enumerate(scores), key=lambda x: x[1]
)[-1]

print(f"Query     : {query}")
print(f"Best match: {documents[best_index]}")
print(f"Score     : {float(best_score):.4f}")
