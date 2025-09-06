from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import BookRecommender
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

df = pd.read_csv('dataset/dataset_clean.csv')
embeddings_array = np.load('dataset/book_embeddings.npy')
genres_matrix = np.load('dataset/genres_matrix.npy')

recommender = BookRecommender(df, embeddings_array, genres_matrix)

@app.route('/recommend', methods=['POST'])
def recommend_books():
    data = request.json
    book_title = data.get('book_title')

    if not book_title:
        return jsonify({'error': 'No book title provided'}), 400
    
    try:
       recommendations = recommender.weighted_recommender(book_title)
       return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 