import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BookRecommender:
    def __init__(self, df, embeddings_array, genres_matrix):
        self.df = df
        self.embeddings_array = embeddings_array
        self.genres_matrix = genres_matrix

    def get_content_scores(self, input_book_idx):
        input_embedding = self.embeddings_array[input_book_idx].reshape(1,-1)
        content_scores = cosine_similarity(input_embedding, self.embeddings_array)[0]
        
        return content_scores
    
    def get_genre_scores(self, input_book_idx):
        input_genre_matrix = self.genres_matrix[input_book_idx].reshape(1, -1)
        genre_scores = cosine_similarity(input_genre_matrix, self.genres_matrix)[0]

        return genre_scores
    
    def get_author_scores(self, input_book_idx):
        input_author = self.df.loc[input_book_idx, 'Author']

        author_scores = np.zeros(len(self.df))

        author_scores = (self.df['Author'] == input_author).astype(int)

        return author_scores
    
    def weighted_recommender(self, input_book_title):
        try:
            input_idx = self.df[self.df['Title'].str.lower() == input_book_title.lower()].index[0]
        except IndexError:
            return []
        
        content_scores = self.get_content_scores(input_idx)
        genre_scores = self.get_genre_scores(input_idx)
        author_scores = self.get_author_scores(input_idx)

        alpha = 50
        beta = 47.5
        gamma = 2.5

        final_scores = (alpha * content_scores) + (beta * genre_scores) + (gamma * author_scores)

        sorted_indices = final_scores.argsort()[::-1]

        top_5_indices = sorted_indices[1:6]
        recommendations_df = self.df.loc[top_5_indices, ['Title', 'Author']]
        recommendations_list = recommendations_df.to_dict('records')

        return recommendations_list

