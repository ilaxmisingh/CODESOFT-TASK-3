import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk
from tkinter import Listbox, Scrollbar

# Load the MovieLens dataset
movies_df = pd.read_csv("movies.csv") 
# The dataset has columns 'movieId', 'title', and 'genres'

# TF-IDF vectorization of movie genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])

# Calculate cosine similarity between movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#rec fun
def get_recommendations(title, num_recommendations=5):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]
#Gui frontend
class MovieRecommendationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation") #title

        self.movie_listbox = Listbox(root)
        self.movie_listbox.pack(fill='both', expand=True)

        self.scrollbar = Scrollbar(root)   #scrollbar for list
        self.scrollbar.pack(side='right', fill='y')
        self.movie_listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.movie_listbox.yview)
        #M-title
        for movie_title in movies_df['title']:
            self.movie_listbox.insert(tk.END, movie_title)

        self.recommend_button = tk.Button(root, text="Get Recommendations", command=self.get_recommendations)
        self.recommend_button.pack()

    def get_recommendations(self):
        selected_movie = self.movie_listbox.get(self.movie_listbox.curselection())
        recommended_movies = get_recommendations(selected_movie)
        self.show_recommendations(recommended_movies)

    def show_recommendations(self, recommendations):
        recommendations_text = "\n".join(recommendations)
        recommendations_window = tk.Toplevel(self.root)
        recommendations_window.title("Recommended Movies")

        recommendations_label = tk.Label(recommendations_window, text=recommendations_text)
        recommendations_label.pack()
    #main method calls for load
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationGUI(root)
    root.mainloop()
