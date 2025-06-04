import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('data/tmdb_5000_movies.csv')
df = df[['title', 'overview']]
df.dropna(inplace=True)

# Preprocess text
df['overview'] = df['overview'].apply(lambda x: x.lower())

# Vectorize overview text
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['overview'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend function
def recommend(movie_title):
    try:
        index = df[df['title'] == movie_title].index[0]
    except IndexError:
        return ["❌ Movie not found! Try another."]

    scores = list(enumerate(cosine_sim[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended = [df.iloc[i[0]]['title'] for i in scores[1:6]]
    return recommended

# Test
movie = input("🎬 Enter a movie name: ")
results = recommend(movie)

print("\n🍿 Recommended Movies:")
for r in results:
    print("👉", r)