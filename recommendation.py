import streamlit as st
st.set_page_config(page_title="Anime Recommendation System", layout="centered")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, sigmoid_kernel
from sklearn.neighbors import NearestNeighbors

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv('anime_merged_with_images.csv')  # Includes image_url
    df['genre'] = df['genre'].fillna('')
    df['type'] = df['type'].fillna('')
    df['user_rating'] = df['user_rating'].fillna(0)
    return df

df = load_data()

# --- Content-based (sigmoid kernel) setup ---
sampled_df = df.sample(n=2000, random_state=42).reset_index(drop=True)
tfidf_sig = TfidfVectorizer(stop_words='english')
tfidf_matrix_sig = tfidf_sig.fit_transform(sampled_df['genre'])
sig_sim = sigmoid_kernel(tfidf_matrix_sig, tfidf_matrix_sig)
sig_indices = pd.Series(sampled_df.index, index=sampled_df['name'].str.lower()).drop_duplicates()

# --- Hybrid setup ---
user_item_matrix = df.pivot_table(index='user_id', columns='anime_id', values='user_rating').fillna(0)
df['content'] = df[['genre', 'type']].astype(str).agg(' '.join, axis=1)
anime_profiles = df.drop_duplicates(subset='anime_id')[['anime_id', 'content']]
tfidf_hybrid = TfidfVectorizer(stop_words='english')
tfidf_matrix_hybrid = tfidf_hybrid.fit_transform(anime_profiles['content'])
content_sim = cosine_similarity(tfidf_matrix_hybrid)
item_knn = NearestNeighbors(metric='cosine', algorithm='brute')
item_knn.fit(user_item_matrix.T)

# --- Collaborative Filtering (User-User) Setup ---
user_similarity_matrix = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# --- Item-Based Setup ---
item_user_matrix = df.pivot_table(index='name', columns='user_id', values='user_rating').fillna(0)
item_similarity_matrix = cosine_similarity(item_user_matrix)
item_sim_df = pd.DataFrame(item_similarity_matrix, index=item_user_matrix.index, columns=item_user_matrix.index)

# --- Display Function ---
def display_recommendations(df_result):
    for i, row in df_result.iterrows():
        cols = st.columns([1, 3])
        with cols[0]:
            if pd.notna(row.get('image_url')):
                st.image(row['image_url'], width=100)
            else:
                st.image("https://via.placeholder.com/100x140?text=No+Image", width=100)
        with cols[1]:
            st.markdown(f"**{row['name']}**")
            if 'genre' in row:
                st.markdown(f"Genre: {row['genre']}")
            st.markdown("---")

# --- Recommender Functions ---
def recommend_sigmoid(title, top_n=10):
    idx = sig_indices.get(title.lower())
    if idx is None:
        return pd.DataFrame([{'name': f"'{title}' not found in sample.", 'image_url': None}])
    scores = list(enumerate(sig_sim[idx].flatten()))
    scores = sorted(scores, key=lambda x: float(x[1]), reverse=True)[1:top_n+1]
    indices = [i[0] for i in scores]
    return sampled_df.iloc[indices][['name', 'genre', 'image_url']].reset_index(drop=True)

def hybrid_recommend(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame([{'name': f"User {user_id} not found.", 'image_url': None}])
    user_ratings = user_item_matrix.loc[user_id]
    watched_anime = user_ratings[user_ratings > 0].index.tolist()
    scores = {}
    anime_idx_map = {aid: idx for idx, aid in enumerate(user_item_matrix.columns)}
    for anime_id in watched_anime:
        try:
            idx = anime_idx_map[anime_id]
            _, indices = item_knn.kneighbors(user_item_matrix.T.iloc[idx, :].values.reshape(1, -1), n_neighbors=top_n+1)
            for neighbor_idx in indices[0][1:]:
                neighbor_id = user_item_matrix.columns[neighbor_idx]
                scores[neighbor_id] = scores.get(neighbor_id, 0) + 0.5
            content_idx = anime_profiles[anime_profiles['anime_id'] == anime_id].index[0]
            sim_scores = list(enumerate(content_sim[content_idx].flatten()))
            sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)[1:top_n+1]
            for idx, sim in sim_scores:
                sim_anime_id = anime_profiles.iloc[idx]['anime_id']
                scores[sim_anime_id] = scores.get(sim_anime_id, 0) + 0.5 * sim
        except Exception as e:
            print(f"Error with anime_id {anime_id}: {e}")
            continue
    ranked = sorted(scores.items(), key=lambda x: float(x[1]), reverse=True)
    recs = [aid for aid, _ in ranked if aid not in watched_anime][:top_n]
    anime_lookup = df.drop_duplicates('anime_id').set_index('anime_id')[['name', 'image_url', 'genre']]
    return anime_lookup.loc[recs].reset_index(drop=True)

def user_cf_recommend(user_id, top_n=5):
    if user_id not in user_sim_df.index:
        return pd.DataFrame([{'name': f"User {user_id} not found.", 'image_url': None}])
    similar_users = user_sim_df[user_id].sort_values(ascending=False).drop(user_id)
    weighted_scores = pd.Series(dtype='float64')
    for other_user, score in similar_users.items():
        other_ratings = user_item_matrix.loc[other_user]
        weighted_scores = weighted_scores.add(other_ratings * score, fill_value=0)
    already_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    weighted_scores = weighted_scores.drop(already_rated, errors='ignore')
    top_anime_ids = weighted_scores.sort_values(ascending=False).head(top_n).index
    anime_lookup = df.drop_duplicates('anime_id').set_index('anime_id')[['name', 'image_url', 'genre']]
    return anime_lookup.loc[top_anime_ids].reset_index(drop=True)

def item_cf_recommend(anime_name, top_n=5):
    if anime_name not in item_sim_df.columns:
        return pd.DataFrame([{'name': f"Anime '{anime_name}' not found.", 'image_url': None}])
    similar = item_sim_df[anime_name].sort_values(ascending=False)[1:top_n+1]
    similar_names = similar.index.tolist()

    # Fixed: avoid KeyError by not including 'name' in [[]]
    anime_lookup = df.drop_duplicates('name').set_index('name')[['image_url', 'genre']]
    
    # Reset index to turn name back into a column
    return anime_lookup.loc[similar_names].reset_index()

# --- Streamlit UI ---
st.title("🎥 Anime Recommendation System")
tabs = st.tabs([
    "Content-Based (Sigmoid)",
    "Hybrid System",
    "Collaborative Filtering",
    "Item-Based Collaborative"
])

with tabs[0]:
    st.subheader("🎭 Genre-Based Recommender")
    anime_title = st.text_input("Enter anime title:", "Naruto")
    if st.button("Get Recommendations", key='sig'):
        result = recommend_sigmoid(anime_title)
        display_recommendations(result)

with tabs[1]:
    st.subheader("🤝 Hybrid (Content + Collaborative)")
    user_id_input = st.selectbox("Select user ID:", user_item_matrix.index.tolist(), key='hybrid_select')
    if st.button("Get Recommendations", key='hybrid'):
        result = hybrid_recommend(user_id_input)
        display_recommendations(result)

with tabs[2]:
    st.subheader("👥 Collaborative Filtering (User-User)")
    user_id_cf = st.selectbox("Choose user ID:", user_item_matrix.index.tolist(), key='cf_select')
    if st.button("Get Recommendations", key='cf'):
        result = user_cf_recommend(user_id_cf)
        display_recommendations(result)

with tabs[3]:
    st.subheader("📺 Item-Based Collaborative Filtering")
    anime_name = st.text_input("Enter anime name you liked:", "Naruto", key='item_based_input')
    if st.button("Get Recommendations", key='item_based_btn'):
        result = item_cf_recommend(anime_name)
        display_recommendations(result)
