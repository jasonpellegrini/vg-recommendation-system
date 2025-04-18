import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt


# read dataset into pandas
df = pd.read_csv("game_info.csv")

# drop irrelevant columns
df = df.drop(columns=["id","slug","website","tba","updated","released"])
df = df.dropna(subset=["genres"])

# discard low activity games
df['ratings_count'].sort_values(ascending = False).head(3000)
df = df[df['ratings_count'] > 50]
df = df.reset_index(drop=True)


# split multi-genre columns into binary vectors
def split_column(df, col):
    return df[col].fillna('').apply(lambda x: x.split("||"))

mlb = MultiLabelBinarizer()
genres = mlb.fit_transform(split_column(df, "genres"))
genres_df = pd.DataFrame(genres, columns=mlb.classes_)

# normalize numerical features
numerical = df[["metacritic", "rating", "playtime", "achievements_count", 
                "ratings_count", "reviews_count", "suggestions_count"]].fillna(0)

scaler = MinMaxScaler()
numerical_scaled = pd.DataFrame(scaler.fit_transform(numerical), columns=numerical.columns)

# build matrix of features
numerical_sparse = csr_matrix(numerical_scaled.values)
genres_sparse = csr_matrix(genres_df.values)

feature_matrix = hstack([numerical_sparse, genres_sparse])

###### Messing with PCA data vis

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(feature_matrix.toarray())


# Add PCA components to df
df['PC1'] = reduced_features[:, 0]
df['PC2'] = reduced_features[:, 1]

# Pick a genre or rating to color by
df['Genre'] = genres_df.idxmax(axis=1)  # dominant genre per row (just for visualization)

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Genre', palette='tab10', legend='full')
plt.title("PCA of Games Colored by Dominant Genre")
plt.show()

# Numerical columns (from original df)
numerical_cols = ["metacritic", "rating", "playtime", "achievements_count", 
                  "ratings_count", "reviews_count", "suggestions_count"]

# Genre columns (from the multilabel binarizer)
genre_cols = mlb.classes_.tolist()

# Combine both
feature_names = numerical_cols + genre_cols

# PCA loading inspection
loadings = pd.DataFrame(pca.components_, columns=feature_names)

print("Top contributors to PC1:")
print(loadings.iloc[0].sort_values(ascending=False).head(10))

print("\nTop contributors to PC2:")
print(loadings.iloc[1].sort_values(ascending=False).head(10))


#######################################

# fit nearest neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(feature_matrix)

# set up index lookup
indices = pd.Series(df.index, index=df['name'])

# function to recommend
def recommend(game_name, top_n=5):
    idx = indices[game_name]
    distances, indices_result = model.kneighbors(feature_matrix[idx], n_neighbors=top_n+1)
    similar_indices = indices_result.flatten()[1:]  # skip the game itself
    return df['name'].iloc[similar_indices]


print(recommend("Call of Duty: Black Ops II"))



