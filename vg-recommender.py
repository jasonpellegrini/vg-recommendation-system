import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

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


print(recommend("Terraria"))

