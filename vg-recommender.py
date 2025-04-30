import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, hstack
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, 
    QListWidget, QComboBox, QTabWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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

# Perform PCA for Visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(feature_matrix.toarray())

df['PC1'] = reduced_features[:, 0]
df['PC2'] = reduced_features[:, 1]
df['Genre'] = genres_df.idxmax(axis=1)  # Assign dominant genre for each game

# Also prepare the PCA loadings for the bar chart
numerical_cols = ["metacritic", "rating", "playtime", "achievements_count", 
                  "ratings_count", "reviews_count", "suggestions_count"]
genre_cols = mlb.classes_.tolist()
feature_names = numerical_cols + genre_cols

loadings = pd.DataFrame(pca.components_, columns=feature_names)
# #######################################

# fit nearest neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(feature_matrix)

# set up index lookup
indices = pd.Series(df.index, index=df['name'])

# function to recommend
def recommend(game_name, preferred_platform=None, top_n=5, weight_mode="Balanced"):
    idx = indices[game_name]

    # Adjust feature matrix based on weight_mode
    if weight_mode == "Genre-heavy":
        weight_numerical = 0.1
        weight_genre = 1.0
    elif weight_mode == "Ratings-heavy":
        weight_numerical = 1.0
        weight_genre = 0.1
    else:  # Balanced
        weight_numerical = 1.0
        weight_genre = 1.0

    # Apply the weights
    weighted_features = hstack([
        numerical_sparse.multiply(weight_numerical),
        genres_sparse.multiply(weight_genre)
    ])

    # Fit a temporary NearestNeighbors model with the adjusted features
    temp_model = NearestNeighbors(metric='cosine', algorithm='brute')
    temp_model.fit(weighted_features)

    distances, indices_result = temp_model.kneighbors(weighted_features[idx], n_neighbors=top_n+20)

    similar_indices = indices_result.flatten()[1:]  # skip the game itself
    candidates = df.iloc[similar_indices].copy()

    if preferred_platform:
        preferred_platform = preferred_platform.lower()

        candidates['platform_match'] = candidates['platforms'].fillna('').apply(
            lambda x: 1 if preferred_platform in x.lower() else 0
        )

        candidates = candidates.sort_values(by=['platform_match'], ascending=False)

    return candidates['name'].head(top_n)

class RecommenderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Game Recommender')
        self.resize(900, 1000)
        self.setStyleSheet("font-size: 16px;")

        self.tabs = QTabWidget()

        self.recommendation_tab = QWidget()
        self.pca_tab = QWidget()
        self.corr_tab = QWidget()
        self.feature_tab = QWidget()
        self.feature2_tab = QWidget()
        self.rating_dist_tab = QWidget()
        self.genre_count_tab = QWidget()
        self.playtime_distribution_tab = QWidget()


        self.tabs.addTab(self.recommendation_tab, "Recommend")
        self.tabs.addTab(self.pca_tab, "PCA Scatter")
        self.tabs.addTab(self.corr_tab, "Correlation Matrix")
        self.tabs.addTab(self.feature_tab, "Feature Importance")
        self.tabs.addTab(self.feature2_tab, "Feature Importance PC2")
        self.tabs.addTab(self.rating_dist_tab, "Rating Distribution")
        self.tabs.addTab(self.genre_count_tab, "Top Genres")
        self.tabs.addTab(self.playtime_distribution_tab, "Platform Distribution")


        self.init_recommendation_tab()
        self.init_pca_tab()
        self.init_corr_tab()
        self.init_feature_tab()
        self.init_feature2_tab()
        self.init_rating_distribution_tab()
        self.init_genre_count_tab()
        self.init_playtime_distribution_tab()


        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def init_recommendation_tab(self):
        layout = QVBoxLayout()

        self.label = QLabel('Enter Game Name:')
        layout.addWidget(self.label)

        self.input_field = QLineEdit()
        layout.addWidget(self.input_field)

        self.platform_label = QLabel('Enter Preferred Platform (optional):')
        layout.addWidget(self.platform_label)

        self.platform_input = QLineEdit()
        layout.addWidget(self.platform_input)

        self.weight_label = QLabel('Select Recommendation Focus:')
        layout.addWidget(self.weight_label)

        self.weight_selector = QComboBox()
        self.weight_selector.addItems(["Balanced", "Genre-heavy", "Ratings-heavy"])
        layout.addWidget(self.weight_selector)

        self.button = QPushButton('Recommend Similar Games')
        self.button.clicked.connect(self.get_recommendations)
        layout.addWidget(self.button)

        self.result_list = QListWidget()
        layout.addWidget(self.result_list)

        self.recommendation_tab.setLayout(layout)

    def init_pca_tab(self):
        layout = QVBoxLayout()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Genre', palette='tab10', legend='full', ax=ax)
        ax.set_title("PCA of Games Colored by Dominant Genre")
        canvas = FigureCanvas(fig)

        layout.addWidget(canvas)
        self.pca_tab.setLayout(layout)

    def init_corr_tab(self):
        layout = QVBoxLayout()

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(numerical.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Matrix")
        canvas = FigureCanvas(fig)

        layout.addWidget(canvas)
        self.corr_tab.setLayout(layout)

    def init_rating_distribution_tab(self):
        layout = QVBoxLayout()

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(df['rating'], bins=20, kde=True, ax=ax)
        ax.set_title("Distribution of Game Ratings")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of Games")
        canvas = FigureCanvas(fig)

        layout.addWidget(canvas)
        self.rating_dist_tab.setLayout(layout)

    
    def init_genre_count_tab(self):
        layout = QVBoxLayout()

        genre_counts = genres_df.sum().sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 5))
        genre_counts.plot(kind='bar', ax=ax)
        ax.set_title("Top 15 Genres by Game Count")
        ax.set_ylabel("Number of Games")
        ax.set_xlabel("Genre")
        canvas = FigureCanvas(fig)

        layout.addWidget(canvas)
        self.genre_count_tab.setLayout(layout)

    
    def init_playtime_distribution_tab(self):
        layout = QVBoxLayout()
        fig, ax = plt.subplots(figsize=(10, 6))

        platform_series = df['platforms'].dropna().apply(lambda x: x.split('||'))
        exploded_platforms = platform_series.explode()

        platform_counts = exploded_platforms.value_counts().sort_values(ascending=False).head(20)  # Top 20 platforms

        platform_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Number of Games per Platform")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Number of Games")
        ax.tick_params(axis='x', rotation=45)

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self.playtime_distribution_tab.setLayout(layout)


    def init_feature_tab(self):
        layout = QVBoxLayout()

        fig, ax = plt.subplots(figsize=(7, 5))
        top_features = loadings.iloc[0].abs().sort_values(ascending=False).head(10)
        top_features.plot(kind='barh', ax=ax)
        ax.set_title("Top Features Contributing to PC1")
        canvas = FigureCanvas(fig)

        layout.addWidget(canvas)
        self.feature_tab.setLayout(layout)
    
    def init_feature2_tab(self):
        layout = QVBoxLayout()

        fig, ax = plt.subplots(figsize=(7, 5))
        top_features_pc2 = loadings.iloc[1].abs().sort_values(ascending=False).head(10)
        top_features_pc2.plot(kind='barh', ax=ax)
        ax.set_title("Top Features Contributing to PC2")
        canvas = FigureCanvas(fig)

        layout.addWidget(canvas)
        self.feature2_tab.setLayout(layout)

    def get_recommendations(self):
        game_name = self.input_field.text()
        platform = self.platform_input.text()
        weight_mode = self.weight_selector.currentText() 

        recommendations = recommend(
            game_name,
            preferred_platform=platform if platform else None,
            weight_mode=weight_mode 
        )
        self.result_list.clear()
        self.result_list.addItems(recommendations)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RecommenderApp()
    window.show()
    sys.exit(app.exec_())
