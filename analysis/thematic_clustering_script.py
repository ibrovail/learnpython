## Cluster Single-Column Feedback
# Script to Normalise and Cluster 1-Column Excel File

"""
This script reads a 1-column Excel file (e.g. "Challenges") where each row contains one feedback item.

It cleans the text, applies TF-IDF vectorisation, reduces dimensionality, and clusters the entries into thematic groups using K-Means.

It also generates a sentence-style summary for each cluster based on the top terms.

Usage:
1. Place your Excel file in the same directory.
2. Set the `INPUT_FILE` variable.
3. Run the script.
4. The output will be a CSV file with feedback, cluster labels, and cluster summaries.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import nltk

# Download NLTK stopwords if needed
nltk.download('stopwords')
from nltk.corpus import stopwords

# Parameters
INPUT_FILE = "rec1.xlsx"
OUTPUT_FILE = "clustered_feedback.csv"
N_CLUSTERS = 6  # Adjust as needed
TOP_N_WORDS = 5  # Number of keywords to describe each cluster

# Read the file (expects 1 column)
df = pd.read_excel(INPUT_FILE)
assert df.shape[1] == 1, "Input file must have exactly 1 column"
col = df.columns[0]

# Clean text
df[col] = df[col].astype(str).str.strip()
df = df[df[col] != ""]

# Preprocess text
STOPWORDS = set(stopwords.words('english'))
df['clean_text'] = (
    df[col].str.lower()
             .str.replace(r'[^a-z ]', ' ', regex=True)
             .apply(lambda x: ' '.join([w for w in x.split() if w not in STOPWORDS]))
)

# Vectorise
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2)
X = vectorizer.fit_transform(df['clean_text'])

# Adjust n_components for SVD based on feature count
n_features = X.shape[1]
n_components = min(50, n_features - 1)

# Dimensionality reduction
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
df['theme'] = kmeans.fit_predict(X_reduced)

# Summarise each cluster in sentence format
terms = vectorizer.get_feature_names_out()
original_X = X.toarray()
df['cluster_summary'] = ""
summaries = {}
for i in range(N_CLUSTERS):
    cluster_indices = df[df['theme'] == i].index
    cluster_tfidf = original_X[cluster_indices]
    mean_tfidf = cluster_tfidf.mean(axis=0)
    top_term_indices = mean_tfidf.argsort()[-TOP_N_WORDS:][::-1]
    top_terms = [terms[j] for j in top_term_indices]
    sentence = f"This cluster is about: {', '.join(top_terms[:-1])}, and {top_terms[-1]}."
    summaries[i] = sentence
    df.loc[df['theme'] == i, 'cluster_summary'] = sentence

# Output results
df[[col, 'theme', 'cluster_summary']].to_csv(OUTPUT_FILE, index=False)
print(f"Clustered feedback with summaries saved to {OUTPUT_FILE}")