import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.neighbors import NearestNeighbors
import joblib


Text_Columns = ["product_name", "brand_name", "short_description", "long_description"]

def build_model(df):
    df = df.copy()
    for col in Text_Columns:
        df[col] = df[col].astype(str).fillna("")

    df["text"] = (
        (df["product_name"] + " ") *3 +
        (df["brand_name"] + " ") * 2 +
        df["short_description"] + " " +
        df["long_description"] 
    ).str.lower()

    custom_words = {
    "product", "item", "buy", "shop", "great",
    "quality", "perfect", "new", "best", "good"
    }
    stop_words = list(ENGLISH_STOP_WORDS.union(custom_words))

    n_neighbors = min(11, len(df))
    vectorizer = TfidfVectorizer(
        stop_words=stop_words, 
        ngram_range=(1, 2),
        min_df = 2,
        max_df = 0.8
    )

    feature_matrix = vectorizer.fit_transform(df["text"])

    model = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
    model.fit(feature_matrix)
    id_to_index = pd.Series(df.index, index = df["product_id"]).to_dict()

    joblib.dump(feature_matrix, "feature_matrix.joblib")
    joblib.dump(model, "knn_model.joblib")
    joblib.dump(vectorizer, "tfid_vectorizer.joblib")
    joblib.dump(id_to_index, "id_to_index.joblib")

    return feature_matrix, model, vectorizer, id_to_index, df

def recommend_by_id(product_id, feature_matrix, model, df, id_to_index, k=10):
    product_index = id_to_index.get(product_id)

    if product_index is None:
        return []

    n_neighbors = min(30, len(df))
    distances, indices = model.kneighbors(
        feature_matrix[product_index],
        n_neighbors = n_neighbors
    )

    results = []
    for dist, i in zip(distances[0], indices[0]):
        if i != product_index:
            results.append({
                "product_id": int(df.iloc[i]["product_id"]),
                "similarity": float(1-dist)
            })
    return results[:k]

def main():
    df = pd.read_csv("cleaned_products.csv")

    feature_matrix, model, vectorizer, id_to_index, df = build_model(df)

    product_id = 197
    recs = recommend_by_id(product_id, feature_matrix, model, df, id_to_index)

    print("Recommended product IDs:")
    for r in recs:
        print(r)

if __name__ == "__main__":
    main()
