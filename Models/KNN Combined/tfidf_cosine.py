from pathlib import Path
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors


df = None
feature_matrix = None
knn_model = None
product_index_map = None


def load_model(save=False):
    global df, feature_matrix, knn_model, product_index_map

    base_path = Path(__file__).resolve().parents[2]
    csv_path = base_path / "data" / "cleaned_data" / "cleaned_products.csv"
    df = pd.read_csv(csv_path)

    combined_features = [
        "product_id",
        "product_name",
        "brand_name",
        "category_name",
        "sub_category_name",
        "short_description",
        "long_description",
        "price",
    ]

    combined_text_features = [
        "product_name",
        "short_description",
        "long_description",
    ]
    for i in combined_text_features:
        df[i] = df[i].fillna("").astype(str)

    combined_categorical_features = [
        "brand_name", 
        "category_name", 
        "sub_category_name"
    ]
    for i in combined_categorical_features:
        df[i] = df[i].fillna("unknown").astype(str)

    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(df["price"].median())

    df["text"] = (
        (df["product_name"] + " ") * 3
        + df["short_description"] + " "
        + df["long_description"]
    )
    df["text"] = df["text"].str.lower().str.strip()
    
    custom_words = {
        "product", "item", "buy", "shop", "great", "quality",
        "perfect", "new", "best", "good", "professional", "customers",
        "this", "strong", "daily", "ideal", "users", "dependable",
        "choice", "polished", "premium", "refined", "support",
        "performance", "smooth", "designed", "digital", "compact",
        "powerful", "refreshing", "feeling", "providing"
    }
    stop_words = list(ENGLISH_STOP_WORDS.union(custom_words))

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=1
    )
    text_features = vectorizer.fit_transform(df["text"])

    encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_features = encoder.fit_transform(df[combined_categorical_features])

    scaler = StandardScaler()
    price_scaled = scaler.fit_transform(df[["price"]])
    price_sparse = csr_matrix(price_scaled)

    feature_matrix = hstack([text_features, categorical_features, price_sparse]).tocsr()

    n_neighbors = min(11, len(df))
    knn_model = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
    knn_model.fit(feature_matrix)

    product_index_map = pd.Series(df.index, index=df["product_id"]).to_dict()

    if save:
        joblib.dump(feature_matrix, "feature_matrix.joblib")
        joblib.dump(knn_model, "knn_model.joblib")
        joblib.dump(df, "products_df.joblib")
        joblib.dump(product_index_map, "product_index_map.joblib")

def recommend_by_id(product_id, k=10):
    """
    Recommends similar products using the trained KNN model and a combined feature matrix.

    Args:
        product_id (int): The ID of the product to find recommendations for.
        k (int): The number of similar products to return.

    Returns:
        list[dict]: A list of dictionaries containing recommended products with their product details similarity scores.
    """
    if df is None or feature_matrix is None or knn_model is None:
        raise ValueError("Model not loaded. Call load_model() first.")

    if product_id not in product_index_map:
        return []

    index = product_index_map[product_id]

    query = min(k + 1, len(df))
    distances, indices = knn_model.kneighbors(feature_matrix[index], n_neighbors=query)

    results = []
    for dist, i in zip(distances[0], indices[0]):
        if i == index:
            continue

        results.append({
            "product_id": int(df.iloc[i]["product_id"]),
            "product_name": df.iloc[i]["product_name"],
            "brand_name": df.iloc[i]["brand_name"],
            "category_name": df.iloc[i]["category_name"],
            "sub_category_name": df.iloc[i]["sub_category_name"],
            "price": float(df.iloc[i]["price"]),
            "similarity_score": round(1 - float(dist), 4)
        })

        if len(results) == k:
            break

    return results


def precision_at_k(product_id, k=5):
    """
    Evaluates the precision of recommendations for a given product ID.
    Args:
        product_id (int): The ID of the product to evaluate.
        k (int): The number of recommendations to consider for precision calculation.
    Returns: 
        float: The precision at K for the given product ID.
    """

    if product_id not in product_index_map:
        return 0.0

    recs = recommend_by_id(product_id, k)
    if not recs:
        return 0.0

    true_category = df.loc[df["product_id"] == product_id, "category_name"].values[0]

    relevant = 0
    for r in recs:
        if r["category_name"] == true_category:
            relevant += 1

    return relevant / k



if __name__ == "__main__":
    load_model(save=True)
    print(recommend_by_id(5, k=5))

    # Test precision for one product
    print("Precision@5:", precision_at_k(5, k=5))

    # Average Precision at K across dataset sample
    scores = []
    sample_size = min(50, len(df))
    for pid in df["product_id"].sample(sample_size, random_state=42):
        scores.append(precision_at_k(pid, k=5))
 
    print("Average Precision@5:", sum(scores) / len(scores))