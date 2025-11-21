from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def extract_features(df, content_column='CONTENT'):
    """
    Extracts features using CountVectorizer and TfidfTransformer.
    Returns the TF-IDF matrix.
    """
    print("\n--- Feature Extraction (Bag of Words) ---")
    count_vectorizer = CountVectorizer()
    X_counts = count_vectorizer.fit_transform(df[content_column])
    print("Initial Features (CountVectorizer) Shape:", X_counts.shape)

    print("\n--- Feature Downscaling (TF-IDF) ---")
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    print("Final Features (TF-IDF) Shape:", X_tfidf.shape)
    
    return X_tfidf
