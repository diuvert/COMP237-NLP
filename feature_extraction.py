from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def extract_features(df, content_column='CONTENT'):
    """
    Extracts features using CountVectorizer and TfidfTransformer.
    Prints basic highlights for the initial (CountVectorizer) features
    and the final (TF-IDF) features.
    Returns the TF-IDF matrix.
    """

    # --- Initial Bag-of-Words Features ---
    print("\n--- Feature Extraction (Bag of Words) ---")
    count_vectorizer = CountVectorizer()
    X_counts = count_vectorizer.fit_transform(df[content_column])

    n_docs, n_features = X_counts.shape
    nnz_counts = X_counts.nnz
    total_cells = n_docs * n_features
    sparsity_counts = 100 * (1 - nnz_counts / total_cells)

    print("Initial Features (CountVectorizer):")
    print(f" - Shape (documents, features): {X_counts.shape}")
    print(f" - Number of documents: {n_docs}")
    print(f" - Vocabulary size: {n_features}")
    print(f" - Non-zero entries: {nnz_counts}")
    print(f" - Sparsity: {sparsity_counts:.2f}% (percentage of zeros)")

    feature_names = count_vectorizer.get_feature_names_out()
    print(f" - Sample feature names: {feature_names[:10]}")

    # --- Downscaling with TF-IDF ---
    print("\n--- Feature Downscaling (TF-IDF) ---")
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    nnz_tfidf = X_tfidf.nnz
    sparsity_tfidf = 100 * (1 - nnz_tfidf / total_cells)

    print("Final Features (TF-IDF):")
    print(f" - Shape (documents, features): {X_tfidf.shape}")
    print(f" - Non-zero entries: {nnz_tfidf}")
    print(f" - Sparsity: {sparsity_tfidf:.2f}% (percentage of zeros)")

    # Show a few TF-IDF values from the first document (if it exists)
    if n_docs > 0:
        first_doc_data = X_tfidf[0].data
        print(f" - Sample TF-IDF values for first document: {first_doc_data[:10]}")

    return X_tfidf
