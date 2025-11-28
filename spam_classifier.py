
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import data_loader
import model_trainer

def main():


    # STEP 1 — Load the dataset
 
    file_path = 'Youtube05-Shakira.csv'
    df = data_loader.try_read(file_path)
    if df is None:
        return
    
    print("\n= Initial Data =")
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))



    # STEP 2 — Basic Data Exploration
    print("\n= Basic Data Exploration =")
    print(df[['CONTENT', 'CLASS']].head())

    print("\nClass distribution:")
    print(df['CLASS'].value_counts())

    print("\nMissing values:")
    print(df[['CONTENT', 'CLASS']].isnull().sum())


    # STEP 3 — CountVectorizer 
    print("\n=== Preparing text using CountVectorizer ===")

    # vectorizer 
    count_vectorizer = CountVectorizer()
    X_counts = count_vectorizer.fit_transform(df['CONTENT'])

    print("\n= CountVectorizer matrix shape =")
    print(X_counts.shape)

    print("\n= Vocabulary of the first 20 words from the vectorizer =")
    vocab = list(count_vectorizer.vocabulary_.keys())
    print(vocab[:20])


    print("\n= Converting the count matrix to TF-IDF =")

    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_counts)

    print("TF-IDF data shape:", X_tfidf.shape)

    # This TF-IDF matrix will now be used for all further steps.

    # 3. Shuffle and Split Data
    # Use pandas.sample to shuffle the dataset, set frac = 1 to get all rows in random order
    print("\n= Shuffling dataset with frac=1 =")
    shuffled_df = df[['CONTENT', 'CLASS']].sample(frac=1, random_state=42)
    shuffled_indices = shuffled_df.index
    print(f"Showing the first 10 shuffled indices: {list(shuffled_indices[:10])}")

    # Reorder the features and labels according to the shuffled indices
    X_shuffled = X_tfidf[shuffled_indices]
    y_shuffled = shuffled_df['CLASS']

    # Split your dataset into 75% for training and 25% for testing
    print("\n--- Splitting Dataset (75% Train, 25% Test) ---")
    split_index = int(0.75 * len(y_shuffled))
    
    X_train = X_shuffled[:split_index]
    y_train = y_shuffled[:split_index]
    
    X_test = X_shuffled[split_index:]
    y_test = y_shuffled[split_index:]
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # 4. Train Model
    clf = model_trainer.train_model(X_train, y_train)

    # 5. Cross Validation
    model_trainer.perform_cross_validation(clf, X_train, y_train, cv=5)

    # 6. Test Model
    model_trainer.evaluate_model(clf, X_test, y_test)

if __name__ == "__main__":
    main()
