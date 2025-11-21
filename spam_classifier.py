# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:09:42 2025

@author: Diuvert
"""

import data_loader
import feature_extraction
import model_trainer

def main():
    # 1. Load Data
    file_path = 'Youtube05-Shakira.csv'
    df = data_loader.load_data(file_path)
    if df is None:
        return

    # 2. Feature Extraction
    # Identify columns: CONTENT and CLASS
    # 1 for spam
    print("\nSelected columns: CONTENT, CLASS")
    X_tfidf = feature_extraction.extract_features(df, content_column='CONTENT')

    # 3. Shuffle and Split Data
    # Use pandas.sample to shuffle the dataset, set frac =1 to get random indices
    print("\n--- Shuffling Dataset ---")
    shuffled_df = df[['CONTENT', 'CLASS']].sample(frac=1, random_state=42) 
    shuffled_indices = shuffled_df.index
    
    # Reorder the features and labels according to the shuffled indices
    X_shuffled = X_tfidf[shuffled_indices]
    y_shuffled = shuffled_df['CLASS']

    # Use pandas split your dataset into 75% for training and 25% for testing
    print("\n--- Splitting Dataset (75% Train, 25% Test) ---")
    split_index = int(0.75 * len(y_shuffled))
    
    X_train = X_shuffled[:split_index]
    y_train = y_shuffled[:split_index]
    
    X_test = X_shuffled[split_index:]
    y_test = y_shuffled[split_index:]
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    # 4. Train Model
    clf = model_trainer.train_model(X_train, y_train)

    # 5. Cross Validation
    model_trainer.perform_cross_validation(clf, X_train, y_train, cv=5)

    # 6. Test Model
    model_trainer.evaluate_model(clf, X_test, y_test)

if __name__ == "__main__":
    main()
