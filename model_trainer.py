from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

def train_model(X_train, y_train):
    
    #Trains a Multinomial Naive Bayes classifier.    
    print("\n= Training Data Fitted into Naive Bayes Classifier Model =")
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    print("Model trained.")
    return clf

def perform_cross_validation(clf, X_train, y_train, cv=5):
    
    # Performs cross-validation and prints the results.    
    print("\n= Cross Validation (5-Fold) =")
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print(f"Cross Validation Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")

def evaluate_model(clf, X_test, y_test):
    # Evaluates the model on the test set.    
    print("\n= Testing Model =")
    y_pred = clf.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
