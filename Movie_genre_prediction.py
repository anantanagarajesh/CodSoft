import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('/content/action.csv')

data.info()
data.shape
data.head()
X = data['description']
y = data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer to convert text data into numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=5000) 

# Fit and transform the vectorizer on the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Naive Bayes
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

def predict_movie_genre(plot_summary):
    plot_summary = tfidf_vectorizer.transform([plot_summary])
    genre = classifier.predict(plot_summary)[0]
    return genre

while True:
    user_input = input("Enter a movie plot summary (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    genre = predict_movie_genre(user_input)
    print(f"Predicted Genre: {genre}")
