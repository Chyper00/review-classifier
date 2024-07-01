import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load the dataset from the CSV file
# The CSV file contains reviews and their respective labels
df = pd.read_csv('reviews_dataset.csv')

# 2. TF-IDF Vectorization
# Converts the review texts into a TF-IDF feature matrix
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['review'])

# 3. Split the data into training and testing sets
# Splits the data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# 4. Initialize the SVM classifier
# The SVM classifier will be used to train the model
classifier = SVC(kernel='linear', random_state=42)

# 5. Train the model
# The SVM classifier is trained with the training data
classifier.fit(X_train, y_train)

# 6. Evaluate the model's performance
# Uses the testing set to predict and evaluate the model's accuracy
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Save the trained model
# Saves the trained model to a .pkl file
joblib.dump(classifier, 'podium_reviews_classifier.pkl')
print("Model saved as podium_reviews_classifier.pkl")

# 8. Load the trained model
# Loads the previously saved model for future use
classifier_loaded = joblib.load('podium_reviews_classifier.pkl')

# 9. Example usage with new reviews
# New reviews for classification
new_reviews = [
    "Excellent experience, highly recommend to everyone!",
    "I was quite disappointed with the service provided.",
    "The place is beautiful, but the food was disappointing."
]

# 10. Vectorize the new reviews
# Converts the new reviews into a TF-IDF feature matrix using the same trained vectorizer
X_new = vectorizer.transform(new_reviews)

# 11. Predict the classes of the new reviews
# Uses the loaded model to predict the class of the new reviews
predictions = classifier_loaded.predict(X_new)

# 12. Display the predictions
# Displays the prediction results
for review, prediction in zip(new_reviews, predictions):
    print(f"Review: {review} => Prediction: {prediction}")
