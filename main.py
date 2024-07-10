import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib  # Import joblib for model serialization
import pickle  # Import pickle for model serialization

# Step 1: Load the CSV dataset
data = pd.read_csv('data.csv')

# Step 2: Data Preprocessing
# Encode the categorical labels into numerical values
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])

# Step 3: Split the Data
X = data['URL']  # Use 'Website' as the feature
y = data['Label']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Text Vectorization (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Save the TF-IDF vectorizer using joblib
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Save the LabelEncoder using joblib
joblib.dump(label_encoder, 'label_encoder.pkl')

# Step 5: Build and Train the KNN Model
k = 1  # You can adjust the value of k as needed
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train_tfidf, y_train)

# Step 6: Make Predictions
y_pred = knn_classifier.predict(X_test_tfidf)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
confusion = confusion_matrix(y_test, y_pred)

# Display and save the confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
#plt.savefig('confusion_matrix.png')  # Save the confusion matrix plot to a file

# Save the trained KNN model using joblib
joblib.dump(knn_classifier, 'model.pkl')

# Save the trained KNN model using pickle in .sav format
with open('model.sav', 'wb') as model_file:
    pickle.dump(knn_classifier, model_file)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)
