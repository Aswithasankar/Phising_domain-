import joblib

# Load the trained KNN model
knn_model = joblib.load('model.pkl')  # Use the correct path to your model file

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = joblib.load(vectorizer_file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = joblib.load(label_encoder_file)

# Function to classify a website
def classify_website(website_name):
    # Create a TF-IDF vector for the input website name
    website_vector = tfidf_vectorizer.transform([website_name])
    
    # Predict the category (0 for phishing, 1 for normal)
    category = knn_model.predict(website_vector)
    
    # Decode the numerical category using the label encoder
    category_label = label_encoder.inverse_transform(category)
    
    # Define categories
    categories = {0: "Bad", 1: "Good"}
    
    # Return the category as "Bad" or "Good"
    return categories[category[0]]

# Example usage:
website_name = "reutregev.com/bin/updates/webscr.html?cmd=_login-run" # Replace with the website name you want to classify
predicted_category = classify_website(website_name)

print(f"Predicted Category: {predicted_category}")
