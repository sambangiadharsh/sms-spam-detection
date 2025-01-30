try:
    import streamlit as st
    import pickle
    import os
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
except ModuleNotFoundError:
    raise ModuleNotFoundError("Streamlit is not installed. Please install it using 'pip install streamlit'.")

# Check if model exists, if not, train and save it
#MODEL_FILE = "spam_model.pkl

def train_and_save_model():
    sms_data = pd.read_csv("/home/rgukt/pt/sms-spam.csv", encoding='latin-1')
    # Instead of assigning column names directly, infer them or use the existing ones
    # sms_data.columns = ['label', 'message'] 
    
    # If the first two columns are 'label' and 'message', select them
    sms_data = sms_data[['label', 'message']] 
    sms_data.columns = ['label', 'message']  
    
    sms_data['label'] = sms_data['label'].map({'ham': 0, 'spam': 1})
    
    # Drop rows with missing values in the 'message' column
    sms_data = sms_data.dropna(subset=['message'])  
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sms_data['message'])
    y = sms_data['label']
    
    model = MultinomialNB()
    model.fit(X, y)
    
    
    print("Model trained and saved successfully.")
    return vectorizer, model

# Load trained model

tfidf, clf = train_and_save_model()

# Streamlit UI
def main():
    st.title("SMS Spam Detector")
    st.write("Enter a message to check if it is spam or not.")
    
    message = st.text_area("Enter Message:")
    if st.button("Check Spam"):
        if message:
            message_vector = tfidf.transform([message])
            prediction = clf.predict(message_vector)[0]
            result = "Spam" if prediction else "Not Spam"
            st.success(f"Prediction: {result}")
        else:
            st.warning("Please enter a message.")

if __name__ == "__main__":
    main()

