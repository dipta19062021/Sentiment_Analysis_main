import streamlit as st
import pickle

# Load all 5 models
def load_models():
    model1 = pickle.load(open("C:\Users\ADMIN\Sentiment-Analysis-main\Models\countVectorizer.pkl", "rb"))
    model2 = pickle.load(open("C:\Users\ADMIN\Sentiment-Analysis-main\Models\model_dt.pkl", "rb"))
    model3 = pickle.load(open("C:\Users\ADMIN\Sentiment-Analysis-main\Models\model_rf.pkl", "rb"))
    model4 = pickle.load(open("C:\Users\ADMIN\Sentiment-Analysis-main\Models\model_xgb.pkl", "rb"))
    model5 = pickle.load(open("C:\Users\ADMIN\Sentiment-Analysis-main\Models\scaler.pkl", "rb"))
    return model1, model2, model3, model4, model5

# Function to make predictions
def predict_sentiment(text, model):
    prediction = model.predict([text])
    return prediction[0]

# Streamlit UI setup
st.title("Sentiment Analysis with Multiple Models")

# Load models once
model1, model2, model3, model4, model5 = load_models()

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ("Model 1", "Model 2", "Model 3", "Model 4", "Model 5")
)

# Text input for the user to input text
user_input = st.text_area("Enter text for sentiment analysis:")

# Perform analysis when user clicks "Analyze"
if st.button("Analyze"):
    if user_input:
        # Select the chosen model based on user input
        if model_choice == "Model 1":
            result = predict_sentiment(user_input, model1)
        elif model_choice == "Model 2":
            result = predict_sentiment(user_input, model2)
        elif model_choice == "Model 3":
            result = predict_sentiment(user_input, model3)
        elif model_choice == "Model 4":
            result = predict_sentiment(user_input, model4)
        else:
            result = predict_sentiment(user_input, model5)

        # Display the result
        st.write(f"Predicted sentiment: {result}")
    else:
        st.write("Please enter some text for analysis.")

