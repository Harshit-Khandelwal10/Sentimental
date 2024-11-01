import streamlit as st
from transformers import pipeline

# Set up the page configuration and title
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

# Load the pipelines for sentiment analysis and summarization
sentiment_analysis = pipeline("sentiment-analysis")
summarization = pipeline("summarization")

# Define the page background style
page_bg = """
<style>
    .stApp {
        background-color: #1f1f2e;
        color: #ffffff;
    }
    .header {
        font-size: 60px;
        color: #ff4b4b;
        font-weight: 700;
        text-align: center;
        font-family: 'Arial Black', Gadget, sans-serif;
        margin-bottom: 20px;
    }
    .footer {
        font-size: 20px;
        color: #cccccc;
        text-align: center;
        margin-top: 40px;
    }
    .result-box {
        font-size: 28px;
        padding: 20px;
        background-color: #2e2e3e;
        border-radius: 10px;
        border: 2px solid #cccccc;
        text-align: center;
        width: 100%;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# Display header
st.markdown("<div class='header'>🔍 Sentiment Analysis App</div>", unsafe_allow_html=True)
st.markdown("""
This application allows you to upload a text file, which will be summarized and analyzed for sentiment using a pre-trained NLP model from Hugging Face.  
Simply upload your text file below to see the sentiment it carries!
""")

# Upload .txt file
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

# Function to convert sentiment to emoji
def sentiment_to_emoji(sentiment):
    if sentiment == "POSITIVE":
        return "😊"
    elif sentiment == "NEGATIVE":
        return "😠"
    else:
        return "😐"

if uploaded_file is not None:
    # Read and decode the text file content
    file_text = uploaded_file.read().decode("utf-8")
    
    # Summarize the text content
    with st.spinner("Summarizing..."):
        summarized_text = summarization(file_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    
    # Display summarized text
    st.markdown("### Summarized Text:")
    st.markdown(f"<div class='result-box'>{summarized_text}</div>", unsafe_allow_html=True)

    # Sentiment analysis on the summarized text
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing..."):
            result = sentiment_analysis(summarized_text)
            sentiment = result[0]['label']
            emoji = sentiment_to_emoji(sentiment)
            st.markdown(f"<div class='result-box'>Sentiment: <b>{sentiment}</b> {emoji}</div>", unsafe_allow_html=True)
else:
    st.info("Please upload a .txt file to proceed with summarization and sentiment analysis.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <div class='footer'>
        Developed by <b>Harshit Khandelwal</b><br>
        Powered by Hugging Face's <i>distilbert-base-uncased-finetuned-sst-2-english</i> model.
    </div>
""", unsafe_allow_html=True)
