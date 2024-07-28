from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import pickle
from better_profanity import profanity
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load student feedback data
df = pd.read_csv('student_feedback.csv')

# Check if the 'comment' column exists
if 'comment' not in df.columns:
    raise KeyError("'comment' column not found in the CSV file")

# Load the tokenizer and LSTM model
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('lstm_model.h5')

# Initialize the profanity filter
profanity.load_censor_words()

# Load Hugging Face BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")

def replace_informal_words(text):
    tokens = word_tokenize(text)
    replaced_tokens = []
    for token in tokens:
        if profanity.contains_profanity(token):
            print("found ", token)
            masked_sentence = text.replace(token, "[MASK]")
            inputs = bert_tokenizer.encode(masked_sentence, return_tensors='pt')
            mask_token_index = torch.where(inputs == bert_tokenizer.mask_token_id)[1]
            with torch.no_grad():
                outputs = bert_model(inputs)
            predictions = outputs[0]
																					 
            mask_token_logits = predictions[0, mask_token_index, :]
            top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
            # Select the best token that is a single word and not punctuation
            for token_id in top_5_tokens:
                replaced_token = bert_tokenizer.decode([token_id]).strip()
                if re.match(r'^[\w-]+$', replaced_token):
                    replaced_tokens.append(replaced_token)
                    break
            else:
                replaced_tokens.append("[MASK]")  # Fallback if no suitable replacement is found
        else:
            replaced_tokens.append(token)
    replaced_summary = ' '.join(replaced_tokens)
    replaced_summary = re.sub(r"\s+([.,!?;'])", r'\1', replaced_summary)  # Remove spaces before punctuation
    print(replaced_summary)
    return replaced_summary

def preprocess_text_for_sentiment(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Define max_len to match the training configuration
max_len = 100  # Replace with the value you used during training

# Function to predict sentiment label
def get_sentiment_label(text):
    preprocessed_text = preprocess_text_for_sentiment(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)[0]
    print(f"Text: {text}, Prediction: {prediction}")  # Debugging output
    labels = ['Very Bad', 'Bad', 'Neutral', 'Good', 'Excellent']
    return labels[np.argmax(prediction)]

	   

# Sample data for professors and comments
data = {
    "professors": df['professor'].unique().tolist(),
    "comments": df.to_dict('records')
}

def calculate_average_label(comments):
    label_map = {"Excellent": 5, "Good": 4, "Neutral": 3, "Bad": 2, "Very Bad": 1}
    inverse_label_map = {v: k for k, v in label_map.items()}
    labels = [label_map[comment['label']] for comment in comments]
    average_score = np.mean(labels)
    average_label = inverse_label_map[round(average_score)]
    return average_label

@app.route('/')
def index():
    return render_template('index.html', professors=data["professors"])

@app.route('/get_comments', methods=['POST'])
def get_comments():
    professor = request.json['professor']
    comments = [c for c in data['comments'] if c['professor'] == professor]
    
    # Process and label comments dynamically
    for comment in comments:
        original_comment = comment['comment']
        cleaned_comment = replace_informal_words(original_comment)
        label = get_sentiment_label(cleaned_comment)
        comment['cleaned_comment'] = cleaned_comment
        comment['label'] = label
    
    average_label = calculate_average_label(comments)
    
    # Add debug prints to ensure values are correct
    print("Comments processed:", comments)
    print("Average label:", average_label)

    return jsonify({"comments": comments, "average_label": average_label})

if __name__ == '__main__':
    app.run(debug=True)
