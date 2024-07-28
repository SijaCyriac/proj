import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import pickle

# Load and preprocess the training data
train_df = pd.read_csv('training_data.csv', encoding='latin1')

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

train_df['cleaned_comments'] = train_df['comment'].apply(preprocess_text)

# Tokenize the text data
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
#assign indices to 10000 most frequent words
tokenizer.fit_on_texts(train_df['cleaned_comments'])
sequences = tokenizer.texts_to_sequences(train_df['cleaned_comments'])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Prepare labels
#Example ['good'] is replaced with 3
labels = train_df['label'].apply(lambda x: {'Excellent': 4, 'Good': 3, 'Neutral': 2, 'Bad': 1, 'Very Bad': 0}[x])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Define LSTM Model
def create_lstm_model():
    model = Sequential()
    model.add(Embedding(max_words, 100, input_length=max_len))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))  # 5 classes for sentiment
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

lstm_model = create_lstm_model()
lstm_model.summary()

# Train the model
batch_size = 32
epochs = 10

history = lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Save the model and tokenizer
lstm_model.save('lstm_model.h5')

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
