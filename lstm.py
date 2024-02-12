import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_texts = train_df['text'].astype(str)
train_labels = train_df['sentiment'].astype(int)
test_texts = test_df['text'].astype(str)

max_features = 500
tokenizer= Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_texts)
X_train = tokenizer.texts_to_sequences(train_texts)
X_test = tokenizer.texts_to_sequences(test_texts)

model = tf.keras.Sequential([
    Embedding(max_features, 128), 
    LSTM(100, dropout=0.2, 
    recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')])

model.copmile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 64
epochs = 10
model.fit(X_train, train_labels, batch_size=batch_size, epochs=epochs)

ss = model.evaluate(X_test, train_labels)
print("Accuracy: %.2f%%" % (ss[1]*100))

