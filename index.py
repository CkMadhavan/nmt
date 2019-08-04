import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from flask import Flask

app = Flask(__name__)

app.route('/')
def main():

    model=keras.models.load_model("nmt.h5")

    t_eng = pickle.load(open("t_eng","rb"))
    t_ger = pickle.load(open("t_ger","rb"))

    string = ["startseq how are you doing endseq"]
    pred_seq = t_eng.texts_to_sequences(string)
    pred = pad_sequences(pred_seq , maxlen=7,padding="post")

    string1 = ["startseq"]

    while string1[0][-6:] != "endseq":
    
        pred_seq1 = t_ger.texts_to_sequences(string1)
        pred1 = pad_sequences(pred_seq1 , maxlen=12,padding="post")

        prediction = model.predict([pred[0].reshape(1,7) , pred1[0].reshape(1,12)])
        string1[0] += ' ' + list(t_ger.word_index.keys())[list(t_ger.word_index.values()).index(np.argmax(prediction[0]))]

    return (string1[0][9:-7])

if __name__ == "__main__":
    app.run()
