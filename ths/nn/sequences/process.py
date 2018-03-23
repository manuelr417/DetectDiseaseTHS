from ths.nn.sequences.tweets import TweetSentiment2LSTM
from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbedding, PadSentences
import numpy as np
import csv
import math

class ProcessTweetsGlove:
    def __init__(self, labeled_tweets_filename, embedding_filename):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename

    def process(self, json_filename, h5_filename):
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    label = r[1]
                    X_all.append(tweet)
                    Y_all.append(label)
                i = i + 1
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        # divide the data into X_train, Y_train, X_test, Y_test
        X_train_sentences = X_all[0: limit]
        Y_train = Y_all[0: limit]
        X_test_sentences = X_all[limit:]
        Y_test = Y_all[limit:]
        print("Data Divided")
        #Get embeeding
        G = GloveEmbedding(self.embedding_filename)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len  = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        print("Train data convert to numpy arrays")
        NN = TweetSentiment2LSTM(max_len, G)
        print("model created")
        NN.build(dense_layer_units=1, first_layer_dropout=0.5, second_layer_dropout=0.7)
        print("model built")
        NN.summary()
        NN.compile(loss="binary_crossentropy", metrics=['binary_accuracy'])
        print("model compiled")
        print("Begin training")
        NN.fit(X_train, Y_train, epochs=80)
        print("Model trained")
        X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        print("Test data mapped")
        X_test_pad = P.pad_list(X_test_indices)
        print("Test data padded")
        X_test = np.array(X_test_pad)
        Y_test = np.array(Y_test)
        print("Test data converted to numpy arrays")
        loss, acc = NN.evaluate(X_test, Y_test)
        print("accuracy: ", acc)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is bad", "i love colombia", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        print("Done!")
