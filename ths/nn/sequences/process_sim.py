from ths.nn.sequences.tweets import TweetSentiment2DCNNv2_1
from ths.nn.sequences.cnn import TweetSentiment2DCNN2Channel, TweetSentiment2DCNN1x12Channel, TweetSentiment2DCNN1x12Channelv2, TweetSentimentInception
from ths.utils.files import GloveEmbedding, Word2VecEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbedding, PadSentences, TrimSentences
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.regularizers import l2
from ths.nn.metrics.f1score import f1, precision, recall, fprate
from ths.utils.synomymos import OverSampleSynonym
from ths.nn.sequences.tweetsimilarity import TweetSimilaryBasic, TweetSimilaryBasicBiDirectional, TweetSimilaryConvInception
from sklearn.utils import class_weight

import numpy as np
import csv
import math
from random import randint

class ProcessTweetsSimBasic:
    def __init__(self, labeled_tweets_filename, embedding_filename):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename

    def map_to_idx(self, S, X_words):
        X_indices, max_len = S.map_sentence_list(X_words)
        if max_len % 2 != 0:
            max_len = max_len + 1
        return X_indices, max_len

    def binarize_aux(self, diseases, labels):
        D = to_categorical(diseases)
        L = to_categorical(labels)
        print("D.shape: ", D.shape)
        print("L.shape: ", L.shape)

        return np.hstack((D, L))

    def process(self, json_filename, h5_filename, prod_json_file, prod_h5_filename, plot=False, epochs = 100, vect_dimensions = 50):
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        All  = [] # array with every row in the file

        # Load all rows into array All
        #with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
        with open(self.labeled_tweets_filename, "r") as f:

            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            ones_count = 0

            for r in csv_file:
                All.append(r)

        # Mix the rows randoming to prevent order dependencies between runs
        np.random.shuffle(All)

        X_one_All = []
        X_two_All = []
        X_three_All = []

        X_one_aux_All = []
        X_two_aux_All = []
        X_three_aux_All = []

        X_one_aux_disease = []
        X_two_aux_disease = []
        X_three_aux_disease = []

        X_one_aux_label = []
        X_two_aux_label = []
        X_three_aux_label = []

        Y_t1_t2_relevance = []
        Y_t1_t3_relevance = []

        for r in All:
            # Collect_Tweets
            X_one_All.append(r[0].lower().strip())
            X_two_All.append(r[3].lower().strip())
            X_three_All.append(r[6].lower().strip())

            #Collect Aux Info
            #X_one_aux_All.append(r[1:3])
            #X_two_aux_All.append(r[4:6])
            #X_three_aux_All.append(r[7:9])
            X_one_aux_disease.append(r[1])
            X_one_aux_label.append(r[2])

            X_two_aux_disease.append(r[4])
            X_two_aux_label.append(r[5])

            X_three_aux_disease.append(r[7])
            X_three_aux_label.append(r[8])

            # Collect Y's
            Y_t1_t2_relevance.append(r[9])
            Y_t1_t3_relevance.append(r[10])
            Y_all.append(r[11])

        # Convert the data to a form the NN can understand
        num_data = len(All)

        #print("All: ", All)
        print("All: ")
        for r in All:
            print(r)
        print("All.len: ", len(All))

        print("X_one_All: ")
        for r in X_one_All:
            print(r)

        G = Word2VecEmbedding(self.embedding_filename, dimensions=vect_dimensions)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_one_indices, max_len1 =  self.map_to_idx(S, X_one_All)

        X_two_indices, max_len2 =  self.map_to_idx(S, X_two_All)
        X_three_indices, max_len3 =  self.map_to_idx(S, X_three_All)

        #print("X_one_indices: ", X_one_indices)
        print("X_one_indices: ")
        for r in X_one_indices:
            print(r)
        print("max_len1 : ", max_len1)

        for r in X_two_indices:
            print(r)
        print("max_len2 : ", max_len2)
        for r in X_three_indices:
            print(r)
        print("max_len3 : ", max_len3)

        # get max len of all
        max_len = max(max_len1, max_len2)
        max_len = max(max_len, max_len3)
        if max_len % 2 !=0:
            max_len = max_len + 1

        # now padd the 3 senteces with 0 to make them all the same size
        P = PadSentences(max_len)
        X_one_train = P.pad_list(X_one_indices)
        X_two_train = P.pad_list(X_two_indices)
        X_three_train = P.pad_list(X_three_indices)

        # now make the sencentes into np.array
        X_one_train  = np.array(X_one_train)
        X_two_train  = np.array(X_two_train)
        X_three_train  = np.array(X_three_train)

        # change to categorical the disease type and the label
        X_one_aux_train = self.binarize_aux(X_one_aux_disease, X_one_aux_label)
        print('X_one_aux_train.shape: ', X_one_aux_train.shape)
        X_two_aux_train = self.binarize_aux(X_two_aux_disease, X_two_aux_label)
        X_three_aux_train = self.binarize_aux(X_three_aux_disease, X_three_aux_label)

        # Create the NN
        labels_dim = 2
        diases_dim = 4
        #NN = TweetSimilaryBasic(max_sentence_len=max_len, embedding_builder=G, labels_dim = labels_dim, diases_dim = diases_dim)
        #NN = TweetSimilaryBasicBiDirectional(max_sentence_len=max_len, embedding_builder=G, labels_dim = labels_dim, diases_dim = diases_dim)
        NN = TweetSimilaryConvInception(max_sentence_len=max_len, embedding_builder=G, labels_dim = labels_dim, diases_dim = diases_dim)

        # Build the NN
        NN.build()
        # Summary
        NN.summary()
        # Compile the NN
        #NN.compile(optimizer='rmsprop', loss=['mean_squared_error','mean_squared_error', 'binary_crossentropy'],
        #           metrics=['mse', 'mse','acc'], loss_weight=[ 1., 1., 1.0])

        NN.compile(optimizer='rmsprop', loss={'R1' : 'mean_squared_error', 'R2' : 'mean_squared_error', 'FINAL' : 'binary_crossentropy'},
                   metrics={'R1' : 'mse', 'R2' : 'mse', 'FINAL' : 'acc'}, loss_weights= {'R1' : 0.25, 'R2' : 0.25, 'FINAL' : 10})

        Y_t1_t2_relevance = np.array(Y_t1_t2_relevance)
        Y_t1_t3_relevance = np.array(Y_t1_t3_relevance)
        Y_all  = np.array(Y_all)
        class_weight_val = class_weight.compute_class_weight('balanced', np.unique(Y_all), Y_all)
        print("type(class_weight_val): ",type(class_weight_val))
        print("class_weight_val", class_weight_val)
        final_class_weight_val = {'R1' : None, 'R2' : None, 'FINAL' : class_weight_val}
        print("final_class_weight_val: ", final_class_weight_val)
        history = NN.fit(X=[X_one_train, X_two_train, X_three_train, X_one_aux_train, X_two_aux_train, X_three_aux_train], Y = [Y_t1_t2_relevance, Y_t1_t3_relevance, Y_all], epochs=epochs, validation_split=0.20,
                         class_weight=final_class_weight_val)


        # Save the model
        NN.save_model_data(json_filename, h5_filename, prod_json_file, prod_h5_filename)
        NN.plot_stats(history)
        #print(history)
        #print(history.history.keys())
        print("Done!")



















