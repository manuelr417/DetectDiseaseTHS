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
from ths.nn.sequences.tweetsimilarity import TweetSimilaryBasic

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

    def process(self, json_filename, h5_filename, plot=False, epochs = 100, vect_dimensions = 50):
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

        for r in All:
            # Collect_Tweets
            X_one_All.append(r[0].lower().strip())
            X_two_All.append(r[3].lower().strip())
            X_three_All.append(r[6].lower().strip())

            #Collect Aux Info
            X_one_aux_All.append(r[1:3])
            X_two_aux_All.append(r[4:6])
            X_three_aux_All.append(r[7:9])

            # Collect Y's
            Y_all.append(r[9:])

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
