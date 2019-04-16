import numpy as np

from keras.models import Model
from keras.layers import TimeDistributed, Dense, Input, Dropout, LSTM, Activation, Bidirectional, BatchNormalization, GRU, Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Flatten, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, Permute, Multiply, TimeDistributed
from keras.layers import Subtract, Multiply, Concatenate, Lambda
from keras.backend import abs

from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from sklearn.preprocessing import LabelBinarizer

from keras.utils import plot_model
def build_auxiliary_input(diases_dim, labels_dim, layer_name='AUX_INPUT'):
    return Input(shape=(diases_dim+labels_dim, ), name=layer_name)

def build_tweet_input(sentence_max_length, layer_name='TWEET_INPUT'):
    return Input(shape=(sentence_max_length,), name=layer_name)

def build_LSTM_Layer(units=64, return_sequences=False, layer_name='LSTM'):
    return LSTM(units, return_sequences=return_sequences, name=layer_name)


class TweetSimilaryBasic:
    def __init__(self, max_sentence_len, embedding_builder, diases_dim, labels_dim):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model_training = None
        self.model_production = None
        self.diases_dim = diases_dim
        self.labels_dim = labels_dim

    def build(self, first_layer_units=128, first_layer_dropout=0.5, second_layer_units=128,
              second_layer_dropout=0.5, dense_layer_units=2):

        # Tweet Input Layers
        tweet_one = build_tweet_input(self.max_sentence_len, layer_name="INPUT_TWEET_1")
        tweet_two = build_tweet_input(self.max_sentence_len, layer_name="INPUT_TWEET_2")
        tweet_three = build_tweet_input(self.max_sentence_len, layer_name="INPUT_TWEET_3")



        # Auxiliary Input Layers
        aux_tweet_one = build_auxiliary_input(diases_dim=self.diases_dim, labels_dim= self.labels_dim,
                                              layer_name="INPUT_AUX_TWEET_1")
        aux_tweet_two = build_auxiliary_input(diases_dim=self.diases_dim, labels_dim= self.labels_dim,
                                              layer_name="INPUT_AUX_TWEET_2")
        aux_tweet_three = build_auxiliary_input(diases_dim=self.diases_dim, labels_dim=self.labels_dim,
                                          layer_name="INPUT_AUX_TWEET_3")

        # Embeddings
        embeddings_layer = self.pretrained_embedding_layer()
        E1  = embeddings_layer(tweet_one)
        E2  = embeddings_layer(tweet_two)
        E3  = embeddings_layer(tweet_three)

        # Combiner - two LSTM in sequence
        # First Part - pass embedded over first LSTM
        lstm_1 = build_LSTM_Layer(units=first_layer_units, return_sequences=True, layer_name="LSTM1")

        X1 = lstm_1(E1)
        X2 = lstm_1(E2)
        X3 = lstm_1(E3)

        # Second Part - pass Xi over sencond LSTM
        lstm_2 = build_LSTM_Layer(units=first_layer_units, return_sequences=False, layer_name="LSTM2")
        X1 = lstm_2(X1)
        X2 = lstm_2(X2)
        X3 = lstm_2(X3)

        # Relevance Component

        # Merge X1 and X2 as per twitter papes
        mult1 = Multiply()([X1, X2])
        sub1 = Subtract()([X1, X2])
        L1 = Lambda(abs)((sub1))
        con1 = Concatenate()([X1, X2, L1, mult1])

        # Merge X1 and X3 as per twitter papes
        mult2 = Multiply()([X1, X3])
        sub2 = Subtract()([X1, X3])
        L2 = Lambda(abs)((sub2))
        con2 = Concatenate()([X1, X3, L2, mult2])

        # Merge all secondary input
        aux_con1 = Concatenate(name="concat1")([aux_tweet_one, aux_tweet_two, con1])
        aux_con2 = Concatenate(name="concat2")([aux_tweet_one, aux_tweet_three, con2])

        # Layer that computes the Relevance
        # Relevance of T1 and T2
        dense1 = Dense(128, activation='relu', name="DENSE_1")
        dense2 = Dense(1, name="DENSE_2")

        # Relevance of T1 and T2
        D1 = dense1(aux_con1)
        V1 = dense2(D1)  # V2 is the relevance between T1 ane T2

        # Relevance of T1 and T3
        D2 = dense1(aux_con2)
        V2 = dense2(D2)

        # Final Dense Layers to classify if the triplet: 0 = T2 is more simular to T1, 1 = T3 is more similar to T1
        dense3 = Dense(16, activation='relu',name="DENSE_3")
        dense4 = Dense(8, activation='relu', name="DENSE_4")
        dense5 = Dense(1, activation='sigmoid', name="FINAL")
        final_concat = Concatenate(name="concatF")([V1, V2])
        F_class = dense3(final_concat)
        F_class = dense4(F_class)
        F_class = dense5(F_class)

        # This is the models used to train the system to get the weights right
        self.model_training = Model(inputs=[tweet_one, tweet_two, tweet_three, aux_tweet_one, aux_tweet_two, aux_tweet_three],
                           outputs=[F_class])
        #print("Joder: ", type(self.model_training))

        # This is the actual model we want to use for pruduction. Given one two tweets T_A and T_B, it gives the relavance of T_B with
        # respect to T_A
        self.model_production = Model(inputs=[tweet_one, tweet_two, aux_tweet_one, aux_tweet_two], outputs=[V1])

    def pretrained_embedding_layer(self):
        # create Keras embedding layer
        word_to_idx, idx_to_word, word_embeddings = self.embedding_builder.read_embedding()
        # vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_embeddings
        # embedding_matrix = np.vstack([word_embeddings, np.zeros((vocabulary_len,))])

        # embedding layer
        embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension, trainable=False,
                                    name="EMBEDDING")
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])
        return embedding_layer

    def summary(self):
        self.model_training.summary()
        self.model_production.summary()


    def plot(self, to_file):
        plot_model(self.model_training, to_file=to_file+"T.jpg")
        plot_model(self.model_production, to_file=to_file+"P.jpg")
