import numpy as np

np.random.seed(7)
from ths.utils.sentences import SentenceToEmbedding

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, BatchNormalization, GRU, Conv1D, \
    Conv2D, MaxPooling2D, Flatten, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, AveragePooling2D, \
    Concatenate, ZeroPadding2D, Multiply, Permute, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU

from keras.layers.embeddings import Embedding


class  TweetSentiment1D:
    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None

    def build(self, first_dropout=0.0, padding='same', filters=4, kernel_size=(1,1), strides=(1,1), activation='relu',
              dense_units=64, second_dropout=0.0):

        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings1 = embeddings_layer(sentence_input)
        width = 1

        X = self.conv_unit_lrelu(activation, embeddings1, 0)

        X = self.conv_unit_lrelu(activation, X, 1)

        X = self.conv_unit_lrelu(activation, X, 2)

        X = self.conv_unit_lrelu(activation, X, 3)

        #Flatten
        X = Flatten()(X)

        # Attention
        #att_dense = 70*20*1
        #attention_probs = Dense(att_dense, activation='softmax', name='attention_probs')(X)
        #attention_mul = Multiply(name='attention_multiply')([X, attention_probs])


        # # First dense layer
        dense_units = 128
        X = Dense(units=int(dense_units), activation='relu', name="DENSE_1")(X)
        X = Dropout(second_dropout, name="DROPOUT_1")(X)

        # # Second dense layer
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_2")(X)
        X = Dropout(second_dropout, name="DROPOUT_2")(X)
        #
        # # Third layer
        X = Dense(units=int(dense_units/4), activation='relu', name="DENSE_3")(X)
        X = Dropout(second_dropout, name="DROPOUT_3")(X)

        # Final layer
        X = Dense(3, activation= "softmax", name="SOFTMAX")(X)
        # create the model
        self.model = Model(input=[sentence_input] , output=X)

    def conv_unit_old(self, activation, prev_layer, level):
        level = str(level)
        X1 = Conv1D(filters=64, kernel_size=1, strides=1, padding='same', activation=activation, name="CONV_1_" + level)(
            prev_layer)
        X2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation=activation, name="CONV_2_" + level)(
            prev_layer)
        X3 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation=activation, name="CONV_3_" + level)(
            prev_layer)
        X4 = Conv1D(filters=64, kernel_size=7, strides=1, padding='same', activation=activation, name="CONV_4_" + level)(
            prev_layer)
        X = Concatenate(name="CONCAT_" + level)([X1, X2, X3, X4])
        X = MaxPooling1D(name="MAX_POOL_1D_" + level)(X)
        return X

    def conv_unit(self, activation, prev_layer, level):
        level = str(level)
        X1 = Conv1D(filters=64, kernel_size=1, strides=1, padding='same', name="CONV_1_" + level)(prev_layer)
        X1 = BatchNormalization(name="BATCH_1_" + level)(X1)
        X1 = Activation(activation=activation, name="ACTIVATION_1_" + level)(X1)

        X2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', name="CONV_2_" + level)(prev_layer)
        X2 = BatchNormalization(name="BATCH_2_" + level)(X2)
        X2 = Activation(activation=activation, name="ACTIVATION_2_" + level)(X2)

        X3 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same', name="CONV_3_" + level)(prev_layer)
        X3 = BatchNormalization(name="BATCH_3_" + level)(X3)
        X3 = Activation(activation=activation, name="ACTIVATION_3_" + level)(X3)

        X4 = Conv1D(filters=64, kernel_size=7, strides=1, padding='same', name="CONV_4_" + level)(prev_layer)
        X4 = BatchNormalization(name="BATCH_4_" + level)(X4)
        X4 = Activation(activation=activation, name="ACTIVATION_4_" + level)(X4)


        X = Concatenate(name="CONCAT_" + level)([X1, X2, X3, X4])
        X = MaxPooling1D(name="MAX_POOL_1D_" + level)(X)
        return X

    def conv_unit_skip_connect(self, activation, prev_layer, level):
        level = str(level)
        X1 = Conv1D(filters=64, kernel_size=1, strides=1, padding='same', name="CONV_1_" + level)(prev_layer)
        X1 = BatchNormalization(name="BATCH_1_" + level)(X1)
        X1 = Activation(activation=activation, name="ACTIVATION_1_" + level)(X1)

        X2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', name="CONV_2_" + level)(prev_layer)
        X2 = BatchNormalization(name="BATCH_2_" + level)(X2)
        X2 = Activation(activation=activation, name="ACTIVATION_2_" + level)(X2)

        X3 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same', name="CONV_3_" + level)(prev_layer)
        X3 = BatchNormalization(name="BATCH_3_" + level)(X3)
        X3 = Activation(activation=activation, name="ACTIVATION_3_" + level)(X3)

        X4 = Conv1D(filters=64, kernel_size=7, strides=1, padding='same', name="CONV_4_" + level)(prev_layer)
        X4 = BatchNormalization(name="BATCH_4_" + level)(X4)
        X4 = Activation(activation=activation, name="ACTIVATION_4_" + level)(X4)


        X = Concatenate(name="CONCAT_" + level)([X1, X2, X3, X4])
        X = MaxPooling1D(name="MAX_POOL_1D" + level)(X)
        return X

    def conv_unit_lrelu(self, activation, prev_layer, level):
        level = str(level)
        X1 = Conv1D(filters=64, kernel_size=1, strides=1, padding='same', name="CONV_1_" + level)(prev_layer)
        X1 = BatchNormalization(name="BATCH_1_" + level)(X1)
        X1 = LeakyReLU(name="ACTIVATION_1_" + level)(X1)

        X2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', name="CONV_2_" + level)(prev_layer)
        X2 = BatchNormalization(name="BATCH_2_" + level)(X2)
        X2 = LeakyReLU(name="ACTIVATION_2_" + level)(X2)

        X3 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same', name="CONV_3_" + level)(prev_layer)
        X3 = BatchNormalization(name="BATCH_3_" + level)(X3)
        X3 = LeakyReLU(name="ACTIVATION_3_" + level)(X3)

        X4 = Conv1D(filters=64, kernel_size=7, strides=1, padding='same', name="CONV_4_" + level)(prev_layer)
        X4 = BatchNormalization(name="BATCH_4_" + level)(X4)
        X4 = LeakyReLU(name="ACTIVATION_4_" + level)(X4)


        X = Concatenate(name="CONCAT_" + level)([X1, X2, X3, X4])
        X = MaxPooling1D(name="MAX_POOL_1D_" + level)(X)
        return X
    def pretrained_embedding_layer(self):
        # create Keras embedding layer
        word_to_idx, idx_to_word, word_embeddings = self.embedding_builder.read_embedding()
        #vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_embeddings
        #embedding_matrix = np.vstack([word_embeddings, np.zeros((vocabulary_len,))])

        # embedding layer
        embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension, trainable=False, name="EMBEDDING")
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])
        return embedding_layer

    def summary(self):
        self.model.summary()

    def compile(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.1, class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split, class_weight=class_weight, verbose=2)

    def evaluate(self, X_test, Y_test):
        return self.model.evaluate(X_test, Y_test)

    def predict(self, X):
        return self.model.predict(X)

    def get_sentiment(self, prediction):
        return np.argmax(prediction)

    #def sentiment_string(self, sentiment):
    #    return self.sentiment_map[sentiment]

    def save_model(self, json_filename, h5_filename):
        json_model = self.model.to_json()
        with open(json_filename, "w") as json_file:
            json_file.write(json_model)
        self.model.save_weights(h5_filename)
        return
