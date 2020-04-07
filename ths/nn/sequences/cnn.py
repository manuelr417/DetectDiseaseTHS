import numpy as np

np.random.seed(7)
from ths.utils.sentences import SentenceToEmbedding

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, BatchNormalization, GRU, Conv1D, \
    Conv2D, MaxPooling2D, Flatten, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, AveragePooling2D, \
    Concatenate, ZeroPadding2D, Multiply, Permute

from keras.layers.embeddings import Embedding


class  TweetSentiment2DCNN2Channel:
    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None

    def build(self, first_dropout=0.0, padding='same', filters=4, kernel_size=(1,1), strides=(1,1), activation='relu',
              dense_units=64, second_dropout=0.0):

        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings1 = embeddings_layer(sentence_input)
        embeddings2 = embeddings_layer(reverse_sentence_input)

        # Reshape
        embeddings1= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings1)
        embeddings2= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings2)

        #stack both input to make it a 2 chanell input
        concat_embeddings = Concatenate(axis = -1)([embeddings1, embeddings2])
        print("concat_embeddings: ", concat_embeddings)
        # Reshape with channels
        #X = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)

        # First convolutional layer
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 3
        kernel_size = (kernel_height, kernel_width)
        X = Conv2D(filters=20, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV2D_1")(concat_embeddings)
        #X  = Conv2D(filters = 66, kernel_size = (kernel_height+2, 1),  strides=(1, 1), padding='same', activation=activation,
        #           name="CONV2D_2")(X)
        #MAX pooling
        pool_height =  self.max_sentence_len - kernel_height + 1  # assumes zero padding and stride of 1
        pool_size = (pool_height, 1)
        X = AveragePooling2D(pool_size=pool_size, name = "MAXPOOL_1")(X)

        #Flatten
        X = Flatten()(X)

        # Attention
        #att_dense = 70*20*1
        #attention_probs = Dense(att_dense, activation='softmax', name='attention_probs')(X)
        #attention_mul = Multiply(name='attention_multiply')([X, attention_probs])


        # # First dense layer
        dense_units = 128
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_1")(X)
        X = Dropout(second_dropout, name="DROPOUT_1")(X)

        # # Second dense layer
        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        X = Dropout(second_dropout, name="DROPOUT_2")(X)
        #
        # # Third layer
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        X = Dropout(second_dropout, name="DROPOUT_3")(X)

        # Final layer
        X = Dense(1, activation= "sigmoid", name="FINAL_SIGMOID")(X)
        # create the model
        self.model = Model(input=[sentence_input, reverse_sentence_input] , output=X)

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

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.0, class_weight=None):
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

class TweetSentiment2DCNN1x12Channel(TweetSentiment2DCNN2Channel):
    def __init__(self, max_sentence_len, embedding_builder):
            super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_dropout=0.0, padding='same', filters=4, kernel_size=(1, 1), strides=(1, 1),
              activation='relu', dense_units=64, second_dropout=0.0):

            # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings1 = embeddings_layer(sentence_input)
        embeddings2 = embeddings_layer(reverse_sentence_input)

        # Reshape
        embeddings1= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings1)
        embeddings2= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings2)

        #stack both input to make it a 2 chanell input
        concat_embeddings = Concatenate(axis = -1)([embeddings1, embeddings2])
        print("concat_embeddings: ", concat_embeddings)

        # one by one convolution
        onebyone = Conv2D(filters=32, kernel_size=(1,1), strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_1X1_1")(concat_embeddings)
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 3
        kernel_size = (kernel_height, kernel_width)
        X = Conv2D(filters=64, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV2D_1")(onebyone)
        # X  = Conv2D(filters = 66, kernel_size = (kernel_height+2, 1),  strides=(1, 1), padding='same', activation=activation,
        #           name="CONV2D_2")(X)
        # MAX pooling
        pool_height = self.max_sentence_len - kernel_height + 1  # assumes zero padding and stride of 1
        pool_size = (3, 1)
        #pool_size = (pool_height, 1)
        X = AveragePooling2D(pool_size=pool_size, name="MAXPOOL_1")(X)

        #X = Conv2D(filters=1, kernel_size=(1,1), strides=(1, 1), padding=padding, activation=activation,
        #           name="CONV2D_2")(X)

        # Flatten
        X = Flatten()(X)

        # # First dense layer
        dense_units = 128
        X = Dense(units=int(dense_units / 2), activation='relu', name="DENSE_1")(X)
        X = Dropout(second_dropout, name="DROPOUT_1")(X)

        # # Second dense layer
        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        X = Dropout(second_dropout, name="DROPOUT_2")(X)
        #
        # # Third layer
        X = Dense(units=int(dense_units / 2), activation='relu', name="DENSE_3")(X)
        X = Dropout(second_dropout, name="DROPOUT_3")(X)

        # Final layer
        X = Dense(1, activation="sigmoid", name="FINAL_SIGMOID")(X)
        # create the model
        self.model = Model(input=[sentence_input, reverse_sentence_input], output=X)


class TweetSentiment2DCNN1x12Channelv2(TweetSentiment2DCNN2Channel):
    def __init__(self, max_sentence_len, embedding_builder):
            super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_dropout=0.0, padding='same', filters=4, kernel_size=(1, 1), strides=(1, 1),
              activation='relu', dense_units=64, second_dropout=0.0):

            # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        reverse_sentence_input = Lambda()
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings1 = embeddings_layer(sentence_input)
        embeddings2 = embeddings_layer(reverse_sentence_input)

        # Reshape
        embeddings1= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings1)
        embeddings2= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings2)

        #stack both input to make it a 2 chanell input
        concat_embeddings = Concatenate(axis = -1)([embeddings1, embeddings2])
        print("concat_embeddings: ", concat_embeddings)

        # one by one convolution
        onebyone = Conv2D(filters=16, kernel_size=(1,1), strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_1X1_1")(concat_embeddings)
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 3
        kernel_size = (kernel_height, kernel_width)
        X = Conv2D(filters=32, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV2D_1")(onebyone)
        # X  = Conv2D(filters = 66, kernel_size = (kernel_height+2, 1),  strides=(1, 1), padding='same', activation=activation,
        #           name="CONV2D_2")(X)
        # MAX pooling
        pool_height = self.max_sentence_len - kernel_height + 1  # assumes zero padding and stride of 1
        pool_size = (pool_height, 1)
        X = AveragePooling2D(pool_size=pool_size, name="MAXPOOL_1")(X)

        # Flatten
        X = Flatten()(X)

        flatonebyone = Flatten()(onebyone)
        # concact one by one with MaxPoling
        X = Concatenate(axis=-1)([X, flatonebyone])
        # # First dense layer
        dense_units = 128
        X = Dense(units=int(dense_units / 2), activation='relu', name="DENSE_1")(X)
        X = Dropout(second_dropout, name="DROPOUT_1")(X)

        # # Second dense layer
        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        X = Dropout(second_dropout, name="DROPOUT_2")(X)
        #
        # # Third layer
        X = Dense(units=int(dense_units / 2), activation='relu', name="DENSE_3")(X)
        X = Dropout(second_dropout, name="DROPOUT_3")(X)

        # Final layer
        X = Dense(1, activation="sigmoid", name="FINAL_SIGMOID")(X)
        # create the model
        self.model = Model(inputs=[sentence_input, reverse_sentence_input], outputs=X)

class TweetSentimentInception(TweetSentiment2DCNN2Channel):
    def __init__(self, max_sentence_len, embedding_builder):
            super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_dropout=0.0, padding='same', filters=4, kernel_size=(1, 1), strides=(1, 1),
              activation='relu', dense_units=64, second_dropout=0.0):

        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings1 = embeddings_layer(sentence_input)
        embeddings2 = embeddings_layer(reverse_sentence_input)

        # Reshape
        embeddings1= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings1)
        embeddings2= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings2)

        #stack both input to make it a 2 chanell input
        concat_embeddings = Concatenate(axis = -1)([embeddings1, embeddings2])
        print("concat_embeddings: ", concat_embeddings)

        #compute 1x1 convolution on input
        onebyone = Conv2D(filters=filters, kernel_size=(1,1), strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_1X1_1")(concat_embeddings)


        #compute 3xdimension convolution on one by one
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 3
        kernel_size = (kernel_height, kernel_width)
        threebydim1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_3xdim_1")(onebyone)

        #compute 3xdimension convolution on input
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 3
        kernel_size = (kernel_height, kernel_width)
        threebydim2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_3xdim_2")(concat_embeddings)

        #compute 5xdimension convolution on one by one
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 5
        kernel_size = (kernel_height, kernel_width)
        fivebydim1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_5xdim_1")(onebyone)
        fivebydim1 = ZeroPadding2D((1, 0))(fivebydim1)

        #compute 5xdimension convolution on input
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 5
        kernel_size = (kernel_height, kernel_width)
        fivebydim2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_5xdim_2")(concat_embeddings)
        fivebydim2 = ZeroPadding2D((1,0))(fivebydim2)

        concat_layer = Concatenate(axis = -1)([threebydim1, threebydim2,fivebydim1, fivebydim2])

        final_onebyone = Conv2D(filters=1, kernel_size=(1,1), strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_1X1_final")(concat_layer)

        #final_onebyone = MaxPooling2D((2,1))(final_onebyone)
        #final_onebyone = AveragePooling2D((2,1))(final_onebyone)

        # Flatten
        X = Flatten()(final_onebyone)
        #X = Dropout(0.10, name="DROPOUT_1")(X)

        # attention
        att_dense = 70
        attention_probs = Dense(att_dense, activation='softmax', name='attention_probs')(X)
        attention_mul = Multiply(name='attention_multiply')([X, attention_probs])

        X = Dense(units=int(dense_units / 1), activation='relu', name="DENSE_1")(attention_mul)
        X = Dense(units=int(dense_units / 2), activation='relu', name="DENSE_2")(X)
        X = Dense(units=int(dense_units / 4), activation='relu', name="DENSE_3")(X)

        # Final layer
        #X = Dense(1, activation="sigmoid", name="FINAL_SIGMOID")(X)
        X = Dense(3, activation="softmax", name="FINAL_SOFTMAX")(X)
        # create the model
        self.model = Model(input=[sentence_input, reverse_sentence_input], output=X)



class TweetSentimentInceptionOneChan(TweetSentiment2DCNN2Channel):
    def __init__(self, max_sentence_len, embedding_builder):
            super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_dropout=0.0, padding='same', filters=4, kernel_size=(1, 1), strides=(1, 1),
              activation='relu', dense_units=64, second_dropout=0.0):

        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        #reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings1 = embeddings_layer(sentence_input)
        #embeddings2 = embeddings_layer(reverse_sentence_input)

        # Reshape
        embeddings1= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings1)
        #embeddings2= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings2)

        #stack both input to make it a 2 chanell input
        #concat_embeddings = Concatenate(axis = -1)([embeddings1, embeddings2])
        #print("concat_embeddings: ", concat_embeddings)

        #compute 1x1 convolution on input
        onebyone = Conv2D(filters=filters, kernel_size=(1,1), strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_1X1_1")(embeddings1)


        #compute 3xdimension convolution on one by one
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 3
        kernel_size = (kernel_height, kernel_width)
        threebydim1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_3xdim_1")(onebyone)

        #compute 3xdimension convolution on input
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 3
        kernel_size = (kernel_height, kernel_width)
        threebydim2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_3xdim_2")(embeddings1)

        #compute 5xdimension convolution on one by one
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 5
        kernel_size = (kernel_height, kernel_width)
        fivebydim1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_5xdim_1")(onebyone)
        fivebydim1 = ZeroPadding2D((1, 0))(fivebydim1)

        #compute 5xdimension convolution on input
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 5
        kernel_size = (kernel_height, kernel_width)
        fivebydim2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_5xdim_2")(embeddings1)
        fivebydim2 = ZeroPadding2D((1,0))(fivebydim2)

        concat_layer = Concatenate(axis = -1)([threebydim1, threebydim2,fivebydim1, fivebydim2])
        #final_onebyone = AveragePooling2D((2,1))(concat_layer)
        final_onebyone = Conv2D(filters=filters*2, kernel_size=(1,1), strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_1X1_final")(concat_layer)

        #final_onebyone = MaxPooling2D((2,1))(final_onebyone)
        #final_onebyone = AveragePooling2D((2,1))(final_onebyone)

        # Flatten
        X = Flatten()(final_onebyone)
        #X = Flatten()(concat_layer)

        #X = Dropout(0.10, name="DROPOUT_1")(X)

        # attention
        # att_dense = 70
        # attention_probs = Dense(att_dense, activation='softmax', name='attention_probs')(X)
        # attention_mul = Multiply(name='attention_multiply')([X, attention_probs])
        #
        # X = Dense(units=int(dense_units / 1), activation='relu', name="DENSE_1")(attention_mul)
        #X = Dense(units=int(dense_units / 2), activation='relu', name="DENSE_2")(X)
        #X = Dense(units=int(dense_units / 4), activation='relu', name="DENSE_3")(X)

        X = Dense(units=128, activation='relu', name="DENSE_2")(X)
        X = Dense(units=64, activation='relu', name="DENSE_3")(X)
        X = Dense(units=32, activation='relu', name="DENSE_4")(X)

        # Final layer
        #X = Dense(1, activation="sigmoid", name="FINAL_SIGMOID")(X)
        X = Dense(3, activation="softmax", name="FINAL_SOFTMAX")(X)
        # create the model
        self.model = Model(input=[sentence_input], output=X)









