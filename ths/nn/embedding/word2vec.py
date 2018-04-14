from keras.layers import Embedding, Input, Dense, Reshape, dot
from keras.models import Model



class Word2VecNegSam:
    def __init__(self, vocabulary_size, vector_dim):
        self.vocabulary_size = vocabulary_size
        self.vector_dim = vector_dim
        self.model = None
        self.validation_model = None


    def build(self):
        # create the input layer for the target word and context word
        # shape is (1,) because each one is just an index of the
        # word in the vocabulary

        input_target = Input((1,), name="INPUT_TARGET")
        input_context = Input((1,), name="INPUT_CONTEXT")

        # now create the embedding layer
        # shape of this layer is vocabulary_size  x vector_dim x 1
        embedding = Embedding(self.vocabulary_size, self.vector_dim, input_length=1, name = "EMBEDDING")

        # now map input target and input context with the embedding
        # This will produce the mapping from index to embedding vector
        # this is what we want this model to lear, how to do this well
        target = embedding(input_target)
        context = embedding(input_context)

        # now to do a reshape to make sure the tensor have the
        # right dimension as a vector_dim x 1 tensor. This part I always
        # see is done, by barely explained why. In this case, it seems
        # is needed to make sure the dot product works correctly
        target = Reshape((self.vector_dim, 1))(target)
        context = Reshape((self.vector_dim, 1))(context)

        #compute the cosine similary between target and context
        # this is need by the validation model
        # axes 0 is used because similary will be done over 2 tensors
        similarity = dot([target, context], axes=0, normalize=True)

        # compute the dot product between target and context
        # axes is 1 becuase the axis = 0 is the batch size
        # the dot will be computed over batches of target,context pairs
        dot_product = dot([target, context], axes=1)
        dot_product = Reshape((1,))(dot_product)

        # now pass dot product throught sigmoid layer
        output_layer = Dense(1, activation='sigmoid', name='SIGMOID')(dot_product)

        #now build the model
        self.model = Model(input=[input_target, input_context], output=output_layer)
        self.validation_model = Model(input=[input_target, input_context], output_layer=similarity)

        return self.model

    def compile(self, optimizer='rmsprop', metrics=['accuracy']):
        self.model.compile(loss= 'binary_crossentropy', optimizer=optimizer, metrics = metrics)

    def get_model(self):
        return self.model

    def get_validation_model(self):
        return self.validation_model
