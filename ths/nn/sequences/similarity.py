import numpy as np
np.random.seed(7)
from ths.utils.sentences import SentenceToEmbedding

from keras.models import Model
from keras.layers import TimeDistributed, Dense, Input, Dropout, LSTM, Activation, Bidirectional, BatchNormalization, GRU, Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Flatten, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, Permute, Multiply, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.regularizers import l2

#np.random.seed(1)

sentence_max_length = 72
categorical_features = 16
lstm1_units = 64
lstm2_units = 64

# Python 3.6.4 (v3.6.4:d48ecebad5, Dec 18 2017, 21:07:28)
# [GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
# Type "help", "copyright", "credits" or "license" for more information.
# >>> enfermedades = ['zika', 'flu', 'miseales', 'diarreha', 'ebola']
# >>> labels = [0, 1, 2]
# >>> from sklearn.preprocessing import LabelBinarizer
# >>> lb = LabelBinarizer()
# >>> sick_encode = lb.fit_transform(enfermedades)
# >>> sick_encode
# array([[0, 0, 0, 0, 1],
#        [0, 0, 1, 0, 0],
#        [0, 0, 0, 1, 0],
#        [1, 0, 0, 0, 0],
#        [0, 1, 0, 0, 0]])
# >>> class_lables = lb.fit_transform(labels)
# >>> class_lables
# array([[1, 0, 0],
#        [0, 1, 0],
#        [0, 0, 1]])
# >>> sick_encode[0]
# array([0, 0, 0, 0, 1])
# >>> import numpy as np
# >>> np.hstack([sick_encode[0], class_lables[0]])
# array([0, 0, 0, 0, 1, 1, 0, 0])
# >>> I  = np.hstack([sick_encode[0], class_lables[0]])
# >>> I.shape
# (8,)
# >>>


input_zero = Input(shape=(sentence_max_length, ), name ="Input_zero")  # query tweet
input_one = Input(shape=(sentence_max_length, ), name ="Input_one") # candidate one
input_two = Input(shape=(sentence_max_length, ), name ="Input_two") # candidate two
input_cat_zero_one = Input(shape=(categorical_features, ), name ="Input_cat_zero_one") # candidate two
input_cat_zero_two = Input(shape=(categorical_features, ), name ="Input_cat_zero_two") # candidate two

emb_zero = None
emb_one = None
emb_two = None

lstm1 = LSTM(lstm1_units, return_sequences=True, name='LSTM_1')
lstm2 = LSTM(lstm2_units, return_sequences=False, name='LSTM_2')


encoded_one = lstm2(lstm1(emb_zero))
encoded_two = lstm2(lstm1(emb_two))

dense1 = Dense(1, name='relevance')

relevance1 = dense1(encoded_one)
relevance2 = dense1(encoded_two)

