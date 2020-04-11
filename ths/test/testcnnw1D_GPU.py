from ths.nn.sequences.processcnnw1d import ProcessTweetsCNN1D
import tensorflow as tf
import sys

def main(epochs):
    print("Working:")
    #P = ProcessTweetsWord2VecOnePass2DCNNv2_1("data/cleantextlabels3.csv", "trained/embedding3.csv")
    #P = ProcessTweetsCNN1D("data/cleantextlabels7.csv", "data/glove.6B.50d.txt")
    #P = ProcessTweetsWord2VecTwoPassLSTMv2_1("data/cleantextlabels4.csv", "trained/embedding3-50d.csv")
    P = ProcessTweetsCNN1D("data/cleantextlabels8.csv", "data/glove.6B.100d.txt")

    #Bueno el model12cnnv2
    # Excelente el de modellstmatt1 con attention
    # El mejor fue modellstmatt2 con attention
    # also good modellstmatt3
    # el 4 con dropout
    # 2, 3 y 4 son buenos
    P.process("trained/modelcnnincepw7.json", "trained/modelcnnincepw7.h5", plot=False, epochs=epochs, vect_dimensions=100)
#joderme
if __name__ == "__main__":

    with tf.device('/GPU:1'):
        argv = sys.argv
        if len(argv) > 1:
            epochs = int(argv[1])
        else:
            epochs = 16
        main(epochs)