from ths.nn.sequences.processcnnw1d import ProcessTweetsCNN1D

def main():
    print("Working:")
    #P = ProcessTweetsWord2VecOnePass2DCNNv2_1("data/cleantextlabels3.csv", "trained/embedding3.csv")
    #P = ProcessTweetsCNN1D("data/cleantextlabels7.csv", "data/glove.6B.50d.txt")
    P = ProcessTweetsCNN1D("data/cleantextlabels7.csv", "data/glove.6B.100d.txt")

    #P = ProcessTweetsWord2VecTwoPassLSTMv2_1("data/cleantextlabels4.csv", "trained/embedding3-50d.csv")

    #Bueno el model12cnnv2
    # Excelente el de modellstmatt1 con attention
    # El mejor fue modellstmatt2 con attention
    # also good modellstmatt3
    # el 4 con dropout
    # 2, 3 y 4 son buenos
    P.process("trained/modelcnnincepw6.json", "trained/modelcnnincepw6.h5", plot=True, epochs=16, vect_dimensions=100)
#joderme
if __name__ == "__main__":
    main()