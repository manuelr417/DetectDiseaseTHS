from ths.nn.sequences.processlstm import ProcessTweetsWord2VecOnePassLSTMv2_1

def main():
    print("Working:")
    #P = ProcessTweetsWord2VecOnePass2DCNNv2_1("data/cleantextlabels3.csv", "trained/embedding3.csv")
    P = ProcessTweetsWord2VecOnePassLSTMv2_1("data/cleantextlabels5.csv", "data/glove.6B.50d.txt")

    #Bueno el model12cnnv2
    P.process("trained/modellstm10.json", "trained/modellstm10.h5", plot=True, epochs=15)
#joderme
if __name__ == "__main__":
    main()