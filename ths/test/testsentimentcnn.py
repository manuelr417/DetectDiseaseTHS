from ths.nn.sequences.processcnn import ProcessTweetsWord2VecOnePass2DCNNv2_1, ProcessTweetsWord2VecOnePass2DCNNv2_1Negate, ProcessTweetsWord2VecOnePass2DCNN2Channelv2_1

def main():
    print("Working:")
    #P = ProcessTweetsWord2VecOnePass2DCNNv2_1("data/cleantextlabels3.csv", "trained/embedding3.csv")
    P = ProcessTweetsWord2VecOnePass2DCNN2Channelv2_1("data/cleantextlabels4.csv", "data/glove.6B.50d.txt")

    #Bueno el model12cnnv2, ccn4, cnn5
    P.process("trained/model2dcnn5.json", "trained/model2dcnn5.h5", plot=True, epochs=20)
#joderme
if __name__ == "__main__":
    main()