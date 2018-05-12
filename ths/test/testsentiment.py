from ths.nn.sequences.process import ProcessTweetsGlove, ProcessTweetsGloveOnePass, ProcessTweetsGloveOnePassSM, ProcessTweetsWord2VecOnePassSM, ProcessTweetsWord2VecOnePassCNN, ProcessTweetsWord2VecOnePass2DCNN

def main():
    print("Working:")
    #P  = ProcessTweetsGlove("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    #P  = ProcessTweetsGloveOnePass("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    #P = ProcessTweetsGloveOnePassSM("data/cleantextlabels2.csv", "data/glove.6B.50d.txt")
    P = ProcessTweetsWord2VecOnePassSM("data/cleantextlabels3.csv", "trained/embedding3.csv")
    #P = ProcessTweetsWord2VecOnePass2DCNN("data/cleantextlabels3.csv", "trained/embedding3.csv")
    #P = ProcessTweetsWord2VecOnePassCNN("data/cleantextlabels3.csv", "trained/embedding3.csv")

    P.process("trained/model17.json", "trained/model17.h5", plot=True, epochs=30)
#joderme
if __name__ == "__main__":
    main()