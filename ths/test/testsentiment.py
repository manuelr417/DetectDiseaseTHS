from ths.nn.sequences.process import ProcessTweetsGlove, ProcessTweetsGloveOnePass, ProcessTweetsGloveOnePassSM, ProcessTweetsWord2VecOnePassSM

def main():
    print("Working:")
    #P  = ProcessTweetsGlove("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    #P  = ProcessTweetsGloveOnePass("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    P = ProcessTweetsGloveOnePassSM("data/cleantextlabels2.csv", "data/glove.6B.50d.txt")
    #P = ProcessTweetsWord2VecOnePassSM("data/cleantextlabels2.csv", "trained/embedding1.csv")
    P.process("trained/model4sm.json", "trained/model4sm.h5", plot=True, epochs=100)
#joderme
if __name__ == "__main__":
    main()