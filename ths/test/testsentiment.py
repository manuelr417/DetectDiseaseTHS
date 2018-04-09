from ths.nn.sequences.process import ProcessTweetsGlove, ProcessTweetsGloveOnePass, ProcessTweetsGloveOnePassSM

def main():
    print("Working:")
    #P  = ProcessTweetsGlove("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    #P  = ProcessTweetsGloveOnePass("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    P = ProcessTweetsGloveOnePassSM("data/cleantextlabels2.csv", "data/glove.6B.50d.txt")
    P.process("trained/model3sm.json", "trained/model3sm.h5", plot=True, epochs=100)
#joderme
if __name__ == "__main__":
    main()