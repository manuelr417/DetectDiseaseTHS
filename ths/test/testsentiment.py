from ths.nn.sequences.process import ProcessTweetsGlove, ProcessTweetsGloveOnePass

def main():
    print("Working:")
    #P  = ProcessTweetsGlove("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    P  = ProcessTweetsGloveOnePass("data/cleantextlabels.csv","data/glove.6B.50d.txt")

    P.process("trained/model15.json", "trained/model15.h5")
#joderme
if __name__ == "__main__":
    main()