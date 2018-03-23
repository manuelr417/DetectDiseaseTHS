from ths.nn.sequences.process import ProcessTweetsGlove

def main():
    print("Working:")
    P  = ProcessTweetsGlove("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    P.process("data/model5.json", "data/model5.h5")
#joderme
if __name__ == "__main__":
    main()