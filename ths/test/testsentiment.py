from ths.nn.sequences.process import ProcessTweetsGlove

def main():
    print("Working:")
    P  = ProcessTweetsGlove("cleantextlabels.csv","glove.6B.50d.txt")
    P.process("model4.json", "model4.h5")
#joder
if __name__ == "__main__":
    main()