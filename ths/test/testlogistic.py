from ths.nn.sequences.processlogistic import ProcessTweetsLogistic
def main():
    print("Working:")
    P = ProcessTweetsLogistic("data/cleantextlabels5.csv", "trained/embedding3.csv")
    P.process(epochs=15)

if __name__ == "__main__":
        main()