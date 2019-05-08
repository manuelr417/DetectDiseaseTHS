from ths.nn.sequences.process_sim import ProcessTweetsSimBasic

def main():
    print("Working:")
    P = ProcessTweetsSimBasic("data/similaritydata.csv", "data/glove.6B.50d.txt")
    P.process("trained/modelsimbasic1.json", "trained/modelsimbasic1.h5", "trained/p_modelsimbasic1.json", "trained/p_modelsimbasic1.h5", plot=False, epochs=3)

#joderme
if __name__ == "__main__":
    main()