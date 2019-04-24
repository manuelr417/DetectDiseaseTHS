from ths.nn.sequences.tweetsimilarity import TweetSimilaryBasic
from ths.utils.files import GloveEmbedding

def main():
    G = GloveEmbedding("data/glove.6B.50d.txt")
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    M = TweetSimilaryBasic(72, G, 5, 3)
    M.build()
    M.summary()
    M.plot("data/model2")


#joderme
if __name__ == "__main__":
    main()