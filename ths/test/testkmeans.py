from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbeddingWithEPSILON, PadSentences
from ths.utils.similarity import matrix_cosine_similary

def main():
    G = GloveEmbedding("../test/data/glove.6B.50d.txt")
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    #print("locon: ", word_to_idx["locon"])
    print("Length dictionary: ", len(word_to_idx))
    s = "I love New York and music locon"
    s = s.lower()
    print("Sentence: ", s)
    S = SentenceToIndices(word_to_idx)
    sentence = S.map_sentence(s)
    print("Sentence to indices: ", sentence)
    print("Padded: ", PadSentences(10).pad(sentence))
    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
    matrix1 = SE.map_sentence(s, max_len=10)

    s2 = "New york rules my man".lower()
    matrix2 = SE.map_sentence(s2, max_len=10);


    print("Matrix: ", matrix1)
    print("Matrix.shape: ", matrix1.shape)
    print("Matrix: ", matrix2)
    print("Matrix.shape: ", matrix2.shape)

    print("Self Similarity: ", matrix_cosine_similary(matrix1, matrix1))
    print("Matrix Similarity: ", matrix_cosine_similary(matrix1, matrix2))

    # print("Embedding i: ", embedding[word_to_idx["i"]])
    #
    # sentences = []
    # sentences.append("I esta malo".lower())
    # sentences.append("Love la musica salsa.".lower())
    # sentences.append("Uff, q mal te va nene".lower())
    # mapped, mlen = S.map_sentence_list(sentences)
    # print("mlen: ", mlen)
    # for s in mapped:
    #     print(s)
    #
    # print("Embedding 0: ", embedding[0])
    # print("Embedding 400001: ", embedding[400001])

if __name__ == "__main__":
    main()