from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToEmbeddingWithEPSILON
import ths.utils.similarity as sim

def main():
    G = GloveEmbedding("data/glove.6B.50d.txt")
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    print("embedding shape: ", embedding.shape)
    print("idx hello: ", word_to_idx["hello"])
    print("word 20: ", idx_to_word[20])
    e = embedding[word_to_idx["hello"]]
    print("embedding hello: ", e)
    print("e.shape: ", e.shape)
    print("<UNK>: ", word_to_idx['<unk>'])
    print("embedding: <UNK>: ", embedding[word_to_idx['<unk>']])

    you = embedding[word_to_idx['you']]
    he = embedding[word_to_idx['he']]
    ise = embedding[word_to_idx['is']]
    crazy = embedding[word_to_idx['crazy']]
    nuts =  embedding[word_to_idx['nuts']]

    print("embedding of you: ", you)
    print("embedding of he: ", he)
    print("embedding of ise: ", ise)
    print("embedding of crazy: ", crazy)
    print("embedding of nuts: ", nuts)

    tweet1 = "You are crazy"
    tweet2 = "You are nuts"
    tweet3 = "He is crazy"
    tweet4 = "You are lazy"
    tweet5 = "You are crazy man"
    tweet6 = "Yes You are crazy"
    tweet7 = "The fast train"

    mapper = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
    emb1 = mapper.map_sentence(tweet1.lower(), 4)
    emb2  = mapper.map_sentence(tweet2.lower(), 4)
    emb3  = mapper.map_sentence(tweet3.lower(), 4)
    emb4  = mapper.map_sentence(tweet4.lower(), 4)
    emb5  = mapper.map_sentence(tweet5.lower(), 4)
    emb6  = mapper.map_sentence(tweet6.lower(), 4)
    emb7  = mapper.map_sentence(tweet7.lower(), 4)


    print("Distance tweet1 vs tweet2: ")
    print("Frobenious: ", sim.Frobenius_Distance(emb1, emb2))
    print("Cos Tri: ", sim.TriUL_sim(emb1, emb2))
    print("Distance tweet1 vs tweet3: ")
    print("Frobenious: ", sim.Frobenius_Distance(emb1, emb3))
    print("Cos Tri: ", sim.TriUL_sim(emb1, emb3))

    print("Distance tweet2 vs tweet3: ")
    print("Frobenious: ", sim.Frobenius_Distance(emb2, emb3))
    print("Cos Tri: ", sim.TriUL_sim(emb2, emb3))

    print("Distance tweet1 vs tweet4: ")
    print("Frobenious: ", sim.Frobenius_Distance(emb1, emb4))
    print("Cos Tri: ", sim.TriUL_sim(emb1, emb4))

    print("Distance tweet1 vs tweet5: ")
    print("Frobenious: ", sim.Frobenius_Distance(emb1, emb5))
    print("Cos Tri: ", sim.TriUL_sim(emb1, emb5))

    print("Distance tweet1 vs tweet6: ")
    print("Frobenious: ", sim.Frobenius_Distance(emb1, emb6))
    print("Cos Tri: ", sim.TriUL_sim(emb1, emb6))

    print("Distance tweet1 vs tweet7: ")
    print("Frobenious: ", sim.Frobenius_Distance(emb1, emb7))
    print("Cos Tri: ", sim.TriUL_sim(emb1, emb7))


    print("Embedding tweet1: ")
    print(emb1)
    print("Embedding tweet6: ")
    print(emb6)

if __name__ == "__main__":
    main()