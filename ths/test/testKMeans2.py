from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbeddingWithEPSILON, SentenceToEmbedding, PadSentences
from ths.utils.similarity import matrix_cosine_similary, distance_similarity_matrix
import numpy as np
import random


def main():
    pass

def max_len_three(file1, file2, file3):
    try:
        data_one = open(file1, "r", encoding='utf-8')
        data_two = open(file2, 'r', encoding='utf-8')
        data_three = open(file3, 'r', encoding='utf-8')
    except Exception as e:
        print(e)
    else:
        max_len1 = 0
        max_len2 = 0
        max_len3 = 0
        SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
        with data_one:
            for line in data_one:
                matrix = SE.map_sentence(line.lower())
                tweet_len = len(matrix)
                if tweet_len > max_len1:
                    max_len1 = tweet_len
        with data_two:
            for line in data_two:
                matrix = SE.map_sentence(line.lower())
                tweet_len = len(matrix)
                if tweet_len > max_len2:
                    max_len2 = tweet_len
        with data_three:
            for line in data_three:
                matrix = SE.map_sentence(line.lower())
                tweet_len = len(matrix)
                if tweet_len > max_len3:
                    max_len3 = tweet_len
        return max(max_len1, max_len2, max_len3)

def get_max_len(file):
    max_len = 0
    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
    for line in file:
        matrix = SE.map_sentence(line.lower())
        tweet_len = len(matrix)
        if tweet_len > max_len:
            max_len = tweet_len
    return max_len

def setCentroidsFromLabel(file1, file2, file3, max_len):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    try:
        data_one = open(file1, "r", encoding='utf-8')
        data_two = open(file2, 'r', encoding='utf-8')
        data_three = open(file3, 'r', encoding='utf-8')
    except Exception as e:
        print(e)
    else:
        SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
        for line in data_one:
            matrix = SE.map_sentence((line.lower()), max_len=max_len)
            cluster1.append(matrix)
        with data_two:
            for line in data_two:
                matrix = SE.map_sentence((line.lower()), max_len=max_len)
                cluster2.append(matrix)
        with data_three:
            for line in data_three:
                matrix = SE.map_sentence((line.lower()), max_len=max_len)
                cluster3.append(matrix)
        #Set centroids
        centroid1 = (1 / len(cluster1)) * np.sum(cluster1, axis=0)
        centroid2 = (1 / len(cluster2)) * np.sum(cluster2, axis=0)
        centroid3 = (1 / len(cluster3)) * np.sum(cluster3, axis=0)
    return centroid1, centroid2, centroid3


def setCentroidsRandom(finaldata, max_len):
    i = random.choice(list(enumerate(finaldata)))[0]
    c1 = finaldata[i]
    j = random.choice(list(enumerate(finaldata)))[0]
    c2 = finaldata[j]
    k = random.choice(list(enumerate(finaldata)))[0]
    c3 = finaldata[k]
    print("centroid 1: ", data[i])
    print("centroid 2: ", data[j])
    print("centroid 3: ", data[k])
    if (i != j) and (i!= k) and (k != j):
        return c1, c2, c3
    else:
        setCentroidsRandom(data, max_len)


def clusterAsignment(finaldata, c1, c2, c3):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for line in finaldata:
        sim1 = matrix_cosine_similary(c1, line)
        sim2 = matrix_cosine_similary(c2, line)
        sim3 = matrix_cosine_similary(c3, line)
        S1 = distance_similarity_matrix(sim1)
        S2 = distance_similarity_matrix(sim2)
        S3 = distance_similarity_matrix(sim3)
        m = min(S1,S2,S3)
        if(m==S1):
            cluster1.append(line)
        elif(m==S2):
            cluster2.append(line)
        else:
            cluster3.append(line)
    return cluster1, cluster2, cluster3


def clusterAsignmentv2(finaldata, c1, c2, c3):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    dictionary = {}
    i = 0
    for line in finaldata:
        sim1 = matrix_cosine_similary(c1, line)
        sim2 = matrix_cosine_similary(c2, line)
        sim3 = matrix_cosine_similary(c3, line)
        S1 = distance_similarity_matrix(sim1)
        S2 = distance_similarity_matrix(sim2)
        S3 = distance_similarity_matrix(sim3)
        m = min(S1,S2,S3)
        if(m==S1):
            cluster1.append(line)
            dictionary[data[i]] = 1
        elif(m==S2):
            cluster2.append(line)
            dictionary[data[i]] = 2
        else:
            cluster3.append(line)
            dictionary[data[i]] = 3
        i += 1
    return cluster1, cluster2, cluster3, dictionary


def moveCentroid(cluster1, cluster2, cluster3, c1, c2, c3):
    sumc1 = np.sum(cluster1, axis=0)
    sumc2 = np.sum(cluster2, axis=0)
    sumc3 = np.sum(cluster3, axis=0)
    if(len(cluster1)>0):
        newc1 = (1.0 / len(cluster1)) * sumc1
        c1 = newc1
    if(len(cluster2)>0):
        newc2 = (1.0 / len(cluster2)) * sumc2
        c2 = newc2
    if(len(cluster3)>0):
        newc3 = (1.0 / len(cluster3)) * sumc3
        c3 = newc3
    return c1, c2, c3

#c = centroid
def centroidsSimilarity(c1, oldc1, c2, oldc2, c3, oldc3):
    if (oldc1 == c1).all() and (oldc2 == c2).all() and (oldc3 == c3).all():
        return True
    else:
        return False


#cl = cluster
def clustersSimilarity(cl1, oldcl1, cl2, oldcl2, cl3, oldcl3):
    if np.array_equal(cl1, oldcl1) and np.array_equal(cl2, oldcl2) and np.array_equal(cl3, oldcl3):
        return True
    else:
        return False


def epsilonSimilarity(c1, oldc1, c2, oldc2, c3, oldc3):
    print("Entered in empsilonSimilarity")
    EPSILON_VALUE = 0.01

    shape = np.array(c1).shape
    EM = np.ones(shape) * EPSILON_VALUE
    #cluster 1
    ans1 = (np.array(c1)- np.array(oldc1) < EM ).all()
    print("Epsilon 1: ", ans1)

    #cluster 2
    ans2 = (np.array(c2)- np.array(oldc2) < EM ).all()
    print("Epsilon 2: ", ans2)

    #cluster 3
    ans3 = (np.array(c3)- np.array(oldc3) < EM ).all()
    print("Epsilon 3: ", ans3)

    if ans1 and ans2 and ans3:
        return True
    else:
        return False


if __name__ == "__main__":
    #Step 1: Set Centroids
    print("Step 1: Starting")
    G = GloveEmbedding("../test/data/glove.twitter.27B.50d.txt")
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    S = SentenceToIndices(word_to_idx)
    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
    data = []
    dictionary1  = {}
    dictionary2  = {}
    try:
        datafile = open("data/small_tweets.txt", "r", encoding='utf-8')
        with datafile as f:
            for line in f:
                newline = " ".join(line.split())
                data.append(newline)
    except Exception as e:
        print(e)
    max_len = get_max_len(data)
    finaldata = []
    for line in data:
        emb = SE.map_sentence(line.lower(), max_len=max_len)
        finaldata.append(emb)
        dictionary1[line] = emb

    c1, c2, c3 = setCentroidsFromLabel("data/clusterone.txt", "data/clustertwo.txt", "data/clusterthree.txt", max_len)
    #c1, c2, c3 = setCentroidsRandom(finaldata, max_len)
    print("Step 1: passed")
    #Step 2: Cluster Asignment
    n = 0
    oldc1 = c1
    oldc2 = c2
    oldc3 = c3
    oldcluster1 = []
    oldcluster2 = []
    oldcluster3 = []
    next = True
    while(next):
        print("Step 2: Starting")
        #cluster1, cluster2, cluster3 = clusterAsignment(finaldata, oldc1, oldc2, oldc3)
        cluster1, cluster2, cluster3, dictionary2 = clusterAsignmentv2(finaldata, oldc1, oldc2, oldc3)
        #cluster1, cluster2, cluster3 = clusterAsignment(finaldata, oldc1, oldc2, oldc3)
        print("Step 2: passed")
        #Step 3: Move Clusters
        print("Step 3: Starting")
        c1, c2, c3 = moveCentroid(cluster1, cluster2, cluster3, oldc1, oldc2, oldc3)
        n+=1
        print("Step 3: passed")
        print("Simulation #:", n, "finished")
        #Step 4: Calculate similarity of clusters and centroids
        if n < 25:
            # result = centroidsSimilarity(c1, oldc1, c2, oldc2, c3, oldc3)
            result = clustersSimilarity(cluster1, oldcluster1, cluster2, oldcluster2, cluster3, oldcluster3)
        else:
            result = epsilonSimilarity(c1, oldc1, c2, oldc2, c3, oldc3)

        if(result):
            next = False
            print("Centroids or Cluster are equals")
        else:
            oldc1 = c1
            oldc2 = c2
            oldc3 = c3
            oldcluster1 = cluster1
            oldcluster2 = cluster2
            oldcluster3 = cluster3
            if n > 50:
                next = False
    print("End Simulation: ")

count = 0
f1 = ""
f2 = ""
f3 = ""
for k, v in dictionary2.items():
    count += 1
    if (v == 1):
        f1 += k + "\n"
    if (v == 2):
        f2 += k + "\n"
    if (v == 3):
        f3 += k + "\n"
    if count == 50:
        break

file1 = open('../test/output/cluster1.txt', 'w', encoding='utf-8')
file1.write(f1)
file1.close()
file2 = open('../test/output/cluster2.txt', 'w', encoding='utf-8')
file2.write(f2)
file2.close()
file3 = open('../test/output/cluster3.txt', 'w', encoding='utf-8')
file3.write(f3)
file3.close()

print("End writing files")



SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
matrix = finaldata[1]
S1 = distance_similarity_matrix(matrix_cosine_similary(matrix, matrix))
print(S1)
"""
    #5: Test Cluster
    isDisease = "make america great again"
    isnotDisease = "i got flu today"
    neither = "the flu is poking holes in hospital cybersecurity and a shot cannot save you"
    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)

    matrix1 = SE.map_sentence(isDisease.lower(), max_len=max_len)
    matrix2 = SE.map_sentence(isnotDisease.lower(), max_len=max_len)
    matrix3 = SE.map_sentence(neither.lower(), max_len=max_len)

    S1 = distance_similarity_matrix(matrix_cosine_similary(c1, matrix1))
    S2 = distance_similarity_matrix(matrix_cosine_similary(c2, matrix1))
    S3 = distance_similarity_matrix(matrix_cosine_similary(c3, matrix1))
    m = min(S1, S2, S3)
    print(S1, S2, S3)
    if (m == S1):
        print("Asignado a Cluster 1: ", isDisease)
    elif (m == S2):
        print("Asignado a Cluster 2: ", isDisease)
    else:
        print("Asignado a Cluster 3: ", isDisease)

    S1 = distance_similarity_matrix(matrix_cosine_similary(c1, matrix2))
    S2 = distance_similarity_matrix(matrix_cosine_similary(c2, matrix2))
    S3 = distance_similarity_matrix(matrix_cosine_similary(c3, matrix2))
    m = min(S1, S2, S3)
    print(S1, S2, S3)
    if (m == S1):
        print("Asignado a Cluster 1: ", isnotDisease)
    elif (m == S2):
        print("Asignado a Cluster 2: ", isnotDisease)
    else:
        print("Asignado a Cluster 3: ", isnotDisease)

    S1 = distance_similarity_matrix(matrix_cosine_similary(c1, matrix3))
    S2 = distance_similarity_matrix(matrix_cosine_similary(c2, matrix3))
    S3 = distance_similarity_matrix(matrix_cosine_similary(c3, matrix3))
    m = min(S1, S2, S3)
    print(S1, S2, S3)
    if (m == S1):
        print("Asignado a Cluster 1: ", neither)
    elif (m == S2):
        print("Asignado a Cluster 2: ", neither)
    else:
        print("Asignado a Cluster 3: ", neither)

"""