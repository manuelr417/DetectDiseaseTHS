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
    for line in data:
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


def setCentroidsRandom(data, max_len):
    centroid1 = random.choice(data)
    centroid2 = random.choice(data)
    centroid3 = random.choice(data)
    print("centroid 1: ", centroid1)
    print("centroid 2: ", centroid2)
    print("centroid 3: ", centroid3)
    if centroid1 != centroid2 and  centroid1 != centroid3 and centroid2 != centroid3 :
        SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
        c1 = SE.map_sentence((centroid1.lower()), max_len=max_len)
        c2 = SE.map_sentence((centroid2.lower()), max_len=max_len)
        c3 = SE.map_sentence((centroid3.lower()), max_len=max_len)
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


def clusterAsignmentv2(finaldata, c1, c2, c3, tweet):
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
            dictionary[tweet[i]] = 1
        elif(m==S2):
            cluster2.append(line)
            dictionary[tweet[i]] = 2
        else:
            cluster3.append(line)
            dictionary[tweet[i]] = 3
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
    sim1 = matrix_cosine_similary(c1, oldc1)
    sim2 = matrix_cosine_similary(c2, oldc2)
    sim3 = matrix_cosine_similary(c3, oldc3)
    print("Sim1: ", sim1)
    print("Distance C1 - oldC1: ", distance_similarity_matrix(sim1))
    print("Distance C2 - oldC2: ", distance_similarity_matrix(sim2))
    print("Distance C3 - oldC3: ", distance_similarity_matrix(sim3))
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

def epsilonSimilarity(cl1, oldcl1, cl2, oldcl2, cl3, oldcl3):
    print("Entered in empsilonSimilarity")
    EPSILON_VALUE = 0.01
    if np.array(cl1).size > 0 and np.array(oldcl1).size > 0 and np.array(cl2).size > 0 and np.array(oldcl2).size > 0 and np.array(cl3).size > 0 and np.array(oldcl3).size > 0:
        EM1 = np.ones(cl1[0].shape) * EPSILON_VALUE
        EM2 = np.ones(cl2[0].shape) * EPSILON_VALUE
        EM3 = np.ones(cl3[0].shape) * EPSILON_VALUE

        #cluster 1
        if len(cl1) == len(oldcl1):
            ans1 = ((np.array(cl1)- np.array(oldcl1)).all() < EM1 ).all()
        else:
            ans1 = False
        print("Answer Epsilon 1: ", ans1)

        #cluster 2
        if len(cl2) == len(oldcl2):
            ans2 = ((np.array(cl2) - np.array(oldcl2)).all() < EM2).all()
        else:
            ans2 = False
        print("Answer Epsilon 1: ", ans2)

        #cluster 3
        if len(cl3) == len(oldcl3):
            ans3 = ((np.array(cl3) - np.array(oldcl3)).all() < EM3).all()
        else:
            ans3 = False
        print("Answer Epsilon 1: ", ans3)

    else:
        ans1 = False
        ans2 = False
        ans3 = False

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
    data = []

    dictionary1  = {}
    dictionary2  = {}
    try:
        datafile = open("data/small_tweets.txt", "r", encoding='utf-8')
        with datafile as f:
            for line in f:
                data.append(line.strip())
                data.append(line)
    except Exception as e:
        print(e)
    max_len = get_max_len(data)

    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)
    finaldata = []
    for line in data:
        emb = SE.map_sentence(line.lower(), max_len=max_len)
        finaldata.append(emb)
        dictionary1[line] = emb
        finaldata.append(SE.map_sentence(line.lower(), max_len=max_len))

    #c1, c2, c3 = setCentroidsFromLabel("data/clusterone.txt", "data/clustertwo.txt", "data/clusterthree.txt", max_len)
    c1, c2, c3 = setCentroidsRandom(data, max_len)
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
        cluster1, cluster2, cluster3, dictionary2 = clusterAsignmentv2(finaldata, oldc1, oldc2, oldc3, data)
        #cluster1, cluster2, cluster3 = clusterAsignment(finaldata, oldc1, oldc2, oldc3)
        print("Step 2: passed")

        #Step 3: Move Clusters
        print("Step 3: Starting")
        c1, c2, c3 = moveCentroid(cluster1, cluster2, cluster3, oldc1, oldc2, oldc3)
        n+=1
        print("Step 3: passed")
        print("Simulation #: ", n, " finished")

        #Step 4: Calculate similarity of clusters and centroids
        #result = centroidsSimilarity(c1, oldc1, c2, oldc2, c3, oldc3)
        if n < 20:
            result = clustersSimilarity(cluster1, oldcluster1, cluster2, oldcluster2, cluster3, oldcluster3)
        else:
            result = epsilonSimilarity(cluster1, oldcluster1, cluster2, oldcluster2, cluster3, oldcluster3)

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

    #5: Test Cluster
    isDisease = "make america great again"
    isnotDisease = "sawyer premium insect repellent w20 picaridin lotion 50 pack"
    neither = "the flu is poking holes in hospital cybersecurity and a shot cannot save you"
    SE = SentenceToEmbeddingWithEPSILON(word_to_idx, idx_to_word, embedding)

    matrix1 = SE.map_sentence(isDisease.lower(), max_len=max_len)
    matrix2 = SE.map_sentence(isnotDisease.lower(), max_len=max_len)
    matrix3 = SE.map_sentence(neither.lower(), max_len=max_len)

    S1 = distance_similarity_matrix(matrix_cosine_similary(c1, matrix1))
    S2 = distance_similarity_matrix(matrix_cosine_similary(c2, matrix1))
    S3 = distance_similarity_matrix(matrix_cosine_similary(c3, matrix1))
    m = min(S1, S2, S3)

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

    if (m == S1):
        print("Asignado a Cluster 1: ", neither)
    elif (m == S2):
        print("Asignado a Cluster 2: ", neither)
    else:
        print("Asignado a Cluster 3: ", neither)

    count = 0
    print("cluster 2:")
    for k,v in dictionary2.items():
        count+=1
        if (v == 2):
            print(k)
        if count == 50:
            break
        print("Asignado a Cluster 3: ", neither)
