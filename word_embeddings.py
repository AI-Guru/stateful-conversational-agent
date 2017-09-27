import os
import json
import nltk
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import pickle

# Download the word embeddings from these locations and put them into the
# "embeddings"-folder.
# https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
# http://devmount.github.io/GermanWordEmbeddings/

# File names for the word embedding. Pick one!
# Note that this might take some time. Depending on the file size.
#word_vectors_model_filename = "embeddings/wiki.de.vec" # FastText. Huge.
word_vectors_model_filename = "embeddings/german.model" # TU Berlin. Smaller.

# The model that will hold the word vectors.
word_vectors_model = None

def main():
    load_word_vectors()
    test_word_vectors()

# Word vectors come either in a plain or a binary format. Attempt both and hope
# that one works.
def load_word_vectors():
    print("Loading word vectors from",  word_vectors_model_filename + "...")
    word_vectors_model_size = os.path.getsize(word_vectors_model_filename) / (1024.0 * 1024.0)
    print("File size is {0:.2f}MB".format(word_vectors_model_size))
    print("Be patient! This might take a while...")
    global word_vectors_model
    try:
        print("Attempting do load as vector-format...")
        word_vectors_model = KeyedVectors.load_word2vec_format(word_vectors_model_filename, binary=False)
        print("Success!")
    except:
        print("Failed!")
        try:
            print("Attempting do load as binary-format...")
            word_vectors_model = KeyedVectors.load_word2vec_format(word_vectors_model_filename, binary=True)
        except:
            print("Failed!")
            exit(0)

    print("Success!")


# Tests the word vector model. Note that not all these methods might be
# successful.
def test_word_vectors():
    print("Doing some tests...")

    # Statistics.
    print("Vocabulary size", len(list(word_vectors_model.wv.vocab.keys())))
    print("Word vector length:", len(word_vectors_model.wv["Mann"]))

    # Getting one word vector.
    print("Word vector for 'Mann'", word_vectors_model.wv["Mann"])

    # Doing word embedding mathemagic. Similarities.
    print("Similarity:", word_vectors_model.similarity("Bundeskanzler", "Bundeskanzlerin"))
    positive = ["Frau", "Bundeskanzler"]
    negative = ["Bundeskanzlerin"]
    print("Most similar:", word_vectors_model.most_similar(positive=positive, negative=negative, topn=1))
    print("Most similar cosmul:", word_vectors_model.most_similar_cosmul(positive=positive, negative=negative, topn=1))

    # Finding the word that does not match.
    print("Does not match:", word_vectors_model.doesnt_match("Frühstück Bundeskanzler Mittagessen Abendessen".split()))

    # Finding word from a vector.
    your_word_vector = word_vectors_model.wv["Merkel"]
    print("Word from a vector:", word_vectors_model.most_similar(positive=[your_word_vector], topn=1))

    print("Most similar top-10:", word_vectors_model.most_similar(positive=["Merkel"], topn=10))

if __name__ == "__main__":
    main()
