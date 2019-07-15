from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np

def train_word2vec(sentence_matrix, vocabulary_inv, num_features=300, min_word_count=1, context=10):
    embedding_model = word2vec.Word2Vec.load("ko.bin")
    embedding_weight = {key: embedding_model[word] if word in embedding_model else np.random.uniform(-0.25, 0.25, embedding_model.vector_size) for key, word in vocabulary_inv.items()}

    return embedding_weight
