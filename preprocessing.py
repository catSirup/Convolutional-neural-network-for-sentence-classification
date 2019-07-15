import numpy as np
import os
import json
import nltk
from konlpy.tag import Okt

# 데이터를 읽어오고 문장과 라벨을 분류
def load_data_and_label(path):
    data = []
    labels = []
    f = open(path, 'r')
    # 첫 줄을 날림.
    f.readline()
    for line in f.readlines():
        data.append(line.split('\t')[1].replace(" ", ""))
        labels.append(line.split('\t')[2][:-1])
    
    data = np.asarray(data)
    labels = np.asarray(labels)

    return data, labels

# 단어 토큰화 하기
def tokenize(sentence):
    okt = Okt()
    tokenized_sentence = []

    # 우선 단어의 기본형으로 모두 살리고, 명사, 동사, 영어만 담는다.
    # 그냥 nouns로 분리하는 것보다 좀 더 정확하고 많은 데이터를 얻을 수 있다.
    for line in sentence:
        result = []
        temp_sentence = okt.pos(line, norm=True, stem=True) # 먼저 형태소 분리해서 리스트에 담고
        print(temp_sentence)
        for i in temp_sentence:                             
            if (i[1] == 'Noun' or i[1] == 'Adjective' or i[1] == 'Alpha'):                  
                result.append(i[0])
            
        tokenized_sentence.append(result)

    return tokenized_sentence

# 가장 긴 단어의 길이를 맞추고 그 외에는 패딩 단어로 바꿈.
def pad_sequence(sentences, padding_word="<PAD/>"):
    maxlen = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = maxlen - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

# 단어들을 인덱스로 바꿔줌.
def build_vocab(sentences):
    tokens = [t for d in sentences for t in d]
    text = nltk.Text(tokens, name='NSMC')
    word_count = text.vocab()
    vocabulary_inv = [x[0] for x in word_count.most_common()]
    vocabulary = {x:i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

# 문장이 너무 많아 토큰화하는 데 시간이 오래 걸려
# json으로 미리 파일을 만들어 놓고, 로드하는 방식으로 만들었다.
def load_data():
    ### 사전에 토큰화해서 저장함 ### 
    # data, labels = load_data_and_label('./Datasets/ratings.txt')    

    # data = tokenize(data)
    # datas = []
    # for i in range(len(data)):
    #     datas.append([data[i], labels[i]])

    # with open('data.json', 'w', encoding='utf-8') as make_file:
    #     json.dump(datas, make_file, ensure_ascii=False, indent='\t')

    ### 저장된 데이터를 불러와서 작업 ### 
    with open('data.json') as f:
        data = json.load(f)

    sentence = []
    labels = []
    for text in data:
        sentence.append(text[0])
        labels.append(text[1])

    sentence_padded = pad_sequence(sentence)
    vocabulary, vocabulary_inv = build_vocab(sentence_padded)
    x, y = build_input_data(sentence_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]