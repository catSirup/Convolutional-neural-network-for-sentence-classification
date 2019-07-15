# Convolutional neural network for sentence classification

이 코드는 Yoon Kim 님의 ["Convolutional Neural Networks for Sentence Classification"](https://www.aclweb.org/anthology/D14-1181) 을 기반으로 구현했습니다. 핵심은 사전 학습한 벡터를 사용할 경우, 매우 간단한 CNN 구조로도 문장 분류에서 상당한 효율을 뽑아낼 수 있다는 것입니다.

## 참조 문서
- [NLP를 위한 CNN (2): Convolutional Neural Network for Sentence Classification](https://reniew.github.io/26/)
- [Alexander Rakhli's Keras-based implementation](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras)
- [[Keras] KoNLPy를 이용한 한국어 영화 리뷰 감정 분석](https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html)

## 개발 환경
- Python 3.6
- Tensorflow 2.0

## Dataset

데이터 셋으로는 [Naver sentiment movie corpus v1.0](https://github.com/e9t/nsmc)을 사용했으며, 한국어로 된 영화 리뷰 데이터입니다. 

## Sentence Tokenize

위의 논문에서는 google news를 활용해 word2vec으로 사전 학습된 임베딩 벡터를 사용했으나, 영어로 사전 학습된 것이기 때문에, 현재 데이터 셋과는 맞지 않아 [박규병 님이 만드신 한국어로 학습된 모델](https://github.com/Kyubyong/wordvectors)을 사용하였습니다.

## 간단한 설명

1. Tensorflow 2.0부터 합쳐진 keras를 [구글에서 본격적으로 밀고 있기 때문에](https://developers-kr.googleblog.com/2019/01/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0.html) keras를 적극 활용하는 방식으로 제작했습니다. 
    - Tensorflow에 비해 모델 구현이 직관적이며 보기가 편하기 때문에 keras를 활용했습니다.

2. 하이퍼 파라미터는 다음과 같습니다
    - ReLU 함수
    - 100 feature map으로 filter의 크기는 3,4,5로 지정
    - Dropout 확률 : 0.5
    - 미니 배치 크기 : 50
    - 파라미터 업데이트 : adadelta

3. CNN-rand / CNN-non-static / CNN-static
    - **CNN-rand** : baseline값으로 사용하기 위해 사용. 모든 단어 벡터를 임의의 값으로 초기화해서 사용했다.
    - **CNN-static** : 앞서 말한 사전 학습된 word2vec 단어 벡터를 사용한 모델이다.
    - **CNN-non-static** : 위의 모델과 같이 학습된 벡터를 사용했지만 각 task에서 벡터값은 update된다.

## CNN-non-static with adam

<pre>
Epoch 1/10
 - 985s - loss: 0.4867 - acc: 0.7572 - val_loss: 0.4057 - val_acc: 0.8187
Epoch 2/10
 - 1007s - loss: 0.3931 - acc: 0.8206 - val_loss: 0.3878 - val_acc: 0.8265
Epoch 3/10
 - 970s - loss: 0.3646 - acc: 0.8371 - val_loss: 0.3819 - val_acc: 0.8259
Epoch 4/10
 - 979s - loss: 0.3471 - acc: 0.8468 - val_loss: 0.3812 - val_acc: 0.8289
Epoch 5/10
 - 991s - loss: 0.3333 - acc: 0.8541 - val_loss: 0.3758 - val_acc: 0.8300
Epoch 6/10
 - 997s - loss: 0.3208 - acc: 0.8607 - val_loss: 0.3816 - val_acc: 0.8295
Epoch 7/10
 - 961s - loss: 0.3108 - acc: 0.8647 - val_loss: 0.3835 - val_acc: 0.8284
Epoch 8/10
 - 881s - loss: 0.3005 - acc: 0.8688 - val_loss: 0.3862 - val_acc: 0.8298
Epoch 9/10
 - 876s - loss: 0.2919 - acc: 0.8738 - val_loss: 0.4100 - val_acc: 0.8288
Epoch 10/10
 - 875s - loss: 0.2831 - acc: 0.8776 - val_loss: 0.3906 - val_acc: 0.8285
</pre>

## CNN-static with adam

<pre>
Epoch 1/10
 - 731s - loss: 0.6245 - acc: 0.6434 - val_loss: 0.5579 - val_acc: 0.7015
Epoch 2/10
 - 777s - loss: 0.5759 - acc: 0.6896 - val_loss: 0.5404 - val_acc: 0.7207
Epoch 3/10
 - 778s - loss: 0.5617 - acc: 0.6999 - val_loss: 0.5233 - val_acc: 0.7279
Epoch 4/10
 - 786s - loss: 0.5530 - acc: 0.7073 - val_loss: 0.5210 - val_acc: 0.7333
Epoch 5/10
 - 786s - loss: 0.5478 - acc: 0.7096 - val_loss: 0.5142 - val_acc: 0.7364
Epoch 6/10
 - 789s - loss: 0.5427 - acc: 0.7142 - val_loss: 0.5088 - val_acc: 0.7373
Epoch 7/10
 - 788s - loss: 0.5402 - acc: 0.7167 - val_loss: 0.5110 - val_acc: 0.7406
Epoch 8/10
 - 793s - loss: 0.5354 - acc: 0.7201 - val_loss: 0.5088 - val_acc: 0.7405
Epoch 9/10
 - 780s - loss: 0.5343 - acc: 0.7209 - val_loss: 0.5065 - val_acc: 0.7432
Epoch 10/10
 - 791s - loss: 0.5319 - acc: 0.7222 - val_loss: 0.4999 - val_acc: 0.7462
</pre>

## CNN-rand with adam
<pre>
Epoch 1/10
180000/180000 - 1072s - loss: 0.4160 - acc: 0.8019 - val_loss: 0.3810 - val_acc: 0.8199
Epoch 2/10
180000/180000 - 1066s - loss: 0.3362 - acc: 0.8520 - val_loss: 0.3808 - val_acc: 0.8243
Epoch 3/10
180000/180000 - 1069s - loss: 0.2881 - acc: 0.8760 - val_loss: 0.4039 - val_acc: 0.8213
Epoch 4/10
180000/180000 - 1071s - loss: 0.2512 - acc: 0.8932 - val_loss: 0.4162 - val_acc: 0.8181
Epoch 5/10
180000/180000 - 1070s - loss: 0.2234 - acc: 0.9056 - val_loss: 0.4504 - val_acc: 0.8138
Epoch 6/10
180000/180000 - 1039s - loss: 0.2025 - acc: 0.9152 - val_loss: 0.4789 - val_acc: 0.8154
Epoch 7/10
180000/180000 - 1035s - loss: 0.1872 - acc: 0.9209 - val_loss: 0.5084 - val_acc: 0.8134
Epoch 8/10
180000/180000 - 1034s - loss: 0.1772 - acc: 0.9250 - val_loss: 0.5325 - val_acc: 0.8101
Epoch 9/10
180000/180000 - 1030s - loss: 0.1686 - acc: 0.9289 - val_loss: 0.5442 - val_acc: 0.8102
Epoch 10/10
180000/180000 - 1031s - loss: 0.1605 - acc: 0.9320 - val_loss: 0.5656 - val_acc: 0.8109
</pre>

## CNN-non-static with adadelta

<pre>
Epoch 1/10
 - 1789s - loss: 0.5819 - acc: 0.6855 - val_loss: 0.4978 - val_acc: 0.7563
Epoch 2/10
 - 971s - loss: 0.4991 - acc: 0.7516 - val_loss: 0.4629 - val_acc: 0.7783
Epoch 3/10
 - 970s - loss: 0.4706 - acc: 0.7709 - val_loss: 0.4389 - val_acc: 0.7898
Epoch 4/10
 - 971s - loss: 0.4527 - acc: 0.7827 - val_loss: 0.4327 - val_acc: 0.7954
Epoch 5/10
 - 971s - loss: 0.4430 - acc: 0.7888 - val_loss: 0.4233 - val_acc: 0.7988
Epoch 6/10
 - 971s - loss: 0.4352 - acc: 0.7939 - val_loss: 0.4260 - val_acc: 0.8027
Epoch 7/10
 - 971s - loss: 0.4296 - acc: 0.7968 - val_loss: 0.4179 - val_acc: 0.8015
Epoch 8/10
 - 1091s - loss: 0.4251 - acc: 0.8003 - val_loss: 0.4125 - val_acc: 0.8060
Epoch 9/10
 - 1133s - loss: 0.4199 - acc: 0.8043 - val_loss: 0.4092 - val_acc: 0.8089
Epoch 10/10
 - 1214s - loss: 0.4186 - acc: 0.8044 - val_loss: 0.4117 - val_acc: 0.8074
</pre>

## CNN-rand with adadelta

<pre>
Epoch 1/10
 - 1326s - loss: 0.4602 - acc: 0.7678 - val_loss: 0.4125 - val_acc: 0.8046
Epoch 2/10
 - 1239s - loss: 0.3932 - acc: 0.8200 - val_loss: 0.3985 - val_acc: 0.8153
Epoch 3/10
 - 1089s - loss: 0.3757 - acc: 0.8316 - val_loss: 0.3916 - val_acc: 0.8202
Epoch 4/10
 - 1116s - loss: 0.3629 - acc: 0.8398 - val_loss: 0.3866 - val_acc: 0.8214
Epoch 5/10
 - 1028s - loss: 0.3527 - acc: 0.8476 - val_loss: 0.3830 - val_acc: 0.8246
Epoch 6/10
 - 1000s - loss: 0.3428 - acc: 0.8523 - val_loss: 0.3845 - val_acc: 0.8249
Epoch 7/10
 - 1022s - loss: 0.3340 - acc: 0.8581 - val_loss: 0.3812 - val_acc: 0.8249
Epoch 8/10
 - 1011s - loss: 0.3257 - acc: 0.8630 - val_loss: 0.3837 - val_acc: 0.8247
Epoch 9/10
 - 1000s - loss: 0.3184 - acc: 0.8670 - val_loss: 0.3917 - val_acc: 0.8253
Epoch 10/10
 - 983s - loss: 0.3117 - acc: 0.8701 - val_loss: 0.3870 - val_acc: 0.8284
 </pre>

 ## CNN-static with adadelta

<pre>
 Epoch 1/10
 - 700s - loss: 0.6254 - acc: 0.6411 - val_loss: 0.5680 - val_acc: 0.7005
Epoch 2/10
 - 727s - loss: 0.5834 - acc: 0.6851 - val_loss: 0.5433 - val_acc: 0.7255
Epoch 3/10
 - 731s - loss: 0.5677 - acc: 0.6979 - val_loss: 0.5298 - val_acc: 0.7279
Epoch 4/10
 - 719s - loss: 0.5574 - acc: 0.7061 - val_loss: 0.5230 - val_acc: 0.7354
Epoch 5/10
 - 729s - loss: 0.5507 - acc: 0.7109 - val_loss: 0.5214 - val_acc: 0.7384
Epoch 6/10
 - 732s - loss: 0.5461 - acc: 0.7145 - val_loss: 0.5221 - val_acc: 0.7365
Epoch 7/10
 - 727s - loss: 0.5444 - acc: 0.7168 - val_loss: 0.5125 - val_acc: 0.7441
Epoch 8/10
 - 725s - loss: 0.5417 - acc: 0.7189 - val_loss: 0.5095 - val_acc: 0.7401
Epoch 9/10
 - 729s - loss: 0.5392 - acc: 0.7207 - val_loss: 0.5048 - val_acc: 0.7504
Epoch 10/10
 - 738s - loss: 0.5368 - acc: 0.7215 - val_loss: 0.5348 - val_acc: 0.7442
 </pre>