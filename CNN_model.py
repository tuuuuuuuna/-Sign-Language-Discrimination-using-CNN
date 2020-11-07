import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
# import keras.backend.tensorflow_backend as K

import tensorflow as tf

"""""""""
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)
"""""""""
# GPU를 사용한 연산을 해보려 했으나 사용가능한 GPU가 없어서 실패

x_train, x_test, y_train, y_test = np.load('./numpy_data/multi_image_data.npy',allow_pickle=True)
# np의 load 메소드를 사용하여, Image.py에서 만들어진
# multi_image_data.npy를 불러와서 train, test set으로 분리하여 저장

"""""""""
print(x_train.shape)
print(x_train.shape[0])
print(x_test.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])
"""""""""
# train set과 test set이 바르게 나누어 졌는지 확인


categories=[chr(ord("A")+i) for i in range(0,26)]
nb_classes = len(categories)

#Generation
x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255
# train set, test set의 픽셀 정보를 255로 나누어 정규화


model = Sequential()
# 모델을 Sequential 객체를 이용하여 생성
model.add(Conv2D(32, (3, 3), padding = "same", input_shape = x_train.shape[1:], activation = 'relu'))
# 필터수 32개, 커널사이즈 (3,3), 스트라이드 (1,1) default값 사용
# 가장자리 정보들이 사라지는 것을 방지하기위해 패딩을 same으로 사용
# 다중클래스 분류 문제이므로 Relu사용

model.add(MaxPooling2D(pool_size = (2,2)))
# (2,2)의 윈도우를 최대값 기준으로 풀링

model.add(Dropout(0.25))
# 오버피팅을 막기 위한 Dropout의 실행



model.add(Flatten())
# 입력 데이터의 Flatten
model.add(Dense(256, activation = 'relu'))
# 256개의 노드를 가지는 은닉층 생성
# 마찬가지로 다중클래스분류이므로 Relu 사용
model.add(Dropout(0,5))
# 다시한번 Dropout
model.add(Dense(nb_classes, activation = 'softmax'))
# 카테고리(nb_classes) 만큼의 노드를 가지는 출력층 생성
# 출력층이므로 활성화 함수로 Softmax를 사용
# 이후 그 값을 확률처럼 사용

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics =['accuracy'])
# 손실함수의 값은 categorical_crossentropy, 최적화 방법은 Adam,
# 측정 값으로는 accuracy 사용

model_dir = './model'
# 모델이 저장되어있는 폴더의 주소값

if not os.path.exists(model_dir):
    os.mkdir(model_dir)
# 모델 폴더의 주소값이 존재하지 않을때, 새로운 폴더 생성

model_path = model_dir + '/multi_img_classification.model'
# 모델의 주소값을 model_path에 저장
checkpoint = ModelCheckpoint(filepath = model_path, monitor = 'val_loss', verbose = 1, save_best_only = True)
# modelcheckpoint를 만들어서 verbose=1로 진행 사항의 출력여부를 정하고,
# save best only로 최고값을 갱신시에만 저장
early_stopping = EarlyStopping(monitor ='val_loss', patience = 6)
# 초기에 최고 성능의 모델이 찾아져서 더 좋은 성능의 모델이 나오지 않을 경우를 대비
# 중간에 더 좋은 성능의 모델이 6번 동안 발견되지 않으면 학습 중단

model.summary()

history = model.fit(x_train, y_train, batch_size = 32, epochs = 1,
                    validation_data = (x_test, y_test),
                    callbacks = [checkpoint, early_stopping])
# train set으로 batch size를 32로 훈련하여 history에 저장
# 이 때, 체크포인트와 조기종료 조건을 적용
print("정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))


y_vloss = history.history['val_loss']
y_loss = history.history['loss']
# 검증 loss와 훈련 loss값을 각각 y_vloss, y_loss에 저장

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker = '.', c = 'red', label = 'val_set_loss')
plt.plot(x_len, y_loss, marker = '.', c = 'blue', label = 'train_set_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()
# epochs이 진행 됨에 따라,
# 빨간 색이 검증 loss, 파란 색을 훈련 loss로 시각화