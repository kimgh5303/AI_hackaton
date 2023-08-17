import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, SimpleRNN, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, LSTM, Dropout, MaxPooling2D, Bidirectional

# CIFAR-100 데이터셋 불러오기
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()  #cifar100 데이터셋을 로드

class_names = [
    'beaver', 'dolphin', 'otter', 'seal', 'whale', 
    'aquarium fish', 'flatfish', 'ray', 'shark', 'trout', 
    'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 
    'bottles', 'bowls', 'cans', 'cups', 'plates', 
    'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 
    'clock', 'computer keyboard', 'lamp', 'telephone', 'television', 
    'bed', 'chair', 'couch', 'table', 'wardrobe', 
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 
    'bear', 'leopard', 'lion', 'tiger', 'wolf', 
    'bridge', 'castle', 'house', 'road', 'skyscraper', 
    'cloud', 'forest', 'mountain', 'plain', 'sea', 
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 
    'fox', 'porcupine', 'possum', 'raccoon, skunk', 
    'crab', 'lobster', 'snail', 'spider', 'worm', 
    'baby', 'boy', 'girl', 'man', 'woman', 
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 
    'maple', 'oak', 'palm', 'pine', 'willow', 
    'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 
    'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'] 

train_images = train_images.reshape((50000, 32, 32, 3))                         #훈련데이터 50000개 32*32크기 RGB영상(3채널)
test_images = test_images.reshape((10000, 32, 32, 3))                           #테스트데이터 10000개 32*32크기 RGB영상(3채널)
train_images, test_images = train_images/255.0, test_images/255.0               #영상의 픽셀값을 0~1로 정규화

#100개의 값을 값을 원-핫 코드로 변경시켜줌
one_hot_train_labels = to_categorical(train_labels, 100)                        #100개의 값을 값을 원-핫 코드로 변경시켜줌                 
one_hot_test_labels = to_categorical(test_labels, 100)

model = Sequential()
model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='same', input_shape = (32, 32, 3)))
model.add(MaxPooling2D((2, 2)))                                                 #필터 2X2
model.add(Dropout(0.2))                                                         #Dropout, 전체노드 중 80%만 사용
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))                                                 #필터 2X2
model.add(Dropout(0.25))                                                        #Dropout, 전체노드 중 75%만 사용
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))                                                 #필터 2X2
model.add(Dropout(0.6))                                                         #Dropout, 전체노드 중 40%만 사용
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))

model.add(Reshape(target_shape = (16, 4*4)))                                    #Reshape
model.add(LSTM(32, input_shape = (16, 4*4), return_sequences = True))           #LSTM 유닛수 :32
model.add(LSTM(32, return_sequences = True))                                    #LSTM 유닛수 :32
model.add(LSTM(32, return_sequences = True))                                    #LSTM 유닛수 :32

model.add(Flatten())                                                            #벡터로 펼쳐줌
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.6))                                                         #Dropout, 전체노드 중 40%만 사용
model.add(Dense(100, activation = 'softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_images, one_hot_train_labels, epochs=100, batch_size=256)

plt.figure(figsize=(12, 4))                                                     #그래프의 가로세로 비율
plt.subplot(1, 1, 1)                                                            #1행 1열의 첫번째 위치
plt.plot(history.history['loss'], 'b--', label = 'loss')                        #loss는 파란색 점선
plt.plot(history.history['accuracy'], 'g-', label = 'Accuracy')                 #accuracy는 녹색실선
plt.xlabel('Epoch')
plt.legend()
plt.show()
print("최적화 완료!")

print("\n================test results=================")
labels=model.predict(test_images)
print("\n Accuracy: %.4f" % (model.evaluate(test_images, one_hot_test_labels)[1]))
print("=============================")

fig = plt.figure()
for i in range(15):                                                             #15장을 그림
  subplot = fig.add_subplot(3, 5, i+1)                                          #세로 2, 가로 5, i+1위치
  subplot.set_xticks([])
  subplot.set_yticks([])
  subplot.set_title('%s' %class_names[np.argmax(labels[i])])
  subplot.imshow(test_images[i].reshape((32, 32, 3)), cmap=plt.cm.brg)
plt.show()

print("=============================")