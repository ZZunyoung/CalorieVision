import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# 하이퍼파라미터 설정 (수업자료: 딥러닝 모델 구성 시 기본적으로 설정하는 파라미터들)
IMG_SIZE = 256  # 이미지 크기를 리사이징된 이미지 크기(256x256)에 맞게 설정
BATCH_SIZE = 32  # 배치 크기: 한 번에 학습할 이미지 샘플 수 (Chapter 2 배치 데이터)
EPOCHS = 20  # 학습 데이터 전체를 학습하는 횟수 (에포크 수)
data_dir = './assets/resized_image'  # GitHub 프로젝트의 assets/resized_image 폴더 경로

# 데이터 로드 함수 정의 (수업자료: 이미지 데이터를 다차원 배열로 표현하는 방법, Chapter 2 텐서 표현)
def load_data(data_dir):
    """
    지정된 디렉토리에서 이미지를 불러와서 리스트에 추가하고, 
    각 이미지에 대해 클래스 라벨을 지정하여 numpy 배열로 반환.
    """
    images = []  # 이미지 데이터를 저장할 리스트
    labels = []  # 이미지의 클래스 라벨을 저장할 리스트
    class_names = os.listdir(data_dir)  # 각 폴더 이름이 클래스 이름으로 간주됨

    # 클래스별 폴더를 순회하며 이미지를 불러옴
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)  # 이미지를 불러옴
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # 지정된 크기로 리사이징
                images.append(img)
                labels.append(label)  # 해당 클래스에 대한 라벨 추가
    return np.array(images), np.array(labels), class_names

# 데이터 불러오기 및 전처리 (수업자료: 데이터를 훈련 및 검증 세트로 분할하는 방법, Chapter 3)
images, labels, class_names = load_data(data_dir)
NUM_CLASSES = len(class_names)  # 클래스의 수를 자동으로 계산
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# 데이터 증강 설정 (수업자료: 데이터 증강으로 모델 일반화 성능 향상, Chapter 3)
train_datagen = ImageDataGenerator(
    rescale=1./255,            # 픽셀 값을 0-1 범위로 정규화 (Chapter 2 이미지 정규화)
    rotation_range=15,          # 이미지를 최대 15도까지 회전 (데이터 증강의 예시)
    width_shift_range=0.1,      # 이미지를 좌우로 최대 10% 이동
    height_shift_range=0.1,     # 이미지를 상하로 최대 10% 이동
    horizontal_flip=True        # 이미지를 좌우로 뒤집음
)
val_datagen = ImageDataGenerator(rescale=1./255)  # 검증 데이터는 정규화만 수행

# 데이터 생성기 설정 (수업자료: 배치 데이터로 모델을 학습하는 과정, Chapter 3)
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# CNN 모델 구성 (수업자료: CNN 아키텍처와 각 레이어의 역할, Chapter 4)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # 첫 번째 Conv 레이어 (Chapter 4 컨볼루션)
    MaxPooling2D((2, 2)),  # 풀링 레이어로 이미지 크기 감소 (Chapter 4 풀링 레이어)
    Conv2D(64, (3, 3), activation='relu'),  # 두 번째 Conv 레이어
    MaxPooling2D((2, 2)),  # 풀링 레이어
    Conv2D(128, (3, 3), activation='relu'),  # 세 번째 Conv 레이어
    MaxPooling2D((2, 2)),  # 풀링 레이어
    Flatten(),  # 2D 데이터를 1D로 변환하여 Dense 레이어에 입력
    Dense(512, activation='relu'),  # 완전 연결된 레이어로 특징 학습 (Chapter 4 밀집 연결)
    Dropout(0.5),  # 과적합 방지를 위해 일부 뉴런을 랜덤하게 비활성화 (Chapter 4 드롭아웃)
    Dense(NUM_CLASSES, activation='softmax')  # 소프트맥스 활성화로 클래스별 확률 출력 (Chapter 4 소프트맥스)
])

# 모델 컴파일 설정 (수업자료: 손실 함수와 옵티마이저, Chapter 2)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 (수업자료: 모델 학습 과정에서 손실과 정확도 모니터링, Chapter 4)
history = model.fit(
    train_generator,            # 학습 데이터 생성기
    epochs=EPOCHS,              # 전체 데이터셋을 20번 학습
    validation_data=val_generator  # 검증 데이터 생성기
)

# 학습된 모델 저장
model.save('classification_model.h5')
print("Model saved as 'classification_model.h5'")
