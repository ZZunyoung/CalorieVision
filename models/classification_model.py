import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class FoodImageClassifier:
    def __init__(self, img_size=224, batch_size=32, epochs=20, train_dir='./assets/resized_image/train', val_dir='./assets/resized_image/validation'):
        """
        초기 설정 및 하이퍼파라미터 정의
        Parameters:
        - img_size (int): 입력 이미지 크기 (모든 이미지가 이 크기로 리사이징됨)
        - batch_size (int): 배치 크기, 한 번에 학습할 데이터 수 (Chapter 2: 배치 데이터 개념)
        - epochs (int): 학습 에포크 수 (데이터셋을 전체 학습할 횟수)
        - train_dir (str): 학습 데이터 폴더 경로
        - val_dir (str): 검증 데이터 폴더 경로
        """
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.model = None  # CNN 모델 초기화
        self.NUM_CLASSES = None  # 클래스 개수 초기화 (데이터 로드 후 설정됨)

    def load_data(self):
        """
        학습 및 검증 데이터 디렉토리에서 클래스 이름을 불러와 클래스 개수 설정
        (데이터는 train 및 validation 디렉토리로 미리 분류되어 있어야 함)
        Returns:
        - class_names (list): 클래스 이름 리스트 (Chapter 2: 다중 클래스 데이터 구조 이해)
        """
        # 클래스 이름 불러오기 (train 폴더 내의 각 폴더 이름이 클래스 이름)
        class_names = os.listdir(self.train_dir)
        # 클래스 개수 설정하여 모델 출력층 노드 수 결정
        self.NUM_CLASSES = len(class_names)
        return class_names

    def preprocess_data(self):
        """
        학습 및 검증 데이터를 위한 데이터 생성기 생성, 증강 적용
        데이터 증강을 통해 학습 데이터 다양성 확보 (Chapter 3: 데이터 증강 중요성)
        Returns:
        - train_generator (ImageDataGenerator): 학습용 데이터 생성기
        - val_generator (ImageDataGenerator): 검증용 데이터 생성기
        """
        # 학습 데이터 증강 설정
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # 모든 픽셀 값을 0-1 범위로 정규화 (Chapter 2: 정규화의 중요성)
            rotation_range=15,  # 회전 적용 (최대 15도)
            width_shift_range=0.1,  # 좌우 이동 (10% 범위)
            height_shift_range=0.1,  # 상하 이동 (10% 범위)
            horizontal_flip=True  # 좌우 반전
        )

        # 검증 데이터는 정규화만 수행, 원본 상태 유지
        val_datagen = ImageDataGenerator(rescale=1./255)

        # train 폴더에서 학습 데이터 불러오기 (flow_from_directory 사용)
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),  # 이미지 크기 통일
            batch_size=self.BATCH_SIZE,
            class_mode='sparse'  # 정수형 라벨로 다중 클래스 분류
        )

        # validation 폴더에서 검증 데이터 불러오기
        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='sparse'
        )
        
        return train_generator, val_generator

    def build_model(self):
        """
        CNN 모델 아키텍처 정의
        Sequential 모델로 레이어를 순차적으로 쌓아 구성 (Chapter 4: CNN 모델 아키텍처 구성)
        """
        self.model = Sequential([
            # 첫 번째 Conv 레이어 (입력 크기: IMG_SIZE x IMG_SIZE x 3)
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3)),
            # 첫 번째 풀링 레이어 (특징 맵 다운샘플링)
            MaxPooling2D((2, 2)),

            # 두 번째 Conv 레이어 (필터 개수 증가)
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            # 세 번째 Conv 레이어 (필터 개수 증가)
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            # Flatten 레이어 (다차원 데이터를 1D 벡터로 변환, 완전 연결 레이어 연결)
            Flatten(),
            # Dense 레이어 (512 뉴런, 특징 학습)
            Dense(512, activation='relu'),
            # Dropout 레이어 (50% 비율로 뉴런 비활성화, 과적합 방지)
            Dropout(0.5),
            # 출력 레이어 (클래스 개수만큼 뉴런 설정, 소프트맥스 활성화로 각 클래스 확률 계산)
            Dense(self.NUM_CLASSES, activation='softmax')
        ])

        # 모델 컴파일 설정 (옵티마이저, 손실 함수, 평가 지표 설정)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, train_generator, val_generator):
        """
        모델 학습 및 평가, 학습 이력 반환
        Parameters:
        - train_generator (ImageDataGenerator): 학습용 데이터 생성기
        - val_generator (ImageDataGenerator): 검증용 데이터 생성기
        Returns:
        - history (History): 학습 이력 객체 (손실과 정확도 기록, 나중에 시각화 가능)
        """
        # 모델 학습, 각 에포크마다 train_generator와 val_generator 사용
        history = self.model.fit(
            train_generator,
            epochs=self.EPOCHS,  # 전체 데이터셋을 학습할 횟수
            validation_data=val_generator  # 검증 데이터로 성능 평가
        )
        return history

    def save_model(self, filepath='classification_model.h5'):
        """
        학습된 모델을 파일로 저장하여 나중에 재사용 가능 (Chapter 4: 학습된 모델 저장)
        Parameters:
        - filepath (str): 모델을 저장할 파일 경로
        """
        self.model.save(filepath)  # 모델 저장
        print(f"Model saved as '{filepath}'")  # 저장 완료 메시지 출력

    def load_and_train(self):
        """
        데이터 로드, 전처리, 학습까지 전체 프로세스 실행
        Returns:
        - history (History): 학습 이력 객체
        """
        # 클래스 이름 불러오기 및 설정 (NUM_CLASSES 설정)
        self.load_data()
        
        # 데이터 전처리 및 증강 설정
        train_generator, val_generator = self.preprocess_data()
        
        # 모델 구성 (build_model 호출)
        self.build_model()
        
        # 모델 학습 (train_model 호출)
        history = self.train_model(train_generator, val_generator)
        
        return history
