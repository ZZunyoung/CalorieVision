import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class FoodImageClassifier:
    def __init__(self, img_size=224, batch_size=32, epochs=20, train_dir='./assets/resized_image/train', val_dir='./assets/resized_image/validation'):
        """
        Parameters:
        - img_size (int): 입력 이미지 크기
        - batch_size (int): 배치 크기
        - epochs (int): 학습 에포크 수
        - train_dir (str): 학습 데이터 폴더 경로
        - val_dir (str): 검증 데이터 폴더 경로
        """
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.model = None  # 모델 초기화
        self.NUM_CLASSES = None  # 클래스 개수 (데이터 로드 후 설정)
        self.class_names = []  # 클래스 이름 리스트

    def load_data(self):
        """
        클래스 이름을 학습 데이터 디렉토리에서 추출하고 설정
        """
        # train 디렉토리의 하위 폴더 이름이 클래스 이름
        self.class_names = sorted(os.listdir(self.train_dir))
        self.NUM_CLASSES = len(self.class_names)  # 클래스 개수 설정
        print(f"Detected classes: {self.class_names}")

    def preprocess_data(self):
        """
        학습 및 검증 데이터를 위한 데이터 생성기 생성
        Returns:
        - train_generator: 학습 데이터 생성기
        - val_generator: 검증 데이터 생성기
        """
        # 데이터 증강 및 정규화
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # 픽셀 값을 0-1로 정규화
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        # 학습 데이터
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical'
        )

        # 검증 데이터
        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical'
        )

        return train_generator, val_generator

    def build_model(self):
        """
        CNN 모델 정의
        """
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.NUM_CLASSES, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, train_generator, val_generator):
        """
        모델 학습
        """
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.EPOCHS
        )
        return history

    def save_model(self, filepath='food_image_classifier.h5'):
        """
        학습된 모델 저장
        """
        self.model.save(filepath)
        print(f"Model saved as '{filepath}'")

    def load_and_train(self):
        """
        전체 실행
        """
        self.load_data()  # 클래스 이름 로드
        train_generator, val_generator = self.preprocess_data()  # 데이터 준비
        self.build_model()  # 모델 빌드
        history = self.train_model(train_generator, val_generator)  # 학습
        return history


# 모델 학습 실행
classifier = FoodImageClassifier(
    train_dir='./assets/resized_image/train',
    val_dir='./assets/resized_image/validation'
)
history = classifier.load_and_train()
classifier.save_model('food_image_classifier.h5')
