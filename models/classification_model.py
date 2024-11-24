import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

class FoodImageClassifier:
    def __init__(self, img_size=224, batch_size=32, epochs=50):  # 초기 에포크를 크게 설정
        base_dir = ".." # CalorieVision/models 경로에서 classfication_model.py를 실행시킨다고 할 때
        self.train_dir = os.path.join(base_dir, "assets", "resized_image", "train")
        self.val_dir = os.path.join(base_dir, "assets", "resized_image", "validation")
        self.test_dir = os.path.join(base_dir, "assets", "resized_image", "test")  # 테스트 데이터 경로 추가

        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.model = None  # 모델 초기화
        self.NUM_CLASSES = None  # 클래스 개수 (데이터 로드 후 설정)
        self.class_names = []  # 클래스 이름 리스트

    def load_data(self):
        self.class_names = sorted(os.listdir(self.train_dir))
        self.NUM_CLASSES = len(self.class_names)  # 클래스 개수 설정
        print(f"Detected classes: {self.class_names}")

    def preprocess_data(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical'
        )
        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical'
        )
        return train_generator, val_generator

    def preprocess_test_data(self):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        return test_generator

    def build_model(self):
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
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, train_generator, val_generator):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.EPOCHS,
            callbacks=[early_stopping]
        )
        return history

    def evaluate_and_plot_confusion_matrix(self, test_generator):
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())
        cm = confusion_matrix(true_classes, predicted_classes)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical')
        ax.set_title("Confusion Matrix", fontsize=16)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        plt.colorbar(disp.im_, ax=ax, orientation='vertical', shrink=0.8)
        plt.tight_layout()
        plt.show()
        print("Confusion Matrix Visualization Completed.")

    def save_model(self, filepath='food_image_classifier.h5'):
        self.model.save(filepath)
        print(f"Model saved as '{filepath}'")

    def generate_training_report(self, history):
        report_data = {
            "Epoch": range(1, len(history.history['loss']) + 1),
            "Training Accuracy": history.history['accuracy'],
            "Validation Accuracy": history.history['val_accuracy'],
            "Training Loss": history.history['loss'],
            "Validation Loss": history.history['val_loss'],
        }
        report_df = pd.DataFrame(report_data)
        print("\nTraining Report:")
        print(report_df)
        return report_df

    def load_and_train(self):
        self.load_data()
        train_generator, val_generator = self.preprocess_data()
        self.build_model()
        history = self.train_model(train_generator, val_generator)
        return history


# 실행
classifier = FoodImageClassifier()
history = classifier.load_and_train()
report_df = classifier.generate_training_report(history)
test_generator = classifier.preprocess_test_data()
classifier.evaluate_and_plot_confusion_matrix(test_generator)
classifier.save_model('food_image_classifier.h5')
