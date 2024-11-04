import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 하이퍼파라미터 설정
input_shape = (224, 224, 3)
num_classes = 5  # Q1 ~ Q5에 해당하는 5가지 분류

# 모델 생성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 데이터 증강 설정
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 데이터 로드 및 학습 (예제용)
train_generator = datagen.flow_from_directory(
    'path_to_train_data',  # 훈련 데이터 폴더 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'  # sparse_categorical_crossentropy에 맞는 레이블 형식
)

validation_generator = datagen.flow_from_directory(
    'path_to_validation_data',  # 검증 데이터 폴더 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20
)

# 모델 저장
model.save('food_quantity_classification_model.h5')
