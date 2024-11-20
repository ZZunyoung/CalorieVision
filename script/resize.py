import cv2
import os
import numpy as np
import random

input_base_folder = "./"  # 기본 폴더 경로
output_folder_train = "./train"  # train 저장할 폴더 이름
output_folder_test = "./validation"  # test 저장할 폴더 이름

def resize_with_padding(image, desired_size=224):
    old_size = image.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # 리사이즈
    image = cv2.resize(image, (new_size[1], new_size[0]))

    # 패딩 추가
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_image

# 입력 폴더에서 new로 시작하는 모든 폴더 가져오기
input_folders = [f for f in os.listdir(input_base_folder) if f.startswith("new") and os.path.isdir(os.path.join(input_base_folder, f))]

for input_folder in input_folders:
    input_folder_path = os.path.join(input_base_folder, input_folder)

    # 입력 폴더에서 파일 목록 가져오기
    file_list = os.listdir(input_folder_path)
    image_files = [f for f in file_list if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]

    # train/test split (80% train, 20% test)
    random.seed(42)
    random.shuffle(image_files)
    train_size = int(0.8 * len(image_files))
    train_files = image_files[:train_size]
    test_files = image_files[train_size:]

    # 폴더 이름 생성
    train_output_folder = os.path.join(output_folder_train, input_folder)
    test_output_folder = os.path.join(output_folder_test, input_folder)

    # 폴더가 없으면 생성
    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(test_output_folder, exist_ok=True)

    # train 이미지 처리 및 저장
    for i, file_name in enumerate(train_files):
        input_path = os.path.join(input_folder_path, file_name)
        output_path = os.path.join(train_output_folder, file_name)

        # 이미지 읽기
        img = cv2.imread(input_path)
        if img is not None:
            # 이미지 리사이즈 및 패딩
            img_resized = resize_with_padding(img, 224)

            # 결과 저장
            cv2.imwrite(output_path, img_resized)
            print(f"Processed and saved to train: {output_path}")
        else:
            print(f"Failed to read image: {input_path}")

    # test 이미지 처리 및 저장
    for i, file_name in enumerate(test_files):
        input_path = os.path.join(input_folder_path, file_name)
        output_path = os.path.join(test_output_folder, file_name)

        # 이미지 읽기
        img = cv2.imread(input_path)
        if img is not None:
            # 이미지 리사이즈 및 패딩
            img_resized = resize_with_padding(img, 224)

            # 결과 저장
            cv2.imwrite(output_path, img_resized)
            print(f"Processed and saved to test: {output_path}")
        else:
            print(f"Failed to read image: {input_path}")
