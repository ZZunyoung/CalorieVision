import cv2
import os
import numpy as np

input_folder = "./08011006" #읽을 폴더 이름
output_folder = "./resize" #저장할 폴더 이름

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

# 입력 폴더에서 파일 목록 가져오기
file_list = os.listdir(input_folder)
image_files = [f for f in file_list if f.endswith(('.jpg', '.jpeg', '.png'))]

# 이미지 전부 처리
for i, file_name in enumerate(image_files):
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    # 이미지 읽기
    img = cv2.imread(input_path)
    if img is not None:
        # 이미지 리사이즈 및 패딩
        img_resized = resize_with_padding(img, 224)
        
        # 결과 저장
        cv2.imwrite(output_path, img_resized)
        print(f"Processed and saved: {output_path}")
    else:
        print(f"Failed to read image: {input_path}")
