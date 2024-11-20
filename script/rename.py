import cv2
import numpy as np
import glob
import os

input_base_folder = "./"  # 기본 폴더 경로

def get_first_3_and_last_8_chars(filename):
    name, ext = os.path.splitext(filename)
    return name[:3] + name[-8:] + ext

def imread_with_unicode_path(file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
    img_array = np.frombuffer(file_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

# 입력 폴더에서 Q로 시작하는 모든 폴더 가져오기
input_folders = [f for f in os.listdir(input_base_folder) if f.startswith("Q") and os.path.isdir(os.path.join(input_base_folder, f))]

for input_folder in input_folders:
    input_folder_path = os.path.join(input_base_folder, input_folder)
    output_folder = os.path.join(input_base_folder, f"new_{input_folder}")  # "new_" 접두사를 붙인 출력 폴더 경로

    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # glob을 사용하여 입력 폴더의 이미지 파일 목록 가져오기
    image_files = glob.glob(f"{input_folder_path}/*.[jJ][pP][gG]") + \
                  glob.glob(f"{input_folder_path}/*.[pP][nN][gG]") + \
                  glob.glob(f"{input_folder_path}/*.[jJ][pP][eE][gG]")

    # 이미지 파일 처리 및 저장
    for file_path in image_files:
        file_name = os.path.basename(file_path)  # 파일 이름 추출
        new_file_name = get_first_3_and_last_8_chars(file_name)  # 처음 3자리와 마지막 8자리로 파일 이름 생성
        output_path = os.path.join(output_folder, new_file_name)

        # 이미지 읽기
        img = imread_with_unicode_path(file_path)
        if img is not None:
            # 결과 저장
            cv2.imwrite(output_path, img)
            print(f"Copied and saved to: {output_path}")
        else:
            print(f"Failed to read image: {file_path}")
