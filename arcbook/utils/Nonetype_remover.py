import os
import cv2

# 이미지 파일이 저장된 디렉토리 경로
directory_path = '/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/book_side/cleansing_book/train/train/'

# 디렉토리 내의 폴더 리스트 가져오기
ID_list = os.listdir(directory_path)
print(f"총 폴더 수: {len(ID_list)}")

count = 0

# 폴더를 순회하면서 작업 수행
for i in ID_list:
    folder_path = os.path.join(directory_path, i)
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)

        try:
            # 이미지 파일 읽기
            image = cv2.imread(file_path)

            # 이미지가 NoneType인 경우 또는 읽은 이미지의 크기가 올바르지 않은 경우 파일 삭제
            if image is None or image.size == 0:
                print(f"{file_path} 파일을 제거합니다. (이미지 읽기 실패)")
                os.remove(file_path)

        except Exception as e:
            # 예기치 않은 오류가 발생한 경우 파일 삭제
            print(f"{file_path} 파일을 제거합니다. (오류: {str(e)})")
            os.remove(file_path)

    # 폴더가 비어있으면 폴더 삭제
    if not os.listdir(folder_path):
        print(f"{folder_path} 폴더를 제거합니다. (비어있음)")
        os.rmdir(folder_path)

print("작업 완료.")