import os
import shutil
import numpy as np

# 디렉토리 경로 설정
directory_path = '/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/book_side/cleansing_book/train/train/'
destination_path = '/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/book_side/cleansing_book/train/val/'

# 사용자가 지정한 샘플 갯수 설정 (예: 500개의 샘플을 선택)
sample_count = 200

# 디렉토리 내의 ID 리스트 가져오기
ID_list = os.listdir(directory_path)
total_samples = len(ID_list)
print(f"Total samples: {total_samples}")

# 샘플 수가 전체 수보다 클 수 없으므로, 예외 처리
if sample_count > total_samples:
    print("Error: 지정한 샘플 수가 전체 데이터 수보다 큽니다.")
else:
    # ID_list를 무작위로 섞은 후 상위 sample_count 개를 선택
    np.random.shuffle(ID_list)
    sampled_IDs = ID_list[:sample_count]

    # 파일 옮기기 및 삭제
    for count, ID in enumerate(ID_list, start=1):
        lis_ = os.listdir(os.path.join(directory_path, ID))
        if lis_:  # 디렉토리가 비어 있지 않은 경우
            if ID in sampled_IDs:
                shutil.move(os.path.join(directory_path, ID), destination_path)
        else:  # 디렉토리가 비어 있으면 삭제
            shutil.rmtree(os.path.join(directory_path, ID))

    # 샘플링 후 남은 ID 리스트 출력
    ID_list_new = os.listdir(directory_path)
    print(f"Remaining samples: {len(ID_list_new)}")