import tritonclient.http as httpclient
import numpy as np
import cv2

# --- 설정 ---
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "bookyolo_detect"
IMAGE_PATH = "./my_test_image.jpg" # 테스트할 이미지 경로

# --- 이미지 로드 및 전처리 ---
# cv2.imread는 BGR 순서로 이미지를 로드합니다.
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

# 모델이 RGB를 기본으로 가정할 경우를 대비해 RGB로 변환하여 전송합니다.
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --- Triton 클라이언트 설정 ---
triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# 파라미터 설정
thr = np.array([0.5], dtype=np.float32)
worker = np.array([8], dtype=np.int32)

# === 입력 생성 변경 ===
inputs = [
    # 이미지 행렬을 UINT8 타입으로 전송
    httpclient.InferInput("INPUT_IMAGE", img_rgb.shape, "UINT8").set_data_from_numpy(img_rgb),
    httpclient.InferInput("thr", [1], "FP32").set_data_from_numpy(thr),
    httpclient.InferInput("worker", [1], "INT32").set_data_from_numpy(worker),
]

outputs = [httpclient.InferRequestedOutput("cropped_images")]

# --- 추론 요청 ---
print(f"Sending image with shape {img_rgb.shape} to model '{MODEL_NAME}'...")
results = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
cropped_images_bytes = results.as_numpy("cropped_images")

# --- 결과 처리 ---
if cropped_images_bytes is not None:
    print(f"Success! Received {len(cropped_images_bytes)} cropped images.")
    # 첫 번째 크롭 이미지를 파일로 저장하여 확인
    if len(cropped_images_bytes) > 0:
        np_arr = np.frombuffer(cropped_images_bytes[0], np.uint8)
        img_mat = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite("output_crop_from_matrix_input.jpg", img_mat)
        print("First cropped image saved.")
else:
    print("Inference returned no results.")