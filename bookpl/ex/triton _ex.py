import tritonclient.http as httpclient
import numpy as np
import time

# --- 설정 ---
TRITON_SERVER_URL = "localhost:8000"  # Recognition 서버 포트가 9000번이라면 "localhost:9000"으로 변경
DETECT_MODEL_NAME = "bookyolo_detect"
RECOGNIZE_MODEL_NAME = "arcbook_recognize"

# 입력 데이터
s3_url = "s3://your-bucket-name/path/to/your/image.jpg"
thr = np.array([0.5], dtype=np.float32)
worker = np.array([8], dtype=np.int32)
# debug_mod = np.array([0], dtype=np.int32) # 단일 처리 모델 config에서 제거했다면 이 라인도 제거

triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)

# --- 1단계: Detection 모델 호출 (이전과 동일) ---
print("Step 1: Calling Detection model...")
detect_inputs = [
    httpclient.InferInput("s3_url", [1], "BYTES").set_data_from_numpy(np.array([s3_url], dtype=object)),
    httpclient.InferInput("thr", [1], "FP32").set_data_from_numpy(thr),
    httpclient.InferInput("worker", [1], "INT32").set_data_from_numpy(worker),
    # httpclient.InferInput("debug_mod", [1], "INT32").set_data_from_numpy(debug_mod),
]
detect_outputs = [httpclient.InferRequestedOutput("cropped_images")]
detect_results = triton_client.infer(model_name=DETECT_MODEL_NAME, inputs=detect_inputs, outputs=detect_outputs)
cropped_images_bytes = detect_results.as_numpy("cropped_images")

if cropped_images_bytes is None or len(cropped_images_bytes) == 0:
    print("No objects detected. Pipeline finished.")
    exit()

print(f"Detection successful! Found {len(cropped_images_bytes)} objects.")

# --- 2단계: Recognition 모델 호출 (★수정된 부분★) ---
print(f"\nStep 2: Calling Recognition model for each of the {len(cropped_images_bytes)} cropped images...")

all_vectors = []
start_time = time.time()

# for 루프를 사용해 각 이미지를 개별적으로 요청
for image_bytes in cropped_images_bytes:
    # 단일 이미지를 입력으로 설정
    # np.array로 감싸고 dtype=object를 지정해야 bytes를 올바르게 처리
    single_image_input = np.array([image_bytes], dtype=object)

    recognize_inputs = [
        # 모델 config에 정의된 입력 이름("CROPPED_IMAGE")과 형태에 맞춤
        httpclient.InferInput("CROPPED_IMAGE", single_image_input.shape, "BYTES").set_data_from_numpy(
            single_image_input)
    ]
    recognize_outputs = [httpclient.InferRequestedOutput("OUTPUT_VECTOR")]

    # 개별 이미지에 대한 추론 요청
    recognize_results = triton_client.infer(model_name=RECOGNIZE_MODEL_NAME, inputs=recognize_inputs,
                                            outputs=recognize_outputs)

    # 반환된 단일 벡터를 리스트에 추가
    output_vector = recognize_results.as_numpy("OUTPUT_VECTOR")
    all_vectors.append(output_vector)

end_time = time.time()

# 모든 벡터를 하나의 NumPy 배열로 합침
final_vectors_array = np.array(all_vectors)

print("\nRecognition successful!")
print(f"Total time for {len(cropped_images_bytes)} recognition requests: {end_time - start_time:.4f} seconds.")
print(f"Received feature vectors with final shape: {final_vectors_array.shape}")
if final_vectors_array.shape[0] > 0:
    print("First 5 elements of the first vector:", final_vectors_array[0][:5])
