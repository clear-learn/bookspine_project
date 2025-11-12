# app_rest.py
import asyncio
import uuid
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel

# 제공해주신 crop 함수를 여기에 포함시킵니다.
from yolobook_crop_image import crop_rotated_box_no_trim

import tritonclient.http as httpclient

TRITON_URL = "localhost:18000"

YOLO_MODEL_NAME = "yolobook"
YOLO_INPUT_IMAGE = "INPUT_IMAGE"
YOLO_INPUT_THR = "thr"
YOLO_INPUT_WORKER = "worker"

YOLO_OUTPUT_RECTS = "output_rects"

# 2. arcbook 모델 (이미지 바이트 -> 벡터)
ARCBOOK_MODEL_NAME = "arcbook"  # 사용자 메모리 기반으로 수정
ARCBOOK_INPUT_NAME = "CROPPED_IMAGE"
ARCBOOK_OUTPUT_NAME = "OUTPUT_VECTOR"  # FP32 [512]

# --- 애플리케이션 설정 ---
ARCBOOK_CONCURRENCY = 64
REQUEST_TIMEOUT_SEC = 8.0

triton_client: httpclient.InferenceServerClient | None = None
app = FastAPI()


# --- Helper Functions ---
def _yolo_infer_sync(rgb_image: np.ndarray, thr: float, worker: int) -> httpclient.InferResult:
    assert triton_client is not None

    if rgb_image.ndim == 3:
        rgb_image = np.expand_dims(rgb_image, axis=0)

    b, h, w, c = rgb_image.shape
    assert c == 3, "INPUT_IMAGE must be BxHxWx3"

    inp_img = httpclient.InferInput(YOLO_INPUT_IMAGE, [b, h, w, c], "UINT8")
    inp_img.set_data_from_numpy(rgb_image)

    inp_thr = httpclient.InferInput(YOLO_INPUT_THR, [1], "FP32")
    inp_thr.set_data_from_numpy(np.array([thr], dtype=np.float32))

    inp_worker = httpclient.InferInput(YOLO_INPUT_WORKER, [1], "INT32")
    inp_worker.set_data_from_numpy(np.array([worker], dtype=np.int32))

    out_rects = httpclient.InferRequestedOutput(YOLO_OUTPUT_RECTS)

    res = triton_client.infer(
        YOLO_MODEL_NAME,
        inputs=[inp_img, inp_thr, inp_worker],
        outputs=[out_rects],
    )
    return res


async def run_yolo(rgb_image: np.ndarray, thr: float, worker: int) -> np.ndarray:
    res = await asyncio.to_thread(_yolo_infer_sync, rgb_image, thr, worker)
    rects = res.as_numpy(YOLO_OUTPUT_RECTS)
    if rects is None:
        return np.empty((0, 5), dtype=np.float32)
    return rects


def _arcbook_infer_one_sync(crop_bytes: bytes, request_id: str) -> np.ndarray:
    assert triton_client is not None
    inp = httpclient.InferInput(ARCBOOK_INPUT_NAME, [1], "BYTES")
    inp.set_data_from_numpy(np.array([crop_bytes], dtype=object))

    out = httpclient.InferRequestedOutput(ARCBOOK_OUTPUT_NAME)
    res = triton_client.infer(ARCBOOK_MODEL_NAME, inputs=[inp], outputs=[out], request_id=request_id)
    vec = res.as_numpy(ARCBOOK_OUTPUT_NAME)
    if vec is None:
        raise RuntimeError("Arcbook model returned no vector")
    return vec.squeeze()


async def run_arcbook_one(crop_bytes: bytes, parent_id: str, idx: int):
    vec = await asyncio.to_thread(_arcbook_infer_one_sync, crop_bytes, f"{parent_id}:{idx}")
    return idx, vec


class InferResponseItem(BaseModel):
    idx: int
    vector: List[float]


class InferResponse(BaseModel):
    parent_id: str
    count: int
    items: List[InferResponseItem]


@app.on_event("startup")
async def on_startup():
    global triton_client
    triton_client = httpclient.InferenceServerClient(url=TRITON_URL)


@app.post("/infer", response_model=InferResponse)
async def infer(
        file: UploadFile = File(...),
        thr: float = Query(0.5, description="threshold"),
        worker: int = Query(4, description="workers"),
):
    parent_id = str(uuid.uuid4())
    image_bytes = await file.read()

    img_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        rects = await asyncio.wait_for(run_yolo(img_rgb, thr, worker), timeout=REQUEST_TIMEOUT_SEC / 2)
    except asyncio.TimeoutError:
        return InferResponse(parent_id=parent_id, count=0, items=[])

    if rects.shape[0] == 0:
        return InferResponse(parent_id=parent_id, count=0, items=[])

    cropped_image_bytes_list = []
    for rect_data in rects:
        rect_tuple = ((rect_data[0], rect_data[1]), (rect_data[2], rect_data[3]), rect_data[4])

        cropped_img = crop_rotated_box_no_trim(img_bgr, rect_tuple)

        if cropped_img is not None:
            resized_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_LINEAR)

            is_success, buffer = cv2.imencode(".jpg", resized_img)
            if is_success:
                cropped_image_bytes_list.append(buffer.tobytes())

    sem = asyncio.Semaphore(ARCBOOK_CONCURRENCY)

    async def call_arcbook(idx: int, crop_b: bytes):
        async with sem:
            return await run_arcbook_one(crop_b, parent_id, idx)

    tasks = [call_arcbook(i, b) for i, b in enumerate(cropped_image_bytes_list)]

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=False),
            timeout=REQUEST_TIMEOUT_SEC
        )
    except asyncio.TimeoutError:
        results = []

    results = [res for res in results if isinstance(res, tuple) and len(res) == 2]
    results.sort(key=lambda t: t[0])

    items = [InferResponseItem(idx=i, vector=v.astype(float).tolist()) for i, v in results]
    return InferResponse(parent_id=parent_id, count=len(items), items=items)
