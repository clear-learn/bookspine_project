import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import concurrent.futures
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Detection 모델 v4.
    이미지 행렬(Numpy array)을 직접 입력으로 받아 객체를 탐지하고,
    탐지된 객체들의 회전된 사각형 좌표([cx, cy, w, h, angle]) 리스트를 반환합니다.
    """

    def initialize(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join(os.path.dirname(__file__), 'bookYOLO_seg_model.pt')
        self.model = YOLO(model_path)
        print(f"Detection model (v4 - Rects Output) loaded: {model_path} on device {self.device}")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                in_image_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
                img = in_image_tensor.as_numpy()

                thr = pb_utils.get_input_tensor_by_name(request, "thr").as_numpy()[0]
                worker = pb_utils.get_input_tensor_by_name(request, "worker").as_numpy()[0]

                rotated_rects_np = self.detect_and_get_rects(
                    image=img,
                    conf_thres=thr,
                    worker=worker
                )

                out_tensor = pb_utils.Tensor("output_rects", rotated_rects_np)

                response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(response)

            except Exception as e:
                import traceback
                error_message = f"{str(e)}\n{traceback.format_exc()}"
                error_response = pb_utils.InferenceResponse(output_tensors=[],
                                                            error=pb_utils.TritonError(error_message))
                responses.append(error_response)

        return responses

    def process_contour_to_rect(self, item, scale_x, scale_y):
        (y_box, x_box, c_obj) = item
        rect = cv2.minAreaRect(c_obj)
        (cx, cy), (w, h), angle = rect

        cx_scaled, cy_scaled = cx * scale_x, cy * scale_y
        w_scaled, h_scaled = w * scale_x, h * scale_y

        return [cx_scaled, cy_scaled, w_scaled, h_scaled, angle]

    def detect_and_get_rects(self, image, conf_thres=0.5, median_mode=False, worker=4):
        start_time = time.time()
        results = self.model.predict(source=image, save=False, imgsz=864, conf=conf_thres, device=self.device,
                                     retina_masks=True, verbose=False)

        img_h, img_w = image.shape[:2]
        if not results[0].masks:
            return np.empty((0, 5), dtype=np.float32)

        masks_data = results[0].masks.data.cpu().numpy()
        pred_h, pred_w = masks_data.shape[1:3]
        scale_x, scale_y = img_w / float(pred_w), img_h / float(pred_h)

        all_contours = []
        for i in range(masks_data.shape[0]):
            mask = (masks_data[i] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                all_contours.append((c, cv2.contourArea(c)))

        if not all_contours:
            return np.empty((0, 5), dtype=np.float32)

        areas = [x[1] for x in all_contours]
        area_threshold = np.median(areas) * 0.3 if median_mode else np.percentile(areas, 15)

        valid_list = []
        for c, c_area in all_contours:
            if c_area < area_threshold: continue
            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
            if (w_b * scale_x) < 10 or (h_b * scale_y) < 10: continue
            valid_list.append((y_b, x_b, c))

        valid_list.sort(key=lambda item: (item[0], item[1]))

        rect_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker) as executor:
            futures = [executor.submit(self.process_contour_to_rect, item, scale_x, scale_y) for item in valid_list]
            for fut in concurrent.futures.as_completed(futures):
                result_rect = fut.result()
                if result_rect:
                    rect_list.append(result_rect)

        print(f"Total detection and rect calculation time: {time.time() - start_time:.4f}s")

        if not rect_list:
            return np.empty((0, 5), dtype=np.float32)

        return np.array(rect_list, dtype=np.float32)

    def finalize(self):
        print('Detection model (v4 - Rects Output) cleaning up...')
