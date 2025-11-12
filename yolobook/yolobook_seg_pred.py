import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
from pytictoc import TicToc

MIN_W = 10
MIN_H = 10 

rainbow_colors = [
    (0, 0, 255),
    (0, 127, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (130, 0, 75),
    (211, 0, 148)
]

def get_random_rainbow_color():
    return random.choice(rainbow_colors)

def visualize_rotated_rect(
    image_path,
    output,
    save_path="output.jpg",
    use_median=False
):
    t_func = TicToc()
    t_func.tic()

    read_func = TicToc()
    read_func.tic()
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]
    read_ms = read_func.tocvalue() * 1000

    masks = output[0].masks.data.cpu().numpy()
    pred_h, pred_w = masks.shape[1], masks.shape[2]
    scale_x = img_w / float(pred_w)
    scale_y = img_h / float(pred_h)

    all_contours = []
    for i in range(masks.shape[0]):
        mask = (masks[i] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            c_area = cv2.contourArea(c)
            all_contours.append((c, c_area))

    if not all_contours:
        cv2.imwrite(save_path, image)
        total_time = t_func.tocvalue() * 1000
        return 0.0, 0, total_time

    areas = [x[1] for x in all_contours]
    if use_median:
        median_val = np.median(areas)
        area_threshold = median_val * 0.3
        print(median_val)
    else:
        area_threshold = np.percentile(areas, 15)

    t_area = TicToc()
    t_area.tic()

    n_masks_processed = 0
    for c, c_area in all_contours:
        if c_area < area_threshold:
            continue

        x_b, y_b, w_b, h_b = cv2.boundingRect(c)
        w_b_scaled = w_b * scale_x
        h_b_scaled = h_b * scale_y
        if w_b_scaled < MIN_W or h_b_scaled < MIN_H:
            continue

        rect = cv2.minAreaRect(c)
        box_pts = cv2.boxPoints(rect)
        box_pts[:, 0] *= scale_x
        box_pts[:, 1] *= scale_y
        box_pts = box_pts.astype(np.int32)
        color = get_random_rainbow_color()
        cv2.drawContours(image, [box_pts], 0, color, 3)
        n_masks_processed += 1

    area_ms = t_area.tocvalue() * 1000

    cv2.imwrite(save_path, image)
    total_time = t_func.tocvalue() * 1000
    print(f"RotatedRect-based box image saved at {save_path}")

    return area_ms, n_masks_processed, total_time


if __name__ == "__main__":
    model = YOLO("pt path")
    input_folder = "input image path"
    output_folder = "output path"
    os.makedirs(output_folder, exist_ok=True)

    t_main = TicToc()
    t_main.tic()

    total_inference_time = 0.0
    total_post_time = 0.0
    total_area_time = 0.0
    total_masks_processed = 0
    count = 0

    USE_MEDIAN = True

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        if not os.path.isfile(img_path):
            continue

        t_infer = TicToc()
        t_infer.tic()
        results = model.predict(
            img_path,
            save=False,
            imgsz=864,
            device= '1',
            conf=0.5,
            show = False,
            save_crop=False,
            retina_masks=True,
            # max_det= 100
        )

        inference_ms = t_infer.tocvalue() * 1000
        total_inference_time += inference_ms

        t_post = TicToc()
        t_post.tic()
        out_path = os.path.join(output_folder, f"rotated_rect_{img_name}")

        area_ms, n_masks, total_func_time = visualize_rotated_rect(
            image_path=img_path,
            output=results,
            save_path=out_path,
            use_median=USE_MEDIAN
        )
        post_ms = t_post.tocvalue() * 1000
        total_post_time += post_ms

        total_area_time += area_ms
        total_masks_processed += n_masks
        count += 1

    main_ms = t_main.tocvalue() * 1000
    avg_inference = total_inference_time / count if count > 0 else 0
    avg_post = total_post_time / count if count > 0 else 0
    avg_total_per_image = main_ms / count if count > 0 else 0
    avg_area_per_mask = total_area_time / total_masks_processed if total_masks_processed > 0 else 0

    print(f"All done! {count} images processed, total={main_ms:.3f} ms")
    print(f" - Average Inference Time (per image) : {avg_inference:.3f} ms")
    print(f" - Average Post Time      (per image) : {avg_post:.3f} ms")
    print(f" - Average Total Time     (per image) : {avg_total_per_image:.3f} ms")
    print(f" - Total Masks Processed : {total_masks_processed}")
    print(f" - Average 면적 연산 Time (per mask)  : {avg_area_per_mask:.3f} ms")
