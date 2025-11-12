import cv2
import numpy as np

def crop_rotated_box_no_trim(orig_image, rect):
    (cx, cy), (w, h), angle = rect
    if angle < -45:
        angle += 90
        w, h = h, w
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    H, W = orig_image.shape[:2]
    corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    transformed_corners = cv2.transform(corners[None, :, :], M)[0]
    min_x, max_x = transformed_corners[:, 0].min(), transformed_corners[:, 0].max()
    min_y, max_y = transformed_corners[:, 1].min(), transformed_corners[:, 1].max()
    needed_w, needed_h = int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y))
    M[0, 2] += (needed_w - W) / 2
    M[1, 2] += (needed_h - H) / 2
    if needed_w < 1 or needed_h < 1: return None
    rotated = cv2.warpAffine(orig_image, M, (needed_w, needed_h), flags=cv2.INTER_LINEAR)
    w_i, h_i = int(w), int(h)
    cx_n, cy_n = needed_w / 2, needed_h / 2
    x1, y1 = int(cx_n - w_i / 2), int(cy_n - h_i / 2)
    x2, y2 = x1 + w_i, y1 + h_i
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(needed_w, x2), min(needed_h, y2)
    if x2 <= x1 or y2 <= y1: return None
    cropped = rotated[y1:y2, x1:x2].copy()
    hh, ww = cropped.shape[:2]
    if ww > hh:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    return cropped
