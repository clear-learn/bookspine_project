import os
from pathlib import Path

import torch
from arcbook.backbones import get_model

from faiss_arcbook import build_or_load_index, topk_search, build_title_lookup_from_csv, safe_lookup_title, _compute_auc_pr_curves

import cv2
import numpy as np
from typing import Optional, Tuple, List
import math
import csv

from collections import Counter, defaultdict

import random


def _collect_topk_from_matches(matches, k=10):
    out = []
    if not matches:
        return out
    for m in matches[:k]:
        try:
            out.append({"id": str(m.get("itemID")), "score": float(m.get("score", 0.0))})
        except Exception:
            pass
    return out

def _update_wrong_id_stats_detailed(
    wrong_pred_counter: Counter,
    wrong_pair_counter: Counter,
    wrong_events_by_pred_id: dict,
    *,
    img_path: str,
    preds_used: list,
    gts: list,
    pairs: list,
    keep_examples_per_id: int = 5,
    keep_topk: int = 10,
):
    from pathlib import Path

    for (pi, gi, iou) in pairs:
        p = preds_used[pi]
        g = gts[gi]

        top1 = p.get("top1") or {}
        pred_id = top1.get("itemID", None)
        pred_id = None if pred_id is None else str(pred_id)
        gt_id = str(g.get("id"))

        if (pred_id is None) or (pred_id == gt_id):
            continue

        wrong_pred_counter[pred_id] += 1
        wrong_pair_counter[(gt_id, pred_id)] += 1

        ex_list = wrong_events_by_pred_id[pred_id]
        if len(ex_list) < keep_examples_per_id:
            ex_list.append({
                "image": Path(img_path).name,
                "crop": str(p.get("crop_path", "") or ""),
                "gt_id": gt_id,
                "pred_id": pred_id,
                "iou": float(iou),
                "top1_score": float(top1.get("score", -1.0)) if top1 else -1.0,
                "top10": _collect_topk_from_matches(p.get("matches", []) or [], k=keep_topk),
            })

def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])
    return float(ap)

def _compute_map_dataset(
    all_results: list,
    iou_thrs: list[float],
    use_passed_only: bool = True,
    score_mode: str = "det"
) -> dict:
    gt_by_img: dict[str, list[dict]] = {}
    hw_by_img: dict[str, tuple[int,int]] = {}

    for R in all_results:
        img_path = R["image"]
        sidecar = Path(img_path).with_suffix(".json")
        gts = _load_labelme_gt(sidecar)
        if gts:
            gt_by_img[img_path] = gts
            im0 = cv2.imread(img_path)
            if im0 is None:
                continue
            H, W = im0.shape[:2]
            hw_by_img[img_path] = (H, W)

    preds_all = []
    for R in all_results:
        img_path = R["image"]
        dets = R["detections"]
        if img_path not in gt_by_img:
            continue

        H, W = hw_by_img[img_path]
        for d in dets:
            if use_passed_only and not d.get("passed", False):
                continue
            poly = d.get("poly4")
            if poly is None:
                x1,y1,x2,y2 = d.get("bbox_xyxy", [0,0,0,0])
                poly = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            conf = float(d.get("conf", 0.0))
            if score_mode == "det":
                score = conf
            elif score_mode == "det_x_cos":
                top1 = d.get("top1") or {}
                cos = float(top1.get("score", 0.0))
                cos = max(cos, 0.0)
                score = conf * cos
            else:
                score = conf
            preds_all.append({"img": img_path, "score": float(score), "poly": poly})

    num_gt = sum(len(v) for v in gt_by_img.values())
    num_pred = len(preds_all)

    preds_all.sort(key=lambda x: x["score"], reverse=True)

    ap_per_iou = {}
    for thr in iou_thrs:
        tp = []
        fp = []
        gt_used = {img: np.zeros(len(gt_by_img[img]), dtype=bool) for img in gt_by_img}

        for p in preds_all:
            img = p["img"]
            if img not in gt_by_img:
                continue
            gts = gt_by_img[img]
            H, W = hw_by_img[img]

            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gts):
                if gt_used[img][j]:
                    continue
                iou = _polygon_iou_by_mask(p["poly"], g["poly"], H, W)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0 and best_iou >= thr:
                tp.append(1.0); fp.append(0.0)
                gt_used[img][best_j] = True
            else:
                tp.append(0.0); fp.append(1.0)

        tp = np.array(tp, dtype=np.float32)
        fp = np.array(fp, dtype=np.float32)
        if tp.size == 0:
            ap_per_iou[f"{thr:.2f}"] = 0.0
            continue

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(num_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        ap = _ap_from_pr(recall, precision)
        ap_per_iou[f"{thr:.2f}"] = ap

    map_50   = ap_per_iou.get("0.50", 0.0)
    keys_50_95 = [f"{x/100:.2f}" for x in range(50, 100, 5)]
    vals = [ap_per_iou[k] for k in keys_50_95 if k in ap_per_iou]
    map_50_95 = float(np.mean(vals)) if len(vals) > 0 else 0.0

    return {
        "num_gt": int(num_gt),
        "num_pred": int(num_pred),
        "ap_per_iou": ap_per_iou,
        "map_50": float(map_50),
        "map_50_95": float(map_50_95),
    }

def _parse_id_from_labelme_label(lbl: str) -> Optional[str]:
    if not lbl:
        return None
    s = lbl.strip()
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    else:
        s = s.split()[0].strip()
    return s if len(s) > 0 else None

def _poly_to_int32(poly):
    p = np.asarray(poly, dtype=np.float32)
    return p.astype(np.int32)

def _polygon_iou_by_mask(polyA, polyB, H, W) -> float:
    ma = np.zeros((H, W), dtype=np.uint8)
    mb = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(ma, [ _poly_to_int32(polyA) ], 1)
    cv2.fillPoly(mb, [ _poly_to_int32(polyB) ], 1)
    inter = (ma & mb).sum()
    union = (ma | mb).sum()
    if union <= 0:
        return 0.0
    return float(inter) / float(union)

def _load_labelme_gt(sidecar_json_path: Path) -> list[dict]:
    import json
    if not sidecar_json_path.exists():
        return []
    with open(sidecar_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gts = []
    shapes = data.get("shapes", []) or []
    for sh in shapes:
        lbl = sh.get("label", "")
        item_id = _parse_id_from_labelme_label(lbl)
        if not item_id:
            continue
        pts = sh.get("points", [])
        if not pts or len(pts) < 4:
            continue
        gts.append({"id": str(item_id), "poly": pts})
    return gts

def _greedy_match_by_iou(preds, gts, H, W, iou_thr: float):
    pairs = []
    if not preds or not gts:
        return pairs

    ious = []
    for pi, p in enumerate(preds):
        ppoly = p.get("poly4", None)
        if ppoly is None:
            continue
        for gi, g in enumerate(gts):
            giou = _polygon_iou_by_mask(ppoly, g["poly"], H, W)
            if giou > 0:
                ious.append((giou, pi, gi))
    ious.sort(key=lambda x: x[0], reverse=True)

    used_p = set()
    used_g = set()
    for iou, pi, gi in ious:
        if iou < iou_thr:
            break
        if pi in used_p or gi in used_g:
            continue
        pairs.append((pi, gi, iou))
        used_p.add(pi)
        used_g.add(gi)
    return pairs

def _eval_topk_hit(pred_det: dict, gt_id: str, ks=(1,3,5)) -> dict:
    result = {}
    mlist = pred_det.get("matches", []) or []
    cand = [str(m.get("itemID")) for m in mlist]
    for k in ks:
        result[f"top{k}"] = (gt_id in cand[:k]) if len(cand) >= 1 else False
    return result

def _order_poly4_tltrbrbl(box4: np.ndarray) -> np.ndarray:
    pts = box4.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _mask_to_orig_size(m: np.ndarray, H: int, W: int) -> np.ndarray:
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    mh, mw = m.shape[-2], m.shape[-1]
    if (mh, mw) != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m

def _poly4_from_mask_findcontours(mask_bin: np.ndarray,
                                  min_area: float = 10.0) -> Optional[np.ndarray]:
    H, W = mask_bin.shape[:2]
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None

    rect = cv2.minAreaRect(c)
    box4 = cv2.boxPoints(rect)
    box4 = _order_poly4_tltrbrbl(box4)

    box4[:, 0] = np.clip(box4[:, 0], 0, W - 1)
    box4[:, 1] = np.clip(box4[:, 1], 0, H - 1)
    return box4.astype(np.float32)

def _warp_crop_from_poly4(img: np.ndarray, poly4: np.ndarray,
                          keep_vertical: bool = True, pad: int = 0,
                          border_value=(114,114,114)) -> Tuple[np.ndarray, float]:
    p = _order_poly4_tltrbrbl(np.asarray(poly4, np.float32))
    w = int(round(np.linalg.norm(p[1] - p[0])))
    h = int(round(np.linalg.norm(p[2] - p[1])))
    w = max(2, w); h = max(2, h)

    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(p, dst)
    patch = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border_value)

    delta = 0.0
    if keep_vertical and w > h:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
        delta = 90.0

    if pad > 0:
        patch = cv2.copyMakeBorder(patch, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border_value)
    return patch, delta

def _edge_angle_deg(poly4: np.ndarray) -> float:
    v = poly4[1] - poly4[0]
    return float(math.degrees(math.atan2(float(v[1]), float(v[0]))))

def _normalize_pm90(theta: float) -> float:
    theta = (theta + 180.0) % 180.0
    if theta > 90.0:
        theta -= 180.0
    return float(theta)

def _rect_wh(poly4: np.ndarray) -> Tuple[int, int]:
    w = float(np.linalg.norm(poly4[1] - poly4[0]))
    h = float(np.linalg.norm(poly4[2] - poly4[1]))
    return max(1, int(round(w))), max(1, int(round(h)))

def _warp_crop_from_poly4(
    img: np.ndarray,
    poly4: np.ndarray,
    keep_vertical: bool = True,
    pad: int = 0,
    border_value=(114,114,114)
) -> Tuple[np.ndarray, float]:

    poly4 = _order_poly4_tltrbrbl(np.asarray(poly4, np.float32))
    w, h = _rect_wh(poly4)

    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(poly4, dst)
    patch = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border_value)

    delta = 0.0
    if keep_vertical and w > h:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
        delta = 90.0

    if pad > 0:
        patch = cv2.copyMakeBorder(patch, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border_value)

    return patch, delta

def _angle_to_nearest_axis(theta_deg: float) -> float:
    k = int(np.round(theta_deg / 90.0))
    return k * 90.0 - theta_deg

def _pca_orientation_deg(pts_xy: np.ndarray) -> float:
    ctr = pts_xy.mean(axis=0, keepdims=True)
    X = pts_xy - ctr
    cov = np.cov(X.T)
    w, v = np.linalg.eig(cov)
    u = v[:, np.argmax(w)]
    theta = np.degrees(np.arctan2(u[1], u[0]))
    theta = (theta + 180.0) % 180.0
    if theta > 90.0:
        theta -= 180.0
    return float(theta)

def _delta_to_vertical(theta: float) -> float:
    delta1 =  90.0 - theta
    delta2 = -90.0 - theta
    return delta1 if abs(delta1) <= abs(delta2) else delta2

def _crop_upright_from_polygon(orig_bgr: np.ndarray, poly_xy: np.ndarray, pad: int = 8):
    H, W = orig_bgr.shape[:2]
    x_min = max(0, int(np.floor(poly_xy[:,0].min())) - pad)
    y_min = max(0, int(np.floor(poly_xy[:,1].min())) - pad)
    x_max = min(W-1, int(np.ceil(poly_xy[:,0].max())) + pad)
    y_max = min(H-1, int(np.ceil(poly_xy[:,1].max())) + pad)
    roi = orig_bgr[y_min:y_max+1, x_min:x_max+1]
    if roi.size == 0:
        raise ValueError("empty ROI")

    pts = poly_xy.copy()
    pts[:,0] -= x_min
    pts[:,1] -= y_min

    theta = _pca_orientation_deg(pts)
    delta = _delta_to_vertical(theta)

    h, w = roi.shape[:2]
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), delta, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0,2] += (new_w/2.0) - w/2.0
    M[1,2] += (new_h/2.0) - h/2.0
    roi_rot = cv2.warpAffine(roi, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(114,114,114))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    mask_rot = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    ys, xs = np.where(mask_rot > 0)
    if len(xs) == 0:
        return roi_rot, float(delta), (0, 0, new_w, new_h)

    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    crop = roi_rot[y0:y1+1, x0:x1+1].copy()
    return crop, float(delta), (int(x0), int(y0), int(x1-x0+1), int(y1-y0+1))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(img_dir: str) -> list[str]:
    p = Path(img_dir)
    files = []
    for ext in IMG_EXTS:
        files.extend(p.rglob(f"*{ext}"))
    return [str(x) for x in sorted(files)]

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1: x2 = min(x1 + 1, w - 1)
    if y2 <= y1: y2 = min(y1 + 1, h - 1)
    return x1, y1, x2, y2

def preprocess_arcbook(bgr_img: np.ndarray, size: int = 224, device: str = "cpu") -> torch.Tensor:
    im = cv2.resize(bgr_img, (size, size))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1)).astype(np.float32)
    t = torch.from_numpy(im).unsqueeze(0).to(device)
    t.div_(255).sub_(0.5).div_(0.5)
    return t

def clean_state_dict(sd):
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {k.replace("module.", ""): v for k, v in sd.items()}

def run_yolo_arcbook_search(
    img_dir: str,
    yolo_weights: str,
    arcbook_arch: str = "r18",
    arcbook_weights: str = "/opt/project/datasets/arcbook_model.pt",
    csv_path: str = "/opt/project/datasets/bookspine_multi_all.csv",
    index_path: str = "items_multi_parts.faiss",
    idmap_path: Optional[str] = None,
    out_json: str = "runs/search/results.json",
    save_crops: bool = True,
    crop_dir: str = "runs/crops",
    save_vis: bool = True,
    vis_dir: str = "runs/vis",
    box_thickness: int = 2,
    draw_top1: bool = True,
    use_seg: bool = True,
    seg_pad: int = 8,
    topk: int = 10,
    tau: float = 0.55,
    reject_if_low_match: bool = True,
    save_rejected_crops: bool = False,
    yolo_imgsz: tuple[int, int] = (1024, 384),
    conf_thres: float = 0.2,
    arcbook_img_size: int = 224,
    save_labelme_pseudo: bool = True,
    labelme_label: str = "book",
    label_as_id: bool = True,
    label_as_id_only_if_passed: bool = False,
    include_id_in_text: bool = False,
    labelme_version: str = "0.4.30",
    labelme_overwrite: bool = True,

    eval_mode: bool = False,
    eval_iou_thr: float = 0.5,
    eval_topk_list: Tuple[int, ...] = (1, 3, 5),
    eval_only_passed: bool = True,
    eval_report_json: Optional[str] = "runs/eval/summary.json",
    eval_pairs_csv: Optional[str] = None,

    eval_compute_map: bool = True,
    eval_map_iou_thresholds: Optional[list] = None,
    eval_map_only_passed: bool = True,
    eval_map_score_mode: str = "det",

):
    import json, time, datetime as dt
    from pathlib import Path
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_device = 0 if torch.cuda.is_available() else None

    raw_pred_total = 0
    rejected_by_tau = 0

    yolo = YOLO(yolo_weights)
    try:
        class_names = yolo.model.names
    except Exception:
        class_names = getattr(yolo, "names", {})

    net = get_model(arcbook_arch, fp16=False).to(device).eval()
    sd = torch.load(arcbook_weights, map_location="cpu")
    net.load_state_dict(clean_state_dict(sd), strict=False)

    index, id_lookup = build_or_load_index(
        csv_path=csv_path,
        auto=True,
        has_header=False,
        index_path=index_path,
        idmap_path=idmap_path
    )

    title_lookup = build_title_lookup_from_csv(
        csv_path=csv_path,
        has_header=False,
        id_col=None,
        title_col=None
    )

    out_p = Path(out_json); out_p.parent.mkdir(parents=True, exist_ok=True)
    crop_root = Path(crop_dir)
    if save_crops: crop_root.mkdir(parents=True, exist_ok=True)
    vis_root = Path(vis_dir)
    if save_vis: vis_root.mkdir(parents=True, exist_ok=True)

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def list_images(d):
        p = Path(d); files=[]
        for ext in IMG_EXTS: files += list(p.rglob(f"*{ext}"))
        return [str(x) for x in sorted(files)]
    image_paths = list_images(img_dir)
    print(f"[info] found {len(image_paths)} images in {img_dir}")

    all_results = []
    t0 = time.time()

    for img_path in image_paths:
        yres = yolo.predict(
            source=img_path,
            imgsz=yolo_imgsz,
            conf=conf_thres,
            device=yolo_device,
            verbose=False,
            retina_masks=True
        )
        res = yres[0]
        orig_bgr = res.orig_img
        H, W = orig_bgr.shape[:2]
        dets = []

        has_seg = (getattr(res, "masks", None) is not None) and (res.masks is not None) \
                  and (getattr(res.masks, "xy", None) is not None)

        if use_seg and has_seg and len(res.masks.xy) > 0:
            n = len(res.masks.xy)
            confs = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None else [1.0]*n
            clss  = res.boxes.cls.cpu().numpy().astype(int).tolist() if res.boxes is not None else [0]*n

            if save_crops:
                stem = Path(img_path).stem
                save_dir = Path(crop_dir) / stem
                save_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n):
                poly4 = None

                masks_data = getattr(res.masks, "data", None)
                if masks_data is not None and i < masks_data.shape[0]:
                    m = masks_data[i].detach().float().cpu().numpy()
                    m = _mask_to_orig_size(m, H, W)
                    mask_bin = (m > 0.5).astype(np.uint8) * 255

                    poly4 = _poly4_from_mask_findcontours(mask_bin, min_area=16.0)

                if poly4 is None:
                    poly_xy_list = getattr(res.masks, "xy", None)
                    if poly_xy_list is not None and i < len(poly_xy_list) and len(poly_xy_list[i]) >= 3:
                        poly_xy = np.asarray(poly_xy_list[i], dtype=np.float32)
                        mask_bin = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(mask_bin, [poly_xy.astype(np.int32)], 255)
                        poly4 = _poly4_from_mask_findcontours(mask_bin, min_area=16.0)

                if poly4 is None:
                    if res.boxes is None or i >= len(res.boxes):
                        continue
                    bb = res.boxes.xyxy[i].cpu().numpy()
                    x1, y1 = max(0, int(bb[0])), max(0, int(bb[1]))
                    x2, y2 = min(W-1, int(bb[2])), min(H-1, int(bb[3]))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    poly4 = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)

                try:
                    crop, delta_upright = _warp_crop_from_poly4(
                        orig_bgr, poly4, keep_vertical=True, pad=seg_pad, border_value=(114,114,114)
                    )
                except Exception:
                    x0, y0 = int(np.floor(poly4[:,0].min())), int(np.floor(poly4[:,1].min()))
                    x1, y1 = int(np.ceil(poly4[:,0].max())),  int(np.ceil(poly4[:,1].max()))
                    x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)
                    crop = orig_bgr[y0:y1+1, x0:x1+1].copy()
                    delta_upright = 0.0
                    poly4 = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)

                with torch.no_grad():
                    t = preprocess_arcbook(crop, size=arcbook_img_size, device=device)
                    feat = net(t)
                    labels, scores, match = topk_search(
                        qvec=feat, index=index, topk=topk, id_lookup=id_lookup, tau=tau
                    )

                top1_score = float(scores[0]) if scores else -1.0
                passed = (len(scores) > 0 and top1_score >= float(tau))

                raw_pred_total += 1
                if reject_if_low_match and not passed:
                    if save_rejected_crops and save_crops:
                        stem = Path(img_path).stem
                        rej_dir = Path(crop_dir) / stem / "_rejected"
                        rej_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str((rej_dir / f"{i:03d}.jpg").as_posix()), crop)
                    rejected_by_tau += 1
                    continue

                crop_path = None
                if save_crops:
                    crop_path = str((save_dir / f"{i:03d}.jpg").as_posix())
                    cv2.imwrite(crop_path, crop)

                cls_id = clss[i] if i < len(clss) else 0
                conf   = confs[i] if i < len(confs) else 1.0
                cls_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else str(cls_id)

                x0, y0 = int(np.floor(poly4[:,0].min())), int(np.floor(poly4[:,1].min()))
                x1, y1 = int(np.ceil(poly4[:,0].max())),  int(np.ceil(poly4[:,1].max()))

                dets.append({
                    "det_id": i,
                    "method": "seg_contour_poly4_warp",
                    "angle_deg": float(delta_upright),
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "poly4": np.asarray(poly4, np.float32).astype(float).tolist(),
                    "conf": float(conf),
                    "cls_id": int(cls_id),
                    "cls_name": cls_name,
                    "crop_path": crop_path,
                    "matches": [{"itemID": labels[k], "score": float(scores[k])} for k in range(len(scores))],
                    "top1": {"itemID": labels[0], "score": float(scores[0])} if len(scores) > 0 else None,
                    "passed": (len(scores) > 0 and float(scores[0]) >= float(tau))
                })

        else:
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy().tolist()
                clss  = res.boxes.cls.cpu().numpy().astype(int).tolist()

                if save_crops:
                    stem = Path(img_path).stem
                    save_dir = Path(crop_dir) / stem
                    save_dir.mkdir(parents=True, exist_ok=True)

                for i, (bb, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
                    x1, y1 = max(0, int(bb[0])), max(0, int(bb[1]))
                    x2, y2 = min(W-1, int(bb[2])), min(H-1, int(bb[3]))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = orig_bgr[y1:y2+1, x1:x2+1].copy()
                    crop_path = None
                    if save_crops:
                        crop_path = str((save_dir / f"{i:03d}.jpg").as_posix())
                        cv2.imwrite(crop_path, crop)

                    with torch.no_grad():
                        t = preprocess_arcbook(crop, size=arcbook_img_size, device=device)
                        feat = net(t)
                        labels, scores, match = topk_search(
                            qvec=feat, index=index, topk=topk, id_lookup=id_lookup, tau=tau
                        )

                    cls_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else str(cls_id)
                    dets.append({
                        "poly4": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        "det_id": i,
                        "method": "bbox",
                        "angle_deg": 0.0,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "conf": float(conf),
                        "cls_id": int(cls_id),
                        "cls_name": cls_name,
                        "crop_path": crop_path,
                        "matches": [{"itemID": labels[k], "score": float(scores[k])} for k in range(len(scores))],
                        "top1": {"itemID": labels[0], "score": float(scores[0])} if len(scores) > 0 else None,
                        "passed": (len(scores) > 0 and float(scores[0]) >= float(tau))
                    })

        annot_path = None
        if save_vis:
            annot = orig_bgr.copy()
            for d in dets:
                pts = np.array(d["poly4"], dtype=np.int32)
                cv2.polylines(annot, [pts], True, (0, 255, 0), box_thickness)
                if draw_top1 and d.get("top1"):
                    x1, y1, x2, y2 = d["bbox_xyxy"]
                    t1 = d["top1"]
                    txt = f"{t1['itemID']} {t1['score']:.2f}"
                    cv2.putText(annot, txt, (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            annot_path = str((Path(vis_dir) / f"{Path(img_path).stem}.jpg").as_posix())
            cv2.imwrite(annot_path, annot)

        if len(dets) > 0:
            for d in dets:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                d["_sortkey"] = (0.5 * (x1 + x2), y1)

            dets.sort(key=lambda d: d["_sortkey"])

            for i, d in enumerate(dets):
                d["order_lr"] = i
                d.pop("_sortkey", None)
        else:
            pass

        if save_labelme_pseudo:
            shapes = []
            for d in dets:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)

                top1 = d.get("top1") or {}
                itemID = str(top1.get("itemID")) if top1 else None
                cos = float(top1.get("score")) if top1 else None
                passed = bool(d.get("passed", False))

                lbl = labelme_label
                if label_as_id and itemID is not None and (passed or not label_as_id_only_if_passed):
                    raw_id = top1.get("itemID") if top1 else None
                    title = safe_lookup_title(title_lookup, raw_id)
                    itemID_str = str(raw_id) if raw_id is not None else ""

                    lbl = itemID_str
                    if title:
                        lbl = f"{itemID_str} ({title})"

                text_field = itemID if (include_id_in_text and itemID is not None) else ""

                p4 = np.asarray(d["poly4"], dtype=np.float32).clip([0, 0], [W - 1, H - 1]).tolist()
                pts = [[float(px), float(py)] for px, py in p4]

                shapes.append({
                    "label": lbl,
                    "text": text_field,
                    "points": pts,
                    "group_id": int(d.get("order_lr", 0)),
                    "shape_type": "polygon",
                    "flags": {
                        "cls_id": int(d.get("cls_id", 0)),
                        "cls_name": d.get("cls_name", ""),
                        "det_conf": float(d.get("conf", 0.0)),
                        "faiss_top1_id": itemID,
                        "faiss_top1_cos": cos,
                        "faiss_passed": passed,
                        "angle_deg": float(d.get("angle_deg", 0.0)),
                        "method": d.get("method", "seg_upright"),
                    }
                })

            labelme_obj = {
                "version": labelme_version,
                "flags": {},
                "shapes": shapes,
                "imagePath": Path(img_path).name,
                "imageData": None,
                "imageHeight": int(H),
                "imageWidth": int(W),
                "text": ""
            }

            out_json_sidecar = Path(img_path).with_suffix(".json")

            if labelme_overwrite and out_json_sidecar.exists():
                out_json_sidecar.unlink()

            with open(out_json_sidecar, "w", encoding="utf-8") as jf:
                json.dump(labelme_obj, jf, ensure_ascii=False, indent=2)

        all_results.append({
            "image": img_path,
            "image_annotated": annot_path,
            "n_dets": len(dets),
            "detections": dets
        })

    payload = {
        "config": {
            "yolo_weights": yolo_weights,
            "arcbook_arch": arcbook_arch,
            "arcbook_weights": arcbook_weights,
            "csv_path": csv_path,
            "index_path": index_path,
            "topk": topk,
            "tau": tau,
            "img_size": arcbook_img_size,
            "yolo_imgsz": list(yolo_imgsz),
            "conf_thres": conf_thres,
            "save_crops": save_crops,
            "crop_dir": crop_dir,
            "save_vis": save_vis,
            "vis_dir": vis_dir,
            "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "results": all_results
    }

    if eval_mode:

        wrong_pred_counter = Counter()
        wrong_pair_counter = Counter()
        wrong_events_by_pred_id = defaultdict(list)

        total_imgs_with_gt = 0
        total_gt = 0
        total_pred = 0
        total_matched = 0

        e2e_tp_top1 = 0
        e2e_pred_cnt = 0
        cls_tp_on_pairs = 0
        cls_pair_cnt = 0

        top_hits = {f"top{k}": 0 for k in eval_topk_list}
        top_counts = {f"top{k}": 0 for k in eval_topk_list}

        e2e_top1_hits = 0
        per_image_records = []

        for R in all_results:
            img_path = R["image"]
            dets = R["detections"]
            if eval_only_passed:
                preds = [d for d in dets if d.get("passed", False)]
            else:
                preds = dets

            sidecar = Path(img_path).with_suffix(".json")
            gts = _load_labelme_gt(sidecar)

            H = None; W = None
            try:
                im0 = cv2.imread(img_path)
                if im0 is not None:
                    H, W = im0.shape[:2]
            except Exception:
                pass
            if H is None or W is None:
                continue

            if len(gts) == 0:
                continue

            total_imgs_with_gt += 1
            total_gt += len(gts)
            total_pred += len(preds)

            pairs = _greedy_match_by_iou(preds, gts, H, W, iou_thr=eval_iou_thr)
            total_matched += len(pairs)
            e2e_pred_cnt += len(preds)

            _update_wrong_id_stats_detailed(
                wrong_pred_counter,
                wrong_pair_counter,
                wrong_events_by_pred_id,
                img_path=img_path,
                preds_used=preds,
                gts=gts,
                pairs=pairs,
                keep_examples_per_id=5,
                keep_topk=10
            )

            correct_on_pairs_top1 = 0
            for (pi, gi, iou) in pairs:
                p = preds[pi]
                g = gts[gi]
                top1 = p.get("top1") or {}
                pred_id_top1 = str(top1.get("itemID")) if top1 else None
                gt_id = str(g["id"])
                if pred_id_top1 is not None and pred_id_top1 == gt_id:
                    correct_on_pairs_top1 += 1

            cls_tp_on_pairs += correct_on_pairs_top1
            cls_pair_cnt    += len(pairs)
            e2e_tp_top1     += correct_on_pairs_top1


            for (pi, gi, iou) in pairs:
                p = preds[pi]
                g = gts[gi]
                gt_id = str(g["id"])

                tk = _eval_topk_hit(p, gt_id, ks=eval_topk_list)
                for k in eval_topk_list:
                    key = f"top{k}"
                    top_hits[key] += int(tk[key])
                    top_counts[key] += 1

                top1 = (p.get("top1") or {})
                pred_id_top1 = str(top1.get("itemID")) if top1 else None
                if pred_id_top1 is not None and pred_id_top1 == gt_id:
                    e2e_top1_hits += 1

                if eval_pairs_csv:
                    per_image_records.append({
                        "image": Path(img_path).name,
                        "pred_idx": pi,
                        "gt_idx": gi,
                        "iou": round(float(iou), 4),
                        "gt_id": gt_id,
                        "pred_top1_id": str(pred_id_top1) if pred_id_top1 is not None else "",
                        "pred_top1_score": float(top1.get("score", -1.0)) if top1 else -1.0
                    })

        prec = (total_matched / total_pred) if total_pred > 0 else 0.0
        rec  = (total_matched / total_gt)   if total_gt   > 0 else 0.0
        f1   = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0

        prec_e2e_top1 = (e2e_tp_top1 / e2e_pred_cnt) if e2e_pred_cnt > 0 else 0.0
        rec_e2e_top1  = (e2e_tp_top1 / total_gt)     if total_gt       > 0 else 0.0
        f1_e2e_top1   = (2*prec_e2e_top1*rec_e2e_top1 / (prec_e2e_top1 + rec_e2e_top1)
                         if (prec_e2e_top1 + rec_e2e_top1) > 0 else 0.0)

        prec_cls_top1 = (cls_tp_on_pairs / cls_pair_cnt) if cls_pair_cnt > 0 else 0.0
        rec_cls_top1  = (cls_tp_on_pairs / total_gt)     if total_gt     > 0 else 0.0
        f1_cls_top1   = (2*prec_cls_top1*rec_cls_top1 / (prec_cls_top1 + rec_cls_top1)
                         if (prec_cls_top1 + rec_cls_top1) > 0 else 0.0)

        topk_acc = {}
        for k in eval_topk_list:
            key = f"top{k}"
            denom = top_counts[key] if top_counts[key] > 0 else 1
            topk_acc[key] = top_hits[key] / denom

        e2e_top1 = (e2e_top1_hits / total_gt) if total_gt > 0 else 0.0

        tp_det = int(total_matched)
        fp_det = int(max(0, total_pred - total_matched))
        fn_det = int(max(0, total_gt - total_matched))
        tn_det = int(max(0, rejected_by_tau))

        tp_rec = int(e2e_tp_top1)
        fp_rec = int(max(0, e2e_pred_cnt - e2e_tp_top1))
        fn_rec = int(max(0, total_gt - e2e_tp_top1))
        tn_rec = int(max(0, rejected_by_tau))

        summary = {
            "images_with_gt": total_imgs_with_gt,
            "gt_instances": total_gt,
            "pred_instances_used": f"{total_pred} (only passed)" if eval_only_passed else total_pred,
            "iou_thr": eval_iou_thr,

            "matched_pairs": total_matched,
            "precision": prec,
            "recall": rec,
            "f1": f1,

            "topk_acc_on_matched": topk_acc,

            "e2e_top1_acc": e2e_top1,

            "recognition_top1": {
                "end_to_end": {
                    "precision": prec_e2e_top1,
                    "recall": rec_e2e_top1,
                    "f1": f1_e2e_top1,
                    "predictions_used": e2e_pred_cnt,
                    "tp": e2e_tp_top1
                },
                "on_matched_pairs": {
                    "precision": prec_cls_top1,
                    "recall": rec_cls_top1,
                    "f1": f1_cls_top1,
                    "pairs_used": cls_pair_cnt,
                    "tp_on_pairs": cls_tp_on_pairs
                }
            }
        }

        summary["rejection"] = {
            "raw_pred_total": int(raw_pred_total),
            "rejected_by_tau": tn_rec,
            "passed_after_tau": int(max(0, raw_pred_total - rejected_by_tau)),
        }

        summary["confusion_matrix"] = {
            "detection": {
                "TP": tp_det, "FP": fp_det, "FN": fn_det, "TN": tn_det
            },
            "recognition_top1_e2e": {
                "TP": tp_rec, "FP": fp_rec, "FN": fn_rec, "TN": tn_rec
            }
        }
        summary["recognition_top1"]["wrong_top1_pred_ids"] = [
            {
                "pred_id": pid,
                "count": cnt,
                "examples": wrong_events_by_pred_id.get(pid, [])
            }
            for pid, cnt in wrong_pred_counter.most_common(10)
        ]

        summary["recognition_top1"]["confusable_pairs_top10"] = [
            {"gt_id": gt, "pred_id": pid, "count": cnt}
            for (gt, pid), cnt in wrong_pair_counter.most_common(10)
        ]

        NUM_SHOW_PIDS = 10
        SAMPLES_PER_PID = 3
        NUM_SHOW_PAIRS = 10

        if wrong_pred_counter:
            all_pids = list(wrong_pred_counter.keys())
            chosen_pids = random.sample(all_pids, k=min(NUM_SHOW_PIDS, len(all_pids)))
            print("\n[Wrong Top-1 predictions] Random-10 (pred_id, count)")
            for rank, pid in enumerate(chosen_pids, 1):
                cnt = wrong_pred_counter[pid]
                exs = wrong_events_by_pred_id.get(pid, []) or []
                if exs:
                    k = min(SAMPLES_PER_PID, len(exs))
                    sample_exs = random.sample(exs, k=k)
                    preview = "; ".join(
                        f"{e.get('image', '')} gt={e.get('gt_id', '')} score={e.get('top1_score', -1.0):.3f}"
                        for e in sample_exs
                    )
                    rnd_ex = random.choice(sample_exs)
                    topk = (rnd_ex.get("top10") or [])[:5]
                    topk_str = ", ".join(f"{c['id']}:{c['score']:.2f}" for c in topk) if topk else ""
                else:
                    preview = "(no examples)"
                    topk_str = ""

                print(f"  {rank:2d}. pred_id={pid}  count={cnt}  | samples: {preview}")
                if topk_str:
                    print(f"       topK(sampled example): {topk_str}")
        else:
            print("\n[Wrong Top-1 predictions] none")

        if wrong_pair_counter:
            pair_items = list(wrong_pair_counter.items())
            chosen_pairs = random.sample(pair_items, k=min(NUM_SHOW_PAIRS, len(pair_items)))
            print("\n[Confusable GT→Pred pairs] Random-10")
            for rank, ((gt, pid), cnt) in enumerate(chosen_pairs, 1):
                print(f"  {rank:2d}. {gt} → {pid} : {cnt}")
        else:
            print("\n[Confusable GT→Pred pairs] none")
        print("\n[Eval] Detection P/R/F1 (IoU ≥ {:.2f})".format(eval_iou_thr))

        print("  GT   :", total_gt)
        print("  Pred :", total_pred)
        print("  Match:", total_matched)
        print("  Prec :", f"{prec:.4f}")
        print("  Rec  :", f"{rec:.4f}")
        print("  F1   :", f"{f1:.4f}")
        print("  Top‑k Acc on matched:", {k: f"{v:.4f}" for k, v in topk_acc.items()})
        print("  E2E@1 (match & top1 correct) over GT:", f"{e2e_top1:.4f}")

        print("\n[Recognition@Top1] End-to-End")
        print("  Prec :", f"{prec_e2e_top1:.4f}")
        print("  Rec  :", f"{rec_e2e_top1:.4f}")
        print("  F1   :", f"{f1_e2e_top1:.4f}")

        print("[Recognition@Top1] on IoU-matched pairs")
        print("  Prec :", f"{prec_cls_top1:.4f}")
        print("  Rec  :", f"{rec_cls_top1:.4f}")
        print("  F1   :", f"{f1_cls_top1:.4f}")

        print("\n[Confusion Matrix] Detection")
        print(f"  TP={tp_det}  FP={fp_det}  FN={fn_det}  TN={tn_det}")

        print("[Confusion Matrix] Recognition@Top1 (End-to-End)")
        print(f"  TP={tp_rec}  FP={fp_rec}  FN={fn_rec}  TN={tn_rec}")

        if eval_compute_map:
            if eval_map_iou_thresholds is None:
                eval_map_iou_thresholds = [x/100.0 for x in range(50, 100, 5)]

            map_result = _compute_map_dataset(
                all_results=all_results,
                iou_thrs=eval_map_iou_thresholds,
                use_passed_only=eval_map_only_passed,
                score_mode=eval_map_score_mode
            )

            print("\n[mAP] ({} predictions) ({} GT)".format(map_result["num_pred"], map_result["num_gt"]))
            for k, v in map_result["ap_per_iou"].items():
                print(f"  AP@{k}: {v:.4f}")
            print(f"  mAP@0.50: {map_result['map_50']:.4f}")
            print(f"  mAP@0.50:0.95: {map_result['map_50_95']:.4f}")

            summary["mAP"] = {
                "ap_per_iou": map_result["ap_per_iou"],
                "mAP50": map_result["map_50"],
                "mAP50_95": map_result["map_50_95"],
                "num_gt": map_result["num_gt"],
                "num_pred": map_result["num_pred"],
                "use_passed_only": eval_map_only_passed,
                "score_mode": eval_map_score_mode,
            }
        eval_gt_dir = img_dir
        eval_iou_thr = 0.5
        eval_tau = tau

        aucpr = _compute_auc_pr_curves(
            all_results=all_results,
            gt_root=eval_gt_dir,
            iou_thr=eval_iou_thr,
            tau=eval_tau,
            vis_dir=vis_dir,
            include_unmatched_as_neg=True
        )

        auc_raw = float(aucpr.get("auc_pr", aucpr.get("auc", 0.0)))

        prevalence = None
        if "prevalence" in aucpr:
            prevalence = float(aucpr["prevalence"])
        elif "n_pos" in aucpr and "n_neg" in aucpr:
            n_pos = int(aucpr["n_pos"]);
            n_neg = int(aucpr["n_neg"])
            total = n_pos + n_neg
            prevalence = (n_pos / total) if total > 0 else None

        if prevalence is None:
            pos = 0
            total_used = 0
            for R in all_results:
                img = R["image"]
                sidecar = Path(img).with_suffix(".json")
                gts = _load_labelme_gt(sidecar)
                if not gts:
                    continue
                im0 = cv2.imread(img)
                if im0 is None:
                    continue
                H, W = im0.shape[:2]

                preds_used = R["detections"] if not eval_only_passed else [d for d in R["detections"] if
                                                                           d.get("passed", False)]
                total_used += len(preds_used)

                pairs = _greedy_match_by_iou(preds_used, gts, H, W, iou_thr=eval_iou_thr)
                pos += len(pairs)

            prevalence = (pos / total_used) if total_used > 0 else 0.0

        if prevalence is not None and prevalence < 1.0:
            auc_norm = (auc_raw - prevalence) / (1.0 - prevalence)
            auc_norm = float(np.clip(auc_norm, 0.0, 1.0))
        else:
            auc_norm = None

        summary["auc_pr_raw"] = auc_raw
        summary["auc_pr"] = aucpr
        summary["auc_pr_normalized"] = {"value": auc_norm, "prevalence": prevalence}
        payload.setdefault("eval", {})
        payload["eval"]["auc_pr"] = aucpr
        payload["eval"]["auc_pr_normalized"] = summary["auc_pr_normalized"]

        if eval_report_json:
            Path(eval_report_json).parent.mkdir(parents=True, exist_ok=True)
            with open(eval_report_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        if eval_pairs_csv and per_image_records:
            Path(eval_pairs_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(eval_pairs_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_image_records[0].keys()))
                writer.writeheader()
                writer.writerows(per_image_records)

        print(f"[Eval] summary saved to: {eval_report_json}")
        if eval_pairs_csv:
            print(f"[Eval] pairs saved to: {eval_pairs_csv}")

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[done] saved JSON: {out_p}  images={len(image_paths)}  elapsed={time.time()-t0:.1f}s")

if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    run_yolo_arcbook_search(
        img_dir="/GT data path/",
        yolo_weights="/your model path/*.pt",
        arcbook_arch="your_model",
        arcbook_weights="/your model path/*.pt",
        csv_path="/csv path/*.csv",
        index_path="items_all.faiss",
        idmap_path=None,
        out_json="runs/search/results.json",
        save_crops=True,
        crop_dir="runs/crops",
        save_vis=True,
        vis_dir="runs/vis",
        box_thickness=2,
        draw_top1=False,
        topk=10,
        tau=0.65,
        yolo_imgsz=(1088, 416),
        conf_thres=0.4,
        arcbook_img_size=224,
        save_labelme_pseudo=False,
        reject_if_low_match=True,
        save_rejected_crops=False,

        eval_mode=True,
        eval_iou_thr=0.5,
        eval_topk_list=(1,3,5),
        eval_only_passed=True,
        eval_report_json="runs/eval/summary.json",
        eval_pairs_csv="runs/eval/pairs.csv",

        eval_compute_map=True,
        eval_map_iou_thresholds=None,
        eval_map_only_passed=True,
        eval_map_score_mode="det"
    )
