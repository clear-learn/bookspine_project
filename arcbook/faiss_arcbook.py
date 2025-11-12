# faiss_search.py
import ast, json
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import faiss
import re

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import cv2
import os



def _parse_id_from_label(lbl: str):
    if lbl is None:
        return None
    m = re.match(r"\s*(\d+)", str(lbl))
    return int(m.group(1)) if m else None

def _load_labelme_gt(json_path: str):
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    H, W = int(j["imageHeight"]), int(j["imageWidth"])
    gts = []
    for sh in j.get("shapes", []):
        pts = np.array(sh.get("points", []), dtype=np.float32)
        if pts.shape[0] < 3:
            continue
        gid = _parse_id_from_label(sh.get("label", ""))
        gts.append({"poly": pts, "id": gid})
    return gts, H, W

def _poly_iou_mask(polyA: np.ndarray, polyB: np.ndarray, H: int, W: int) -> float:
    xs = np.concatenate([polyA[:,0], polyB[:,0]])
    ys = np.concatenate([polyA[:,1], polyB[:,1]])
    x0 = max(0, int(np.floor(xs.min())) - 2)
    y0 = max(0, int(np.floor(ys.min())) - 2)
    x1 = min(W-1, int(np.ceil(xs.max())) + 2)
    y1 = min(H-1, int(np.ceil(ys.max())) + 2)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    mA = np.zeros((h, w), dtype=np.uint8)
    mB = np.zeros((h, w), dtype=np.uint8)
    pA = polyA.copy(); pA[:,0]-=x0; pA[:,1]-=y0
    pB = polyB.copy(); pB[:,0]-=x0; pB[:,1]-=y0
    cv2.fillPoly(mA, [pA.astype(np.int32)], 1)
    cv2.fillPoly(mB, [pB.astype(np.int32)], 1)
    inter = (mA & mB).sum()
    union = (mA | mB).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def _match_preds_to_gts_by_iou(pred_dets: list, gt_shapes: list, H: int, W: int, iou_thr: float=0.5):
    matched = []
    gt_used = [False] * len(gt_shapes)

    for d in pred_dets:
        ppoly = np.array(d["poly4"], dtype=np.float32)  # TL,TR,BR,BL
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_shapes):
            if gt_used[j]:
                continue
            iou = _poly_iou_mask(ppoly, gt["poly"], H, W)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            gt_used[best_j] = True
            matched.append({"pred": d, "gt_id": gt_shapes[best_j]["id"], "iou": best_iou})
        else:
            matched.append({"pred": d, "gt_id": None, "iou": best_iou})
    return matched

def _compute_auc_pr_curves(
    all_results: list,
    gt_root: str,
    iou_thr: float = 0.5,
    tau: float = 0.5,
    vis_dir: Optional[str] = None,
    roc_png: Optional[str] = None,
    pr_png: Optional[str] = None,
    include_unmatched_as_neg: bool = True
):
    y_true, y_score = [], []


    for item in all_results:
        img_path = item["image"]
        dets = item.get("detections", [])
        gt_json = str(Path(gt_root) / (Path(img_path).stem + ".json"))
        if not os.path.exists(gt_json):
            continue
        gts, H, W = _load_labelme_gt(gt_json)

        matched = _match_preds_to_gts_by_iou(dets, gts, H, W, iou_thr=iou_thr)
        for m in matched:
            d = m["pred"]
            t1 = d.get("top1") or {}
            pid = t1.get("itemID", None)
            pscore = float(t1.get("score", -1.0))
            if pid is None:
                pscore = -1.0

            if m["gt_id"] is None:
                if include_unmatched_as_neg:
                    y_true.append(0)
                    y_score.append(pscore)
            else:
                y_true.append(1 if str(pid) == str(m["gt_id"]) else 0)
                y_score.append(pscore)

    if len(y_true) == 0:
        return {
            "n_samples": 0,
            "roc_auc": None,
            "best_f1": None,
            "best_precision": None,
            "best_recall": None,
            "best_threshold": None,
            "precision_at_tau": None,
            "recall_at_tau": None,
            "f1_at_tau": None,
            "roc_curve_png": None,
            "pr_curve_png": None
        }

    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)

    fpr, tpr, roc_thrs = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    prec, rec, pr_thrs = precision_recall_curve(y_true, y_score)
    eps = 1e-9
    f1_vals = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + eps)
    best_idx = int(np.nanargmax(f1_vals))
    best_f1 = float(f1_vals[best_idx])
    best_p  = float(prec[best_idx])
    best_r  = float(rec[best_idx])
    best_thr = float(pr_thrs[best_idx])

    y_pred_tau = (y_score >= float(tau)).astype(np.int32)
    tp = int(((y_pred_tau == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_tau == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_tau == 0) & (y_true == 1)).sum())
    precision_tau = float(tp / (tp + fp + eps))
    recall_tau    = float(tp / (tp + fn + eps))
    f1_tau        = float(2 * precision_tau * recall_tau / (precision_tau + recall_tau + eps))

    roc_out, pr_out = None, None
    out_dir = Path(vis_dir) if vis_dir else Path("runs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    roc_out = roc_png or str(out_dir / "roc_curve.png")
    pr_out  = pr_png  or str(out_dir / "pr_curve.png")

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_out, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(rec, prec)

    plt.scatter([best_r],[best_p])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(pr_out, dpi=150)
    plt.close()

    return {
        "n_samples": int(y_true.size),
        "roc_auc": float(roc_auc),
        "best_f1": best_f1,
        "best_precision": best_p,
        "best_recall": best_r,
        "best_threshold": best_thr,
        "precision_at_tau": precision_tau,
        "recall_at_tau": recall_tau,
        "f1_at_tau": f1_tau,
        "roc_curve_png": roc_out,
        "pr_curve_png": pr_out
    }

def safe_lookup_title(lookup: dict, item_id) -> str:
    if item_id is None:
        return ""
    if item_id in lookup:
        return lookup[item_id]
    s = str(item_id)
    if s in lookup:
        return lookup[s]
    try:
        i = int(s)
        if i in lookup:
            return lookup[i]
    except Exception:
        pass
    return ""
def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s

def _looks_like_vector(cell: str) -> bool:
    s = cell.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s.startswith("[")


def _int_candidates(tokens):
    out = []
    for t in tokens:
        ts = _strip_quotes(t)
        if re.fullmatch(r"\d+", ts):
            out.append(int(ts))
    return out

def _split_row_quote_bracket_aware(s: str, delim: str = ","):
    out, buf = [], []
    depth = 0; in_quote = False; quote_ch = None; esc = False
    for ch in s:
        if esc:
            buf.append(ch); esc=False; continue
        if ch == "\\":
            buf.append(ch); esc=True; continue
        if in_quote:
            buf.append(ch)
            if ch == quote_ch:
                in_quote = False
            continue
        if ch in ('"', "'"):
            in_quote = True; quote_ch = ch; buf.append(ch); continue
        if ch == "[":
            depth += 1; buf.append(ch); continue
        if ch == "]" and depth > 0:
            depth -= 1; buf.append(ch); continue
        if ch == delim and depth == 0 and not in_quote:
            out.append("".join(buf)); buf=[]
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out

def build_title_lookup_from_csv(
    csv_path: str,
    has_header: bool = False,
    id_col: str = None,
    title_col: str = None,
) -> dict:
    lookup: dict = {}

    if has_header and id_col and title_col:
        for enc in ("utf-8","utf-8-sig","cp949","euc-kr"):
            try:
                df = pd.read_csv(csv_path, encoding=enc, engine="python", low_memory=False)
                break
            except Exception:
                df = None
        if df is None:
            raise RuntimeError("로드 실패함.")
        if id_col not in df.columns or title_col not in df.columns:
            raise KeyError(f"col 없음: {id_col}, {title_col}")

        for _, row in df[[id_col, title_col]].iterrows():
            rid = row[id_col]
            title = str(row[title_col]).strip()
            if title == "" or pd.isna(title):
                continue
            try:
                rid_int = int(str(rid))
                lookup[rid_int] = title
                lookup[str(rid_int)] = title
            except Exception:
                lookup[str(rid)] = title
        return lookup

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            toks = _split_row_quote_bracket_aware(line)
            if not toks:
                continue
            vec_idx = None
            for j, t in enumerate(toks):
                if _looks_like_vector(t):
                    vec_idx = j
                    break
            if vec_idx is None:
                continue

            title = _strip_quotes(toks[vec_idx - 1]) if vec_idx - 1 >= 0 else ""
            if title == "":
                for j in range(vec_idx - 1, -1, -1):
                    ts = _strip_quotes(toks[j])
                    if ts and not re.fullmatch(r"\d+(\.\d+)?", ts, flags=re.I) and ts.lower() not in ("true","false"):
                        title = ts
                        break
            if title == "":
                continue

            ids = _int_candidates(toks)
            if not ids:
                continue
            rid_int = max(ids, key=lambda x: (len(str(x)), x))
            lookup[rid_int] = title
            lookup[str(rid_int)] = title

    return lookup

def _parse_vec(cell: Any, line_no: int = -1) -> np.ndarray:
    if isinstance(cell, (list, tuple, np.ndarray)):
        arr = np.asarray(cell, dtype=np.float32)
    else:
        s = str(cell).strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1].strip()
        if "[" not in s and "]" not in s and "," in s and any(ch.isdigit() for ch in s):
            s = f"[{s}]"
        try:
            arr = np.asarray(json.loads(s), dtype=np.float32)
        except Exception:
            arr = np.asarray(ast.literal_eval(s), dtype=np.float32)

    arr = np.ravel(arr).astype(np.float32, copy=False)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"[line {line_no}] invalid vector")
    return arr

def _ensure_unit_float32(v: Any) -> np.ndarray:
    try:
        import torch
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
    except Exception:
        pass
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _split_row_quote_bracket_aware(s: str, delim: str = ",") -> List[str]:
    out, buf = [], []
    depth = 0; in_quote = False; quote_ch = None; esc = False
    for ch in s:
        if esc: buf.append(ch); esc=False; continue
        if ch == "\\": buf.append(ch); esc=True; continue
        if in_quote:
            buf.append(ch)
            if ch == quote_ch: in_quote = False
            continue
        if ch in ('"', "'"):
            in_quote = True; quote_ch = ch; buf.append(ch); continue
        if ch == "[": depth += 1; buf.append(ch); continue
        if ch == "]" and depth > 0: depth -= 1; buf.append(ch); continue
        if ch == delim and depth == 0 and not in_quote:
            out.append("".join(buf)); buf=[]
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out

def _looks_vector_like(tok: str) -> bool:
    s = str(tok).strip().strip('"').strip("'")
    return s.startswith("[")

def _looks_int_token(tok: str) -> bool:
    s = str(tok).strip().strip('"').strip("'")
    return bool(re.fullmatch(r"\d+", s))

def _read_csv_auto(csv_path: str, has_header: bool = True) -> pd.DataFrame:
    ids: List[str] = []
    vecs: List[np.ndarray] = []
    bad = 0

    for enc in ("utf-8","utf-8-sig","cp949","euc-kr"):
        try:
            with open(csv_path, "r", encoding=enc, errors="replace") as f:
                first = True
                for ln, line in enumerate(f, start=1):
                    if first and has_header:
                        first = False
                        continue
                    s = line.rstrip("\n")
                    if not s:
                        continue
                    toks = _split_row_quote_bracket_aware(s)

                    vec_j = None
                    for j in range(len(toks)-1, -1, -1):
                        if _looks_vector_like(toks[j]):
                            vec_j = j; break
                    if vec_j is None:
                        bad += 1
                        if bad <= 5:
                            print(f"[warn] line {ln}: vector-like token not found -> skip")
                        continue

                    id_cands = []
                    for j, tok in enumerate(toks):
                        if _looks_int_token(tok):
                            val = str(tok).strip().strip('"').strip("'")
                            id_cands.append((j, val))
                    if not id_cands:
                        bad += 1
                        if bad <= 5:
                            print(f"[warn] line {ln}: integer ID token not found -> skip")
                        continue
                    j_best, id_str = max(id_cands, key=lambda p: (len(p[1]), -p[0]))
                    try:
                        vec = _parse_vec(toks[vec_j], line_no=ln)
                        ids.append(int(id_str))  # 숫자 ID로 저장
                        vecs.append(vec)
                    except Exception as e:
                        bad += 1
                        if bad <= 5:
                            print(f"[warn] line {ln}: {e} -> skip")
                        continue
            break
        except UnicodeDecodeError:
            ids, vecs, bad = [], [], 0
            continue

    if not ids:
        raise RuntimeError("CSV 자동 파싱 실패: 인코딩/포맷을 확인하세요.")
    if bad:
        print(f"[info] skipped {bad} malformed rows")
    return pd.DataFrame({"itemID": ids, "vectorDB": vecs})

def _read_csv_named(csv_path: str, id_col: str, vec_col: str) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp949","euc-kr"):
        try:
            df = pd.read_csv(
                csv_path, encoding=enc, engine="python",
                sep=",", quotechar='"', escapechar="\\",
                low_memory=False, on_bad_lines="error"
            )
            if id_col not in df.columns or vec_col not in df.columns:
                raise KeyError("필수 컬럼 없음")
            return df[[id_col, vec_col]]
        except Exception:
            pass
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n")
    cols = _split_row_quote_bracket_aware(header)
    assert id_col in cols and vec_col in cols, "헤더에 필요한 컬럼이 없습니다."
    expected = len(cols)
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            toks = _split_row_quote_bracket_aware(line.rstrip("\n"))
            if len(toks) == expected:
                rows.append(toks)
            elif len(toks) > expected:
                toks = toks[:expected-2] + [",".join(toks[expected-2:-1])] + [toks[-1]]
                rows.append(toks)
            else:
                continue
    df = pd.DataFrame(rows, columns=cols)
    return df[[id_col, vec_col]]

def _read_csv_positional(csv_path: str, id_idx: int, vec_idx: int,
                         has_header: bool) -> pd.DataFrame:
    ids, vecs = [], []
    for enc in ("utf-8","utf-8-sig","cp949","euc-kr"):
        try:
            with open(csv_path, "r", encoding=enc, errors="replace") as f:
                first = True
                for line in f:
                    if first and has_header:
                        first = False
                        continue
                    s = line.rstrip("\n")
                    if not s:
                        continue
                    toks = _split_row_quote_bracket_aware(s)
                    iid = toks[id_idx if id_idx >= 0 else len(toks)+id_idx]
                    vsv = toks[vec_idx if vec_idx >= 0 else len(toks)+vec_idx]
                    ids.append(iid)
                    vecs.append(_parse_vec(vsv))
            break
        except UnicodeDecodeError:
            ids, vecs = [], []
            continue
    if not ids:
        raise RuntimeError("CSV를 읽지 못했습니다. 인코딩/인덱스를 확인하세요.")
    return pd.DataFrame({"itemID": ids, "vectorDB": vecs})

def build_or_load_index(
    csv_path: str,
    id_col: Optional[str] = None,
    vec_col: Optional[str] = None,
    id_idx: Optional[int] = None,
    vec_idx: Optional[int] = None,
    has_header: bool = True,
    auto: bool = False,
    index_path: str = "items.faiss",
    idmap_path: Optional[str] = "id_map.parquet",
) -> Tuple[faiss.Index, Optional[Dict[int, str]]]:

    p_index = Path(index_path)
    p_map   = Path(idmap_path) if idmap_path else None

    if p_index.exists():
        index = faiss.read_index(str(p_index))
        id_lookup: Optional[Dict[int, str]] = None
        if p_map and p_map.exists():
            df_map = pd.read_parquet(p_map) if p_map.suffix.lower()==".parquet" else pd.read_csv(p_map)
            id_lookup = df_map.set_index("int64_id")["itemID"].to_dict()
        return index, id_lookup

    if auto or (id_col is None and vec_col is None and id_idx is None and vec_idx is None):
        df = _read_csv_auto(csv_path, has_header=has_header)
    elif id_col is not None and vec_col is not None:
        df = _read_csv_named(csv_path, id_col=id_col, vec_col=vec_col)
        df = df.rename(columns={id_col: "itemID", vec_col: "vectorDB"})
    else:
        if id_idx is None or vec_idx is None:
            raise ValueError("헤더가 없으면 id_idx/vec_idx를 지정하거나 auto=True를 사용하세요.")
        df = _read_csv_positional(csv_path, id_idx=id_idx, vec_idx=vec_idx, has_header=has_header)

    vecs = np.stack(df["vectorDB"].apply(lambda x: _parse_vec(x)).values).astype("float32", copy=False)
    faiss.normalize_L2(vecs)
    d = vecs.shape[1]
    base = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap2(base)

    ids_series = df["itemID"]
    if np.issubdtype(ids_series.dtype, np.number):
        ids = ids_series.astype(np.int64).values
        id_lookup = None
    else:
        if ids_series.map(lambda x: bool(re.fullmatch(r"\d+", str(x)))).all():
            ids = ids_series.astype(np.int64).values
            id_lookup = None
        else:
            codes, uniques = pd.factorize(ids_series, sort=False)
            ids = codes.astype(np.int64)
            idmap_df = pd.DataFrame({"int64_id": np.arange(len(uniques), dtype=np.int64),
                                     "itemID": uniques})
            if p_map is None:
                p_map = Path("id_map.parquet")
            if p_map.suffix.lower() == ".parquet":
                idmap_df.to_parquet(p_map, index=False)
            else:
                idmap_df.to_csv(p_map, index=False)
            id_lookup = idmap_df.set_index("int64_id")["itemID"].to_dict()

    mask = np.isfinite(vecs).all(axis=1)
    if mask.sum() != len(vecs):
        vecs, ids = vecs[mask], ids[mask]

    index.add_with_ids(vecs, ids)
    faiss.write_index(index, str(p_index))
    return index, id_lookup

def topk_search(
    qvec: Any,
    index: faiss.Index,
    topk: int = 10,
    id_lookup: Optional[Dict[int, Any]] = None,
    tau: Optional[float] = None,
) -> Tuple[List[Any], List[float], Optional[Any]]:
    q = qvec.squeeze() if hasattr(qvec, "squeeze") else qvec
    q = _ensure_unit_float32(q)
    D, I = index.search(q[None, :], topk)
    ids    = I[0].tolist()
    scores = D[0].tolist()
    labels = [id_lookup.get(i, i) if id_lookup else i for i in ids]
    match  = labels[0] if (tau is not None and len(scores) > 0 and scores[0] >= tau) else None
    return labels, scores, match
