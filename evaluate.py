"""Scoring — read cached preds + dataset GT, align, compute aggregate + boundary
metrics per image. Pure functions of the cache; safe to re-run when a metric
definition changes. Emits long-format rows (one dict per image) for results.csv.
"""
from pathlib import Path
import numpy as np

from align import scale_shift_align
from score.aggregate import aggregate_metrics
from score import boundary as B
from score.pointcloud import flying_pixel_fraction
import data.loaders as loaders
from predict import load_cached_pred, precision_tag, CACHE_ROOT
from models.registry import MODELS


def score_sample(sample, pred_raw, info, model_kind="relative", align_mode="ss",
                 compute_boundary=True, boundary_ks=(3,)):
    md, xd = info["min_depth"], info["max_depth"]
    # Alignment is a property of the MODEL, not the dataset: relative/affine-
    # invariant models predict disparity and need scale-shift; metric models
    # output meters directly and must NOT be aligned (that is the clean gate).
    if model_kind == "relative" and align_mode != "none":
        aligned, (s, t), resid = scale_shift_align(
            pred_raw, sample.gt_depth, sample.valid_mask, mode=align_mode,
            pred_is_disparity=True, min_depth=md, max_depth=xd)
    else:
        aligned, s, t, resid = np.asarray(pred_raw, float), np.nan, np.nan, np.nan

    row = aggregate_metrics(aligned, sample.gt_depth, sample.valid_mask, md, xd)
    row.update(align_s=s, align_t=t, align_resid=resid)

    if compute_boundary:
        gt_edges = sample.gt_edges
        derived = gt_edges is None
        if derived:
            gt_edges = B.depth_discontinuities(sample.gt_depth, sample.valid_mask, percentile=98)
        acc, com = B.dbe(aligned, gt_edges, sample.valid_mask)
        sharp, width = B.edge_sharpness(aligned, gt_edges, sample.valid_mask, k=2)
        row.update(dbe_acc=acc, dbe_com=com, edge_sharpness=sharp, edge_width=width,
                   n_gt_edge=int(np.sum(gt_edges)), edges_derived=int(derived))
        for k in boundary_ks:
            bm = B.boundary_masked_error(aligned, sample.gt_depth, gt_edges,
                                         sample.valid_mask, k=k, min_depth=md, max_depth=xd)
            row[f"boundary_absrel_k{k}"] = bm["boundary_absrel"]
            row[f"interior_absrel_k{k}"] = bm["interior_absrel"]
            row[f"n_boundary_k{k}"] = bm["n_boundary"]
        # 3D structure: flying pixels, only where intrinsics exist
        if sample.K is not None:
            fp = flying_pixel_fraction(aligned, sample.gt_depth, gt_edges,
                                       sample.valid_mask, k=3)
            row["flying_frac"] = fp["flying_frac"]
            row["n_flying"] = fp["n_flying"]
            row["n_edge_band"] = fp["n_edge_band"]
    return row


def evaluate_run(dataset_name, model_name, spec, align_mode="ss", subset=None,
                 compute_boundary=True, boundary_ks=(3,), cache_root=CACHE_ROOT,
                 dataset_kwargs=None):
    """Score one (dataset, model, precision) run. Returns (rows, info)."""
    tag = precision_tag(spec)
    gen, info = loaders.build(dataset_name, subset=subset, **(dataset_kwargs or {}))
    mspec = MODELS[model_name]
    rows = []
    for s in gen():
        pred = load_cached_pred(cache_root, dataset_name, model_name, tag, s.id)
        if pred is None:
            continue
        r = score_sample(s, pred, info, model_kind=mspec["kind"], align_mode=align_mode,
                         compute_boundary=compute_boundary, boundary_ks=boundary_ks)
        r.update(dataset=dataset_name, model=model_name, backbone=mspec["backbone"],
                 family=mspec["family"], kind=mspec["kind"], precision=tag,
                 w_bits=spec.get("w_bits", 16), a_bits=spec.get("a_bits", 16),
                 component=spec.get("component", "all"), align=align_mode, image_id=s.id)
        rows.append(r)
    return rows, info


def write_rows(rows, csv_path, append=True):
    """Column-safe append via pandas (concat aligns by NAME, not position, and
    fills missing columns with NaN). A hand-rolled per-batch DictWriter misaligns
    columns when later batches have a different key order/set."""
    import pandas as pd
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return csv_path
    new = pd.DataFrame(rows)
    if append and csv_path.exists():
        new = pd.concat([pd.read_csv(csv_path), new], ignore_index=True)
    new.to_csv(csv_path, index=False)
    return csv_path
