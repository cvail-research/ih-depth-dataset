"""
utils/metrics.py
 
Evaluate distance estimates against lidar ground truth.
Extensible design: add a line to the METRICS dict to include a new metric
without touching the rest of the code.
"""
 
import json
import os
import numpy as np
 
 
# ── metrics dictionary ───────────────────────────────────────────────────────
# Add any new metric here. Signature: f(pred, gt) → float.
# pred and gt are 1D arrays of valid pixels (no NaN).
 
METRICS = {
    # Metrics from the original paper
    'MAE':      lambda p, g: np.mean(np.abs(p - g)),
    'RMSE':     lambda p, g: np.sqrt(np.mean((p - g) ** 2)),
 
    # Additional metrics
    'MedAE':    lambda p, g: np.median(np.abs(p - g)),
    'RelMAE':   lambda p, g: np.mean(np.abs(p - g) / np.clip(g, 1e-6, None)),
    'delta_1':  lambda p, g: np.mean(np.maximum(p / np.clip(g, 1e-6, None),
                                                  g / np.clip(p, 1e-6, None)) < 1.25),
    # To add a new metric:
    # 'NombreMetrica': lambda pred, gt: <formula>,
}
 
 
# ── main evaluation function ─────────────────────────────────────────────────
 
def evaluate(pred, gt, method_name='', verbose=True):
    """
    Evaluate a distance estimate against lidar ground truth.
 
    Only evaluates pixels where both pred and gt are valid (not NaN).
 
    Parameters
    ----------
    pred : np.ndarray (H, W)
        Estimated distance map (m). NaN = invalid pixel.
    gt : np.ndarray (H, W)
        Lidar ground-truth distance map (m). NaN = no measurement.
    method_name : str
        Method name (for logging).
    verbose : bool
        If True, prints the results.
 
    Returns
    -------
    results : dict
        Dictionary with all metrics + metadata.
        {
            'method': str,
            'n_valid': int,          # evaluated pixels
            'coverage': float,       # fraction of valid pixels
            'MAE': float,
            'RMSE': float,
            ...
        }
    """
    # Mask of pixels valid in both arrays
    valid = ~np.isnan(pred) & ~np.isnan(gt)
    n_valid = int(np.sum(valid))
    n_total = pred.size
 
    if n_valid == 0:
        print(f"  [{method_name}] No valid pixels to evaluate.")
        return {
            'method':   method_name,
            'n_valid':  0,
            'coverage': 0.0,
            **{k: np.nan for k in METRICS}
        }
 
    p = pred[valid].flatten()
    g = gt[valid].flatten()
 
    # Compute all metrics
    results = {
        'method':   method_name,
        'n_valid':  n_valid,
        'coverage': n_valid / n_total,
    }
 
    for name, fn in METRICS.items():
        try:
            results[name] = float(fn(p, g))
        except Exception as e:
            results[name] = np.nan
            print(f"  Warning: metric '{name}' failed — {e}")
 
    if verbose:
        print(f"\n  {'Metric':<12} {'Value':>10}")
        print(f"  {'-'*24}")
        print(f"  {'Method':<12} {method_name:>10}")
        print(f"  {'N valid':<12} {n_valid:>10d}")
        print(f"  {'Coverage':<12} {results['coverage']:>9.1%}")
        print(f"  {'-'*24}")
        for name in METRICS:
            val = results[name]
            fmt = f"{val:>10.4f}" if not np.isnan(val) else f"{'NaN':>10}"
            print(f"  {name:<12} {fmt}")
 
    return results
 
 
# ── save results ─────────────────────────────────────────────────────────────
 
def save_results(results, out_dir, scene_name, method_name):
    """
    Save the metrics dictionary as JSON.
 
    Parameters
    ----------
    results : dict
        Output of evaluate().
    out_dir : str
        Output directory.
    scene_name : str
        Scene name (without extension).
    method_name : str
        Method name.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{scene_name}_{method_name}_metrics.json')
 
    # Convert NaN to null for valid JSON
    serializable = {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in results.items()
    }
 
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
 
    print(f"  Metrics saved: {out_path}")
    return out_path