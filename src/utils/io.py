"""File, config and checkpoint I/O, plus readers for saved eval and sweep outputs."""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, Any, List
import itertools
import copy
import importlib
import numpy as np
import pandas as pd
import torch


# Directory paths ----------------------------------------------------------------------------------------------------

def get_root() -> str:
    """Repository root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def add_root(dir_name=None) -> str:
    '''Returns dir_name if it's already rooted at get_root(), otherwise joins with get_root().'''
    if dir_name is None:
        return None
    if dir_name.startswith(get_root()):
        return dir_name
    return os.path.join(get_root(), dir_name)

def localize_path(path=None) -> str:
    '''Re-root a path under the local repo root.

    Absolute paths saved on another machine are re-rooted at the last path segment
    named like the repo root. Relative paths are joined with get_root(); existing or
    unrecognisable absolute paths are returned unchanged.
    '''
    if path is None:
        return None
    root = get_root()
    if not os.path.isabs(path):
        return os.path.normpath(os.path.join(root, path))
    if path.startswith(root) or os.path.exists(path):
        return path
    # Foreign absolute path: split at the last repo-root-named segment and re-root.
    root_name = os.path.basename(root)
    parts = os.path.normpath(path).split(os.sep)
    if root_name in parts:
        idx = len(parts) - 1 - parts[::-1].index(root_name)
        return os.path.join(root, *parts[idx + 1:])
    return path

def merge_output_dir(config, cli_output_dir):
    """Set config["output_dir"] with precedence CLI > config > default."""
    default_dir = f"./outputs/default"
    if cli_output_dir is not None:
        if "output_dir" in config and config["output_dir"] != cli_output_dir:
            print(f"Warning: Overriding config output_dir '{config['output_dir']}'\
                    with CLI argument '{cli_output_dir}'.")
        config["output_dir"] = cli_output_dir
    elif "output_dir" not in config or not config["output_dir"]:
        config["output_dir"] = default_dir
    return config


# Config management ---------------------------------------------------------------------------------------------------

def save_json(d: Dict[str, Any], save_path: str) -> None:
    """Save a dict as indented JSON (non-serialisable values via str)."""
    with open(save_path, 'w') as f:
        json.dump(d, f, indent=2, default=str)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file as a dict."""
    try:
        with open(file_path, 'r') as f:
            d = json.load(f)
            return d
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration file: {e}")

def nested_dicts_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Recursively compare two nested dicts; leaves are compared with ==."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, dict) and isinstance(vb, dict):
            if not nested_dicts_equal(va, vb):
                return False
        else:
            if va != vb:
                return False
    return True

_SCALAR_TYPES = (int, float, str, bool, type(None))

def _collect_sweep_lists(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, List]:
    """Recursively find keys whose value is a flat list of scalars (sweep dimension)."""
    out = {}
    for k, v in cfg.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, list):
            if len(v) > 0 and all(isinstance(x, _SCALAR_TYPES) for x in v):
                out[path] = v
        elif isinstance(v, dict):
            out.update(_collect_sweep_lists(v, path))
    return out


def _set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """Set a nested key by dot-separated path (e.g. 'settings.lr' or 'components.dynamics.params.hidden_dim')."""
    parts = path.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value


def expand_job_configs(config):
    """Expand every list-of-scalars field (at any depth) into the Cartesian product of job configs.

    Lists of dicts are not expanded; without sweep fields, returns [config].
    """
    sweep_params = _collect_sweep_lists(config)
    if not sweep_params:
        return [config]

    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    configs = []
    for combo in itertools.product(*values):
        new_cfg = copy.deepcopy(config)
        for k, val in zip(keys, combo):
            _set_by_path(new_cfg, k, val)
        configs.append(new_cfg)
    return configs


# Component loading ------------------------------------------------------------------------------------------------

def import_from_path(path: str) -> Any:
    """Import a class from 'module.submodule:ClassName'."""
    if ":" not in path:
        raise ValueError(f"Invalid path argument '{path}'. Expected format: 'module.submodule:ClassName'")
    module_path, class_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"Module '{module_path}' has no attribute '{class_name}'")

def load_run_config(run_dir: str) -> Dict[str, Any]:
    """Load run_dir/config.json."""
    try:
        return load_json(os.path.join(run_dir, "config.json"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {os.path.join(run_dir, 'config.json')}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration file: {e}")

def _get_component_spec(cfg: Dict[str, Any], component_name: str) -> Dict[str, Any]:
    """Validated component spec {'kind': ..., 'params': {...}}; missing params become {}."""
    comps = cfg.get("components", None)
    if comps is None or not isinstance(comps, dict):
        raise KeyError("Run config does not contain a 'components' dict.")

    if component_name not in comps:
        raise KeyError(
            f"Component '{component_name}' not found in run config.")

    spec = comps[component_name]
    if not isinstance(spec, dict):
        raise ValueError(f"Component spec for '{component_name}' must be a dict, got {type(spec)}")

    if "kind" not in spec:
        raise ValueError(f"Component spec for '{component_name}' missing required key 'kind'")
    if "params" not in spec:
        spec = dict(spec)
        spec["params"] = {}

    if not isinstance(spec["params"], dict):
        raise ValueError(
            f"Component spec params for '{component_name}' must be a dict, got {type(spec['params'])}")

    return spec

def build_component_from_run(run_dir: str, component_name: str) -> Any:
    """Instantiate a component from run_dir/config.json without loading weights."""
    cfg = load_run_config(run_dir)
    spec = _get_component_spec(cfg, component_name)

    from src.experiments.base import Component
    params = copy.deepcopy(spec.get("params", {}))
    comp = Component(component=component_name, kind=spec["kind"], params=params)
    return comp.build()

def load_component_from_run(run_dir: str, name: str, device: torch.device = None) -> Any:
    """Build a component from run_dir/config.json and load run_dir/{name}.pth into it."""
    weights_path = os.path.join(run_dir, f"{name}.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    obj = build_component_from_run(run_dir, name)

    if not hasattr(obj, "load_state_dict"):
        raise TypeError(
            f"Built component '{name}' (type={type(obj)}) does not implement load_state_dict(). "
             "This loader is intended for torch.nn.Module components.")

    if device is None:
        device = torch.device("cpu")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Accept both a bare state_dict and {'state_dict': ...}.
    if isinstance(state_dict, dict) and "state_dict" in state_dict and all(
        isinstance(k, str) for k in state_dict["state_dict"].keys()):
        state_dict = state_dict["state_dict"]

    # Drop unexpected keys, e.g. the recognition B, which is re-synced via set_B_from_dynamics.
    model_keys = set(obj.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    obj.to(device)
    obj.load_state_dict(state_dict, strict=True)

    return obj


# Eval-output loading ------------------------------------------------------------------------------------------------
# Readers for saved eval outputs; they never run the eval experiment.

def load_eval_data(eval_dir: str) -> dict:
    """Load metrics, example trajectories, B summary and affine maps from an EvalISSM output dir."""
    data = {"eval_dir": eval_dir}

    metrics_path = os.path.join(eval_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.csv not found in {eval_dir}")
    data["metrics"] = pd.read_csv(metrics_path)

    npz_path = os.path.join(eval_dir, "example_plot_data.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"example_plot_data.npz not found in {eval_dir}")
    npz = np.load(npz_path)
    data["u"]       = npz["u"]
    data["y_true"]  = npz["y_true"]
    data["y_mean"]  = npz["y_mean"]
    data["y_std"]   = npz["y_std"]
    data["x_true"]  = npz["x_true"]
    data["x_pdc_mean"] = npz["x_pdc_aligned_mean"]
    data["x_pdc_std"]  = npz["x_pdc_aligned_std"]
    data["dt"]      = float(npz["dt"]) if "dt" in npz else 1.0

    b_path = os.path.join(eval_dir, "B_matrix_summary.npz")
    if os.path.exists(b_path):
        b_npz = np.load(b_path)
        data["B_abs_mean"] = b_npz["B_abs_mean"]
        data["B_abs_std"]  = b_npz["B_abs_std"]
        data["B_true"]     = b_npz["B_true"]

    affine_path = os.path.join(eval_dir, "affine_maps_per_seed.json")
    if os.path.exists(affine_path):
        with open(affine_path) as fh:
            data["affine_maps"] = json.load(fh)

    return data


def _build_learned_flow_fn(dynamics, perm, d, c, device):
    """Return a callable flow function in true latent coordinates."""
    perm = np.asarray(perm)
    inv_perm = np.argsort(perm)
    d_arr = np.asarray(d, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)

    @torch.no_grad()
    def f(pts):
        x_aligned = pts * d_arr[None, :] + c_arr[None, :]
        x_learned  = x_aligned[:, inv_perm]
        t = torch.tensor(x_learned, dtype=torch.get_default_dtype(), device=device)
        v_learned  = dynamics.flow(t).cpu().numpy()
        v_aligned  = v_learned[:, perm]
        return v_aligned / (d_arr[None, :] + 1e-12)
    return f


def _build_true_flow_fn(system, device):
    @torch.no_grad()
    def f(pts):
        t = torch.tensor(pts, dtype=torch.get_default_dtype(), device=device)
        return system.flow(t).cpu().numpy()
    return f


def load_best_seed_flow_fns(eval_data: dict, device: torch.device):
    """(true_fn, learned_fn, x_true) for the seed with the highest jac_cosine, or three Nones."""
    metrics      = eval_data["metrics"]
    affine_maps  = eval_data.get("affine_maps")

    if affine_maps is None:
        return None, None, None
    if "jac_cosine" not in metrics.columns or metrics["jac_cosine"].isna().all():
        return None, None, None

    best_seed = int(metrics.loc[metrics["jac_cosine"].idxmax(), "seed"])
    seed_key  = f"seed_{best_seed}"
    if seed_key not in affine_maps:
        return None, None, None

    seed_info = affine_maps[seed_key]
    run_dir   = localize_path(seed_info.get("run_dir"))
    if run_dir is None or not os.path.isdir(run_dir):
        warnings.warn(f"run_dir not found for {seed_key}: {run_dir}")
        return None, None, None

    try:
        dynamics = load_component_from_run(run_dir, "dynamics", device)
        system   = build_component_from_run(run_dir, "system")
    except Exception as exc:
        warnings.warn(f"Could not load model for flow field: {exc}")
        return None, None, None

    perm = seed_info["perm"]
    d    = seed_info["d"]
    c    = seed_info["c_diag"]

    true_fn    = _build_true_flow_fn(system, device)
    learned_fn = _build_learned_flow_fn(dynamics, perm, d, c, device)
    x_data     = eval_data["x_true"]

    return true_fn, learned_fn, x_data


def load_scale_aligned_B(eval_data: dict, device: torch.device = None,
                         eps: float = 1e-6) -> "np.ndarray | None":
    """Median across seeds of B in true latent coordinates, shape (D, M), or None.

    Latent scale is only identifiable up to a per-latent factor, so row j of each
    permuted B is divided by the affine slope d[j] (x_learned = d * x_true + c), as
    for the learned flow; the sign of d also fixes sign flips. Seeds with any
    |d| < eps are skipped.
    """
    affine = eval_data.get("affine_maps")
    if not affine:
        return None
    device = device or torch.device("cpu")

    aligned = []
    for seed_key in sorted(affine):
        info    = affine[seed_key]
        run_dir = localize_path(info.get("run_dir"))
        if run_dir is None or not os.path.isdir(run_dir):
            warnings.warn(f"run_dir not found for {seed_key}: {run_dir}")
            continue
        try:
            dynamics = load_component_from_run(run_dir, "dynamics", device)
            B = dynamics.B_effective.detach().cpu().numpy()
        except Exception as exc:
            warnings.warn(f"Could not load B for {seed_key}: {exc}")
            continue

        perm = np.asarray(info["perm"], dtype=int)
        d    = np.asarray(info["d"], dtype=float)
        if B.shape[0] != perm.size or d.size != perm.size:
            warnings.warn(
                f"{seed_key}: B has {B.shape[0]} rows but perm/d have "
                f"{perm.size}/{d.size}; skipping.")
            continue
        if np.any(np.abs(d) < eps):
            warnings.warn(
                f"{seed_key}: degenerate affine scale d={d}; skipping in the "
                "scale-aligned B.")
            continue
        aligned.append(B[perm, :] / d[:, None])

    if not aligned:
        return None
    return np.median(np.stack(aligned), axis=0)


# Sweep loading (metric vs. a swept config parameter across job_*/ dirs) ----------------------------------------------

def _resolve_dotted(config: dict, dotted_key: str):
    """Resolve a dotted path like 'settings.D' into config['settings']['D']."""
    parts = dotted_key.split(".")
    node = config
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Key '{part}' not found (path: '{dotted_key}')")
        node = node[part]
    return node


def _load_eval_metric(job_dir: Path, metric: str, eval_subdir: str) -> List[float]:
    """Load metric values from a job's eval metrics.csv (one row per seed)."""
    csv_path = job_dir / eval_subdir / "metrics.csv"
    try:
        df = pd.read_csv(csv_path)
        if metric not in df.columns:
            warnings.warn(f"Metric '{metric}' not in {csv_path}.")
            return []
        return df[metric].dropna().tolist()
    except Exception as e:
        warnings.warn(f"Could not read {csv_path}: {e}.")
        return []


def load_sweep(run_dir: str, metric: str, config_param: str,
               eval_subdir: str = "eval") -> list:
    """Sorted [(x_value, mean, sem, raw_array)] across job_*/ dirs in run_dir.

    Jobs without eval_subdir/metrics.csv are skipped; raises if none has eval results.
    """
    run_path = Path(run_dir)
    job_dirs = sorted(run_path.glob("job_*"))
    if not job_dirs:
        warnings.warn(f"No job_* subdirs found in {run_dir}; returning empty records.")
        return []

    records = []
    n_missing_eval = 0
    for job_dir in job_dirs:
        seed_dirs = sorted(job_dir.glob("seed_*"))
        if not seed_dirs:
            warnings.warn(f"No seed dirs in {job_dir}, skipping.")
            continue

        config_path = seed_dirs[0] / "config.json"
        try:
            with open(config_path) as f:
                config = json.load(f)
            x_val = _resolve_dotted(config, config_param)
        except (FileNotFoundError, KeyError) as e:
            warnings.warn(f"Could not resolve '{config_param}' in {config_path}: {e}. Skipping.")
            continue

        if not (job_dir / eval_subdir / "metrics.csv").exists():
            n_missing_eval += 1
            warnings.warn(f"No {eval_subdir}/metrics.csv in {job_dir}; run the eval "
                          "experiment first. Skipping this job.")
            continue

        values = _load_eval_metric(job_dir, metric, eval_subdir)
        if not values:
            warnings.warn(f"No valid values for {job_dir.name}, skipping.")
            continue

        arr = np.array(values)
        sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        records.append((x_val, arr.mean(), sem, arr))

    if not records and n_missing_eval:
        raise FileNotFoundError(
            f"No eval results found under any job_* in {run_dir}. Run the eval "
            f"experiment first, e.g.:\n"
            f"    python -m src.experiments.run_ex -x <EvalISSM|"
            f"EvalKianiISSM|EvalHuangISSM> -s group_dir=<job_dir> -o <job_dir>/{eval_subdir}")

    records.sort(key=lambda r: r[0])
    return records


def scale_records(records: list, scale: float) -> list:
    """Multiply the x-value of each sweep record by scale."""
    return [(x * scale, m, s, vals) for x, m, s, vals in records]


# Sweep-dir discovery helpers ----------------------------------------------------------------------------------------

def available_dir(path: "str | None") -> bool:
    """True if path is a non-empty string naming an existing directory."""
    return bool(path and os.path.isdir(path))


def read_dt_from_dir(train_dir: str) -> float:
    """Read dt from the first seed_*/config.json found in train_dir (default 1.0)."""
    cfg_path = next(Path(train_dir).glob("seed_*/config.json"), None)
    if cfg_path is None:
        return 1.0
    with open(cfg_path) as f:
        cfg = json.load(f)
    return float(cfg.get("settings", {}).get("dt", 1.0))


def read_stim_scale(sweep_dir: str) -> float:
    """Return T * n_train_samples from any config.json found under sweep_dir."""
    cfg_path = next(Path(sweep_dir).glob("job_*/seed_*/config.json"), None)
    if cfg_path is None:
        raise FileNotFoundError(f"No config.json found under {sweep_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    s = cfg["settings"]
    n_train = s["n_samples"] - s.get("n_val_samples", 0) - s.get("n_test_samples", 0)
    return float(s["T"] * n_train)


def _read_stim_rate(job_dir: Path) -> "float | None":
    """Read stim_rate from the first seed's config.json in job_dir."""
    cfg_path = next(job_dir.glob("seed_*/config.json"), None)
    if cfg_path is None:
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    try:
        return float(cfg["components"]["stimulator"]["params"]["stim_rate"])
    except (KeyError, TypeError):
        return None


def find_stim_job(vdir: Path, which: str) -> "Path | None":
    """Return vdir's job_* subdir with the max ('max') or zero ('zero') stim_rate."""
    jobs = sorted(vdir.glob("job_*"))
    rated = [(r, j) for j in jobs if (r := _read_stim_rate(j)) is not None]
    if not rated:
        return None
    pick = min(rated, key=lambda rj: abs(rj[0])) if which == "zero" \
        else max(rated, key=lambda rj: rj[0])
    return pick[1]
