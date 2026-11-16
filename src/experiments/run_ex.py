"""
Entry point for running experiments.

    python -m src.experiments.run_ex -x TrainISSM -c <config.json> -o <output_dir>
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import json
import time
try:
    import wandb
except ImportError:
    wandb = None
import copy
from src.utils.io import load_json, save_json, merge_output_dir, expand_job_configs
from src.experiments.base import create_experiment


# Helper functions ----------------------------------------------------------------------------------------------

def _to_plain_dict(wb_config):
    """Convert a wandb config object to a plain dictionary."""
    try:
        return wb_config.as_dict()
    except Exception:
        try:
            return dict(wb_config)
        except Exception:
            return {}

def _deep_update(base: dict, override: dict) -> dict:
    """Deep-merge override into base, returning a new dict."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time as 'Xh Ymin Z.XXs' (omits zero h/min)."""
    h, r = divmod(int(seconds), 3600)
    m, _ = divmod(r, 60)
    secs = seconds - 3600 * h - 60 * m
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}min")
    parts.append(f"{secs:.2f}s")
    return " ".join(parts)


# Main function ------------------------------------------------------------------------------------------------

def main():
    """Run and log an experiment, e.g.

    python -m src.experiments.run_ex -x TrainISSM -c configs/toymodels/saddle.json
        -o outputs/saddle -j 0 --group_seeds --seed_idx 0
    """
    parser = argparse.ArgumentParser(description="Run and log an experiment.")
    parser.add_argument("--experiment", "-x", type=str, required=True,
                        help="Name of the Experiment subclass, e.g. TrainISSM")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to JSON config file (optional; omit to use registry defaults).")
    parser.add_argument("--setting", "-s", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Override a settings key, e.g. -s group_dir=outputs/myrun. Repeatable.")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Optional override for output directory.")
    parser.add_argument("--job_id", "-j", type=int, default=None,
                        help="Optional job ID for array jobs (0-indexed).")
    parser.add_argument("--group_seeds", action="store_true",
                        help="Use per-seed subdirectories under the job dir (for array jobs).")
    parser.add_argument("--seed_idx", type=int, default=0,
                        help="Seed index for this run; actual seed = base_seed + seed_idx (e.g. an array index).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output (overrides config).")
    args = parser.parse_args()

    if args.seed_idx < 0:
        raise ValueError(f"seed_idx must be >= 0, got {args.seed_idx}.")

    # Parse --setting KEY=VALUE overrides
    settings_overrides: dict = {}
    for kv in args.setting:
        if "=" not in kv:
            raise ValueError(f"--setting must be KEY=VALUE, got: {kv!r}")
        k, v = kv.split("=", 1)
        raw = v.strip()
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        settings_overrides[k.strip()] = raw

    # Load config (optional); fall back to empty dict when -c is omitted
    base_config = load_json(args.config) if args.config is not None else {}

    # Apply --setting overrides into settings dict
    if settings_overrides:
        base_config.setdefault("settings", {})
        base_config["settings"].update(settings_overrides)

    if args.job_id is not None:
        expanded_configs = expand_job_configs(base_config)
        if args.job_id >= len(expanded_configs):
            raise ValueError(f"job_id {args.job_id} out of range (found {len(expanded_configs)} configs).")
        config = expanded_configs[args.job_id]
        print(f"Loaded job-specific config for job_id={args.job_id}/{len(expanded_configs)-1}")
    else:
        config = base_config

    config["job_id"] = args.job_id if args.job_id is not None else 0

    # Merge/override output_dir
    config["experiment"] = args.experiment
    config = merge_output_dir(config, args.output_dir)

    # If array job, nest under job folder
    if args.job_id is not None:
        config["output_dir"] = os.path.join(config["output_dir"], f"job_{args.job_id}")

    # Seed: base_seed from config (or 0) + seed_idx
    base_seed = config.get("seed", 0)
    config["seed"] = base_seed + args.seed_idx

    # CLI overrides for verbose
    if args.verbose:
        config["verbose"] = True

    # Optional per-seed output subdirectory
    if args.group_seeds:
        actual_seed = config["seed"]
        config["output_dir"] = os.path.join(config["output_dir"], f"seed_{actual_seed:03d}")

    # Initialize logging
    _EVAL_EXPERIMENTS = {"EvalISSM", "EvalKianiISSM", "EvalHuangISSM", "AttractorHuangISSM"}
    log_cfg = config.get("logging", {})
    logger_type = log_cfg.get("logger", "local")
    log_params = log_cfg.get("params", {}) if isinstance(log_cfg, dict) else {}
    if args.experiment in _EVAL_EXPERIMENTS and logger_type == "wandb":
        raise ValueError(f"{args.experiment} does not support wandb logging.")

    run = None
    effective_config = config

    if logger_type == "wandb":
        if wandb is None:
            raise ImportError("logging.logger is 'wandb' but wandb is not installed.")
        project = log_params.get("project", "default-project")
        entity = log_params.get("entity", None)
        run_name = log_params.get("run_name", None)
        mode = log_params.get("mode", "online")

        if mode != "disabled":
            os.makedirs(config["output_dir"], exist_ok=True)
            run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                mode=mode,
                config=config,
                dir=config["output_dir"],
                reinit=True
            )
            print(f"wandb run initialized: {wandb.run.name}")

            # Read back the final config (with sweep overrides) and deep-merge
            wb_cfg_dict = _to_plain_dict(wandb.config)
            effective_config = _deep_update(config, wb_cfg_dict)

            # Create separate directory for wandb runs
            run_id = run.id
            output_dir = os.path.join(config["output_dir"], f"wandb_{run_id}")
            os.makedirs(output_dir, exist_ok=True)
            effective_config["output_dir"] = output_dir

        else:
            print("wandb mode set to 'disabled'; proceeding without logging")
    else:
        print(f"Logging mode: {logger_type}")

    # Create and validate experiment
    ex = create_experiment(args.experiment, effective_config)
    if ex.verbose:
        print(ex)

    # preprocess_config may have changed output_dir; ensure it exists
    os.makedirs(ex.output_dir, exist_ok=True)

    # Save resolved config.json (ex.config is resolved/ validated)
    config_file = os.path.join(ex.output_dir, "config.json")
    save_json(ex.config, config_file)
    if logger_type == "wandb" and run is not None:
        wandb.save(config_file)

    # Run experiment
    start_time = time.perf_counter()
    try:
        ex.run()
        elapsed = time.perf_counter() - start_time
        print(f"Experiment completed successfully. Outputs saved to {ex.output_dir}")
        print(f"Run time: {_format_elapsed(elapsed)}")

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"Experiment failed: {e}")
        print(f"Run time: {_format_elapsed(elapsed)}")
        # Report/ log error
        error_file = os.path.join(ex.output_dir, "error.txt")
        with open(error_file, "w") as f:
            f.write(str(e))
        if logger_type == "wandb" and run is not None:
            wandb.alert(title="Experiment failed", text=str(e))
        raise e

    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
