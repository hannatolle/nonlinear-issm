"""
Base class for experiments.
"""

import os
import json
import copy
import warnings
from dataclasses import dataclass, field, asdict, is_dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple
import importlib
import torch
from src.utils.io import import_from_path


# Params no longer accepted by any component; dropped with a warning when a saved
# config.json still contains them.
DEPRECATED_COMPONENT_PARAMS = {"residual_euler", "max_speed", "encode_u", "init"}


class DotDict(dict):
    """Dictionary that allows dot notation access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

@dataclass(frozen=True)
class Component():
    component: str            # e.g., "system", "emission"
    kind: str                 # e.g., "DynamicAttractor"
    params: Union[dict, Any]  # dataclass or dict

    @property
    def config(self) -> dict:
        return self.get_config()

    def get_config(self) -> dict:
        config = {
            'kind': self.kind,
            'params': asdict(self.params) if is_dataclass(self.params) else self.params,
        }
        return config

    def build(self) -> Any:
        """Build a component object using the import_path from the component registry."""
        component_registry, _ = _load_registries()
        if self.kind not in component_registry:
            raise KeyError(f"Component kind '{self.kind}' not found in component registry.")
        entry = component_registry[self.kind]
        import_path = entry.get("import_path")
        if not import_path:
            raise ValueError(f"Component kind '{self.kind}' has no 'import_path' in registry.")
        class_obj = import_from_path(import_path)
        # Drop unsupported params so saved configs still build.
        params = asdict(self.params) if is_dataclass(self.params) else dict(self.params)
        for dk in DEPRECATED_COMPONENT_PARAMS & set(params):
            warnings.warn(f"{self.kind}: dropping deprecated param '{dk}'.",
                          DeprecationWarning)
            params.pop(dk)
        return class_obj(**params)

    def __repr__(self):
        return self.print()

    def print(self):
        result = f"{self.component}: {self.kind}"
        return result

@dataclass(frozen=True)
class Experiment(ABC):
    # Basic setup
    seed: int = 0
    output_dir: str = './outputs/default'
    verbose: bool = False
    job_id: int = 0
    device: str = 'auto'
    logging: dict = field(default_factory=lambda: {'logger': 'local', 
                                                    'params': {}}) 

    # Components
    settings: dict = field(default_factory=dict)
    components: dict = field(default_factory=dict)

    def __post_init__(self):
        self.check_config()

    @property
    def experiment(self) -> str:
        return self.__class__.__name__

    @property
    def config(self) -> dict:
        return self.get_config()

    @property
    def logger(self) -> str:
        return self.logging['logger']

    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
    
    def fetch_settings(self) -> DotDict:
        """Get settings as dot dictionary"""
        return DotDict(self.settings)
    
    def fetch_components(self) -> DotDict:
        """Get components as dot dictionary"""
        return DotDict(self.components)

    def get_config(self) -> dict:
        """Return the resolved config (settings and components with defaults filled)."""
        config = {
            "seed": self.seed,
            "output_dir": self.output_dir,
            "verbose": self.verbose,
            "experiment": self.experiment,
            "job_id": self.job_id,
            "device": self.device,
            "logging": self.logging,
            "settings": dict(self.settings),
            "components": {},
        }
        for c, v in self.components.items():
            config["components"][c] = v.config
        return config
        
    def __repr__(self):
        return self.print()

    def print(self):
        header = f"Experiment(type={self.experiment}, seed={self.seed}, job_id={self.job_id})"
        comp_summary = "\n- ".join([v.print() for v in self.components.values()])
        return f"{header}\nComponents:\n- {comp_summary}"

    def match_settings(self, strict=True, skip=None):
        """
        Ensure all components use the same values for keys
        that appear in settings and in component params.

        Args:
            skip: optional dict mapping component names to sets of keys to
                  exempt from the check, e.g. {'system': {'D'}}.
        """
        settings = self.settings
        skip = skip or {}
        mismatches = []
        for cname, comp in self.components.items():
            params = comp.params
            skip_keys = skip.get(cname, set())
            for k, v in settings.items():
                if k in skip_keys:
                    continue
                if k in params and params[k] != v:
                    msg = (f"Inconsistent setting for '{k}' in component '{cname}': "
                           f"settings={v}, component={params[k]}")
                    if strict:
                        raise ValueError(msg)
                    else:
                        mismatches.append(msg)
        if mismatches:
            print(f"Found {len(mismatches)} inconsistent settings:")
            for m in mismatches:
                print(" -", m)
        else:
            print("Settings are consistent across all components.")

    @classmethod
    def preprocess_config(cls, config: dict) -> dict:
        """Optionally transform the raw config before resolve_config (e.g. derived
        dimensions). Must return the (possibly modified) config; no-op by default.
        """
        return config

    @abstractmethod
    def check_config(self):
        """Make sure all components are compatible with each other."""
        pass

    @abstractmethod
    def run(self):
        """Run the experiment."""
        pass


# Helper functions -------------------------------------------------------------

def _load_registries() -> Tuple[dict, dict]:
    """Load component and settings registries from cfg_registries."""
    this_dir = os.path.dirname(__file__)
    comp_path = os.path.join(this_dir, "cfg_registries", "components.json")
    settings_path = os.path.join(this_dir, "cfg_registries", "settings.json")
    if not os.path.exists(comp_path):
        raise FileNotFoundError(f"Component registry not found: {comp_path}")
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Settings registry not found: {settings_path}")
    with open(comp_path, "r") as f:
        component_registry = json.load(f)
    with open(settings_path, "r") as f:
        settings_registry = json.load(f)
    return component_registry, settings_registry

def resolve_config(
    ex_name: str,
    raw_settings: dict,
    raw_components: dict) -> Tuple[dict, dict]:
    """
    Resolve raw config into fully filled, validated config. Idempotent: if
    input is already resolved, returns equivalent output. Never mutates inputs.

    Args:
        ex_name: Experiment class name (e.g. 'TrainISSM').
        raw_settings: User-provided settings dict (may omit defaults).
        raw_components: Dict of name -> {kind, params} (params may omit defaults).

    Returns:
        (resolved_settings, resolved_components_config) where
        resolved_components_config is name -> {kind, params} with params filled.
    """
    # Get component and settings registries
    component_registry, settings_registry = _load_registries()

    # Validate settings -------------------------------------------------------------
    if ex_name not in settings_registry:
        raise KeyError(
            f"Settings registry does not contain configuration for class '{ex_name}'")
    exp_cfg = settings_registry[ex_name]
    settings_defaults = exp_cfg.get("defaults", {})
    settings_required = exp_cfg.get("required", [])
    all_known_settings = set(settings_defaults.keys()) | set(settings_required)

    # Initialize with defaults, then override with user inputs
    resolved_settings = copy.deepcopy(settings_defaults)
    for key, val in raw_settings.items():
        if key not in all_known_settings:
            raise ValueError(
                f"Unrecognized settings for {ex_name}: {key}. ")
        resolved_settings[key] = copy.deepcopy(val)
    missing_required = [k for k in settings_required if k not in resolved_settings]
    if missing_required:
        raise ValueError(f"{ex_name} is missing required settings: {missing_required}")

    # Validate components -------------------------------------------------------------
    # For each component, merge defaults + settings + raw params
    resolved_components_config = {}
    for cname, spec in raw_components.items():
        kind = spec.get("kind")
        if not kind:
            raise ValueError(f"Invalid component configuration for '{cname}': missing 'kind'")
        raw_params = spec.get("params", {})

        if kind not in component_registry:
            raise KeyError(
                f"Component kind '{kind}' (for '{cname}') not found in component registry."
            )
        comp_cfg = component_registry[kind]
        defaults = copy.deepcopy(comp_cfg.get("defaults", {}))
        required = comp_cfg.get("required", [])

        # Initialize with defaults, then override with user inputs 
        resolved_params = copy.deepcopy(defaults)
        for key, val in resolved_settings.items():
            if key in (set(defaults.keys()) | set(required)) and key not in raw_params:
                resolved_params[key] = copy.deepcopy(val)
        for key, val in raw_params.items():
            resolved_params[key] = copy.deepcopy(val)
        missing_required = [k for k in required if k not in resolved_params]
        if missing_required:
            raise ValueError(
                f"{cname} (kind={kind}) is missing required parameters: {missing_required}")
        resolved_components_config[cname] = {"kind": kind, "params": resolved_params}

    return resolved_settings, resolved_components_config

def create_experiment(ex_name: str, config: Dict[str, Any]) -> Experiment:
    """
    Create an experiment object from a configuration dictionary. Raw config
    is resolved (defaults filled, validated) before construction.

    Args:
        ex_name (str): Name of the Experiment subclass to instantiate
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        Experiment: An instance of the specified experiment class
    """
    # Allow the experiment class to adjust the config before resolution
    try:
        experiments_module = importlib.import_module("src.experiments")
        experiment_class = getattr(experiments_module, ex_name, None)
        if experiment_class is not None and issubclass(experiment_class, Experiment):
            config = experiment_class.preprocess_config(config)
    except ImportError:
        pass

    raw_settings = config.get("settings", {})
    raw_components_spec = config.get("components", {})

    # Validate configs and fill defaults
    resolved_settings, resolved_components_config = resolve_config(
        ex_name, raw_settings, raw_components_spec
    )

    # Build basic params dict
    basic_params = {
        "seed": config.get("seed", 0),
        "output_dir": config.get("output_dir", "./outputs"),
        "verbose": config.get("verbose", False),
        "job_id": config.get("job_id", 0),
        "device": config.get("device", "auto"),
        "logging": config.get("logging", {"logger": "local"}),
    }

    # Build Components and settings from resolved configs
    components = {}
    for cname, spec in resolved_components_config.items():
        params_copy = copy.deepcopy(spec["params"])
        components[cname] = Component(component=cname, 
                                      kind=spec["kind"], 
                                      params=params_copy)
    settings = copy.deepcopy(resolved_settings)

    # Load and build experiment
    try:
        experiments_module = importlib.import_module("src.experiments")
        if not hasattr(experiments_module, ex_name):
            raise ValueError(f"Experiment class '{ex_name}' not found in experiments module")
        experiment_class = getattr(experiments_module, ex_name)
        if not issubclass(experiment_class, Experiment):
            raise ValueError(f"'{ex_name}' is not a subclass of Experiment")
        experiment = experiment_class(
            **basic_params, settings=settings, components=components)
        return experiment
    except ImportError as e:
        raise ValueError(f"Could not import experiments module: {e}")
    except Exception as e:
        raise ValueError(f"Failed to create experiment instance: {e}")