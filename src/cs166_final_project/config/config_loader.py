from pathlib import Path

from omegaconf import OmegaConf
from cs166_final_project.config.config import ExperimentConfig


def load_config(path: str, overrides: list[str] | None = None) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"not found: {path}")

    schema = OmegaConf.structured(ExperimentConfig)

    user = OmegaConf.load(path)
    configs = [schema, user]

    if overrides:
        configs.append(OmegaConf.from_dotlist(overrides))

    merged = OmegaConf.merge(*configs)

    return OmegaConf.to_object(merged)

