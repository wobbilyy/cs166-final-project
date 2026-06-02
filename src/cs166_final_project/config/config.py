from dataclasses import dataclass, field


@dataclass
class DataConfig:
    name: str = "MNIST"
    image_size: int = 28
    mask_size: int = 14

@dataclass
class ModelConfig:
    name: str = "unet"

@dataclass
class UNetConfig:
    hidden_dim: int = 64

@dataclass
class DDPMConfig:
    hidden_dim: int = 64
    num_timesteps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2

@dataclass
class FlowMatchingConfig:
    hidden_dim: int = 64
    num_steps: int = 25

@dataclass
class TransformerConfig:
    hidden_dim: int = 64
    num_layers: int = 2

@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 250
    lr: float = 1e-4
    weight_decay: float = 1e-6
    seed: int = 0
    device: str = "cpu"

@dataclass
class EvalConfig:
    save_plots: bool = True
    mask_seed: int = 0
    max_images: int = 8

@dataclass
class ExperimentConfig:
    name: str = "clean"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    ddpm: DDPMConfig = field(default_factory=DDPMConfig)
    flow: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

