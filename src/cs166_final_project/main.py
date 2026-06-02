import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pprint

from cs166_final_project.config.config import ExperimentConfig
from cs166_final_project.config.config_loader import load_config

from cs166_final_project.data.dataloaders import (
    get_cifar_dataloaders,
    get_ffhq_dataloaders,
    get_mnist_dataloaders,
)
from cs166_final_project.models.ddpm.inpainter import DDPMInpainter
from cs166_final_project.models.flow.inpainter import FlowInpainter
from cs166_final_project.models.unet.inpainter import UnetInpainter

from cs166_final_project.evaluate import evaluate_model
from cs166_final_project.train import train_model

from cs166_final_project.utils.visualization import save_data_examples


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to .yaml configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="path to output directory",
    )

    return parser.parse_known_args()


def _get_model_params(config: ExperimentConfig) -> str:
    """
    unet: unet-{hidden}
    diffusion: ddpm-{steps}-{hidden}
    flow: flow-{steps}-{hidden}
    """
    model_name = config.model.name

    if model_name == "unet":
        return f"h{config.unet.hidden_dim}"

    if model_name == "ddpm":
        return f"T{config.ddpm.num_timesteps}-h{config.ddpm.hidden_dim}"

    if model_name == "flow":
        return f"T{config.flow.num_steps}-h{config.flow.hidden_dim}"

    return "fuhhed"


def _get_experiment_params(config: ExperimentConfig) -> str:
    """
    all: {epochs}-{batch}
    """
    return f"ep{config.train.epochs}-bt{config.train.batch_size}"


def _get_run_dir(output_dir: Path, config: ExperimentConfig) -> tuple[Path, str]:
    """
    {dataset}-{model}-{specifications}-{epochs}-{batch}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.data.name}-{config.model.name}-{_get_model_params(config)}-{_get_experiment_params(config)}"

    run_dir = output_dir / timestamp / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, timestamp


def _save_config(
    run_dir: Path,
    config: ExperimentConfig,
):
    payload = {
        "config": asdict(config),
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(
            payload,
            f,
            indent=2,
        )


def _get_dataloaders(config: ExperimentConfig, data_dir: Path):
    dataset = config.data.name

    if dataset == "mnist":
        return get_mnist_dataloaders(
            data_dir=data_dir,
            batch_size=config.train.batch_size,
            image_size=config.data.image_size,
        )

    if dataset == "cifar":
        return get_cifar_dataloaders(
            data_dir=data_dir,
            batch_size=config.train.batch_size,
            image_size=config.data.image_size,
        )

    if dataset == "ffhq":
        return get_ffhq_dataloaders(
            data_dir=data_dir,
            batch_size=config.train.batch_size,
            image_size=config.data.image_size,
        )

    raise ValueError(f"bad dataset name: {config.data.name}")


def _build_model(config: ExperimentConfig):
    model_name = config.model.name
    image_channels = 1 if config.data.name.lower() == "mnist" else 3

    if model_name == "unet":
        return UnetInpainter(
            image_channels=image_channels,
            hidden_dim=config.unet.hidden_dim,
        )

    if model_name == "ddpm":
        return DDPMInpainter(
            image_channels=image_channels,
            hidden_dim=config.ddpm.hidden_dim,
            num_timesteps=config.ddpm.num_timesteps,
            beta_start=config.ddpm.beta_start,
            beta_end=config.ddpm.beta_end,
        )

    if model_name == "flow":
        return FlowInpainter(
            image_channels=image_channels,
            hidden_dim=config.flow.hidden_dim,
            num_steps=config.flow.num_steps,
        )

    raise ValueError(f"bad model name: {config.model.name}")


def main():
    """
    entrypooint

    1. config
    2. saves data examples
    3. train
    4. eval
    """

    args, overrides = parse_args()

    config: ExperimentConfig = load_config(args.config, overrides=overrides)

    print("(1) CONFIG:")
    pprint(config)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = config.data.name
    model_name = config.model.name

    run_dir, timestamp = _get_run_dir(output_dir, config)
    _save_config(run_dir, config)
    print(f"  OUTPUT DIR: {run_dir}")

    train_dataloader, val_dataloader, test_dataloader = _get_dataloaders(config, data_dir)
    model = _build_model(config)

    print("(2) SAVE DATA EXAMPLES:")
    save_data_examples(
        val_dataloader,
        run_dir / f"{dataset_name}_data_examples.png",
        mask_size=config.data.mask_size,
        mask_seed=config.eval.mask_seed,
        max_images=config.eval.max_images,
     )

    print("(3) TRAIN:")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        output_dir=run_dir,
    )

    print("(4) EVALUATE:")
    evaluate_model(
        model=model,
        dataloader=test_dataloader,
        config=config,
        output_dir=run_dir,
    )


if __name__ == "__main__":
    main()

