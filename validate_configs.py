#!/usr/bin/env python3
"""
Validation script for Startup Fine-Tuning Cookbook configs.

Checks that all 5 use-case configs load correctly and have required fields.
"""

import sys
from pathlib import Path
import yaml


REQUIRED_KEYS = {
    "model": ["name", "load_in_4bit"],
    "lora": ["r", "alpha", "dropout", "target_modules", "task_type"],
    "data": ["train_path", "eval_path", "format", "max_seq_length"],
    "training": ["output_dir", "num_train_epochs", "per_device_train_batch_size", "learning_rate"],
    "merge": ["output_dir", "export_gguf"],
}

USE_CASES = [
    "support-chatbot",
    "code-generation",
    "domain-qa",
    "sentiment-analysis",
    "instruction-following",
]


def validate_config(config_path: Path) -> tuple[bool, list[str]]:
    """Validate a single config file. Returns (is_valid, errors)."""
    errors = []

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Failed to parse YAML: {e}"]

    if not isinstance(config, dict):
        return False, ["Config root must be a dictionary"]

    # Check required top-level sections
    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            errors.append(f"Missing required section: {section}")
            continue
        for key in keys:
            if key not in config[section]:
                errors.append(f"Missing required key: {section}.{key}")

    # Validate specific values
    if "model" in config:
        if config["model"].get("load_in_4bit") and "bnb_4bit_compute_dtype" not in config["model"]:
            errors.append("model.load_in_4bit=true requires bnb_4bit_compute_dtype")

    if "lora" in config:
        r = config["lora"].get("r")
        if r is not None and (not isinstance(r, int) or r <= 0):
            errors.append("lora.r must be positive integer")

        alpha = config["lora"].get("alpha")
        if alpha is not None and (not isinstance(alpha, int) or alpha <= 0):
            errors.append("lora.alpha must be positive integer")

    if "training" in config:
        epochs = config["training"].get("num_train_epochs")
        if epochs is not None:
            try:
                epochs_val = float(epochs)
                if epochs_val <= 0:
                    errors.append("training.num_train_epochs must be positive number")
            except (ValueError, TypeError):
                errors.append("training.num_train_epochs must be a valid number")

        lr = config["training"].get("learning_rate")
        if lr is not None:
            try:
                lr_val = float(lr)
                if lr_val <= 0:
                    errors.append("training.learning_rate must be positive number")
            except (ValueError, TypeError):
                errors.append("training.learning_rate must be a valid number")

    # Check dataset file exists
    if "data" in config:
        train_path = Path(config["data"].get("train_path", ""))
        if train_path and not train_path.exists():
            errors.append(f"Dataset not found: {train_path}")

    return len(errors) == 0, errors


def main():
    """Validate all use-case configs."""
    base_path = Path(__file__).parent / "use-cases"
    all_valid = True

    print("=" * 60)
    print("Startup Fine-Tuning Cookbook - Config Validation")
    print("=" * 60)

    for use_case in USE_CASES:
        config_path = base_path / use_case / "config.yaml"
        print(f"\nValidating: {use_case}")

        if not config_path.exists():
            print(f"  ❌ Config not found: {config_path}")
            all_valid = False
            continue

        valid, errors = validate_config(config_path)

        if valid:
            print(f"  ✅ Valid")
        else:
            print(f"  ❌ Invalid")
            for err in errors:
                print(f"     - {err}")
            all_valid = False

    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All configs validated successfully!")
        return 0
    else:
        print("❌ Some configs failed validation")
        return 1


if __name__ == "__main__":
    sys.exit(main())