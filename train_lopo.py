#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.baselines import PersistenceBaseline, RidgePerHorizonBaseline, XGBoostDirectBaseline
from src.data import FEATURES, PerturbConfig, load_subject_series, prepare_split_data
from src.eval import aggregate_rows, compute_metrics, save_rows_table
from src.models import GRUMixerModel, LSTMTinyModel, TinyTCNModel, TinyTransformerModel
from src.models.vqc import pennylane_available
from src.splits import lopo_folds, sample_subject_fraction, split_train_val_subjects
from src.train import count_parameters, fit_model, make_dataloader, predict_model
from src.utils import ensure_dir, parse_float_csv, parse_int_csv, save_json, set_seed, setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LOPO experiments on BIDMC Numerics")
    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument(
        "--model",
        type=str,
        default="gru_mixer",
        choices=["gru_mixer", "lstm_tiny", "tcn", "transformer", "ridge", "persistence", "xgboost_direct"],
    )
    parser.add_argument("--mixer", type=str, default="none", choices=["none", "mlp", "vqc"])
    parser.add_argument("--n_qubits", type=int, default=6)
    parser.add_argument("--vqc_depth", type=int, default=3)
    parser.add_argument("--vqc_entangle", type=int, default=1, choices=[0, 1])
    parser.add_argument("--vqc_vectorized", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--vqc_backend",
        type=str,
        default="auto",
        choices=["auto", "lightning.gpu", "lightning.qubit", "default.qubit", "default.qubit.torch"],
    )
    parser.add_argument("--L", type=int, default=240)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--horizons", type=str, default="15,30,60")
    parser.add_argument("--target_mode", type=str, default="value", choices=["value", "delta"])
    parser.add_argument("--nan_strategy", type=str, default="ffill_bfill", choices=["ffill_bfill", "drop"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="runs/standard")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--lstm_hidden_dim", type=int, default=56)
    parser.add_argument("--tcn_channels", type=int, default=32)
    parser.add_argument("--tcn_levels", type=int, default=3)
    parser.add_argument("--tcn_kernel_size", type=int, default=3)
    parser.add_argument("--tcn_dropout", type=float, default=0.1)
    parser.add_argument("--xgb_n_estimators", type=int, default=16)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_n_jobs", type=int, default=1)
    parser.add_argument("--mlp_dim", type=int, default=6)
    parser.add_argument("--max_folds", type=int, default=0, help="0 means all LOPO folds")
    parser.add_argument(
        "--test_subject",
        type=str,
        default="",
        help="Run only the LOPO fold with this test subject id (e.g. 1 or 01)",
    )

    parser.add_argument(
        "--exp_mode",
        type=str,
        default="standard",
        choices=["standard", "data_efficiency", "robustness"],
    )
    parser.add_argument("--data_fractions", type=str, default="1.0,0.5,0.25")
    parser.add_argument("--perturb_target", type=str, default="test", choices=["test", "train_test"])
    parser.add_argument("--perturb_type", type=str, default="missingness", choices=["missingness", "noise"])
    parser.add_argument("--missing_rates", type=str, default="0.0,0.1,0.3")
    parser.add_argument("--noise_sigmas", type=str, default="0.0,0.01,0.05")
    parser.add_argument("--mask_channel", type=int, default=0, choices=[0, 1])
    parser.add_argument("--missing_impute", type=str, default="ffill_bfill", choices=["ffill_bfill", "zero"])
    return parser


def resolve_device(requested: str, force_cpu: bool = False) -> torch.device:
    requested = requested.lower()
    if force_cpu:
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def check_vqc_dependency(mixer: str) -> None:
    if mixer == "vqc" and not pennylane_available():
        raise RuntimeError(
            "mixer=vqc richiesto ma PennyLane non e disponibile. "
            "Installa `pennylane` oppure usa --mixer mlp."
        )


def build_torch_model(
    setting: dict[str, Any],
    input_dim: int,
    horizons: list[int],
    l_window: int,
    prefer_gpu: bool,
) -> torch.nn.Module:
    model_name = setting["model"]
    if model_name == "gru_mixer":
        return GRUMixerModel(
            input_dim=input_dim,
            horizons=horizons,
            mixer=setting["mixer"],
            mlp_dim=setting["mlp_dim"],
            n_qubits=setting["n_qubits"],
            vqc_depth=setting["vqc_depth"],
            vqc_entangle=bool(setting["vqc_entangle"]),
            vqc_backend=setting["vqc_backend"],
            vqc_prefer_gpu=prefer_gpu,
            vqc_vectorized=bool(setting["vqc_vectorized"]),
        )
    if model_name == "lstm_tiny":
        return LSTMTinyModel(
            input_dim=input_dim,
            horizons=horizons,
            hidden_dim=setting["lstm_hidden_dim"],
        )
    if model_name == "tcn":
        return TinyTCNModel(
            input_dim=input_dim,
            horizons=horizons,
            channels=setting["tcn_channels"],
            levels=setting["tcn_levels"],
            kernel_size=setting["tcn_kernel_size"],
            dropout=setting["tcn_dropout"],
        )
    if model_name == "transformer":
        return TinyTransformerModel(
            input_dim=input_dim,
            horizons=horizons,
            max_len=max(2048, l_window + max(horizons) + 32),
        )
    raise ValueError(f"Unsupported torch model: {model_name}")


def run_single_setting(
    setting: dict[str, Any],
    split_data: dict[str, np.ndarray],
    args: argparse.Namespace,
    horizons: list[int],
    output_dir: Path,
    run_name: str,
    logger: Any,
) -> dict[str, float] | None:
    x_train = split_data["x_train"]
    y_train = split_data["y_train"]
    x_val = split_data["x_val"]
    y_val = split_data["y_val"]
    x_test = split_data["x_test"]
    y_test = split_data["y_test"]
    y_test_abs = split_data["y_test_abs"]
    anchor_test = split_data["anchor_test"]

    if x_train.shape[0] == 0:
        logger.warning("Skipping run=%s because x_train is empty", run_name)
        return None
    if x_test.shape[0] == 0:
        logger.warning("Skipping run=%s because x_test is empty", run_name)
        return None

    if setting["model"] == "persistence":
        baseline = PersistenceBaseline(
            n_features=len(FEATURES),
            target_mode=setting["target_mode"],
        )
        baseline.fit(x_train, y_train)
        y_pred_target = baseline.predict(x_test)
    elif setting["model"] == "ridge":
        baseline = RidgePerHorizonBaseline(alpha=setting["ridge_alpha"])
        baseline.fit(x_train, y_train)
        y_pred_target = baseline.predict(x_test)
    elif setting["model"] == "xgboost_direct":
        baseline = XGBoostDirectBaseline(
            n_estimators=setting["xgb_n_estimators"],
            max_depth=setting["xgb_max_depth"],
            learning_rate=setting["xgb_learning_rate"],
            subsample=setting["xgb_subsample"],
            colsample_bytree=setting["xgb_colsample_bytree"],
            seed=args.seed,
            n_jobs=setting["xgb_n_jobs"],
        )
        baseline.fit(x_train, y_train)
        y_pred_target = baseline.predict(x_test)
        logger.info(
            "run=%s model=xgboost_direct effective_params~%d",
            run_name,
            baseline.effective_param_count(),
        )
    else:
        check_vqc_dependency(setting["mixer"])
        requested_device = resolve_device(args.device, force_cpu=False)

        model = build_torch_model(
            setting=setting,
            input_dim=x_train.shape[2],
            horizons=horizons,
            l_window=args.L,
            prefer_gpu=requested_device.type == "cuda",
        )
        device = requested_device
        if setting["model"] == "gru_mixer" and setting["mixer"] == "vqc":
            uses_gpu_backend = bool(getattr(model.mixer_block, "uses_gpu_backend", False))
            backend_name = str(getattr(model.mixer_block, "device_name", "unknown"))
            vectorized_enabled = bool(getattr(model.mixer_block, "vectorized_enabled", False))
            if requested_device.type == "cuda" and not uses_gpu_backend:
                device = torch.device("cpu")
                logger.info(
                    "run=%s vqc_backend=%s is CPU-only; falling back to device=cpu (vectorized=%s)",
                    run_name,
                    backend_name,
                    vectorized_enabled,
                )
            else:
                logger.info(
                    "run=%s using vqc_backend=%s on device=%s (vectorized=%s)",
                    run_name,
                    backend_name,
                    device,
                    vectorized_enabled,
                )

        logger.info(
            "run=%s model=%s mixer=%s params=%d device=%s",
            run_name,
            setting["model"],
            setting["mixer"],
            count_parameters(model),
            device,
        )

        train_loader = make_dataloader(
            x=x_train,
            y=y_train,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + 101,
        )
        val_loader = make_dataloader(
            x=x_val,
            y=y_val,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed + 103,
        )
        test_loader = make_dataloader(
            x=x_test,
            y=y_test,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed + 107,
        )
        if train_loader is None or test_loader is None:
            logger.warning("Skipping run=%s due to empty loader", run_name)
            return None

        checkpoint_path, history = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            output_dir=output_dir,
            run_name=run_name,
            patience=args.patience,
            logger=logger,
        )
        save_rows_table(history, output_dir / "histories" / f"{run_name}.xlsx")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        y_pred_target, _ = predict_model(model=model, loader=test_loader, device=device)

    y_pred_abs = target_to_absolute(
        y_pred_target=y_pred_target,
        anchors=anchor_test,
        target_mode=setting["target_mode"],
    )
    return compute_metrics(y_true=y_test_abs, y_pred=y_pred_abs, horizons=horizons, features=FEATURES)


def base_setting_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "mixer": args.mixer if args.model == "gru_mixer" else "none",
        "target_mode": args.target_mode,
        "n_qubits": args.n_qubits,
        "vqc_depth": args.vqc_depth,
        "vqc_entangle": args.vqc_entangle,
        "vqc_vectorized": args.vqc_vectorized,
        "vqc_backend": args.vqc_backend,
        "mlp_dim": args.mlp_dim if args.mlp_dim > 0 else args.n_qubits,
        "ridge_alpha": args.ridge_alpha,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "tcn_channels": args.tcn_channels,
        "tcn_levels": args.tcn_levels,
        "tcn_kernel_size": args.tcn_kernel_size,
        "tcn_dropout": args.tcn_dropout,
        "xgb_n_estimators": args.xgb_n_estimators,
        "xgb_max_depth": args.xgb_max_depth,
        "xgb_learning_rate": args.xgb_learning_rate,
        "xgb_subsample": args.xgb_subsample,
        "xgb_colsample_bytree": args.xgb_colsample_bytree,
        "xgb_n_jobs": args.xgb_n_jobs,
    }


def get_variant_settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = base_setting_from_args(args)
    none_variant = {
        **base,
        "model": "gru_mixer",
        "mixer": "none",
        "variant": "gru_none",
    }
    vqc_variant = {
        **base,
        "model": "gru_mixer",
        "mixer": "vqc",
        "variant": "gru_vqc",
    }
    return [none_variant, vqc_variant]


def target_to_absolute(
    y_pred_target: np.ndarray,
    anchors: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    if target_mode == "value":
        return y_pred_target.astype(np.float32)
    if target_mode == "delta":
        if anchors.ndim != 2 or anchors.shape[0] != y_pred_target.shape[0]:
            raise ValueError(
                f"Invalid anchors shape {anchors.shape} for predictions {y_pred_target.shape}"
            )
        return (y_pred_target + anchors[:, None, :]).astype(np.float32)
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def trim_folds(
    folds: list[tuple[str, list[str]]], max_folds: int
) -> list[tuple[str, list[str]]]:
    if max_folds <= 0:
        return folds
    return folds[:max_folds]


def _canonical_subject_id(subject: str, available: list[str]) -> str:
    if not available:
        raise ValueError("No available subjects found")
    wanted = subject.strip()
    if not wanted:
        raise ValueError("Empty test subject")

    if wanted in available:
        return wanted
    try:
        wanted_int = int(wanted)
    except ValueError as exc:
        raise ValueError(f"Invalid test subject '{subject}'") from exc

    for candidate in available:
        if int(candidate) == wanted_int:
            return candidate
    raise ValueError(
        f"Requested test subject '{subject}' not found. "
        f"Available: {', '.join(sorted(available, key=int))}"
    )


def select_folds(
    folds: list[tuple[str, list[str]]],
    test_subject: str,
    max_folds: int,
) -> list[tuple[str, list[str]]]:
    selected = folds
    if test_subject.strip():
        canonical = _canonical_subject_id(
            subject=test_subject,
            available=[fold_subject for fold_subject, _ in folds],
        )
        selected = [fold for fold in folds if fold[0] == canonical]
    return trim_folds(selected, max_folds=max_folds)


def run_standard_mode(
    args: argparse.Namespace,
    subject_series: dict[str, np.ndarray],
    folds: list[tuple[str, list[str]]],
    horizons: list[int],
    output_dir: Path,
    logger: Any,
) -> None:
    setting = base_setting_from_args(args)
    rows: list[dict[str, Any]] = []

    for fold_idx, (test_subject, train_pool) in enumerate(folds):
        train_subjects, val_subjects = split_train_val_subjects(
            train_pool=train_pool,
            val_ratio=args.val_ratio,
            seed=args.seed + fold_idx,
        )
        split_data = prepare_split_data(
            subject_series=subject_series,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subject=test_subject,
            l_window=args.L,
            horizons=horizons,
            target_mode=args.target_mode,
            window_stride=args.window_stride,
            mask_channel=False,
            missing_impute=args.missing_impute,
            seed=args.seed + fold_idx * 1000,
        )

        run_name = f"fold_{test_subject}_{setting['model']}_{setting['mixer']}"
        metrics = run_single_setting(
            setting=setting,
            split_data=split_data,
            args=args,
            horizons=horizons,
            output_dir=output_dir,
            run_name=run_name,
            logger=logger,
        )
        if metrics is None:
            continue

        row = {
            "exp_mode": "standard",
            "fold_idx": fold_idx,
            "test_subject": test_subject,
            "model": setting["model"],
            "mixer": setting["mixer"],
            "target_mode": setting["target_mode"],
            "n_qubits": setting["n_qubits"],
            "vqc_depth": setting["vqc_depth"],
            "vqc_entangle": setting["vqc_entangle"],
            "vqc_vectorized": setting["vqc_vectorized"],
            "vqc_backend": setting["vqc_backend"],
            "lstm_hidden_dim": setting["lstm_hidden_dim"],
            "tcn_channels": setting["tcn_channels"],
            "tcn_levels": setting["tcn_levels"],
            "tcn_kernel_size": setting["tcn_kernel_size"],
            "tcn_dropout": setting["tcn_dropout"],
            "xgb_n_estimators": setting["xgb_n_estimators"],
            "xgb_max_depth": setting["xgb_max_depth"],
            "xgb_learning_rate": setting["xgb_learning_rate"],
            "xgb_subsample": setting["xgb_subsample"],
            "xgb_colsample_bytree": setting["xgb_colsample_bytree"],
            "xgb_n_jobs": setting["xgb_n_jobs"],
            "window_stride": args.window_stride,
            "n_train_subjects": len(train_subjects),
            "n_val_subjects": len(val_subjects),
            "n_train_windows": int(split_data["x_train"].shape[0]),
            "n_val_windows": int(split_data["x_val"].shape[0]),
            "n_test_windows": int(split_data["x_test"].shape[0]),
        }
        row.update(metrics)
        rows.append(row)

    save_rows_table(rows, output_dir / "lopo_fold_metrics.xlsx")
    summary = aggregate_rows(rows, group_keys=["model", "mixer"])
    save_rows_table(summary, output_dir / "lopo_aggregate.xlsx")


def run_data_efficiency_mode(
    args: argparse.Namespace,
    subject_series: dict[str, np.ndarray],
    folds: list[tuple[str, list[str]]],
    horizons: list[int],
    output_dir: Path,
    logger: Any,
) -> None:
    fractions = parse_float_csv(args.data_fractions)
    variants = get_variant_settings(args)
    rows: list[dict[str, Any]] = []

    for fold_idx, (test_subject, train_subjects_full) in enumerate(folds):
        train_pool, val_subjects = split_train_val_subjects(
            train_pool=train_subjects_full,
            val_ratio=args.val_ratio,
            seed=args.seed + fold_idx,
        )

        for frac_idx, fraction in enumerate(fractions):
            sampled_train = sample_subject_fraction(
                subjects=train_pool,
                fraction=fraction,
                seed=args.seed + fold_idx * 100 + frac_idx,
            )
            logger.info(
                "fold=%s fraction=%.3f train_subjects=%d val_subjects=%d",
                test_subject,
                fraction,
                len(sampled_train),
                len(val_subjects),
            )

            for variant in variants:
                split_data = prepare_split_data(
                    subject_series=subject_series,
                    train_subjects=sampled_train,
                    val_subjects=val_subjects,
                    test_subject=test_subject,
                    l_window=args.L,
                    horizons=horizons,
                    target_mode=args.target_mode,
                    window_stride=args.window_stride,
                    mask_channel=False,
                    missing_impute=args.missing_impute,
                    seed=args.seed + fold_idx * 1000 + frac_idx * 10,
                )
                run_name = (
                    f"de_fold_{test_subject}_frac_{fraction:.2f}_"
                    f"{variant['model']}_{variant['mixer']}"
                )
                metrics = run_single_setting(
                    setting=variant,
                    split_data=split_data,
                    args=args,
                    horizons=horizons,
                    output_dir=output_dir,
                    run_name=run_name,
                    logger=logger,
                )
                if metrics is None:
                    continue
                row = {
                    "exp_mode": "data_efficiency",
                    "fold_idx": fold_idx,
                    "test_subject": test_subject,
                    "variant": variant["variant"],
                    "model": variant["model"],
                    "mixer": variant["mixer"],
                    "target_mode": variant["target_mode"],
                    "vqc_vectorized": variant["vqc_vectorized"],
                    "vqc_backend": variant["vqc_backend"],
                    "lstm_hidden_dim": variant["lstm_hidden_dim"],
                    "tcn_channels": variant["tcn_channels"],
                    "tcn_levels": variant["tcn_levels"],
                    "tcn_kernel_size": variant["tcn_kernel_size"],
                    "tcn_dropout": variant["tcn_dropout"],
                    "xgb_n_estimators": variant["xgb_n_estimators"],
                    "xgb_max_depth": variant["xgb_max_depth"],
                    "xgb_learning_rate": variant["xgb_learning_rate"],
                    "xgb_subsample": variant["xgb_subsample"],
                    "xgb_colsample_bytree": variant["xgb_colsample_bytree"],
                    "xgb_n_jobs": variant["xgb_n_jobs"],
                    "window_stride": args.window_stride,
                    "train_fraction": float(fraction),
                    "n_train_subjects": len(sampled_train),
                    "n_val_subjects": len(val_subjects),
                    "n_train_windows": int(split_data["x_train"].shape[0]),
                    "n_val_windows": int(split_data["x_val"].shape[0]),
                    "n_test_windows": int(split_data["x_test"].shape[0]),
                }
                row.update(metrics)
                rows.append(row)

    save_rows_table(rows, output_dir / "data_efficiency_fold_metrics.xlsx")
    summary = aggregate_rows(rows, group_keys=["variant", "train_fraction"])
    save_rows_table(summary, output_dir / "data_efficiency_summary.xlsx")


def run_robustness_mode(
    args: argparse.Namespace,
    subject_series: dict[str, np.ndarray],
    folds: list[tuple[str, list[str]]],
    horizons: list[int],
    output_dir: Path,
    logger: Any,
) -> None:
    variants = get_variant_settings(args)
    if args.perturb_type == "missingness":
        levels = parse_float_csv(args.missing_rates)
    else:
        levels = parse_float_csv(args.noise_sigmas)
    use_mask_channel = bool(args.mask_channel and args.perturb_type == "missingness")

    rows: list[dict[str, Any]] = []
    for fold_idx, (test_subject, train_pool) in enumerate(folds):
        train_subjects, val_subjects = split_train_val_subjects(
            train_pool=train_pool,
            val_ratio=args.val_ratio,
            seed=args.seed + fold_idx,
        )
        for level_idx, level in enumerate(levels):
            perturb = PerturbConfig(perturb_type=args.perturb_type, level=float(level))
            if args.perturb_target == "test":
                train_perturb = None
                val_perturb = None
                test_perturb = perturb
            else:
                train_perturb = perturb
                val_perturb = perturb
                test_perturb = perturb

            for variant in variants:
                split_data = prepare_split_data(
                    subject_series=subject_series,
                    train_subjects=train_subjects,
                    val_subjects=val_subjects,
                    test_subject=test_subject,
                    l_window=args.L,
                    horizons=horizons,
                    target_mode=args.target_mode,
                    window_stride=args.window_stride,
                    train_perturb=train_perturb,
                    val_perturb=val_perturb,
                    test_perturb=test_perturb,
                    mask_channel=use_mask_channel,
                    missing_impute=args.missing_impute,
                    seed=args.seed + fold_idx * 1000 + level_idx * 10,
                )
                run_name = (
                    f"rob_fold_{test_subject}_{args.perturb_type}_{level:.4f}_"
                    f"{variant['model']}_{variant['mixer']}"
                )
                metrics = run_single_setting(
                    setting=variant,
                    split_data=split_data,
                    args=args,
                    horizons=horizons,
                    output_dir=output_dir,
                    run_name=run_name,
                    logger=logger,
                )
                if metrics is None:
                    continue

                row = {
                    "exp_mode": "robustness",
                    "fold_idx": fold_idx,
                    "test_subject": test_subject,
                    "variant": variant["variant"],
                    "model": variant["model"],
                    "mixer": variant["mixer"],
                    "target_mode": variant["target_mode"],
                    "vqc_vectorized": variant["vqc_vectorized"],
                    "vqc_backend": variant["vqc_backend"],
                    "lstm_hidden_dim": variant["lstm_hidden_dim"],
                    "tcn_channels": variant["tcn_channels"],
                    "tcn_levels": variant["tcn_levels"],
                    "tcn_kernel_size": variant["tcn_kernel_size"],
                    "tcn_dropout": variant["tcn_dropout"],
                    "xgb_n_estimators": variant["xgb_n_estimators"],
                    "xgb_max_depth": variant["xgb_max_depth"],
                    "xgb_learning_rate": variant["xgb_learning_rate"],
                    "xgb_subsample": variant["xgb_subsample"],
                    "xgb_colsample_bytree": variant["xgb_colsample_bytree"],
                    "xgb_n_jobs": variant["xgb_n_jobs"],
                    "window_stride": args.window_stride,
                    "perturb_type": args.perturb_type,
                    "perturb_target": args.perturb_target,
                    "perturb_level": float(level),
                    "mask_channel": int(use_mask_channel),
                    "n_train_subjects": len(train_subjects),
                    "n_val_subjects": len(val_subjects),
                    "n_train_windows": int(split_data["x_train"].shape[0]),
                    "n_val_windows": int(split_data["x_val"].shape[0]),
                    "n_test_windows": int(split_data["x_test"].shape[0]),
                }
                row.update(metrics)
                rows.append(row)

    save_rows_table(rows, output_dir / "robustness_fold_metrics.xlsx")
    summary = aggregate_rows(
        rows,
        group_keys=["variant", "perturb_type", "perturb_target", "mask_channel", "perturb_level"],
    )
    save_rows_table(summary, output_dir / "robustness_summary.xlsx")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    horizons = sorted(parse_int_csv(args.horizons))
    if not horizons:
        raise ValueError("At least one horizon is required")
    if args.window_stride <= 0:
        raise ValueError("--window_stride must be > 0")

    output_dir = ensure_dir(args.output_dir)
    ensure_dir(output_dir / "histories")
    logger = setup_logger(output_dir=output_dir, name="train_lopo")
    set_seed(args.seed)

    save_json(output_dir / "config.json", vars(args))
    logger.info("Loading subject data from %s", args.data_dir)
    subject_series = load_subject_series(
        data_dir=args.data_dir,
        features=FEATURES,
        nan_strategy=args.nan_strategy,
    )
    all_folds = lopo_folds(sorted(subject_series.keys(), key=int))
    folds = select_folds(
        folds=all_folds,
        test_subject=args.test_subject,
        max_folds=args.max_folds,
    )
    if args.test_subject.strip():
        logger.info("Filtered to single test subject=%s", folds[0][0])
    logger.info("Subjects=%d Folds=%d Horizons=%s", len(subject_series), len(folds), horizons)

    if args.exp_mode == "standard":
        run_standard_mode(args, subject_series, folds, horizons, output_dir, logger)
    elif args.exp_mode == "data_efficiency":
        run_data_efficiency_mode(args, subject_series, folds, horizons, output_dir, logger)
    elif args.exp_mode == "robustness":
        run_robustness_mode(args, subject_series, folds, horizons, output_dir, logger)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported exp_mode: {args.exp_mode}")

    logger.info("Finished. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
