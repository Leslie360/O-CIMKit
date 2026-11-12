import os
import sys
import time
import json
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.ecg_cardio.dataset import load_ecg_data
from applications.ecg_cardio.model import OptoelectronicReservoir, ECGQATClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_validation_splits(X_feat, y, groups=None, validation="beat-kfold"):
    if validation == "beat-kfold":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return list(splitter.split(X_feat, y))
    if validation != "patient-group-kfold":
        raise ValueError(f"Unknown validation strategy: {validation}")
    if groups is None:
        raise ValueError("Record identifiers are required for patient-group-kfold validation.")
    groups = np.asarray(groups)
    if len(groups) != len(y) or len(np.unique(groups)) < 5:
        raise ValueError("Grouped validation needs one record identifier per beat and at least five records.")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    return list(splitter.split(X_feat, y, groups=groups))


def train_and_evaluate_ecg(X_feat, y, out_classes, device_profile=None, hidden_dim=256, lr=0.005,
                           epochs=150, dropout_p=0.3, seed=42, groups=None, validation="beat-kfold",
                           prediction_records=None, model_name=None):
    splits = make_validation_splits(X_feat, y, groups=groups, validation=validation)
    folds = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        random.seed(seed + fold)
        np.random.seed(seed + fold)
        torch.manual_seed(seed + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + fold)

        X_train, X_test = X_feat[train_idx], X_feat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        model = ECGQATClassifier(
            in_features=X_feat.shape[1],
            hidden_dim=hidden_dim,
            out_classes=out_classes,
            device_profile=device_profile,
            dropout_p=dropout_p,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        generator = torch.Generator().manual_seed(seed + fold)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)

        model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                noise = torch.randn_like(bx) * 0.05
                outputs = model(bx + noise)
                loss = criterion(outputs, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            preds = outputs.argmax(dim=-1).cpu().numpy()
        acc = accuracy_score(y_test, preds) * 100
        f1 = f1_score(y_test, preds, average="macro", labels=[0, 1], zero_division=0)
        if prediction_records is not None:
            if not model_name:
                raise ValueError("model_name is required when collecting prediction evidence.")
            probabilities = torch.softmax(outputs, dim=-1).cpu().numpy()
            for sample_index, prediction, probability in zip(test_idx, preds, probabilities):
                prediction_records.append({
                    "sample_index": int(sample_index),
                    "fold": fold + 1,
                    "record_id": str(groups[sample_index]) if groups is not None else "",
                    "model": model_name,
                    "y_true": int(y[sample_index]),
                    "y_pred": int(prediction),
                    "normal_score": float(probability[0]),
                    "abnormal_score": float(probability[1]),
                })
        fold_result = {"fold": fold + 1, "accuracy_percent": float(acc), "f1_macro": float(f1),
                       "test_samples": int(len(test_idx)),
                       "test_class_counts": {str(label): int(count) for label, count in
                                             zip(*np.unique(y_test, return_counts=True))}}
        if validation == "patient-group-kfold":
            train_records = set(np.asarray(groups)[train_idx])
            test_records = set(np.asarray(groups)[test_idx])
            if train_records & test_records:
                raise RuntimeError("Record leakage detected between training and test folds.")
            fold_result.update({"train_record_count": len(train_records), "test_record_count": len(test_records),
                                "test_records": sorted(test_records)})
        folds.append(fold_result)
        print(f"  Fold {fold + 1}/5 - Accuracy: {acc:.2f}%, F1-Score: {f1:.4f}")

    accs = np.array([fold["accuracy_percent"] for fold in folds])
    f1s = np.array([fold["f1_macro"] for fold in folds])
    return {
        "accuracy_mean_percent": float(np.mean(accs)),
        "accuracy_std_percent": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "folds": folds,
    }


def summarize_prediction_records(prediction_records):
    summaries = {}
    for model_name in ["float_baseline", "measured_device_simulation"]:
        records = [record for record in prediction_records if record["model"] == model_name]
        if not records:
            raise ValueError(f"No prediction records for {model_name}")
        y_true = np.array([record["y_true"] for record in records])
        y_pred = np.array([record["y_pred"] for record in records])
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        support = matrix.sum(axis=1)
        recalls = np.divide(matrix.diagonal(), support, out=np.zeros(2, dtype=float), where=support > 0)
        summaries[model_name] = {
            "confusion_matrix": matrix.tolist(),
            "support": support.tolist(),
            "normal_recall_percent": float(recalls[0] * 100),
            "abnormal_recall_percent": float(recalls[1] * 100),
            "accuracy_percent": float(accuracy_score(y_true, y_pred) * 100),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        }
    return summaries


def select_prediction_examples(prediction_records, waveforms):
    model_names = {"float_baseline", "measured_device_simulation"}
    by_sample = {}
    for record in prediction_records:
        by_sample.setdefault(record["sample_index"], {})[record["model"]] = record
    examples = []
    for class_index, class_name in [(0, "正常心搏"), (1, "异常心搏")]:
        candidates = []
        for sample_index, models in by_sample.items():
            if not model_names.issubset(models):
                continue
            if any(models[name]["y_true"] != class_index or models[name]["y_pred"] != class_index
                   for name in model_names):
                continue
            score = models["measured_device_simulation"][f"{'normal' if class_index == 0 else 'abnormal'}_score"]
            candidates.append((score, sample_index, models))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, sample_index, models = candidates[(len(candidates) - 1) // 2]
        measured = models["measured_device_simulation"]
        examples.append({
            "sample_index": int(sample_index),
            "record_id": measured["record_id"],
            "fold": measured["fold"],
            "class_index": class_index,
            "class_name": class_name,
            "waveform": np.round(np.asarray(waveforms[sample_index], dtype=float), 6).tolist(),
            "amplitude": "z-score normalized within each 180-sample heartbeat window",
            "float_prediction": models["float_baseline"]["y_pred"],
            "measured_prediction": measured["y_pred"],
            "float_scores": {
                "normal": models["float_baseline"]["normal_score"],
                "abnormal": models["float_baseline"]["abnormal_score"],
            },
            "measured_scores": {
                "normal": measured["normal_score"],
                "abnormal": measured["abnormal_score"],
            },
        })
    return examples


def export_prediction_evidence(output_json_path, prediction_records, waveforms):
    import csv

    output_dir = os.path.dirname(os.path.abspath(output_json_path))
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.csv")
    fieldnames = ["sample_index", "fold", "record_id", "model", "y_true", "y_pred",
                  "normal_score", "abnormal_score"]
    with open(predictions_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(prediction_records, key=lambda record: (
            record["sample_index"], record["model"])))
    examples = select_prediction_examples(prediction_records, waveforms)
    selection_rule = ("For each class, select the median measured-model true-class score among held-out beats "
                      "correctly classified by both models; ties are resolved by sample index.")
    with open(os.path.join(output_dir, "prediction_examples.json"), "w", encoding="utf-8") as handle:
        json.dump({"selection_rule": selection_rule, "examples": examples}, handle, ensure_ascii=False, indent=2)
    return {
        "predictions_csv": "predictions.csv",
        "examples_json": "prediction_examples.json",
        "labels": {"0": "正常心搏", "1": "异常心搏"},
        "selection_rule": selection_rule,
        "models": summarize_prediction_records(prediction_records),
        "examples": examples,
    }


def resolve_profile_path(value, project_root):
    if os.path.isfile(value):
        return os.path.abspath(value)
    candidate = os.path.join(project_root, "profiles", "repository", value)
    if not candidate.endswith(".json"):
        candidate += ".json"
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Device profile not found: {value}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ECG classification under device-profile constraints")
    parser.add_argument("--dataset", default="mitdb", choices=["mitdb", "ptbdb"], help="ECG dataset to use")
    parser.add_argument("--data-dir", default=None, help="Optional ECG source directory")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs per fold")
    parser.add_argument("--device-profile", default="PFO_AlOx_NonVolatile",
                        help="Nonvolatile profile name or JSON path")
    parser.add_argument("--validation", default="beat-kfold", choices=["beat-kfold", "patient-group-kfold"],
                        help="Beat-level folds reproduce the historical baseline; grouped folds isolate MIT-BIH records")
    parser.add_argument("--seed", type=int, default=42, help="Seed for repeatable model optimization")
    parser.add_argument("--output-json", default=None, help="Save fold-level results and validation metadata")
    parser.add_argument("--export-predictions", action="store_true",
                        help="Export held-out predictions, confusion matrices, and representative heartbeat inputs")
    args = parser.parse_args()
    if args.validation == "patient-group-kfold" and args.dataset != "mitdb":
        parser.error("patient-group-kfold requires MIT-BIH record identifiers.")
    if args.export_predictions and not args.output_json:
        parser.error("--export-predictions requires --output-json.")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    volatile_profile_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_Volatile.json")
    nonvolatile_profile_path = resolve_profile_path(args.device_profile, project_root)
    volatile_profile = DeviceProfile.from_json(volatile_profile_path) if os.path.exists(volatile_profile_path) else None
    nonvolatile_profile = DeviceProfile.from_json(nonvolatile_profile_path)

    print("=" * 60)
    print(f"ECG classification ({args.dataset.upper()}) | Validation: {args.validation}")
    print(f"Device profile: {nonvolatile_profile.device_name} | Seed: {args.seed}")
    print("=" * 60)

    X_ecg, y_ecg, record_groups = load_ecg_data(
        dataset_name=args.dataset, data_dir=args.data_dir, return_groups=args.dataset == "mitdb")
    print(f"  Input size: {X_ecg.shape}, Abnormal ratio: {np.mean(y_ecg) * 100:.2f}%")
    ecg_res = OptoelectronicReservoir(device_profile=volatile_profile, n_inputs=1, n_reservoir=2000,
                                      seed=args.seed, device=device)
    ecg_res.to(device)
    ecg_res.eval()

    print("Extracting reservoir features...")
    ecg_features = []
    batch_size = 256
    t0 = time.time()
    for i in range(0, len(X_ecg), batch_size):
        batch_X = torch.FloatTensor(X_ecg[i:i + batch_size]).to(device)
        with torch.no_grad():
            states = ecg_res(batch_X)
            feat = ecg_res.extract_features(states)
            ecg_features.append(feat.cpu().numpy())
    ecg_features = np.concatenate(ecg_features, axis=0)
    print(f"  Feature shape: {ecg_features.shape}, Extraction time: {time.time() - t0:.2f}s")

    prediction_records = [] if args.export_predictions else None
    print("Phase 1: Floating-point baseline...")
    float_result = train_and_evaluate_ecg(
        ecg_features, y_ecg, out_classes=2, device_profile=None, hidden_dim=256, lr=0.005,
        epochs=args.epochs, dropout_p=0.3, seed=args.seed, groups=record_groups, validation=args.validation,
        prediction_records=prediction_records, model_name="float_baseline")
    print("Phase 2: Device-profile-constrained model...")
    hardware_result = train_and_evaluate_ecg(
        ecg_features, y_ecg, out_classes=2, device_profile=nonvolatile_profile, hidden_dim=256,
        lr=0.005, epochs=args.epochs, dropout_p=0.3, seed=args.seed,
        groups=record_groups, validation=args.validation,
        prediction_records=prediction_records, model_name="measured_device_simulation")

    acc_float = float_result["accuracy_mean_percent"]
    acc_hw = hardware_result["accuracy_mean_percent"]
    print("  SIMULATION RESULTS & COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} | {'Ideal Float':<12} | {'Device profile':<16}")
    print("-" * 60)
    print(f"  {'Classification Acc (%)':<25} | {acc_float:<12.2f} | {acc_hw:<16.2f}")
    print(f"  {'F1-Score':<25} | {float_result['f1_macro_mean']:<12.4f} | {hardware_result['f1_macro_mean']:<16.4f}")
    print(f"  {'Accuracy Delta (pp)':<25} | {'N/A':<12} | {acc_hw - acc_float:<16.2f}")
    print("=" * 60)
    print(f"Final Result: Accuracy = {acc_hw:.2f}% ± {hardware_result['accuracy_std_percent']:.2f}%, "
          f"F1-Score = {hardware_result['f1_macro_mean']:.4f}")

    if args.output_json:
        report = {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "application": "ecg_cardio",
            "dataset": args.dataset,
            "data_origin": "public MIT-BIH recordings" if args.dataset == "mitdb" else "synthetic PTB-DB example waveforms",
            "sample_count": int(len(y_ecg)),
            "record_count": int(len(set(record_groups))) if record_groups is not None else None,
            "class_counts": {str(label): int(count) for label, count in zip(*np.unique(y_ecg, return_counts=True))},
            "validation_strategy": args.validation,
            "validation": ("StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), "
                           "groups are MIT-BIH record identifiers" if args.validation == "patient-group-kfold"
                           else "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), beats sampled across records"),
            "epochs_per_fold": args.epochs,
            "seed": args.seed,
            "device_profile": nonvolatile_profile.device_name,
            "device_profile_path": os.path.relpath(nonvolatile_profile_path, project_root),
            "device_profile_source": nonvolatile_profile.source_metadata,
            "float_baseline": float_result,
            "measured_device_simulation": hardware_result,
            "accuracy_delta_percentage_points": float(acc_hw - acc_float),
            "interpretation": "Device-profile-constrained simulation; not a clinical or fabricated-chip result.",
        }
        if args.export_predictions:
            report["prediction_evidence"] = export_prediction_evidence(
                args.output_json, prediction_records, X_ecg)
        output_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"Fold-level evidence saved to: {output_path}")


if __name__ == "__main__":
    main()
