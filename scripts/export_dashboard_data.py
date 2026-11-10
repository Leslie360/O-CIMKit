import json
from pathlib import Path

import re

import numpy as np
from openpyxl import load_workbook


def read_reliability_report(report_path):
    values = [[], [], [], []]
    active_task = False
    time_points = {"Fresh (0h)": 0, "1 Month (720h)": 1, "10 Years (87.6k)": 2}
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        task = cells[0].replace("*", "")
        if task:
            active_task = task.startswith("Face Rec (Yale Grayscale)")
        if not active_task or cells[1] not in time_points:
            continue
        time_index = time_points[cells[1]]
        for method_index, cell in enumerate(cells[2:6]):
            match = re.search(r"(\d+(?:\.\d+)?)%", cell)
            if not match:
                raise ValueError(f"Missing reliability value in {report_path}: {line}")
            values[method_index].append((time_index, float(match.group(1))))
    ordered = [[value for _, value in sorted(items)] for items in values]
    if any(len(items) != 3 for items in ordered):
        raise ValueError(f"Expected three Yale aging points in {report_path}")
    return {
        "labels": ["全新器件", "1 个月", "10 年"],
        "methods": [
            {"name": "无补偿", "color": "#d34b4b", "values": ordered[0]},
            {"name": "全局缩放", "color": "#d59120", "values": ordered[1]},
            {"name": "参考校准", "color": "#7367c9", "values": ordered[2]},
            {"name": "O-CIMKit 统计补偿", "color": "#0f8b7c", "values": ordered[3]},
        ],
        "source": "reports/top_journal_benchmark_report.md",
        "scope": "Yale small-sample recognition with model-based device aging; not physical ten-year operation",
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    workbook_path = project_root / "data" / "devices" / "LTP-LTD.xlsx"
    profile_path = project_root / "profiles" / "repository" / "XJTU_Measured_Memristor.json"
    result_path = project_root / "reports" / "xjtu_measured_ecg_results.json"
    patient_result_path = project_root / "reports" / "xjtu_measured_ecg_patient_group_results.json"

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    sheet_names = [name for name in workbook.sheetnames if name.strip().startswith("单个")]
    devices = []
    for name in sheet_names:
        sheet = workbook[name]
        values = [float(row[1]) for row in sheet.iter_rows(values_only=True) if row[1] is not None]
        devices.append({"name": name, "values": np.round(values, 6).tolist()})
    if not devices:
        raise ValueError("No single-device sheets found in the measured workbook.")

    curves = np.array([device["values"] for device in devices], dtype=float)
    with open(profile_path, encoding="utf-8") as handle:
        profile = json.load(handle)
    with open(result_path, encoding="utf-8") as handle:
        ecg = json.load(handle)
    ecg.setdefault("validation_strategy", "beat-kfold")
    if patient_result_path.exists():
        with open(patient_result_path, encoding="utf-8") as handle:
            patient_ecg = json.load(handle)
    else:
        patient_ecg = None

    fitting = profile["fitting_parameters"]
    dynamic = profile["dynamic_parameters"]
    response_min = fitting["conductance_min"]
    response_max = fitting["conductance_max"]
    data = {
        "measured": {
            "sourceFile": workbook_path.name,
            "devices": devices,
            "mean": np.round(curves.mean(axis=0), 6).tolist(),
            "min": np.round(curves.min(axis=0), 6).tolist(),
            "max": np.round(curves.max(axis=0), 6).tolist(),
            "profile": {
                "name": profile["device_name"],
                "states": dynamic["discrete_states_count"],
                "responseMin": response_min,
                "responseMax": response_max,
                "dynamicRange": response_max / response_min,
                "noiseRatio": fitting["noise_std_ratio"],
                "ltpPoly": fitting["ltp_poly_coefficients"],
                "ltdPoly": fitting["ltd_poly_coefficients"],
                "source": profile.get("source_metadata", {}),
            },
        },
        "ecg": ecg,
        "patient_ecg": patient_ecg,
        "reliability": read_reliability_report(project_root / "reports" / "top_journal_benchmark_report.md"),
        "meta": {
            "applicationCount": len([
                path for path in (project_root / "applications").iterdir()
                if path.is_dir() and (path / "train.py").exists()
            ]),
        },
    }
    output_path = project_root / "applications" / "web_dashboard" / "data.js"
    output_path.write_text("window.REVIEW_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"Dashboard evidence exported to: {output_path}")


if __name__ == "__main__":
    main()
