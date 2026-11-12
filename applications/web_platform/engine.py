import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from profiles.parser import read_curve, read_excel_table


class PlatformEngine:
    def __init__(self, project_root, jobs_root=None):
        self.project_root = Path(project_root).resolve()
        self.dashboard_root = self.project_root / "applications" / "web_dashboard"
        self.jobs_root = Path(jobs_root) if jobs_root else self.project_root / "data" / "platform_jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.example_path = self.project_root / "data" / "devices" / "LTP-LTD.xlsx"
        self.dataset_root = self.project_root / "data" / "datasets" / "mitdb"
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="o-cimkit-training")
        self.job_lock = threading.RLock()

    def close(self):
        self.executor.shutdown(wait=False)

    def health(self):
        records = sorted(self.dataset_root.glob("*.hea")) if self.dataset_root.is_dir() else []
        return {
            "status": "ok",
            "engine": "local-python",
            "example_available": self.example_path.is_file(),
            "mitdb_available": bool(records),
            "mitdb_record_count": len(records),
            "defaults": {"epochs": 150, "seed": 42, "validation": "patient-group-kfold"},
        }

    def job_directory(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", str(job_id)):
            raise FileNotFoundError("任务编号无效")
        directory = self.jobs_root / job_id
        if not directory.is_dir():
            raise FileNotFoundError("任务不存在或已被清理")
        return directory

    def write_job(self, job_id, values):
        directory = self.job_directory(job_id)
        with self.job_lock:
            path = directory / "job.json"
            temporary = directory / "job.json.tmp"
            temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, path)

    def read_job(self, job_id):
        directory = self.job_directory(job_id)
        path = directory / "job.json"
        if not path.is_file():
            raise FileNotFoundError("任务状态不可用")
        return json.loads(path.read_text(encoding="utf-8"))

    def update_job(self, job_id, **updates):
        with self.job_lock:
            job = self.read_job(job_id)
            job.update(updates)
            self.write_job(job_id, job)

    def curve_summary(self, input_path):
        input_path = Path(input_path)
        if input_path.suffix.lower() == ".xlsx":
            excel = pd.ExcelFile(input_path, engine="openpyxl")
            sheets = [name for name in excel.sheet_names if name.strip().startswith("单个")] or excel.sheet_names[:1]
            curves = [read_curve(read_excel_table(excel, sheet)) for sheet in sheets]
            names = sheets
        else:
            frame = pd.read_csv(input_path, sep=None, engine="python")
            curves = [read_curve(frame)]
            names = [input_path.stem]
        if len({len(curve) for curve in curves}) != 1:
            raise ValueError("所有器件曲线必须具有相同脉冲数")
        stacked = np.abs(np.stack(curves))
        return {
            "devices": [{"name": name, "values": np.round(curve, 6).tolist()}
                        for name, curve in zip(names, curves)],
            "mean": np.round(stacked.mean(axis=0), 6).tolist(),
            "min": np.round(stacked.min(axis=0), 6).tolist(),
            "max": np.round(stacked.max(axis=0), 6).tolist(),
        }

    def prepare_input(self, job_id, input_path, measurement_unit, example=False):
        directory = self.job_directory(job_id)
        profile_path = directory / "profile.json"
        command = [
            sys.executable,
            str(self.project_root / "profiles" / "parser.py"),
            "--file", str(input_path),
            "--name", f"Platform_Device_{job_id[:8]}",
            "--type", "nonvolatile",
            "--measurement-type", "current",
            "--output", str(profile_path),
        ]
        if measurement_unit:
            command.extend(["--measurement-unit", measurement_unit])
        completed = subprocess.run(command, cwd=self.project_root, capture_output=True, text=True, timeout=90)
        (directory / "ingestion.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not profile_path.is_file():
            raise ValueError("数据解析失败：" + (completed.stdout + completed.stderr).strip().splitlines()[-1])
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        prepared = {
            "id": job_id,
            "status": "prepared",
            "progress": 0,
            "message": "器件画像已生成，训练尚未开始",
            "example": example,
            "profile": profile,
            "measured": self.curve_summary(input_path),
            "source_file": input_path.name,
        }
        self.write_job(job_id, prepared)
        return prepared

    def prepare_example(self):
        if not self.example_path.is_file():
            raise FileNotFoundError("示例工作表不存在")
        job_id = uuid.uuid4().hex
        directory = self.jobs_root / job_id
        input_directory = directory / "input"
        input_directory.mkdir(parents=True)
        input_path = input_directory / self.example_path.name
        shutil.copyfile(self.example_path, input_path)
        return self.prepare_input(job_id, input_path, "", example=True)

    def prepare_upload(self, filename, content, measurement_unit=""):
        if not filename or Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("文件名无效")
        if Path(filename).suffix.lower() not in {".xlsx", ".csv", ".txt"}:
            raise ValueError("仅支持 .xlsx、.csv、.txt 器件工作表")
        if not content:
            raise ValueError("上传文件为空")
        if len(content) > 20 * 1024 * 1024:
            raise ValueError("单个工作表不能超过 20 MB")
        if measurement_unit not in {"", "uA", "µA", "mA", "A", "S"}:
            raise ValueError("不支持的测量单位")
        job_id = uuid.uuid4().hex
        directory = self.jobs_root / job_id
        input_directory = directory / "input"
        input_directory.mkdir(parents=True)
        input_path = input_directory / filename
        input_path.write_bytes(content)
        try:
            return self.prepare_input(job_id, input_path, measurement_unit)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def start_training(self, prepared_id, epochs=150, seed=42, validation="patient-group-kfold"):
        directory = self.job_directory(prepared_id)
        job = self.read_job(prepared_id)
        if job["status"] != "prepared":
            raise ValueError("该画像已提交训练，请重新导入数据或加载示例")
        if not isinstance(epochs, int) or isinstance(epochs, bool) or not 1 <= epochs <= 300:
            raise ValueError("训练轮数必须是 1–300 的整数")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2147483647:
            raise ValueError("随机种子无效")
        if validation not in {"beat-kfold", "patient-group-kfold"}:
            raise ValueError("验证协议无效")
        if not list(self.dataset_root.glob("*.hea")):
            raise FileNotFoundError("本地未配置 MIT-BIH 数据，不能启动训练")
        parameters = {"epochs": epochs, "seed": seed, "validation": validation}
        self.update_job(prepared_id, status="queued", progress=1, message="训练任务已排队", parameters=parameters)
        self.executor.submit(self.run_training, prepared_id, parameters)
        return self.read_job(prepared_id)

    def run_training(self, job_id, parameters):
        directory = self.job_directory(job_id)
        output_path = directory / "report.json"
        command = [
            sys.executable,
            str(self.project_root / "applications" / "ecg_cardio" / "train.py"),
            "--dataset", "mitdb",
            "--epochs", str(parameters["epochs"]),
            "--device-profile", str(directory / "profile.json"),
            "--validation", parameters["validation"],
            "--seed", str(parameters["seed"]),
            "--output-json", str(output_path),
            "--export-predictions",
        ]
        try:
            self.update_job(job_id, status="running", progress=3, message="加载心电数据与器件画像")
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(command, cwd=self.project_root, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1, env=environment)
            phase = 0
            with (directory / "training.log").open("w", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if "Feature shape:" in line:
                        self.update_job(job_id, progress=10, message="心电特征提取完成")
                    elif "Phase 1:" in line:
                        phase = 1
                        self.update_job(job_id, progress=15, message="浮点基线训练")
                    elif "Phase 2:" in line:
                        phase = 2
                        self.update_job(job_id, progress=65, message="实测画像约束训练")
                    else:
                        match = re.search(r"Fold (\d+)/5", line)
                        if match and phase:
                            fold = int(match.group(1))
                            progress = 15 + fold * 9 if phase == 1 else 65 + fold * 6
                            self.update_job(job_id, progress=progress,
                                            message=f"{'浮点基线' if phase == 1 else '画像约束'}：第 {fold}/5 折完成")
            return_code = process.wait(timeout=7200)
            if return_code != 0:
                lines = (directory / "training.log").read_text(encoding="utf-8").splitlines()
                raise RuntimeError("训练失败：" + "\n".join(lines[-12:]))
            report = json.loads(output_path.read_text(encoding="utf-8"))
            if "prediction_evidence" not in report:
                raise RuntimeError("训练结束但未生成逐样本预测证据")
            self.update_job(job_id, status="completed", progress=100,
                            message="训练、可视化数据与分析报告已生成")
        except Exception as error:
            self.update_job(job_id, status="failed", progress=0, message="任务失败", error=str(error))

    def result(self, job_id):
        job = self.read_job(job_id)
        if job["status"] != "completed":
            raise ValueError("训练尚未完成")
        directory = self.job_directory(job_id)
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        return {
            "id": job_id,
            "job": job,
            "report": report,
            "analysis": build_result_analysis(report),
            "artifacts": {
                "report": f"/api/artifacts/{job_id}/report.json",
                "predictions": f"/api/artifacts/{job_id}/predictions.csv",
                "profile": f"/api/artifacts/{job_id}/profile.json",
                "examples": f"/api/artifacts/{job_id}/prediction_examples.json",
            },
        }

    def artifact(self, job_id, filename):
        if filename not in {"report.json", "predictions.csv", "profile.json", "prediction_examples.json", "training.log"}:
            raise FileNotFoundError("下载文件不存在")
        path = self.job_directory(job_id) / filename
        if not path.is_file():
            raise FileNotFoundError("下载文件不存在")
        return path


def build_result_analysis(report):
    baseline = report["float_baseline"]
    measured = report["measured_device_simulation"]
    delta = report["accuracy_delta_percentage_points"]
    grouped = report.get("validation_strategy") == "patient-group-kfold"
    protocol = "记录隔离五折" if grouped else "心搏级内部五折"
    statements = [
        f"{protocol}：浮点基线准确率 {baseline['accuracy_mean_percent']:.2f}%，"
        f"实测画像约束准确率 {measured['accuracy_mean_percent']:.2f}%，差值 {delta:+.2f} 个百分点。",
    ]
    evidence = report.get("prediction_evidence", {}).get("models", {})
    if evidence:
        pooled_baseline = evidence["float_baseline"]["accuracy_percent"]
        pooled_measured = evidence["measured_device_simulation"]["accuracy_percent"]
        statements.append(f"五折均值对各折等权；合并 {report['sample_count']} 个留出心搏后按样本数加权："
                          f"浮点 {pooled_baseline:.2f}%，画像约束 {pooled_measured:.2f}%。")
        normal_recall = evidence["measured_device_simulation"]["normal_recall_percent"]
        abnormal_recall = evidence["measured_device_simulation"]["abnormal_recall_percent"]
        statements.append(f"画像约束模型正常心搏召回率 {normal_recall:.2f}%，异常心搏召回率 {abnormal_recall:.2f}%；"
                          "两类样本应分别核查，不能只看整体准确率。")
    folds = measured["folds"]
    lowest = min(folds, key=lambda fold: fold["accuracy_percent"])
    statement = f"最低折为第 {lowest['fold']} 折（{lowest['accuracy_percent']:.2f}%）"
    if lowest.get("test_records"):
        statement += f"，测试记录为 {', '.join(lowest['test_records'])}"
    statements.append(statement + "，反映折间差异，应结合记录来源继续分析。")
    if grouped:
        statements.append("记录隔离结果不能表述为临床泛化；下一步应扩大记录与器件批次，并优先检查异常心搏漏检和跨记录校准。")
    else:
        statements.append("心搏级划分可能让同一记录进入训练与测试，只用于验证数据链路，不作为跨患者性能证据。")
    return statements
