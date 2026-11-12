import argparse
import time
from pathlib import Path

import wfdb

RECORD_IDS = ["100", "101", "103", "105", "106", "108", "109", "111",
              "112", "113", "114", "115", "116", "117", "118"]
REQUIRED_SUFFIXES = [".dat", ".hea", ".atr"]


def missing_files(target):
    target = Path(target)
    return [f"{record}{suffix}" for record in RECORD_IDS for suffix in REQUIRED_SUFFIXES
            if not (target / f"{record}{suffix}").is_file()]


def prepare_ecg_data(target, overwrite=False, downloader=None):
    target = Path(target)
    if target.is_symlink() and not target.exists():
        target.unlink()
    target.mkdir(parents=True, exist_ok=True)
    missing = missing_files(target)
    if missing or overwrite:
        download = downloader or wfdb.dl_database
        for attempt in range(3):
            try:
                download("mitdb", str(target), records=RECORD_IDS, annotators=["atr"],
                         keep_subdirs=False, overwrite=overwrite)
                if not missing_files(target):
                    break
            except Exception:
                if attempt == 2:
                    raise
            if attempt < 2:
                print(f"下载暂时失败或文件不完整，正在重试（{attempt + 1}/2）……")
                time.sleep(2 ** attempt)
    missing = missing_files(target)
    if missing:
        raise RuntimeError(f"MIT-BIH 数据未准备完整，缺少：{', '.join(missing)}")
    print(f"MIT-BIH 数据已就绪：{target}（{len(RECORD_IDS)} 条记录，含波形与标注）")
    return target


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download the 15 public MIT-BIH records used by O-CIMKit")
    parser.add_argument("--output-dir", default=str(project_root / "data" / "datasets" / "mitdb"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_ecg_data(args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
