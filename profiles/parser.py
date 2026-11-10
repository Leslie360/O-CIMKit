import os
import argparse
import numpy as np
import pandas as pd

try:
    from .device_profile import DeviceProfile
    from .fitter import fit_ltp_ltd, fit_volatile_decay
except ImportError:
    from device_profile import DeviceProfile
    from fitter import fit_ltp_ltd, fit_volatile_decay


def parse_args():
    parser = argparse.ArgumentParser(description="Parse raw experimental device measurements to generate a DeviceProfile JSON.")
    parser.add_argument("--file", type=str, required=True, help="Path to raw Excel (.xlsx) or Text (.txt/.csv) file.")
    parser.add_argument("--name", type=str, required=True, help="Name of the device profile (e.g. MyOECT).")
    parser.add_argument("--type", type=str, choices=["volatile", "nonvolatile", "dual"], default="nonvolatile",
                        help="Type of the device profile.")
    parser.add_argument("--states", type=int, default=None,
                        help="Discrete state count. Multi-device LTP/LTD workbooks default to the LTP pulse count; other inputs default to 64.")
    parser.add_argument("--noise-ratio", type=float, default=None,
                        help="Noise ratio relative to the response range. Multi-device workbooks estimate D2D CV automatically when omitted.")
    parser.add_argument("--measurement-type", choices=["current", "conductance"], default="current",
                        help="Physical meaning of the source values; current is never relabelled as absolute conductance.")
    parser.add_argument("--measurement-unit", default=None, help="Source unit, for example uA or S. Leave empty if the workbook does not specify it.")
    parser.add_argument("--output", default=None, help="Output JSON path (default: profiles/repository/<name>.json).")
    return parser.parse_args()


def read_excel_table(xls, sheet_name):
    frame = xls.parse(sheet_name, header=None).dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError(f"Sheet '{sheet_name}' contains no data.")
    if any(isinstance(value, str) for value in frame.iloc[0].tolist()):
        frame.columns = [str(value) for value in frame.iloc[0]]
        frame = frame.iloc[1:].reset_index(drop=True)
    else:
        frame.columns = [f"column_{index}" for index in range(frame.shape[1])]
    return frame


def read_curve(frame):
    if frame.shape[1] < 2:
        raise ValueError("A device sheet needs a pulse/index column and a measurement column.")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 4:
        raise ValueError("At least four measurements are required per device.")
    return values


def estimate_d2d_cv(curves, response_min, response_max, half):
    scale = response_max - response_min
    if scale <= 0:
        return 0.0
    endpoints = np.array([(curve[half - 1] - response_min) / scale for curve in curves])
    endpoint_mean = float(np.mean(endpoints))
    if np.isclose(endpoint_mean, 0.0):
        return 0.0
    return float(np.std(endpoints) / abs(endpoint_mean))


def main():
    args = parse_args()
    file_path = args.file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    xls = None
    if ext in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(file_path, engine="openpyxl")
        source_sheets = [name for name in xls.sheet_names if name.strip().startswith("单个")]
        if not source_sheets:
            source_sheets = [xls.sheet_names[0]]
        df = read_excel_table(xls, source_sheets[0])
    else:
        df = pd.read_csv(file_path, sep=None, engine="python")
        source_sheets = []

    print(f"Reading {file_path}")
    print(f"Columns: {list(df.columns)}")

    profile_kwargs = {
        "is_volatile": False,
        "is_nonvolatile": False,
        "noise_std_ratio": args.noise_ratio if args.noise_ratio is not None else 0.02,
    }
    source_metadata = {
        "source_file": os.path.basename(file_path),
        "source_sheets": source_sheets,
        "measurement_type": args.measurement_type,
        "measurement_unit": args.measurement_unit or ("S" if args.measurement_type == "conductance" else "not specified in source"),
    }

    if args.type == "volatile":
        profile_kwargs["is_volatile"] = True
        time_col = [c for c in df.columns if "time" in c.lower() or "时间" in c.lower()]
        curr_col = [c for c in df.columns if "current" in c.lower() or "电流" in c.lower() or "decay" in c.lower()]
        if not time_col and not curr_col and df.shape[1] >= 2:
            time_col, curr_col = [df.columns[0]], [df.columns[1]]

        if time_col and curr_col:
            time_vals = pd.to_numeric(df[time_col[0]], errors="coerce").dropna().to_numpy(dtype=float)
            curr_vals = pd.to_numeric(df[curr_col[0]], errors="coerce").dropna().to_numpy(dtype=float)
            fitted_tau = fit_volatile_decay(time_vals, curr_vals)
            profile_kwargs["tau_volatile"] = float(fitted_tau)
            profile_kwargs["conductance_min"] = float(np.abs(curr_vals).min())
            profile_kwargs["conductance_max"] = float(np.abs(curr_vals).max())
            print(f"Fitted relaxation time constant: {fitted_tau:.4f} s")
        else:
            raise ValueError("Could not detect time/current columns. Use named columns or a two-column table.")

    elif args.type == "nonvolatile":
        profile_kwargs["is_nonvolatile"] = True
        curves = None
        if xls is not None and len(source_sheets) > 1:
            curves = [read_curve(read_excel_table(xls, sheet)) for sheet in source_sheets]
            lengths = {len(curve) for curve in curves}
            if len(lengths) != 1:
                raise ValueError(f"All device sheets must have the same pulse count; found {sorted(lengths)}.")
            if len(curves[0]) % 2:
                raise ValueError("A full LTP/LTD cycle must contain an even number of pulses.")
            half = len(curves[0]) // 2
            stacked = np.abs(np.stack(curves))
            mean_curve = stacked.mean(axis=0)
            ltp_raw, ltd_raw = mean_curve[:half], mean_curve[half:]
            fit_results = fit_ltp_ltd(ltp_raw, ltd_raw)
            response_min = fit_results["conductance_min"]
            response_max = fit_results["conductance_max"]
            if args.noise_ratio is None:
                profile_kwargs["noise_std_ratio"] = estimate_d2d_cv(stacked, response_min, response_max, half)
            source_metadata.update({
                "device_count": len(curves),
                "pulses_per_device": len(curves[0]),
                "aggregation": "arithmetic mean across device sheets",
                "d2d_cv_at_ltp_end": profile_kwargs["noise_std_ratio"],
            })
            print(f"Aggregated {len(curves)} measured devices, {len(curves[0])} pulses each.")
        else:
            ltp_col = [c for c in df.columns if "ltp" in c.lower() or "增强" in c.lower() or "blue" in c.lower()]
            ltd_col = [c for c in df.columns if "ltd" in c.lower() or "抑制" in c.lower() or "red" in c.lower()]
            if not ltp_col and not ltd_col and len(df.columns) >= 2:
                curr_vals = read_curve(df)
                if len(curr_vals) % 2:
                    raise ValueError("A full LTP/LTD cycle must contain an even number of pulses.")
                half = len(curr_vals) // 2
                ltp_raw, ltd_raw = curr_vals[:half], curr_vals[half:]
                source_metadata.update({"device_count": 1, "pulses_per_device": len(curr_vals)})
            else:
                if not ltp_col or not ltd_col:
                    raise ValueError("Both LTP and LTD columns are required when using separate update columns.")
                ltp_raw = pd.to_numeric(df[ltp_col[0]], errors="coerce").dropna().to_numpy(dtype=float)
                ltd_raw = pd.to_numeric(df[ltd_col[0]], errors="coerce").dropna().to_numpy(dtype=float)
                source_metadata.update({"device_count": 1, "pulses_per_device": len(ltp_raw) + len(ltd_raw)})
            fit_results = fit_ltp_ltd(ltp_raw, ltd_raw)

        if len(ltp_raw) < 2 or len(ltd_raw) < 2:
            raise ValueError("LTP/LTD data are insufficient for curve fitting.")
        profile_kwargs.update(fit_results)
        profile_kwargs["discrete_states_count"] = args.states or (len(ltp_raw) if curves is not None else 64)
        response_range = fit_results["conductance_max"] - fit_results["conductance_min"]
        print(f"Response range: {fit_results['conductance_min']:.6g} to {fit_results['conductance_max']:.6g} ({source_metadata['measurement_unit']})")
        print(f"Dynamic range ratio: {(fit_results['conductance_max'] / fit_results['conductance_min']):.4f}x")
        print(f"Noise ratio: {profile_kwargs['noise_std_ratio']:.6g}; states: {profile_kwargs['discrete_states_count']}")
        if response_range <= 0:
            raise ValueError("The measured response range must be positive.")
    else:
        raise ValueError("Dual profiles require separate volatile and nonvolatile inputs; run the parser for each component.")

    profile_kwargs["source_metadata"] = source_metadata
    profile = DeviceProfile(name=args.name, device_type="memristor", **profile_kwargs)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = args.output or os.path.join(project_root, "profiles", "repository", f"{args.name}.json")
    profile.to_json(json_path)
    print(f"Device profile saved to: {json_path}")


if __name__ == "__main__":
    main()
