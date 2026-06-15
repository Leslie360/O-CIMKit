import os
import argparse
import json
import numpy as np
import pandas as pd
from device_profile import DeviceProfile
from fitter import fit_ltp_ltd, fit_volatile_decay

def parse_args():
    parser = argparse.ArgumentParser(description="Parse raw experimental device measurements to generate a DeviceProfile JSON.")
    parser.add_argument("--file", type=str, required=True, help="Path to raw Excel (.xlsx) or Text (.txt/.csv) file.")
    parser.add_argument("--name", type=str, required=True, help="Name of the device profile (e.g. MyOECT).")
    parser.add_argument("--type", type=str, choices=["volatile", "nonvolatile", "dual"], default="nonvolatile", 
                        help="Type of the device profile.")
    parser.add_argument("--states", type=int, default=64, help="Discrete state count for nonvolatile mode.")
    parser.add_argument("--noise-ratio", type=float, default=0.02, help="Cycle-to-cycle noise ratio relative to dynamic range.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    file_path = args.file
    name = args.name
    device_type = args.type
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return
        
    print(f"🔄 Parsing raw device data from: {file_path}")
    
    # Read the file
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        # Excel reading
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        sheet_name = xls.sheet_names[0]
        df = xls.parse(sheet_name)
    else:
        # Delimited file reading
        df = pd.read_csv(file_path, sep=None, engine='python')
        
    print(f"📋 Columns found: {list(df.columns)}")
    
    # Process based on type
    profile_kwargs = {
        "is_volatile": False,
        "is_nonvolatile": False,
        "noise_std_ratio": args.noise_ratio
    }
    
    if device_type == "volatile":
        profile_kwargs["is_volatile"] = True
        
        # Check standard columns for volatile decay
        time_col = [c for c in df.columns if "time" in c.lower() or "时间" in c.lower()]
        curr_col = [c for c in df.columns if "current" in c.lower() or "电流" in c.lower() or "decay" in c.lower()]
        
        if len(time_col) > 0 and len(curr_col) > 0:
            time_vals = df[time_col[0]].dropna().values
            curr_vals = df[curr_col[0]].dropna().values
            
            # Simple tau estimation
            curr_norm = (curr_vals - curr_vals.min()) / (curr_vals.max() - curr_vals.min())
            max_curr = curr_norm[0]
            threshold = max_curr / np.e
            tau = time_vals[-1]
            for i, c in enumerate(curr_norm):
                if c < threshold:
                    tau = time_vals[i]
                    break
            
            # Fit exponential decay for better accuracy
            fitted_tau = fit_volatile_decay(time_vals, curr_vals)
            print(f"⚡ Estimated time constant (1/e): {tau:.4f}s, Fitted time constant: {fitted_tau:.4f}s")
            
            profile_kwargs["tau_volatile"] = float(fitted_tau)
            profile_kwargs["conductance_min"] = float(curr_vals.min())
            profile_kwargs["conductance_max"] = float(curr_vals.max())
        else:
            print("⚠️ Warning: Could not auto-detect time/current columns. Using default values.")
            profile_kwargs["tau_volatile"] = 3.64
            
    elif device_type == "nonvolatile":
        profile_kwargs["is_nonvolatile"] = True
        profile_kwargs["discrete_states_count"] = args.states
        
        # Look for LTP/LTD columns or multi-column data
        ltp_col = [c for c in df.columns if "ltp" in c.lower() or "增强" in c.lower() or "blue" in c.lower()]
        ltd_col = [c for c in df.columns if "ltd" in c.lower() or "抑制" in c.lower() or "red" in c.lower()]
        
        # If no specific columns, check if we have 2 main columns (e.g. Time vs Current)
        if len(ltp_col) == 0 and len(ltd_col) == 0 and len(df.columns) >= 2:
            print("ℹ️ Mapping general current curve to LTP/LTD updates...")
            curr_vals = df[df.columns[1]].dropna().values
            # Split the curve in half to simulate LTP/LTD if it is a full cycle
            half = len(curr_vals) // 2
            ltp_raw = curr_vals[:half]
            ltd_raw = curr_vals[half:]
        else:
            ltp_raw = df[ltp_col[0]].dropna().values if len(ltp_col) > 0 else []
            ltd_raw = df[ltd_col[0]].dropna().values if len(ltd_col) > 0 else []
            
        if len(ltp_raw) > 1 and len(ltd_raw) > 1:
            fit_results = fit_ltp_ltd(ltp_raw, ltd_raw)
            profile_kwargs.update(fit_results)
            print(f"📈 Fitted LTP/LTD non-linear polynomials (Range: {fit_results['conductance_min']:.2e} ~ {fit_results['conductance_max']:.2e} S/A)")
        else:
            print("⚠️ Warning: LTP/LTD data insufficient for curve fitting. Using linear default.")
            profile_kwargs["conductance_min"] = 1e-10
            profile_kwargs["conductance_max"] = 2e-9
            profile_kwargs["ltp_poly_coefficients"] = [0.0, 0.0, 0.0, 0.05]
            profile_kwargs["ltd_poly_coefficients"] = [0.0, 0.0, 0.0, -0.05]

    # Create the DeviceProfile and save to repository
    profile = DeviceProfile(name=name, device_type=device_type, **profile_kwargs)
    json_path = f"profiles/repository/{name}.json"
    profile.to_json(json_path)
    
    print(f"🎉 Success! Device profile saved to: {json_path}")

if __name__ == "__main__":
    main()
