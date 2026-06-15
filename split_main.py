import os

with open('main.py', 'r') as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

os.makedirs('cli', exist_ok=True)

with open('cli/__init__.py', 'w') as f:
    f.write("")

# Extract components
imports_and_refs = get_lines(1, 48)
run_app_code = get_lines(49, 117)
exec_eval_code = get_lines(118, 188)
run_bench_code = get_lines(189, 350)
run_diag_code = get_lines(351, 537)
run_codesign_code = get_lines(538, 670)
main_code = get_lines(671, 745)

# cli/run.py
with open('cli/run.py', 'w') as f:
    f.write("import os\nimport sys\nimport subprocess\n\n")
    f.write(run_app_code)

# cli/benchmark.py
with open('cli/benchmark.py', 'w') as f:
    f.write("import os\nimport sys\nimport subprocess\nimport re\nimport json\n\n")
    f.write("from profiles.device_profile import DeviceProfile\n\n")
    
    # Just grab REFERENCE_ACCURACIES
    ref_idx = imports_and_refs.find("REFERENCE_ACCURACIES")
    if ref_idx != -1:
        f.write(imports_and_refs[ref_idx:])
        
    f.write("\n")
    f.write(exec_eval_code)
    f.write("\n")
    f.write(run_bench_code)

# cli/diagnostics.py
with open('cli/diagnostics.py', 'w') as f:
    f.write("import os\nimport json\nimport sys\n\nfrom profiles.device_profile import DeviceProfile\n\n")
    f.write(run_diag_code)

# cli/codesign.py
with open('cli/codesign.py', 'w') as f:
    f.write("import os\nimport json\nimport sys\n\nfrom profiles.device_profile import DeviceProfile\n\n")
    f.write(run_codesign_code)

# cli/main.py
with open('cli/main.py', 'w') as f:
    f.write("import argparse\nimport sys\n\n")
    f.write("from cli.run import run_application\n")
    f.write("from cli.benchmark import run_benchmark\n")
    f.write("from cli.diagnostics import run_diagnostics\n")
    f.write("from cli.codesign import run_codesign\n\n")
    f.write(main_code)

# root main.py
with open('main.py', 'w') as f:
    f.write("#!/usr/bin/env python3\n")
    f.write("import sys\n")
    f.write("import os\n")
    f.write("sys.path.append(os.path.dirname(os.path.abspath(__file__)))\n")
    f.write("from cli.main import main\n\n")
    f.write("if __name__ == '__main__':\n")
    f.write("    main()\n")

print("Successfully split main.py!")
