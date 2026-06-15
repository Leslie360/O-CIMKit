# Contributing to Organic CIM Simulation Platform

Thank you for your interest in contributing to our project! We welcome contributions from the community to make this framework as robust, high-fidelity, and feature-rich as possible.

## 🛠️ Development Setup

To set up a local development environment, clone this repository and install the package in editable mode:

```bash
git clone https://github.com/Leslie360/CIM_application_project.git
cd CIM_application_project
pip install -e .
```

## 🧪 Running Unit Tests

We use Python's built-in `unittest` framework to verify changes. Before submitting a Pull Request, please run all unit tests to ensure no regressions:

```bash
python run_tests.py
```

All new modules, layers, or optimization strategies should be accompanied by corresponding unit tests inside the `tests/` directory.

## 📂 Code Style and Architecture

- **Modularity**: Keep device physics separate from neural network applications. If you add a new physical noise model or device dynamics, place it in `core/physics.py` or `core/layers.py`.
- **Formatting**: We follow standard PEP 8 naming conventions (e.g. `lower_snake_case` for filenames and functions, `PascalCase` for classes).
- **Documentation**: All public classes and functions should include descriptive docstrings detailing arguments and returns.

## 🚀 Proposing a New CIM Application

If you want to contribute a new neuromorphic application or benchmark dataset:
1. Create a subfolder inside `applications/` (e.g., `applications/my_new_task/`).
2. Implement `dataset.py`, `model.py`, and `train.py`.
3. Register your application in `main.py` (under the `run_application` and `run_benchmark` mappings) and `REFERENCE_ACCURACIES`.
4. Run the benchmark to verify it runs end-to-end.

## 🐛 Reporting Bugs & Feedback

If you find a bug or have a feature request, please open an Issue on GitHub, providing:
- A clear description of the issue.
- A minimal reproducible code snippet.
- Expected vs. actual behavior.
