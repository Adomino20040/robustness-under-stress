"""Phase 3/4 -- Top-level experiment driver.

Usage:
    python run_audit.py --config configs/experiment_config.yaml

For every (domain, model) cell in the 2x2 matrix:
    1. load frozen pipeline from outputs/models/
    2. load held-out test split from data/processed/
    3. results = BlackBoxAuditor(model, ...).run(X_test, y_test)
    4. save tidy CSV to outputs/results/{domain}__{model}.csv
Then aggregate all four result frames, compute degradation slopes,
write outputs/results/summary_table.csv, and render all figures.
"""
from __future__ import annotations

import argparse

# TODO(Phase 3): implement main() per docstring above.


def main(config_path: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    main(parser.parse_args().config)
