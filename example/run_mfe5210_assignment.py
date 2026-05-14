"""
One-click entry for the MFE5210 alpha-factor assignment workflow.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor.factor_analysis.run_alpha_suite import run_alpha_suite


def main() -> None:
    result = run_alpha_suite(
        start_date="20250101",
        end_date="20251219",
        rebalance_freq="monthly",
        n_groups=5,
        min_universe=80,
        max_corr=0.5,
    )

    print("=" * 80)
    print("MFE5210 assignment workflow completed")
    print("=" * 80)
    print(f"Results folder: {result['output_dir']}")
    print(f"README file:    {result['readme_path']}")


if __name__ == "__main__":
    main()
