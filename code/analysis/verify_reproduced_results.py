"""Compare regenerated numerical tables with their frozen original results."""

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    reports = []
    for stage in a.runs_root.iterdir():
        if not (stage / "run_status.json").exists():
            continue
        status = json.loads((stage / "run_status.json").read_text(encoding="utf8"))
        for script in status["scripts"]:
            folder = Path(script).parent.parent
            for kind in ["source_data", "tables"]:
                for new in (stage / folder / kind).glob("*.csv"):
                    rel = new.relative_to(stage)
                    old = a.input_root / rel
                    if not old.exists():
                        continue
                    left = pd.read_csv(old, dtype=str, keep_default_na=False)
                    right = pd.read_csv(new, dtype=str, keep_default_na=False)
                    row = {
                        "stage": status["stage"],
                        "file": rel.as_posix(),
                        "rows": len(right),
                        "columns": len(right.columns),
                        "max_absolute_numeric_difference": 0.0,
                    }
                    errors = []
                    if left.shape != right.shape or list(left.columns) != list(
                        right.columns
                    ):
                        errors.append("shape or column mismatch")
                    else:
                        for col in left.columns:
                            lnum = pd.to_numeric(left[col], errors="coerce")
                            rnum = pd.to_numeric(right[col], errors="coerce")
                            numeric = lnum.notna() & rnum.notna()
                            if numeric.any():
                                lv = lnum[numeric].to_numpy(float)
                                rv = rnum[numeric].to_numpy(float)
                                if not np.allclose(
                                    lv, rv, rtol=1e-8, atol=1e-10, equal_nan=True
                                ):
                                    errors.append(col + ": numerical mismatch")
                                finite = np.isfinite(lv) & np.isfinite(rv)
                                if finite.any():
                                    row["max_absolute_numeric_difference"] = max(
                                        row["max_absolute_numeric_difference"],
                                        float(np.max(np.abs(lv[finite] - rv[finite]))),
                                    )
                            if not left.loc[~numeric, col].equals(
                                right.loc[~numeric, col]
                            ):
                                errors.append(col + ": text mismatch")
                    row["passed"] = not errors
                    row["errors"] = errors
                    reports.append(row)
    result = {
        "compared_files": len(reports),
        "passed_files": sum(x["passed"] for x in reports),
        "failed_files": [x for x in reports if not x["passed"]],
        "comparisons": reports,
    }
    a.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "comparisons"}, ensure_ascii=False
        )
    )
    if result["failed_files"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
