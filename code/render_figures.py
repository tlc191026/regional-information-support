"""Recreate the four main and seven supplementary figures from public source data."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """Return the SHA-256 digest of a source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_inputs(work, data_root):
    """Assemble plotting tables without fitting models or changing estimates."""
    mapping = json.loads((ROOT / "code/figures/input_map.json").read_text("utf-8"))
    provenance = []
    for relative, spec in mapping.items():
        sources = [data_root / name for name in spec["sources"]]
        frames = [
            pd.read_csv(
                path,
                keep_default_na=False,
                dtype={"patch": str},
                float_precision="round_trip",
            )
            for path in sources
        ]
        frame = pd.concat(frames, ignore_index=True).drop_duplicates()
        if spec["operation"] == "ecdf":
            groups = []
            for (measure, region), values in frame.groupby(
                ["measure", "region"], sort=False
            ):
                values = values.sort_values("score", kind="stable")
                scores = np.repeat(
                    values.score.to_numpy(), values.frequency.to_numpy(int)
                )
                n = len(scores)
                assert n == int(values.n_players.iloc[0])
                groups.append(
                    pd.DataFrame(
                        {
                            "measure": measure,
                            "region": region,
                            "score": scores,
                            "cumulative_fraction": np.arange(1, n + 1) / n,
                        }
                    )
                )
            frame = pd.concat(groups, ignore_index=True)
        elif spec["operation"] == "spline":
            common = ["region", "maintenance_difference_sd"]
            fields = [c for c in frame.columns if c.endswith("_spline")]
            frame = frame[common + fields].rename(columns={c: c[:-7] for c in fields})
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        provenance.append(
            {
                "input": relative,
                "sources": spec["sources"],
                "source_sha256": [digest(p) for p in sources],
                "operation": spec["operation"],
                "rows": len(frame),
            }
        )
    target = work / "source_data/main/relationships/metric_dictionary.csv"
    shutil.copy2(ROOT / "code/figures/metric_labels.csv", target)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Choose an empty output directory.")
    out.mkdir(parents=True, exist_ok=True)
    work = out / "working"
    (work / "validation").mkdir(parents=True)
    (work / "qa").mkdir(parents=True)
    shutil.copytree(ROOT / "code/figures", work / "code/figures")
    provenance = prepare_inputs(work, args.data_root.resolve())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    for name in ["draw_main.py", "draw_supplement.py"]:
        result = subprocess.run(
            [sys.executable, "-B", str(work / "code/figures" / name)],
            cwd=work,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (out / (Path(name).stem + ".log")).write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(
                f'Figure generation failed; see {out/(Path(name).stem+".log")}'
            )
    for name in ["main", "supplementary"]:
        shutil.copytree(work / "figures" / name, out / name)
    figures = list((out / "main").glob("*.png")) + list(
        (out / "supplementary").glob("*.png")
    )
    if len(figures) != 11:
        raise RuntimeError(f"Expected 11 figures, found {len(figures)}.")
    report = {
        "figures": 11,
        "formats": ["png", "pdf", "svg", "tiff"],
        "inputs": provenance,
    }
    (out / "reproduction.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Recreated 4 main and 7 supplementary figures in {out}")


if __name__ == "__main__":
    main()
