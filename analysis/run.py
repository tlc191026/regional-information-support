"""List or run the current analysis and figure workflows."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REV = ROOT / "revision"
EXT = REV / "extensions"
SAMPLES = ("discovery", "holdout", "strict")


def steps(stage):
    if stage == "figures":
        folder = EXT / "code/figures"
        return [(folder / f"fig{i}.py", [], "main") for i in range(1, 6)] + [
            (folder / "supp_a.py", [], "supplementary"),
            (folder / "supp_b.py", [], "supplementary")]
    if stage == "regional":
        return [(REV / "theory/code/regional_tests.py", [], None)]
    if stage == "role_pairs":
        return [(REV / "theory/code/role_pair_tests.py", ["--refit"], None)]
    if stage == "regional_summaries":
        return [(EXT / "code/reexpress_results.py", [], None)]
    if stage == "cross_activity":
        return [(EXT / "code/peer_extensions.py", ["--sample", s], None) for s in SAMPLES] + [
            (EXT / "code/peer_extensions.py", ["--collect"], None),
            (EXT / "code/additional_summaries.py", [], None)]
    if stage == "objective":
        return [(REV / "code/regional_performance/revise_performance.py", [], None)]
    if stage == "time_windows":
        return [(EXT / "code/timeline_extensions.py", ["--sample", s], None) for s in SAMPLES]
    raise ValueError(stage)


STAGES = ("figures", "regional", "role_pairs", "regional_summaries",
          "cross_activity", "objective", "time_windows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--list", action="store_true", help="List workflow stages.")
    parser.add_argument("--execute", action="store_true", help="Execute the selected workflow; otherwise print its commands.")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "outputs/current_figures")
    args = parser.parse_args()
    if args.list:
        print("\n".join(STAGES))
        return
    if args.stage is None:
        parser.error("Choose --stage or --list.")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               RIS_REVISION_ROOT=str(REV))
    for script, extra, group in steps(args.stage):
        command = [sys.executable, "-B", str(script), *extra]
        print(subprocess.list2cmdline(command), flush=True)
        if args.execute:
            if group:
                output = args.output_dir.resolve() / group
                output.mkdir(parents=True, exist_ok=True)
                env["RIS_FIGURE_OUTPUT"] = str(output)
                env["MPLCONFIGDIR"] = str(args.output_dir.resolve() / "matplotlib_cache")
            subprocess.run(command, cwd=ROOT.parent, env=env, check=True)


if __name__ == "__main__":
    main()
