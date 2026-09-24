"""Run a selected analysis in an isolated copy of its fixed input files."""

from pathlib import Path
import argparse, shutil, subprocess, sys, json, hashlib, os

PACKAGE = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent / "project"
PREFIX = "manuscript_20260915/"
REPRO = PREFIX + "03_reproducibility/"
TIMELINE = PREFIX + "13_timeline_dynamics_20260919/"
SCORES = [
    "exp_20260717/data/context_unresidualized_participant_scores.parquet",
    "exp_20260717/data/participant_metrics_transformed_z.parquet",
    "exp_20260717/data/bootstrap_weights_primary_region_tier_1000.npy",
]
RAW = "outputs/study0/data/study0_canonical_participant_match_sample.pkl"
STAGES = {
    "raw_prepare": (
        [REPRO + "code/prepare_inputs.py"],
        [
            RAW,
            "exp_20260717/data/canonical_unique_match_metadata.parquet",
            SCORES[0],
            "outputs/study1_information_support_validation/data/study1_match_difference_table_scored.parquet",
        ],
    ),
    "relationships": (
        [PREFIX + "09_variable_relationships_20260917/code/analyse_relationships.py"],
        ["exp_20260717/data/player_metric_and_process_scores_minimal.parquet"],
    ),
    "regional": (
        [
            PREFIX
            + "10_overall_and_process_differences_20260918/code/analyse_section.py"
        ],
        SCORES,
    ),
    "signals": (
        [PREFIX + "12_integrated_regional_practices_20260918/code/analyse_signals.py"],
        SCORES
        + [
            RAW,
            PREFIX
            + "10_overall_and_process_differences_20260918/data/adjusted_player_scores.parquet",
        ],
    ),
    "context": (
        [PREFIX + "14_context_results_20260919/code/analyse_context.py"],
        [SCORES[0], SCORES[2], RAW],
    ),
    "supplement_additions": (
        [PREFIX + "16_supplementary_materials_20260920/code/analyse_additions.py"],
        SCORES + [REPRO + "data/analysis_scored_0_10.parquet"],
    ),
    "minute_profiles": (
        [TIMELINE + "code/analyse_minutes.py"],
        [
            TIMELINE + "data/minute_shards/*.parquet",
            REPRO + "data/timeline_features_0_10.parquet",
            REPRO + "data/all_participant_disjoint_matches.parquet",
        ],
    ),
    "objective_curves": (
        [TIMELINE + "code/analyse_function.py"],
        [REPRO + "data/analysis_scored_0_10.parquet"],
    ),
    "timeline_models": (
        [REPRO + "code/run_models.py"],
        [
            REPRO + "data/timeline_features_0_10.parquet",
            REPRO + "data/timeline_features_0_15.parquet",
        ],
    ),
    "timeline_sensitivity": (
        [REPRO + "code/run_sensitivity.py"],
        [
            REPRO + "data/*.parquet",
            "exp_20260717/data/timeline_match_differences_nonduplicated_scores.parquet",
        ],
    ),
    "prediction": (
        [REPRO + "code/run_prediction.py"],
        [
            REPRO + "data/analysis_scored_0_10.parquet",
            REPRO + "data/focal_player_match_split.parquet",
            REPRO + "data/canonical_focal_links.parquet",
        ],
    ),
    "tests": (
        [
            REPRO + "code/test_parser.py",
            REPRO + "code/test_alignment.py",
            REPRO + "code/test_statistics.py",
        ],
        [],
    ),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument(
        "--input-root",
        type=Path,
        help="Original project root containing fixed analytical inputs",
    )
    p.add_argument("--output-root", type=Path, help="New, empty analysis workspace")
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    scripts, patterns = STAGES[a.stage]
    source = a.input_root.resolve() if a.input_root else None
    if patterns and source is None:
        p.error("--input-root is required for this stage")
    files = []
    missing = []
    for pattern in patterns:
        matches = list(source.glob(pattern))
        if not matches:
            missing.append(pattern)
        files.extend(x for x in matches if x.is_file())
    if missing:
        raise FileNotFoundError("Missing fixed inputs: " + ", ".join(missing))
    report = {
        "stage": a.stage,
        "files": len(files),
        "bytes": sum(x.stat().st_size for x in files),
        "scripts": scripts,
    }
    if a.check_only:
        print(json.dumps(report, indent=2))
        return
    target = (a.output_root or PACKAGE / "outputs/analysis_runs" / a.stage).resolve()
    if source is not None and (target == source or target in source.parents):
        raise ValueError("The output cannot replace the source directory")
    if target.exists() and any(target.iterdir()):
        raise ValueError("Use a new empty output directory")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROJECT, target, dirs_exist_ok=True)

    if source is not None:
        support = [
            "exp_final_construct_v3/tables/*.csv",
            "exp_final_construct_v3/tables/data_audit_canonical_invariants.json",
            "exp_final_construct_v3/timeline_machine_readable_summary.json",
            "exp_20260717/tables/*.csv",
            PREFIX + "02_timeline/tables/*.csv",
            REPRO + "data/*scale*.json",
            TIMELINE + "qa/minute_parse_summary.json",
            PREFIX + "10_overall_and_process_differences_20260918/source_data/*.csv",
            PREFIX + "16_supplementary_materials_20260920/source_data/regional/*.csv",
        ]
        for pattern in support:
            for src in source.glob(pattern):
                if src.is_file():
                    rel = src.relative_to(source)
                    dst = target / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

    provenance = []
    for src in dict.fromkeys(files):
        rel = src.relative_to(source)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        provenance.append(
            {"path": rel.as_posix(), "bytes": src.stat().st_size, "sha256": digest(dst)}
        )
    for code in target.rglob("code"):
        for name in ["data", "tables", "qa", "logs", "source_data", "figures"]:
            (code.parent / name).mkdir(parents=True, exist_ok=True)
    (target / "input_manifest.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    env["MPLCONFIGDIR"] = str(target / "matplotlib_cache")
    for script in scripts:
        logfile = target / (Path(script).stem + ".log")
        with logfile.open("w", encoding="utf-8") as f:
            result = subprocess.run(
                [sys.executable, "-u", str(target / script)],
                cwd=target,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError("Analysis failed; inspect " + str(logfile))
    report["completed"] = True
    report["output_root"] = str(target)
    (target / "run_status.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
