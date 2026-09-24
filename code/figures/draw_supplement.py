"""Render the seven supplementary figures using the shared figure styles."""

from pathlib import Path
import os, json, shutil

OUT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(OUT / "validation/mplconfig")
import draw_main as renderer

ns = vars(renderer)
plt = ns["plt"]
np = ns["np"]
pd = ns["pd"]
fig = ns["fig"]
ax = ns["ax"]
panel = ns["panel"]
txt = ns["txt"]
legend = ns["legend"]
error = ns["error"]
save = ns["save"]
style = ns["style"]
REG = ns["REG"]
COL = ns["COL"]
MARK = ns["MARK"]
LS = ns["LS"]
TIERS = ns["TIERS"]
ABBR = ns["ABBR"]
style()


def read(name, folder="tables"):
    return pd.read_csv(
        OUT / folder / "supplementary" / name,
        keep_default_na=False,
        dtype={"patch": str},
    )


stems = [
    "Supplementary_FigS1_regional_relationships",
    "Supplementary_FigS2_distributions",
    "Supplementary_FigS3_all_signal_tier_profiles",
    "Supplementary_FigS4_minute_regional_contrasts",
    "Supplementary_FigS5_early_tier_profiles",
    "Supplementary_FigS6_functional_sensitivity",
    "Supplementary_FigS7_coverage_and_versions",
]

# Render shared panels from the same numerical inputs and style functions.
ns["FIGURES"] = OUT / "figures/supplementary"
ns["FIGURES"].mkdir(parents=True, exist_ok=True)


def shared_save(f, stem):
    if stem in [stems[0], stems[2], stems[3]]:
        save(f, stem)
    else:
        plt.close(f)


ns["save"] = shared_save
ns["relationships"]()
ns["context"]()
ns["timeline"]()
ns["save"] = save


def selected_save(f, stem):
    if stem != stems[1]:
        plt.close(f)
        return
    # Highlight the empirical upper-tail crossing.
    a = f.axes[4]
    a.axvspan(96, 99, color="#D9D9E0", alpha=0.5, zorder=0)
    a.annotate(
        "96th–99th percentiles",
        xy=(97.5, -0.085),
        xytext=(46, -0.19),
        fontsize=5.4,
        color="#555560",
        arrowprops={"arrowstyle": "-", "color": "#858590", "lw": 0.5},
    )
    save(f, stem)


ns["save"] = selected_save
ns["regional"]()
ns["save"] = save

# Early-tier means use one patch reference; the existing conditional slopes use
# the frozen global predictor SD and their original full outcome samples.
m = read("early_tier_common_patch_means.csv")
sens = read("timeline_extended/sensitivity_models.csv")
f = fig(95)
legend(f)
panel(f, 2, 10, "a")
panel(f, 98, 10, "b")
a = ax(f, 17, 23, 62, 49)
txt(f, 48, 17, "Early maintenance", ha="center", fontsize=6.5)
for r in REG:
    v = m[m.region == r].set_index("tier").loc[TIERS]
    x = np.arange(7)
    a.fill_between(x, v.ci_low, v.ci_high, color=COL[r], alpha=0.13, lw=0)
    a.plot(x, v.estimate, color=COL[r], ls=LS[r], marker=MARK[r], ms=3, lw=1)
a.set_xticks(range(7), ABBR)
a.set_xlabel("Sampling tier", labelpad=4)
a.set_ylabel("Maintenance score (common units)", labelpad=4)
a.set_ylim(-0.94, 1.06)
a.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
a = ax(f, 113, 23, 44, 49)
txt(f, 135, 17, "Subsequent objective", ha="center", fontsize=6.5)
a.axvline(1.1, color="#B5B5BF", lw=0.65, ls=(0, (3, 3)))
for i, t in enumerate(TIERS):
    r = sens[sens.specification == "tier_" + t].iloc[0]
    a.errorbar(
        r.odds_ratio,
        6 - i,
        xerr=[[r.odds_ratio - r.or_low], [r.or_high - r.odds_ratio]],
        fmt="o",
        color="#454550",
        ms=3,
        capsize=2,
        lw=0.9,
    )
    a.text(
        65 / 44,
        6 - i,
        f"{int(r.n_matches):,}",
        transform=a.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=5.3,
    )
txt(f, 178, 17, "Matches", ha="right", fontsize=5.6)
a.set_yticks(range(6, -1, -1), ABBR)
a.set_ylim(-0.6, 6.6)
a.set_xlim(1.082, 1.118)
a.set_xticks([1.09, 1.10, 1.11])
a.set_xlabel("Odds ratio per common SD", labelpad=4)
a.tick_params(axis="y", length=0)
a.spines["left"].set_visible(False)
save(f, stems[4])

# Functions, components and clearly grouped choices, preserving their own scales.
diff = read("S5a_spline_minus_linear.csv", "source_data")
f = fig(157)
legend(f)
panel(f, 2, 10, "a")
a = ax(f, 23, 22, 149, 31)
for r in REG:
    v = diff[diff.region == r]
    a.plot(
        v.maintenance_difference_sd,
        v.difference_percentage_points,
        color=COL[r],
        ls=LS[r],
        lw=1.1,
    )
a.axhline(0, color="#AAAAB2", lw=0.6)
a.set_xlim(-2, 2)
a.set_xticks([-2, -1, 0, 1, 2])
a.set_yticks([-0.3, 0, 0.3])
a.set_xlabel("Early maintenance difference (common SD)", labelpad=4)
a.set_ylabel("Spline − linear probability\n(percentage points)", labelpad=4)
panel(f, 2, 69, "b")
a = ax(f, 25, 85, 46, 47)
v = sens[sens.specification == "three_components_simultaneously"]
for i, comp in enumerate(
    ["valid_wards_placed", "valid_ward_kills", "control_ward_purchases"]
):
    r = v[v.predictor == comp + "_diff_z"].iloc[0]
    a.errorbar(
        r.odds_ratio,
        2 - i,
        xerr=[[r.odds_ratio - r.or_low], [r.or_high - r.odds_ratio]],
        fmt="o",
        color="#454550",
        ms=3.2,
        capsize=2,
        lw=1,
    )
a.axvline(1, color="#B0B0B8", lw=0.6, ls=(0, (3, 3)))
a.set_yticks([2, 1, 0], ["Placement", "Clearing", "Purchases"])
a.set_ylim(-0.5, 2.5)
a.set_xlim(0.99, 1.092)
a.set_xticks([1, 1.04, 1.08])
a.tick_params(axis="y", length=0)
a.spines["left"].set_visible(False)
a.set_xlabel("Odds ratio per component SD", labelpad=4)
panel(f, 84, 69, "c")
a = ax(f, 133, 81, 31, 63)
specs = [
    ("strict_nominal_10min_earlier_state_frame", "Strict 10-min cutoff", 9),
    ("window_0_15", "15-min window", 8),
    ("purchase_net_of_undo", "Net purchases", 6.7),
    ("first_legally_assigned_objective", "First assigned objective", 5.7),
    ("omit_sampling_time_tier", "Without sampling tier", 4.7),
    ("add_final_duration", "Final duration adjusted", 3.7),
    ("region_tier_balanced_matches", "Balanced strata", 2.4),
    ("coverage_inverse_probability_weighted", "Coverage weighting", 1.4),
    ("all_ten_participants_disjoint", "No repeated participants", 0.4),
]
for spec, label, y in specs:
    r = sens[sens.specification == spec].iloc[0]
    a.errorbar(
        r.odds_ratio,
        y,
        xerr=[[r.odds_ratio - r.or_low], [r.or_high - r.odds_ratio]],
        fmt="o",
        color="#454550",
        ms=3,
        capsize=2,
        lw=0.9,
    )
    a.text(
        46 / 31,
        y,
        f"{int(r.n_matches)/1e6:.3f}",
        transform=a.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=5.1,
    )
txt(f, 179, 74, "N (millions)", ha="right", fontsize=5.3)
a.set_yticks([s[2] for s in specs], [s[1] for s in specs])
a.tick_params(axis="y", length=0, labelsize=5.4)
a.spines["left"].set_visible(False)
a.set_ylim(-0.65, 9.65)
a.set_xlim(1.075, 1.12)
a.set_xticks([1.08, 1.10, 1.12])
a.set_xlabel("Odds ratio per score SD", labelpad=4, fontsize=6)
save(f, stems[5])

flow = read("timeline_extended/sample_flow_region_tier_0_10.csv")
flow.region = flow.region.replace({"NA1": "NA", "EUW1": "EUW"})
coverage = (
    100
    * flow.pivot(index="region", columns="tier", values="eligible").loc[REG, TIERS]
    / flow.pivot(index="region", columns="tier", values="canonical").loc[REG, TIERS]
)
v = sens[sens.specification.str.startswith("patch_")].copy()
v["patch"] = v.specification.str.replace("patch_", "", regex=False)
v["order"] = v.patch.map(lambda p: tuple(map(int, p.split("."))))
v = v.sort_values("order").reset_index(drop=True)
save_data = v.drop(columns="order")
save_data.to_csv(
    OUT / "source_data/supplementary/S7bc_patch_associations.csv",
    index=False,
    encoding="utf-8-sig",
)
flow.assign(coverage_percent=100 * flow.eligible / flow.canonical).to_csv(
    OUT / "source_data/supplementary/S7a_coverage.csv",
    index=False,
    encoding="utf-8-sig",
)
f = fig(163)
panel(f, 2, 5, "a")
a = ax(f, 22, 15, 134, 29)
im = a.imshow(coverage.to_numpy(), cmap="Greys", vmin=97, vmax=100, aspect="auto")
for i in range(3):
    for j in range(7):
        val = coverage.iloc[i, j]
        a.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white" if val >= 98.8 else "#292929",
        )
a.set_xticks(range(7), ABBR)
a.set_yticks(range(3), REG)
a.tick_params(length=0)
a.set_xlabel("Sampling tier", labelpad=4)
cax = f.add_axes([160 / 180, 1 - 44 / 163, 3 / 180, 29 / 163])
cb = f.colorbar(im, cax=cax)
cb.set_label("Valid timelines (%)", fontsize=6)
cb.ax.tick_params(labelsize=5.5, length=2)
cb.set_ticks([97, 98, 99, 100])
panel(f, 2, 56, "b")
a = ax(f, 22, 68, 150, 39)
x = np.arange(len(v))
a.axhline(1, color="#B0B0B8", lw=0.6, ls=(0, (3, 3)))
a.errorbar(
    x,
    v.odds_ratio,
    yerr=[v.odds_ratio - v.or_low, v.or_high - v.odds_ratio],
    fmt="o",
    color="#454550",
    ms=2.1,
    lw=0.7,
    capsize=1,
)
a.set_xlim(-0.8, len(v) - 0.2)
a.set_xticks([])
a.set_ylabel("Odds ratio per common SD", labelpad=4)
a.set_ylim(min(0.85, v.or_low.min() - 0.015), max(1.2, v.or_high.max() + 0.015))
a.yaxis.set_major_locator(ns["MaxNLocator"](4))
panel(f, 2, 115, "c")
a = ax(f, 22, 120, 150, 23)
a.bar(x, v.n_matches, color="#96969F", width=0.72)
a.set_yscale("log")
a.set_ylim(800, 6e5)
a.set_yticks([1e3, 1e4, 1e5], ["1,000", "10,000", "100,000"])
a.minorticks_off()
a.set_xlim(-0.8, len(v) - 0.2)
a.set_xticks(x, v.patch, rotation=90, fontsize=5.2)
a.set_ylabel("Matches\n(log scale)", labelpad=4)
a.set_xlabel("Game version", labelpad=4)
save(f, stems[6])
(OUT / "validation/layout_audit.json").write_text(
    json.dumps(ns["AUDIT"], ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT / "validation/supplementary_figure_stems.json").write_text(
    json.dumps(stems, indent=2), encoding="utf-8"
)
