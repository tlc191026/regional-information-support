"""Supplementary figures S1-S6 (measurement and regional configuration)."""
from style import *
from matplotlib.colors import LinearSegmentedColormap

setup()
SD = BASE / "source_data"
DIV = LinearSegmentedColormap.from_list("div", [(0, "#184f95"), (0.25, "#6da7ec"), (0.5, "#f0efec"),
                                                (0.75, "#ef9796"), (1, "#c62f2f")])
M12 = ["wards_placed", "wards_killed", "control_wards_bought", "enemy_missing_pings", "enemy_vision_pings",
       "command_pings", "need_vision_pings", "assist_me_pings", "on_my_way_pings", "push_pings", "all_in_pings",
       "get_back_pings"]
LAB12 = ["Placed", "Cleared", "Purchased", "Enemy missing", "Enemy vision", "Command", "Need vision", "Assist me",
         "On my way", "Push", "All in", "Retreat"]
SIG9 = M12[3:]
SIGLAB = dict(zip(M12, LAB12))
fmt = lambda v, d=2: f"{v:.{d}f}".replace("-", "−")


def rd(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def reg_point(ax, x, y, reg, lo=None, hi=None, horizontal=True, ms=None):
    if lo is not None:
        if horizontal:
            ax.plot([lo, hi], [y, y], color=REG[reg], lw=1.0, zorder=2)
        else:
            ax.plot([x, x], [lo, hi], color=REG[reg], lw=1.0, zorder=2)
    ax.plot([x], [y], marker=REG_MARK[reg], color=REG[reg], ms=ms or (4.6 if reg == "KR" else 4.2), mec="white",
            mew=0.5, ls="none", zorder=3)


def reg_legend(ax, **kw):
    hs = [ax.plot([], [], marker=REG_MARK[r], color=REG[r], ls="none", ms=4.6 if r == "KR" else 4.2, label=r)[0]
          for r in REG_ORDER]
    return ax.legend(handles=hs, handletextpad=0.2, fontsize=6.8, **kw)


# --------------------------------------------------------------------------- S1
def figS1():
    mat = rd(SD / "FigS1_matrices.csv")
    mat["row"] = mat["row"].str.replace("_pm", "", regex=False)
    mat["column"] = mat["column"].str.replace("_pm", "", regex=False)
    fig = plt.figure(figsize=(FULL_W, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.42, left=0.11, right=0.985, top=0.95, bottom=0.12)
    summ = []
    for k, reg in enumerate(REG_ORDER):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        d = mat[mat.region == reg].pivot_table(index="row", columns="column", values="pearson_r").loc[M12, M12]
        A = d.values.copy()
        mask = np.triu(np.ones_like(A, dtype=bool), k=1)
        ax.imshow(np.ma.masked_where(mask, A), cmap=DIV, vmin=-1, vmax=1, aspect="equal")
        for i in range(12):
            for j in range(i):
                v = A[i, j]
                lab = f"{abs(v):.2f}"[1:]
                lab = lab if (v >= 0 or lab == ".00") else "−" + lab
                ax.text(j, i, lab, ha="center", va="center",
                        fontsize=4.6, color="white" if abs(v) > 0.55 else INK)
        ax.plot([-0.5, 2.5, 2.5], [2.5, 2.5, 11.5], color=INK, lw=0.8)
        ax.set_xticks(range(12))
        ax.set_xticklabels(LAB12, rotation=60, ha="right", fontsize=6.2)
        ax.set_yticks(range(12))
        ax.set_yticklabels(LAB12, fontsize=6.2)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        panel_title(ax, "abc"[k], reg, x=-0.38, y=1.02)
        idx = {m: i for i, m in enumerate(M12)}
        wm = [abs(A[i, j]) for i in range(3) for j in range(i)]
        ws = [abs(A[i, j]) for i in range(3, 12) for j in range(3, i)]
        bt = [abs(A[i, j]) for i in range(3, 12) for j in range(3)]
        summ.append((reg, np.mean(wm), np.mean(ws), np.mean(bt)))
    ax = fig.add_subplot(gs[1, 1])
    cats = ["Within maintenance\n(3 pairs)", "Within signalling\n(36 pairs)", "Between processes\n(27 pairs)"]
    for k, (reg, a, b, c) in enumerate(summ):
        for i, v in enumerate((a, b, c)):
            y = 2 - i + (0.2, 0, -0.2)[k]
            ax.plot([0, v], [y, y], color=REG[reg], lw=1.2, alpha=0.5, zorder=1)
            reg_point(ax, v, y, reg)
            ax.text(v + 0.012, y, f"{v:.3f}", ha="left", va="center", fontsize=6.2, color=INK2)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(cats, fontsize=6.8, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.6, 2.6)
    hgrid(ax, "x")
    ax.set_xlabel("Mean absolute Pearson correlation")
    reg_legend(ax, loc="lower right")
    panel_title(ax, "d", "Correlations are higher within than between processes", x=-0.62, y=1.02)
    save(fig, "FigS1_correlation_structure")
    return summ


# --------------------------------------------------------------------------- S2
def figS2():
    ec = pd.read_csv(SD / "player_ecdf.csv", keep_default_na=False)
    occ = rd(SD / "signal_occurrence.csv")
    occ["metric"] = occ["metric"].str.replace("_pm", "", regex=False)
    fig = plt.figure(figsize=(FULL_W, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.5, wspace=0.32, left=0.09, right=0.985,
                          top=0.95, bottom=0.08)
    titles = {"overall": "Overall information work", "maintenance": "Maintenance", "signalling": "Signalling"}
    for k, meas in enumerate(["overall", "maintenance", "signalling"]):
        ax = fig.add_subplot(gs[0, k])
        for reg in REG_ORDER:
            d = ec[(ec.measure == meas) & (ec.region == reg)].sort_values("score")
            q = np.linspace(0.002, 0.998, 400)
            xs = np.interp(q, d.cumulative_fraction.values, d.score.values)
            ax.plot(xs, q * 100, color=REG[reg], lw=1.4)
        ax.set_xlim(-2.5, 3.5)
        ax.set_ylim(0, 100)
        hgrid(ax, "y")
        ax.set_xlabel("Player mean score (SD)")
        if k == 0:
            ax.set_ylabel("Players at or below (%)")
            hs = [ax.plot([], [], color=REG[r], lw=1.6, label=r)[0] for r in REG_ORDER]
            ax.legend(handles=hs, loc="lower right", fontsize=6.8)
        panel_title(ax, "abc"[k], titles[meas], x=-0.22, y=1.04)
    order = occ[occ.region == "NA"].sort_values("prevalence").metric.tolist()
    ax = fig.add_subplot(gs[1, :2])
    for i, m in enumerate(order):
        for reg, dy in (("KR", 0.22), ("EUW", 0.0), ("NA", -0.22)):
            r = occ[(occ.metric == m) & (occ.region == reg)].iloc[0]
            reg_point(ax, r.prevalence * 100, i + dy, reg, r.prevalence_ci_low * 100, r.prevalence_ci_high * 100)
        g = occ[(occ.metric == m)].set_index("region").prevalence * 100
        if m in ("get_back_pings", "all_in_pings"):
            ax.text(g.max() + 1.5, i, "  ".join(f"{r} {g[r]:.1f}%" for r in ["KR", "EUW", "NA"]), ha="left",
                    va="center", fontsize=6.2, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SIGLAB[m] for m in order], fontsize=7, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(order) - 0.4)
    hgrid(ax, "x")
    ax.set_xlabel("Player–match records with at least one use (%)")
    reg_legend(ax, loc="lower right")
    panel_title(ax, "d", "KR uses retreat signals in more matches and all-in signals in fewer", x=-0.18, y=1.03)
    ax = fig.add_subplot(gs[1, 2], sharey=ax)
    for i, m in enumerate(order):
        for reg, dy in (("KR", 0.22), ("EUW", 0.0), ("NA", -0.22)):
            r = occ[(occ.metric == m) & (occ.region == reg)].iloc[0]
            reg_point(ax, r.conditional_mean_per_10_min, i + dy, reg, r.conditional_ci_low, r.conditional_ci_high)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 3.2)
    hgrid(ax, "x")
    ax.set_xlabel("Uses per 10 min when used")
    panel_title(ax, "e", "Frequency once used", x=-0.04, y=1.03)
    save(fig, "FigS2_distributions_and_signal_use")


# --------------------------------------------------------------------------- S3
def figS3():
    dec = rd(SD / "champion_decomposition.csv")
    pp = rd(SD / "patch_profiles.csv")
    pp["region"] = pp["region"].str.replace("1", "", regex=False)
    piv = pp.pivot_table(index="patch", columns="region", values="player_matches")
    keep = piv[(piv >= 5000).all(axis=1)].index.tolist()
    fig = plt.figure(figsize=(FULL_W, 5.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.62, wspace=1.2, left=0.08, right=0.985, top=0.93,
                          bottom=0.1)
    comps = [("KR-NA1", "KR − NA"), ("KR-EUW1", "KR − EUW"), ("EUW1-NA1", "EUW − NA")]
    parts = [("total_gap", "total_ci_low", "total_ci_high", "Total", INK, "o"),
             ("within_champion", "within_ci_low", "within_ci_high", "Within champion", "#6b6a66", "s"),
             ("composition", "composition_ci_low", "composition_ci_high", "Champion composition", "#b4b2ab", "D")]
    for k, (met, lab) in enumerate([("control_wards_bought", "Control-ward purchases"),
                                    ("wards_killed", "Wards cleared")]):
        ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
        for i, (c, cl) in enumerate(comps):
            r = dec[(dec.metric == met) & (dec.contrast == c)].iloc[0]
            for j, (est, lo, hi, plab, col, mk) in enumerate(parts):
                y = 2 - i + (0.22, 0, -0.22)[j]
                ax.plot([r[lo], r[hi]], [y, y], color=col, lw=1.0)
                ax.plot([r[est]], [y], marker=mk, color=col, mfc=col, ms=4.2, ls="none", mec="white", mew=0.5)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([c[1] for c in comps], fontsize=7, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(-0.08, 0.4)
        ax.set_ylim(-0.6, 2.6)
        hgrid(ax, "x")
        ax.set_xlabel("Regional gap (uses per player per 10 min)")
        if k == 1:
            hs = [ax.plot([], [], marker=mk, color=col, ls="none", ms=4.2, label=plab)[0]
                  for _, _, _, plab, col, mk in parts]
            ax.legend(handles=hs, loc="lower right", fontsize=6.6, handletextpad=0.2)
        panel_title(ax, "ab"[k], f"{lab}: gaps hold within champions", x=-0.22, y=1.04)
    for k, (met, lab) in enumerate([("control_wards_bought", "Control-ward purchases"),
                                    ("wards_killed", "Wards cleared"), ("all_in_pings", "All-in signals")]):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        xs = np.arange(len(keep))
        for reg in REG_ORDER:
            v = pp[pp.region == reg].set_index("patch").loc[keep, met]
            ax.plot(xs, v.values, color=REG[reg], lw=1.3, marker=REG_MARK[reg], ms=3.4, mec="white", mew=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{p:g}" if isinstance(p, float) else str(p) for p in keep], rotation=60, fontsize=6)
        hgrid(ax, "y")
        ax.set_ylabel("Per player per 10 min" if k == 0 else "")
        ax.set_xlabel("Game version")
        if k == 0:
            reg_legend(ax, loc="upper left", ncol=3)
        panel_title(ax, "cde"[k], lab, x=-0.2, y=1.04)
    save(fig, "FigS3_champions_and_versions")
    return keep


# --------------------------------------------------------------------------- S4
def figS4():
    tm = rd(SD / "tier_adjusted_means.csv")
    fig, axs = plt.subplots(3, 3, figsize=(FULL_W, 6.4), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.08, hspace=0.42, wspace=0.28)
    x = np.arange(len(TIER_ORDER))
    for k, m in enumerate(SIG9):
        ax = axs[k // 3, k % 3]
        for reg in REG_ORDER:
            d = tm[(tm.region == reg) & (tm.measure == m + "_pm")].set_index("level").loc[TIER_ORDER]
            ax.fill_between(x, d.ci_low, d.ci_high, color=REG[reg], alpha=0.16, lw=0)
            ax.plot(x, d.estimate, color=REG[reg], lw=1.3, marker=REG_MARK[reg], ms=3.2, mec="white", mew=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(["Ir", "Br", "Si", "Go", "Pl", "Em", "Di"], fontsize=6.6)
        ax.set_ylim(bottom=0)
        hgrid(ax, "y")
        if k % 3 == 0:
            ax.set_ylabel("Per player per 10 min", fontsize=6.8)
        panel_title(ax, "abcdefghi"[k], SIGLAB[m], x=-0.2, y=1.04)
    reg_legend(axs[0, 0], loc="lower right", ncol=1)
    save(fig, "FigS4_signals_by_tier")


# --------------------------------------------------------------------------- S5 (tier gaps; moved from v10 Fig. 3)
def figS5():
    tc = {"maintenance": read("source_data/main/Fig3c.csv"), "signalling": read("source_data/main/Fig3d.csv")}
    tch = pd.read_csv(TAB / "tier_process_changes.csv", keep_default_na=False)
    COMPC = {"KR − NA": (REG["KR"], "o", "-"), "KR − EUW": ("#6b6a66", "s", (0, (4, 2))),
             "EUW − NA": (REG["EUW"], "s", "-")}
    fig = plt.figure(figsize=(FULL_W, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.55, wspace=0.2, left=0.08, right=0.97, top=0.93,
                          bottom=0.08)
    x = np.arange(len(TIER_ORDER))
    axs = []
    titles = {"maintenance": ("a", "Maintenance gaps involving KR are larger at higher tiers"),
              "signalling": ("b", "KR's signalling lead shrinks at higher tiers")}
    for j, proc in enumerate(["maintenance", "signalling"]):
        ax = fig.add_subplot(gs[0, j], sharey=axs[0] if axs else None)
        axs.append(ax)
        d = tc[proc]
        d = d[d["sample"] == "discovery"]
        for comp, (col, mk, ls) in COMPC.items():
            dd = d[d.comparison == comp].set_index("tier").loc[TIER_ORDER]
            ax.fill_between(x, dd.ci_low, dd.ci_high, color=col, alpha=0.14, lw=0)
            ax.plot(x, dd.estimate, color=col, lw=1.4, ls=ls)
            ax.plot(x, dd.estimate, marker=mk, color=col, ms=3.6, mec="white", mew=0.4, ls="none")
            ax.text(len(TIER_ORDER) - 1 + 0.2, dd.estimate.iloc[-1] + {"KR − NA": 0, "KR − EUW": -0.03,
                                                                      "EUW − NA": 0.03}[comp] *
                    (1 if proc == "signalling" else 0), comp, ha="left", va="center", fontsize=6.4, color=INK2)
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([TIER_LABEL[t] for t in TIER_ORDER])
        ax.set_xlim(-0.4, len(TIER_ORDER) + 0.6)
        ax.set_ylim(-0.12, 0.66)
        hgrid(ax, "y")
        ax.set_xlabel("Roster tier (low → high)")
        if j == 0:
            ax.set_ylabel("Regional gap (SD)")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        L, T = titles[proc]
        panel_title(ax, L, T, x=-0.12 if j == 0 else -0.03, y=1.05)
    ax = fig.add_subplot(gs[1, :])
    rows = [(p, c) for p in ["maintenance", "signalling"] for c in ["KR-NA1", "KR-EUW1", "EUW1-NA1"]]
    clab = {"KR-NA1": "KR − NA", "KR-EUW1": "KR − EUW", "EUW1-NA1": "EUW − NA"}
    plab = {"maintenance": "Maintenance", "signalling": "Signalling"}
    for i, (p, c) in enumerate(rows):
        y = len(rows) - 1 - i + (0 if p == "signalling" else 0.6)
        for k, s in enumerate(SAMPLES):
            r = tch[(tch["sample"] == s) & (tch.contrast == c) & (tch.process == p)].iloc[0]
            st = SAMPLE_STYLE[s]
            yy = y + (0.22, 0, -0.22)[k]
            ax.plot([r.ci_low, r.ci_high], [yy, yy], color=MUTED if s == "holdout" else INK, lw=1.0)
            ax.plot([r.estimate], [yy], marker=st["marker"], mfc=st["mfc"], mec=st["mec"], mew=0.9,
                    ms={"o": 4.4, "s": 4.0, "D": 3.8}[st["marker"]], ls="none")
    ax.axvline(0, color=INK2, lw=0.7)
    ys = [len(rows) - 1 - i + (0 if p == "signalling" else 0.6) for i, (p, c) in enumerate(rows)]
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{plab[p]}  {clab[c]}" for p, c in rows], fontsize=7, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.62, 0.42)
    hgrid(ax, "x")
    ax.set_xlabel("Change in the regional gap from Bronze to Diamond (SD)")
    hs = [ax.plot([], [], marker=SAMPLE_STYLE[s]["marker"], mfc=SAMPLE_STYLE[s]["mfc"], mec=SAMPLE_STYLE[s]["mec"],
                  mew=0.9, ls="none", ms=4.2, label=SAMPLE_LABEL[s])[0] for s in SAMPLES]
    ax.legend(handles=hs, loc="lower left", fontsize=6.6, handletextpad=0.2)
    panel_title(ax, "c", "Endpoint changes in the three samples", x=-0.12, y=1.04)
    save(fig, "FigS5_regional_gaps_by_tier")


# --------------------------------------------------------------------------- S6 (distances)
def figS6():
    dist = rd(SD / "tier_distances.csv")
    fig, axs = plt.subplots(2, 3, figsize=(FULL_W, 4.6), sharey="row")
    fig.subplots_adjust(left=0.08, right=0.93, top=0.86, bottom=0.11, hspace=0.55, wspace=0.12)
    tiers = TIER_ORDER[1:]
    PAIR_LS = {"KR-NA1": "-", "KR-EUW1": (0, (4, 2)), "EUW1-NA1": (0, (1, 1.5))}
    x = np.arange(len(tiers))
    for r, kind in enumerate(["level", "content"]):
        for c, s in enumerate(SAMPLES):
            ax = axs[r, c]
            d = dist[(dist["sample"] == s) & (dist.kind == kind)]
            ends = []
            for comp in ["KR-NA1", "KR-EUW1", "EUW1-NA1"]:
                dd = d[d.contrast == comp].set_index("tier").reindex(tiers)
                ax.plot(x, dd.estimate, color="#a3a19a", lw=1.0, ls=PAIR_LS[comp])
                lab = {"KR-NA1": "KR–NA", "KR-EUW1": "KR–EUW", "EUW1-NA1": "EUW–NA"}[comp]
                ends.append([dd.estimate.iloc[-1], lab])
            if c == 2:
                # direct labels at the line ends, spaced so that they never overlap
                lo, hi = (0.02, 0.42) if kind == "level" else (0.015, 0.125)
                gap = 0.075 * (hi - lo)
                ends.sort(key=lambda e: e[0])
                for i in range(1, len(ends)):
                    ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
                for yv, lab in ends:
                    ax.text(len(tiers) - 1 + 0.15, yv, lab, ha="left", va="center", fontsize=6.0, color=MUTED)
            dd = d[d.contrast == "mean_three_pairs"].set_index("tier").reindex(tiers)
            ax.fill_between(x, dd.ci_low, dd.ci_high, color=INK, alpha=0.12, lw=0)
            ax.plot(x, dd.estimate, color=INK, lw=1.5, marker="o", ms=3, mec="white", mew=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(["Br", "Si", "Go", "Pl", "Em", "Di"], fontsize=6.6)
            hgrid(ax, "y")
            if c == 0:
                ax.set_ylabel("Level distance (SD)" if kind == "level" else "Content distance (JS)", fontsize=7)
            t = SAMPLE_LABEL[s]
            panel_title(ax, "abcdef"[r * 3 + c], t, x=-0.12 if c else -0.3, y=1.04)
    hs = [axs[0, 0].plot([], [], color="#a3a19a", lw=1.0, ls=PAIR_LS[k], label=v)[0]
          for k, v in (("KR-NA1", "KR–NA"), ("KR-EUW1", "KR–EUW"), ("EUW1-NA1", "EUW–NA"))]
    hs.append(axs[0, 0].plot([], [], color=INK, lw=1.5, marker="o", ms=3, mec="white", mew=0.4,
                             label="Mean of three pairs (95% CI)")[0])
    fig.legend(handles=hs, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.6,
               handlelength=2.6, columnspacing=1.6)
    save(fig, "FigS6_regional_distances_by_tier")


if __name__ == "__main__":
    print(figS1())
    figS2()
    print(figS3())
    figS4()
    figS5()
    figS6()
