"""Supplementary figures S7-S11 (team relations and early maintenance)."""
from style import *
from matplotlib.colors import LinearSegmentedColormap

setup()
SUPP = BASE / "source_data/supplementary"
PERF = ROOT / "tables/regional_performance"
DIV = LinearSegmentedColormap.from_list("div", [(0, "#184f95"), (0.25, "#6da7ec"), (0.5, "#f0efec"),
                                                (0.75, "#ef9796"), (1, "#c62f2f")])
MS = {"o": 4.4, "s": 4.0, "D": 3.8}
f3 = lambda v: f"{v:.3f}".replace("-", "−")


def rd(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def spt(ax, x, y, s, lo, hi, horizontal=True):
    st = SAMPLE_STYLE[s]
    col = MUTED if s == "holdout" else INK
    if horizontal:
        ax.plot([lo, hi], [y, y], color=col, lw=1.0, zorder=2)
    else:
        ax.plot([x, x], [lo, hi], color=col, lw=1.0, zorder=2)
    ax.plot([x], [y], marker=st["marker"], mfc=st["mfc"], mec=st["mec"], mew=0.9, ms=MS[st["marker"]], ls="none",
            zorder=3)


def sample_legend(ax, **kw):
    hs = [ax.plot([], [], marker=SAMPLE_STYLE[s]["marker"], mfc=SAMPLE_STYLE[s]["mfc"], mec=SAMPLE_STYLE[s]["mec"],
                  mew=0.9, ls="none", ms=MS[SAMPLE_STYLE[s]["marker"]], label=SAMPLE_LABEL[s])[0] for s in SAMPLES]
    return ax.legend(handles=hs, handletextpad=0.2, fontsize=6.6, **kw)


# --------------------------------------------------------------------------- S7 role-pair matrices
def figS7():
    rs = rd(TAB / "role_structure.csv")
    rs = rs[(rs.unit == "common_sd") & (rs.inference == "two_way") & rs.contrast.str.contains("|", regex=False)]
    outs = [("wards_placed", "Placement"), ("control_wards", "Control-ward purchases"), ("wards_killed", "Clearing")]
    fig, axs = plt.subplots(3, 3, figsize=(FULL_W, 6.9))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.08, hspace=0.28, wspace=0.12)
    short = {"TOP": "Top", "JUNGLE": "Jgl", "MIDDLE": "Mid", "BOTTOM": "Bot", "UTILITY": "Sup"}
    lim = 0.15
    for i, s in enumerate(SAMPLES):
        for j, (o, olab) in enumerate(outs):
            ax = axs[i, j]
            d = rs[(rs["sample"] == s) & (rs.outcome == o)]
            M = np.full((5, 5), np.nan)
            for _, r in d.iterrows():
                a, b = r.contrast.split("|")
                M[ROLE_ORDER.index(a), ROLE_ORDER.index(b)] = r.estimate
            ax.imshow(np.ma.masked_invalid(M), cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
            for p in range(5):
                for q in range(5):
                    if p == q:
                        ax.add_patch(plt.Rectangle((q - 0.5, p - 0.5), 1, 1, fc="white", ec="none"))
                        continue
                    v = M[p, q]
                    ax.text(q, p, f3(v), ha="center", va="center", fontsize=5.6,
                            color="white" if abs(v) > 0.075 else INK)
            for k in range(6):
                ax.axhline(k - 0.5, color="white", lw=1.0)
                ax.axvline(k - 0.5, color="white", lw=1.0)
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_xticklabels([short[r] for r in ROLE_ORDER] if i == 2 else [], fontsize=6.8)
            ax.set_yticklabels([short[r] for r in ROLE_ORDER] if j == 0 else [], fontsize=6.8)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                pass  # Outcome labels are specified in the caption.
            ax.text(-0.06, 1.02, "abcdefghi"[i * 3 + j], transform=ax.transAxes, fontsize=9, fontweight="bold",
                    color=INK, ha="right", va="bottom")
            if j == 0:
                ax.set_ylabel(SAMPLE_LABEL[s], fontsize=7.5, color=INK, labelpad=6)
    fig.text(0.55, 0.015, "Teammate whose behaviour in the match is observed", ha="center", fontsize=7, color=INK2)
    fig.text(0.012, 0.5, "Teammate with the higher usual behaviour", rotation=90, va="center", fontsize=7,
             color=INK2)
    save(fig, "FigS7_role_pair_matrices")


# --------------------------------------------------------------------------- S8 robustness of role structure
def figS8():
    a = rd(SUPP / "FigS7a.csv")
    b = rd(SUPP / "FigS7b.csv")
    c = rd(SUPP / "FigS7c.csv")
    d = rd(SUPP / "FigS7d.csv")
    specs = [("base", "Base"), ("recipient_champion", "Recipient champion × role"),
             ("focal_champion", "Focal champion"), ("both_champions", "Both champion adjustments"),
             ("complete_units", "Complete five-member units")]
    fig = plt.figure(figsize=(FULL_W, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.75, left=0.2, right=0.985, top=0.93, bottom=0.08)
    for k, (df, title, xl) in enumerate([(a, "Support as recipient stays negative", (-0.1, 0.0)),
                                         (b, "Non-support pairs stay small and positive", (0.0, 0.025))]):
        ax = fig.add_subplot(gs[0, k])
        for i, (sp, slab) in enumerate(specs):
            for m, s in enumerate(SAMPLES):
                r = df[(df["sample"] == s) & (df.spec == sp)].iloc[0]
                spt(ax, r.estimate, len(specs) - 1 - i + (0.22, 0, -0.22)[m], s, r.ci_low, r.ci_high)
        ax.set_yticks(range(len(specs)))
        ax.set_yticklabels([s[1] for s in specs][::-1], fontsize=6.8, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(*xl)
        hgrid(ax, "x")
        ax.set_xlabel("Association (common SD)")
        if k == 0:
            sample_legend(ax, loc="lower right")
        panel_title(ax, "ab"[k], title, x=-0.62 if k == 0 else -0.62, y=1.04)
    rows = [("JUNGLE|UTILITY", "Jungle → support"), ("UTILITY|JUNGLE", "Support → jungle"),
            ("non_support", "Non-support pairs (mean)")]
    for k, (df, title, xl, xlab) in enumerate([(c, "Natural units", (-0.42, 0.1),
                                                "Teammate placements per 10 min"),
                                               (d, "Role-specific SD", (-0.13, 0.06), "Association (role SD)")]):
        ax = fig.add_subplot(gs[1, k])
        for i, (con, lab) in enumerate(rows):
            for m, s in enumerate(SAMPLES):
                r = df[(df["sample"] == s) & (df.contrast == con)].iloc[0]
                spt(ax, r.estimate, len(rows) - 1 - i + (0.22, 0, -0.22)[m], s, r.ci_low, r.ci_high)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[1] for r in rows][::-1], fontsize=6.8, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(*xl)
        hgrid(ax, "x")
        ax.set_xlabel(xlab)
        panel_title(ax, "cd"[k], title, x=-0.62, y=1.04)
    save(fig, "FigS8_role_structure_robustness")


# --------------------------------------------------------------------------- S9 minute trajectories
def figS9():
    comps = [("Fig5b.csv", "Wards placed"), ("Fig5c.csv", "Wards cleared"), ("Fig5a.csv", "Control wards purchased")]
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.7))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.17, wspace=0.28)
    for k, (f, lab) in enumerate(comps):
        ax = axs[k]
        d = read("source_data/main/" + f)
        for lo, hi, wl in ((0, 5, "0–5"), (5, 8, "5–8"), (8, 10, "8–10")):
            ax.axvspan(lo, hi, color=NEUTRAL if wl != "5–8" else "white", lw=0, zorder=0)
            ax.text((lo + hi) / 2, 1.0, wl, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6.2, color=MUTED)
        for reg in REG_ORDER:
            dd = d[d.region == reg].sort_values("minute")
            ax.fill_between(dd.time_midpoint, dd.ci_low, dd.ci_high, color=REG[reg], alpha=0.18, lw=0)
            ax.plot(dd.time_midpoint, dd.mean_rate, color=REG[reg], lw=1.4, marker=REG_MARK[reg], ms=3.2,
                    mec="white", mew=0.4)
        ax.set_xlim(0, 10)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        hgrid(ax, "y")
        ax.set_xlabel("Minute of the match")
        if k == 0:
            ax.set_ylabel("Events per team per minute")
            hs = [ax.plot([], [], marker=REG_MARK[r], color=REG[r], ls="-", lw=1.2, ms=3.6, label=r)[0]
                  for r in REG_ORDER]
            ax.legend(handles=hs, loc="lower right", fontsize=6.6)
        panel_title(ax, "abc"[k], lab, x=-0.2, y=1.1)
    save(fig, "FigS9_minute_maintenance_trajectories")


# --------------------------------------------------------------------------- S10 probability curves
def figS10():
    cur = rd(PERF / "performance_probability_curves.csv")
    sup = rd(PERF / "performance_support.csv")
    for df in (cur, sup):
        df["sample"] = df["sample"].replace({"holdout_strict": "strict"})
        df["region"] = df["region"].str.replace("1", "", regex=False)
    fig, axs = plt.subplots(2, 3, figsize=(FULL_W, 3.6), sharex=True,
                            gridspec_kw=dict(height_ratios=[3.2, 1]))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.14, hspace=0.08, wspace=0.12)
    for k, s in enumerate(SAMPLES):
        ax = axs[0, k]
        for reg in REG_ORDER:
            d = cur[(cur["sample"] == s) & (cur.region == reg)].sort_values("maintenance_difference_sd")
            ax.fill_between(d.maintenance_difference_sd, d.ci_low * 100, d.ci_high * 100, color=REG[reg],
                            alpha=0.18, lw=0)
            ax.plot(d.maintenance_difference_sd, d.probability * 100, color=REG[reg], lw=1.4)
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_ylim(40, 60)
        hgrid(ax, "y")
        if k == 0:
            ax.set_ylabel("P(blue takes next objective), %")
            hs = [ax.plot([], [], color=REG[r], lw=1.6, label=r)[0] for r in REG_ORDER]
            ax.legend(handles=hs, loc="upper left", fontsize=6.6)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        panel_title(ax, "abc"[k], SAMPLE_LABEL[s], x=-0.02, y=1.04)
        ax2 = axs[1, k]
        for reg in REG_ORDER:
            d = sup[(sup["sample"] == s) & (sup.region == reg)].sort_values("mid")
            d = d[(d.mid >= -2.5) & (d.mid <= 2.5)]
            share = d["count"] / sup[(sup["sample"] == s) & (sup.region == reg)]["count"].sum() * 100
            ax2.plot(d.mid, share, color=REG[reg], lw=1.0)
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(bottom=0)
        hgrid(ax2, "y")
        ax2.set_xlabel("Relative maintenance (SD)")
        if k == 0:
            ax2.set_ylabel("Matches\n(% per bin)", fontsize=6.6)
        else:
            plt.setp(ax2.get_yticklabels(), visible=False)
    save(fig, "FigS10_probability_curves")


# --------------------------------------------------------------------------- S11 windows and natural counts
def figS11():
    win = pd.concat([read(f"source_data/supplementary/FigS8{k}.csv") for k in "abc"])
    fig, axs = plt.subplots(2, 3, figsize=(FULL_W, 5.0))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.1, hspace=0.6, wspace=0.14)
    wins = ["w0_5", "w5_8", "w8_10"]
    wlab = ["0–5", "5–8", "8–10"]
    for k, s in enumerate(SAMPLES):
        ax = axs[0, k]
        for g, dg in enumerate(["near", "middle", "far"]):
            d = win[(win["sample"] == s) & (win.delay_group == dg)].copy()
            d["w"] = d.predictor.str.extract(r"score_(w\d+_\d+)__")[0]
            d = d.set_index("w").loc[wins]
            xs = np.arange(3) + (-0.18, 0, 0.18)[g]
            ax.plot(xs, d.estimate, color=DELAY[dg], lw=1.3, marker="o", ms=3.6, mec="white", mew=0.4)
            for xx, (_, r) in zip(xs, d.iterrows()):
                ax.plot([xx, xx], [r.ci_low, r.ci_high], color=DELAY[dg], lw=0.9)
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(wlab)
        ax.set_ylim(-0.2, 2.25)
        hgrid(ax, "y")
        ax.set_xlabel("Maintenance window (min)")
        if k == 0:
            ax.set_ylabel("Δ P(objective), pp per window SD")
            hs = [ax.plot([], [], color=DELAY[g], lw=1.4, marker="o", ms=3.6, label=DELAY_LABEL[g])[0]
                  for g in ["near", "middle", "far"]]
            ax.legend(handles=hs, loc="upper left", fontsize=6.2)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        panel_title(ax, "abc"[k], SAMPLE_LABEL[s], x=-0.02, y=1.04)
    comps = [("valid_wards_placed", "Placement"), ("valid_ward_kills", "Clearing"),
             ("control_ward_purchases", "Purchases")]
    nat = {s: rd(SUPP / f"FigS8{'def'[i]}_{s}.csv") for i, s in enumerate([])}
    for k, (cp, lab) in enumerate(comps):
        ax = axs[1, k]
        for m, s in enumerate(SAMPLES):
            f = SUPP / f"FigS8{'def'[k]}_{s}.csv"
            d = rd(f)
            d["w"] = d.predictor.str.extract(r"_(w\d+_\d+)_diff")[0]
            d = d.set_index("w").loc[wins]
            for i, (_, r) in enumerate(d.iterrows()):
                spt(ax, i + (-0.2, 0, 0.2)[m], r.estimate, s, r.ci_low, r.ci_high, horizontal=False)
        ax.set_xticks(range(3))
        ax.set_xticklabels(wlab)
        ax.set_ylim(0, 1.15)
        hgrid(ax, "y")
        ax.set_xlabel("Maintenance window (min)")
        if k == 0:
            ax.set_ylabel("Δ P(objective), pp per event")
            sample_legend(ax, loc="upper left")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        panel_title(ax, "def"[k], lab, x=-0.02, y=1.04)
    save(fig, "FigS11_windows_and_components")


if __name__ == "__main__":
    figS7()
    figS8()
    figS9()
    figS10()
    figS11()
