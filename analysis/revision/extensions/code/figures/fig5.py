from style import *

setup()
SUPP = BASE / "source_data/supplementary"
fmt2 = lambda v: f"{v:.2f}".replace("-", "−")
MS = {"o": 4.6, "s": 4.2, "D": 3.9}


def pt(ax, x, y, s, lo, hi):
    st = SAMPLE_STYLE[s]
    ax.plot([lo, hi], [y, y], color=MUTED if s == "holdout" else INK, lw=1.0, zorder=2)
    ax.plot([x], [y], marker=st["marker"], mfc=st["mfc"], mec=st["mec"], mew=0.9, ms=MS[st["marker"]], ls="none",
            zorder=3)


def vpt(ax, x, y, s, lo, hi):
    st = SAMPLE_STYLE[s]
    ax.plot([x, x], [lo, hi], color=MUTED if s == "holdout" else INK, lw=1.0, zorder=2)
    ax.plot([x], [y], marker=st["marker"], mfc=st["mfc"], mec=st["mec"], mew=0.9, ms=MS[st["marker"]], ls="none",
            zorder=3)


def fig5():
    ame = read("source_data/main/Fig5e.csv")
    ame["sample"] = ame["sample"].replace({"holdout_strict": "strict"})
    win = pd.concat([read(f"source_data/supplementary/FigS8{k}.csv") for k in "abc"])

    fig = plt.figure(figsize=(FULL_W, 3.15))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.82, 2.0], wspace=0.2, left=0.075, right=0.995, top=0.78,
                             bottom=0.16)
    # ---- a
    ax = fig.add_subplot(outer[0, 0])
    regs = ["pooled", "KR", "EUW", "NA"]
    lab = {"pooled": "All regions", "KR": "KR", "EUW": "EUW", "NA": "NA"}
    for i, rg in enumerate(regs):
        y = len(regs) - 1 - i
        for k, s in enumerate(SAMPLES):
            r = ame[(ame["sample"] == s) & (ame.region == rg)].iloc[0]
            pt(ax, 100 * r.prob_change, y + (0.22, 0, -0.22)[k], s, 100 * r.prob_change_low, 100 * r.prob_change_high)
            if s == "discovery":
                ax.text(100 * r.prob_change_high + 0.04, y + 0.22, fmt2(100 * r.prob_change), ha="left", va="center",
                        fontsize=6.6, color=INK)
    ax.axhline(2.5, color=GRID, lw=0.6)
    ax.set_yticks(range(len(regs)))
    ax.set_yticklabels([lab[r] for r in regs[::-1]], fontsize=7.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(1.2, 2.65)
    ax.set_ylim(-0.6, 3.6)
    hgrid(ax, "x")
    ax.set_xlabel("Change in P(next objective), pp\nper 1 SD more maintenance than opponents", linespacing=1.3)
    hs = [ax.plot([], [], marker=SAMPLE_STYLE[s]["marker"], mfc=SAMPLE_STYLE[s]["mfc"], mec=SAMPLE_STYLE[s]["mec"],
                  mew=0.9, ms=MS[SAMPLE_STYLE[s]["marker"]], ls="none", label=SAMPLE_LABEL[s])[0] for s in SAMPLES]
    fig.legend(handles=hs, loc="center right", bbox_to_anchor=(0.995, 0.875), ncol=3, fontsize=6.8,
               handletextpad=0.2, columnspacing=1.2)
    fig.text(0.008, 0.955, "a", fontsize=9, fontweight="bold", color=INK, ha="left", va="center")

    # ---- b-d small multiples
    inner = outer[0, 1].subgridspec(1, 3, wspace=0.12)
    wins = ["w0_5", "w5_8", "w8_10"]
    wlab = ["0–5", "5–8", "8–10"]
    dlab = {"near": "Objective ≤60 s after 10:00", "middle": "61–180 s after", "far": ">180 s after"}
    axs = []
    for j, dg in enumerate(["near", "middle", "far"]):
        ax = fig.add_subplot(inner[0, j], sharey=axs[0] if axs else None)
        axs.append(ax)
        ax.axhline(0, color=INK2, lw=0.7)
        for k, s in enumerate(SAMPLES):
            d = win[(win["sample"] == s) & (win.delay_group == dg)].copy()
            d["w"] = d.predictor.str.extract(r"score_(w\d+_\d+)__")[0]
            d = d.set_index("w").loc[wins]
            xs = np.arange(3) + (-0.2, 0, 0.2)[k]
            if s == "discovery":
                ax.plot(xs, d.estimate, color=INK, lw=1.2, zorder=1)
            for xx, (_, r) in zip(xs, d.iterrows()):
                vpt(ax, xx, r.estimate, s, r.ci_low, r.ci_high)
        dd = win[(win["sample"] == "discovery") & (win.delay_group == dg)].copy()
        dd["w"] = dd.predictor.str.extract(r"score_(w\d+_\d+)__")[0]
        dd = dd.set_index("w").loc[wins]
        imax = int(np.argmax(dd.estimate.values))
        ax.text(imax - 0.2, dd.estimate.iloc[imax] + 0.32, fmt2(dd.estimate.iloc[imax]), ha="center", va="bottom",
                fontsize=6.6, color=INK, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(wlab)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.45, 2.45)
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        hgrid(ax, "y")
        ax.text(0.5, 1.0, dlab[dg], transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=INK)
        if j == 0:
            ax.set_ylabel("Change in P(next objective), pp\nper 1 SD of the window's maintenance gap", linespacing=1.3)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if j == 1:
            ax.set_xlabel("Maintenance window (minutes into the match)")
    fig.text(0.315, 0.955, "b", fontsize=9, fontweight="bold", color=INK, ha="left", va="center")
    save(fig, "Fig5_relative_maintenance_and_objectives")


if __name__ == "__main__":
    fig5()
