from style import *
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtrans

setup()
fmt3 = lambda v: f"{v:.3f}".replace("-", "−")


def sample_point(ax, x, y, s, lo=None, hi=None, color=INK, horizontal=True):
    st = SAMPLE_STYLE[s]
    if lo is not None:
        ax.plot([lo, hi], [y, y], color=st["mec"] if s != "strict" else INK, lw=1.0, zorder=2)
    ax.plot([x], [y], marker=st["marker"], mfc=st["mfc"], mec=st["mec"], mew=0.9,
            ms={"o": 4.6, "s": 4.2, "D": 3.9}[st["marker"]], ls="none", zorder=3)


def sample_legend(ax, **kw):
    hs = []
    for s in SAMPLES:
        st = SAMPLE_STYLE[s]
        hs.append(ax.plot([], [], marker=st["marker"], mfc=st["mfc"], mec=st["mec"], mew=0.9, ls="none",
                          ms={"o": 4.6, "s": 4.2, "D": 3.9}[st["marker"]], label=SAMPLE_LABEL[s])[0])
    return ax.legend(handles=hs, handletextpad=0.2, fontsize=6.8, **kw)


def fig4():
    pe = pd.read_csv(TAB / "peer_extensions.csv", keep_default_na=False, na_values=[""])
    pe = pe[(pe.spec == "base") & (pe.inference == "two_way") & (pe.unit == "common_sd")]
    team = pd.read_csv(TAB / "team_placement_associations.csv", keep_default_na=False)
    opp = read("source_data/main/Fig4d.csv")

    fig = plt.figure(figsize=(FULL_W, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], height_ratios=[1.15, 1], hspace=0.5, wspace=0.5,
                          left=0.115, right=0.985, top=0.93, bottom=0.085)

    # ---- a: heatmap of directed placement associations
    ax = fig.add_subplot(gs[0, 0])
    d = pe[(pe["sample"] == "discovery") & (pe.exposure == "wards_placed") & (pe.outcome == "wards_placed")]
    d = d[d.contrast.str.contains("|", regex=False)]
    M = np.full((5, 5), np.nan)
    for _, r in d.iterrows():
        a, b = r.contrast.split("|")
        M[ROLE_ORDER.index(a), ROLE_ORDER.index(b)] = r.estimate
    cmap = LinearSegmentedColormap.from_list("div", [(0, "#184f95"), (0.25, "#6da7ec"), (0.5, "#f0efec"),
                                                     (0.75, "#ef9796"), (1, "#c62f2f")])
    lim = 0.13
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(5):
        for j in range(5):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="none"))
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color=MUTED)
                continue
            v = M[i, j]
            ax.text(j, i, fmt3(v), ha="center", va="center", fontsize=6.4,
                    color="white" if abs(v) > 0.075 else INK, fontweight="bold" if abs(v) > 0.075 else "normal")
    # white gaps between cells
    for k in range(6):
        ax.axhline(k - 0.5, color="white", lw=1.2)
        ax.axvline(k - 0.5, color="white", lw=1.2)
    ax.set_xticks(range(5))
    ax.set_xticklabels([ROLE_LABEL[r] for r in ROLE_ORDER])
    ax.set_yticks(range(5))
    ax.set_yticklabels([ROLE_LABEL[r] for r in ROLE_ORDER], color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Teammate whose placement in the match is observed")
    ax.set_ylabel("Teammate with higher usual placement")
    panel_title(ax, "a", "Negative links concentrate in pairs involving support", x=-0.36, y=1.04)

    # ---- b: cross-activity to support
    ax = fig.add_subplot(gs[0, 1])
    acts = [("wards_placed", "Placement"), ("control_wards", "Purchases"), ("wards_killed", "Clearing")]
    srcs = [("JUNGLE|UTILITY", "Jungle"), ("non_jungle_to_support", "Other roles")]
    rows, labels, y = {}, [], 0
    ylab = []
    for a, alab in acts[::-1]:
        for c, clab in srcs[::-1]:
            rows[(a, c)] = y
            ylab.append((y, clab))
            y += 1
        y += 0.7
    for (a, c), yy in rows.items():
        for k, s in enumerate(SAMPLES):
            r = pe[(pe["sample"] == s) & (pe.exposure == "wards_placed") & (pe.outcome == a) & (pe.contrast == c)].iloc[0]
            sample_point(ax, r.estimate, yy + (0.22, 0, -0.22)[k], s, r.ci_low, r.ci_high)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks([p for p, _ in ylab])
    ax.set_yticklabels([l for _, l in ylab], fontsize=7, color=INK)
    tr = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
    for a, alab in acts:
        top = max(rows[(a, c)] for c, _ in srcs)
        ax.text(-0.02, top + 0.62, f"Support's {alab.lower()}", transform=tr, ha="right", va="center",
                fontsize=7, color=INK, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.14, 0.012)
    ax.set_xticks([-0.12, -0.09, -0.06, -0.03, 0])
    ax.set_xticklabels(["−0.12", "−0.09", "−0.06", "−0.03", "0"])
    ax.set_ylim(-0.6, y - 0.3)
    hgrid(ax, "x")
    ax.set_xlabel("Association with the teammate's usual placement (common SD)")
    sample_legend(ax, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    panel_title(ax, "b", "Support's purchases and clearing are also lower", x=-0.36, y=1.04)

    # ---- c: team-to-own ratio
    ax = fig.add_subplot(gs[1, 0])
    rat = team[team.outcome == "team_to_own_ratio"]
    disc = rat[rat["sample"] == "discovery"].set_index("role")["estimate"].sort_values(ascending=False)
    order = list(disc.index)
    for i, role in enumerate(order):
        for k, s in enumerate(SAMPLES):
            r = rat[(rat["sample"] == s) & (rat.role == role)].iloc[0]
            sample_point(ax, r.estimate, i + (0.22, 0, -0.22)[k], s, r.ci_low, r.ci_high)
        r = rat[(rat["sample"] == "discovery") & (rat.role == role)].iloc[0]
        if role in ("UTILITY", "JUNGLE"):
            ax.text(r.ci_low - 0.008, i + 0.22, f"{r.estimate:.3f}", ha="right", va="center", fontsize=6.6, color=INK)
    ax.axvline(1, color=INK2, lw=0.7)
    ax.text(1.004, len(order) - 0.45, "team slope = own slope", ha="left", va="center", fontsize=6.4, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([ROLE_LABEL[r] for r in order], fontsize=7.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.66, 1.1)
    ax.set_ylim(-0.6, len(order) - 0.2)
    hgrid(ax, "x")
    ax.set_xlabel("Team placement slope ÷ own placement slope")
    panel_title(ax, "c", "Team placement slopes are smaller than own slopes", x=-0.36, y=1.04)

    # ---- d: teammate vs opponent clearing
    ax = fig.add_subplot(gs[1, 1])
    ax.axvspan(-0.02, 0.02, color=NEUTRAL, zorder=0, lw=0)
    ax.text(0.0215, 1.86, "← shaded band: ±0.02 SD\n    equivalence bounds", ha="left", va="center", fontsize=6.3,
            color=MUTED, linespacing=1.2)
    for i, rel in enumerate(["opponent", "teammate"]):
        for k, s in enumerate(SAMPLES):
            r = opp[(opp["sample"] == s) & (opp.relation == rel)].iloc[0]
            sample_point(ax, r.b_std, i + (0.22, 0, -0.22)[k], s, r.ci_low_std, r.ci_high_std)
            if s == "discovery":
                ax.text(r.ci_high_std + 0.002, i + 0.22, fmt3(r.b_std), ha="left", va="center", fontsize=6.6,
                        color=INK)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Opponents'\nclearing", "Teammates'\nclearing"], fontsize=7.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.03, 0.062)
    ax.set_xticks([-0.02, 0, 0.02, 0.04, 0.06])
    ax.set_xticklabels(["−0.02", "0", "0.02", "0.04", "0.06"])
    ax.set_ylim(-0.6, 2.05)
    hgrid(ax, "x")
    ax.set_xlabel("Association with a player's usual placement (common SD)")
    panel_title(ax, "d", "Opponents, not teammates, clear more wards", x=-0.36, y=1.04)
    save(fig, "Fig4_teammates_and_opponents")


if __name__ == "__main__":
    fig4()
