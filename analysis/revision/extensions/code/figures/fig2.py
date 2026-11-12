from style import *

setup()


def fig2():
    tests = read("source_data/main/Fig2a_tests.csv")
    nat = pd.read_csv(NATURAL, keep_default_na=False)
    nat = nat[nat["scale"] == "per_10_min"].copy()
    nat["region"] = nat["region"].str.replace("1", "", regex=False)
    shares = read("source_data/main/Fig2d.csv")

    fig = plt.figure(figsize=(FULL_W, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.9], wspace=0.5, hspace=0.62,
                          left=0.03, right=0.99, top=0.93, bottom=0.085)

    # ---- a: EUW position between NA (0) and KR (1)
    ax = fig.add_subplot(gs[0, 0])
    rows = {"maintenance": 1.0, "signalling": 0.0}
    offs = {"discovery": 0.17, "holdout": 0.0, "strict": -0.17}
    for proc, y0 in rows.items():
        ax.plot([0, 1], [y0, y0], color=TRACK, lw=9, solid_capstyle="round", zorder=0)
        ax.plot([0], [y0], marker=REG_MARK["NA"], color=REG["NA"], ms=6, mec="white", mew=0.8, zorder=3)
        ax.plot([1], [y0], marker=REG_MARK["KR"], color=REG["KR"], ms=6.5, mec="white", mew=0.8, zorder=3)
        for s in SAMPLES:
            r = tests[(tests["sample"] == s) & (tests["measure"] == proc)].iloc[0]
            yy = y0 + offs[s]
            ax.plot([r.ci_low, r.ci_high], [yy, yy], color=REG["EUW"], lw=1.1, zorder=2)
            st = dict(marker="s", ms=5, mec=REG["EUW"], mew=1.0, zorder=4)
            fc = {"discovery": REG["EUW"], "holdout": "#8fd8bd", "strict": "white"}[s]
            ax.plot([r.estimate], [yy], mfc=fc, ls="none", **st)
            if s == "discovery":
                ax.text(r.ci_high + 0.03, yy, f"{r.estimate:.3f}", ha="left", va="center", fontsize=6.8, color=INK)
    ax.text(0, 1.47, "NA", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.text(1, 1.47, "KR", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.annotate("", xy=(0.97, 1.42), xytext=(0.03, 1.42),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, mutation_scale=6))
    ax.text(0.5, 1.47, "position of EUW on the NA→KR gap", ha="center", va="bottom", fontsize=6.8, color=MUTED)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["Information\nmaintenance", "Team\nsignalling"], fontsize=7.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.42, 1.62)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_xlabel("Relative position r = (EUW − NA) / (KR − NA)")
    hgrid(ax, "x")
    d = tests[(tests["sample"] == "discovery") & (tests["measure"] == "signalling_minus_maintenance")].iloc[0]
    ax.text(0.30, 0.50, f"r(signalling) − r(maintenance)\n= {d.estimate:.2f} [{d.ci_low:.2f}, {d.ci_high:.2f}]",
            ha="center", va="center", fontsize=6.8, color=INK)
    # sample key
    kx, ky = 0.60, 0.62
    for i, s in enumerate(SAMPLES):
        fc = {"discovery": REG["EUW"], "holdout": "#8fd8bd", "strict": "white"}[s]
        ax.plot([kx], [ky - i * 0.13], marker="s", ms=4.5, mfc=fc, mec=REG["EUW"], mew=0.9, ls="none")
        ax.text(kx + 0.04, ky - i * 0.13, SAMPLE_LABEL[s], va="center", fontsize=6.6, color=INK2)
    panel_title(ax, "a", "Same ranking, different relative positions", x=-0.3, y=1.04)

    # ---- c: content composition of pings
    ax = fig.add_subplot(gs[1, 0])
    fam = [("caution", "Risk"), ("initiation", "Initiation"), ("intent", "Intent"), ("request", "Requests"),
           ("enemy_information", "Enemy information"), ("tactical", "Tactical attention")]
    sh = shares[shares["sample"] == "discovery"]
    na_share = sh[sh.region == "NA"].set_index("family")["estimate"]
    fam = sorted(fam, key=lambda f: na_share[f[0]])
    for i, (f, lab) in enumerate(fam):
        vals = sh[sh.family == f].set_index("region")
        for reg, dy in (("KR", 0.2), ("EUW", 0.0), ("NA", -0.2)):
            v = vals.loc[reg]
            ax.plot([v.ci_low * 100, v.ci_high * 100], [i + dy] * 2, color=REG[reg], lw=1.0, zorder=2)
            ax.plot([v.estimate * 100], [i + dy], marker=REG_MARK[reg], color=REG[reg],
                    ms=4.6 if reg == "KR" else 4.2, mec="white", mew=0.5, ls="none", zorder=3)
        if f in ("caution", "initiation"):
            k = vals.loc["KR"].estimate * 100
            ax.text(k * (1.1 if f == "caution" else 1 / 1.1), i + 0.2, f"{k:.1f}%", fontsize=6.4, color=INK2,
                    ha="left" if f == "caution" else "right", va="center")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.set_xticklabels(["1%", "2%", "5%", "10%", "20%", "40%"])
    ax.minorticks_off()
    ax.set_xlim(1.0, 45)
    ax.set_yticks(range(len(fam)))
    ax.set_yticklabels([l for _, l in fam], fontsize=7, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(fam) - 0.3)
    hgrid(ax, "x")
    ax.set_xlabel("Share of all pings (log scale)")
    hs = [ax.plot([], [], marker=REG_MARK[r], color=REG[r], ls="none", ms=5 if r == "KR" else 4.6, label=r)[0]
          for r in REG_ORDER]
    ax.legend(handles=hs, loc="lower right", ncol=3, fontsize=6.8, handletextpad=0.1, columnspacing=0.8)
    panel_title(ax, "c", "More risk warnings and fewer initiation signals in KR", x=-0.3, y=1.04)

    # ---- b: activity rates relative to NA
    ax = fig.add_subplot(gs[:, 1])
    maint = [("control_wards_bought", "Control-ward purchases"), ("wards_killed", "Wards cleared"),
             ("wards_placed", "Wards placed")]
    sig = [("get_back_pings", "Retreat"), ("command_pings", "Command"), ("enemy_missing_pings", "Enemy missing"),
           ("assist_me_pings", "Assist me"), ("on_my_way_pings", "On my way"), ("need_vision_pings", "Need vision"),
           ("enemy_vision_pings", "Enemy vision"), ("all_in_pings", "All in"), ("push_pings", "Push")]

    def ratio(metric, reg):
        a = nat[(nat.region == reg) & (nat.metric == metric)].iloc[0]
        b = nat[(nat.region == "NA") & (nat.metric == metric)].iloc[0]
        sa = (a.ci_high - a.ci_low) / 3.92 / a.estimate
        sb = (b.ci_high - b.ci_low) / 3.92 / b.estimate
        lr = np.log(a.estimate / b.estimate)
        se = np.sqrt(sa ** 2 + sb ** 2)
        return np.exp(lr), np.exp(lr - 1.96 * se), np.exp(lr + 1.96 * se)

    def order(items):
        return sorted(items, key=lambda m: -ratio(m[0], "KR")[0])

    maint, sig = order(maint), order(sig)
    labels, ys = [], []
    y = 0
    gap = 1.5
    pos = {}
    for m, lab in sig[::-1]:
        pos[m] = y; labels.append(lab); ys.append(y); y += 1
    ysig_top = y - 1
    y += gap
    for m, lab in maint[::-1]:
        pos[m] = y; labels.append(lab); ys.append(y); y += 1
    ymaint_top = y - 1
    ax.axvline(1, color=INK2, lw=0.7, zorder=1)
    for m, lab in maint + sig:
        for reg, dy in (("EUW", -0.13), ("KR", 0.13)):
            est, lo, hi = ratio(m, reg)
            ax.plot([lo, hi], [pos[m] + dy] * 2, color=REG[reg], lw=1.0, zorder=2)
            ax.plot([est], [pos[m] + dy], marker=REG_MARK[reg], color=REG[reg], ms=4.6 if reg == "EUW" else 5,
                    mec="white", mew=0.6, ls="none", zorder=3)
    ax.set_xscale("log", base=2)
    ticks = [0.5, 0.75, 1, 1.5, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.5×", "0.75×", "1×", "1.5×", "2×"])
    ax.minorticks_off()
    ax.set_xlim(0.42, 2.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, ymaint_top + 1.3)
    hgrid(ax, "x")
    ax.axhline(ysig_top + gap / 2 + 0.5, color=GRID, lw=0.6)
    import matplotlib.transforms as mtrans
    tr = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(-0.012, ymaint_top + 0.85, "Maintenance activities", transform=tr, ha="right", va="center",
            fontsize=7, color=INK, fontweight="bold")
    ax.text(-0.012, ysig_top + 0.85, "Signal types", transform=tr, ha="right", va="center",
            fontsize=7, color=INK, fontweight="bold")
    for m in ("control_wards_bought", "get_back_pings", "push_pings"):
        est, lo, hi = ratio(m, "KR")
        if est > 1:
            ax.text(hi * 1.035, pos[m] + 0.13, f"{est:.2f}×", ha="left", va="center", fontsize=6.6, color=INK2)
        else:
            ax.text(lo / 1.035, pos[m] + 0.13, f"{est:.2f}×", ha="right", va="center", fontsize=6.6, color=INK2)
    ax.set_xlabel("Rate relative to NA (per player per 10 min, log scale)")
    h1 = ax.plot([], [], marker=REG_MARK["EUW"], color=REG["EUW"], ls="none", ms=4.6, label="EUW ÷ NA")[0]
    h2 = ax.plot([], [], marker=REG_MARK["KR"], color=REG["KR"], ls="none", ms=5, label="KR ÷ NA")[0]
    ax.legend(handles=[h2, h1], loc="lower right", bbox_to_anchor=(1.0, 0.0), handletextpad=0.2, fontsize=7)
    panel_title(ax, "b", "Where the regional differences lie", x=-0.36, y=1.02)
    save(fig, "Fig2_same_ranking_different_positions")


if __name__ == "__main__":
    fig2()
