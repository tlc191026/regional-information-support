from style import *

setup()
f3 = lambda v: f"{v:+.3f}".replace("-", "−")


def fig3():
    rm = {"maintenance": read("source_data/main/Fig3e.csv"), "signalling": read("source_data/main/Fig3f.csv")}
    tm = {"maintenance": read("source_data/main/Fig3a.csv"), "signalling": read("source_data/main/Fig3b.csv")}
    tch = pd.read_csv(TAB / "tier_process_changes.csv", keep_default_na=False)

    fig = plt.figure(figsize=(FULL_W, 5.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.55, wspace=0.18,
                          left=0.1, right=0.965, top=0.93, bottom=0.08)
    # role order by KR-NA maintenance gap (largest at top)
    m = rm["maintenance"].pivot_table(index="level", columns="region", values="estimate")
    gap = (m["KR"] - m["NA"]).sort_values()
    roles = list(gap.index)
    ypos = {r: i for i, r in enumerate(roles)}

    titles = {"maintenance": ("a", "Maintenance gaps are largest for support and jungle"),
              "signalling": ("b", "Signalling gaps are similar across roles")}
    xl = {"maintenance": (-0.75, 2.05), "signalling": (-0.75, 2.05)}
    top_axes = []
    for j, proc in enumerate(["maintenance", "signalling"]):
        ax = fig.add_subplot(gs[0, j], sharey=top_axes[0] if top_axes else None)
        top_axes.append(ax)
        d = rm[proc]
        for r in roles:
            v = d[d.level == r].set_index("region")
            y = ypos[r]
            for reg, dy in (("KR", 0.17), ("EUW", 0.0), ("NA", -0.17)):
                e = v.loc[reg]
                ax.plot([e.ci_low, e.ci_high], [y + dy] * 2, color=REG[reg], lw=1.0, zorder=2)
                ax.plot([e.estimate], [y + dy], marker=REG_MARK[reg], color=REG[reg],
                        ms=4.6 if reg == "KR" else 4.2, mec="white", mew=0.5, ls="none", zorder=3)
            g = v.loc["KR"].estimate - v.loc["NA"].estimate
            ax.text(xl[proc][1] - 0.02, y, f"{g:.2f}", ha="right", va="center", fontsize=6.6, color=INK2)
        ax.text(xl[proc][1] - 0.02, len(roles) - 0.35, "KR − NA", ha="right", va="center", fontsize=6.4,
                color=MUTED)
        ax.set_xlim(*xl[proc])
        ax.set_ylim(-0.6, len(roles) - 0.1)
        hgrid(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Adjusted mean (SD of the process score)")
        if j == 0:
            ax.set_yticks(range(len(roles)))
            ax.set_yticklabels([ROLE_LABEL[r] for r in roles], fontsize=7.5, color=INK)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        L, T = titles[proc]
        panel_title(ax, L, T, x=-0.15 if j == 0 else -0.03, y=1.05)
    hs = [top_axes[1].plot([], [], marker=REG_MARK[r], color=REG[r], ls="none", ms=5 if r == "KR" else 4.6,
                           label=r)[0] for r in ["KR", "EUW", "NA"]]
    top_axes[1].legend(handles=hs, loc="center right", bbox_to_anchor=(0.86, 0.5), fontsize=7, handletextpad=0.2)

    tiers = TIER_ORDER
    x = np.arange(len(tiers))
    titles = {"maintenance": ("c", "KR's maintenance lead is larger at higher tiers"),
              "signalling": ("d", "KR's signalling lead shrinks at higher tiers")}
    bot_axes = []
    for j, proc in enumerate(["maintenance", "signalling"]):
        ax = fig.add_subplot(gs[1, j], sharey=bot_axes[0] if bot_axes else None)
        bot_axes.append(ax)
        d = tm[proc]
        ends = {}
        for reg in REG_ORDER:
            dd = d[d.region == reg].set_index("level").loc[tiers]
            ax.fill_between(x, dd.ci_low, dd.ci_high, color=REG[reg], alpha=0.16, lw=0, zorder=1)
            ax.plot(x, dd.estimate, color=REG[reg], lw=1.6, zorder=2)
            ax.plot(x, dd.estimate, marker=REG_MARK[reg], color=REG[reg], ms=4.2 if reg == "KR" else 3.8,
                    mec="white", mew=0.5, ls="none", zorder=3)
            ends[reg] = dd.estimate.iloc[-1]
        # direct labels at the right end, nudged apart
        order = sorted(ends, key=ends.get)
        ys = [ends[r] for r in order]
        for k in range(1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + 0.075)
        for r, yv in zip(order, ys):
            ax.text(len(tiers) - 1 + 0.2, yv, f"{r} {ends[r]:.2f}", ha="left", va="center", fontsize=6.6,
                    color=INK2)
        ch = tch[(tch.contrast == "KR-NA1") & (tch.process == proc)].set_index("sample")
        s = ch.loc["discovery"]
        txt = (f"KR − NA gap, Bronze → Diamond: {f3(s.estimate)} [{f3(s.ci_low)}, {f3(s.ci_high)}]\n"
               f"player / strict holdout: {f3(ch.loc['holdout'].estimate)} / {f3(ch.loc['strict'].estimate)}")
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK,
                linespacing=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels([TIER_LABEL[t] for t in tiers])
        ax.set_xlim(-0.4, len(tiers) + 0.6)
        ax.set_ylim(-0.75, 1.0)
        hgrid(ax, "y")
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_xlabel("Roster tier (low → high)")
        if j == 0:
            ax.set_ylabel("Adjusted mean (SD)")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        L, T = titles[proc]
        panel_title(ax, L, T, x=-0.15 if j == 0 else -0.03, y=1.05)
    save(fig, "Fig3_roles_and_tiers")


if __name__ == "__main__":
    fig3()
