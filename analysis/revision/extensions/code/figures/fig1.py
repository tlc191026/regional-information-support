from style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

setup()
W, H = 190, 80
BOX = "#f5f4f1"
EDGE = "#c9c7c0"
ACC = "#e8f0fb"
ACCE = "#9ec5f4"


def box(ax, x, y, w, h, fc=BOX, ec=EDGE, lw=0.7, r=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def rq(ax, x, y, tag, text):
    ax.text(x, y, tag, ha="left", va="top", fontsize=6.6, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc=INK, ec="none"))
    ax.text(x + 11.5, y + 0.15, text, ha="left", va="top", fontsize=6.6, color=INK, linespacing=1.3)


def fig1():
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    box(ax, 45, 70, 100, 8, fc="white")
    ax.text(95, 74, "Shared platform tools: the same wards and pings for every group", ha="center", va="center",
            fontsize=7.2, color=INK)
    arrow(ax, 95, 70, 95, 65.6)

    box(ax, 4, 43, 182, 22.5, fc=ACC, ec=ACCE)
    ax.text(8, 61.6, "Information work", ha="left", va="center", fontsize=7.6, color=INK, fontweight="bold")
    ax.text(182, 61.6, "configuration: level, content and role distribution of the two types of work",
            ha="right", va="center", fontsize=6.6, color=INK2, style="italic")
    box(ax, 8, 45.5, 85, 13, fc="white", ec=ACCE)
    ax.text(12, 55.0, "Information maintenance", ha="left", va="center", fontsize=7.2, color=INK, fontweight="bold")
    ax.text(12, 50.2, "provision  ·  control  ·  resource preparation", ha="left", va="center", fontsize=6.6,
            color=INK2)
    box(ax, 97, 45.5, 85, 13, fc="white", ec=ACCE)
    ax.text(101, 55.0, "Team signalling", ha="left", va="center", fontsize=7.2, color=INK, fontweight="bold")
    ax.text(101, 50.2, "state  ·  request  ·  action", ha="left", va="center", fontsize=6.6, color=INK2)

    cw, top, bot = 56, 36.5, 3
    cols = [4, 67, 130]
    heads = ["Across groups", "Within and between teams", "In contested tasks"]
    for x, head in zip(cols, heads):
        arrow(ax, x + cw / 2, 43, x + cw / 2, top + 0.1)
        box(ax, x, bot, cw, top - bot)
        ax.text(x + 3, top - 4.2, head, ha="left", va="center", fontsize=7.6, color=INK, fontweight="bold")
    # connections between levels
    for (x0, x1, lab) in ((cols[0] + cw, cols[1], "roles"), (cols[1] + cw, cols[2], "opponents")):
        arrow(ax, x0 + 0.6, 17, x1 - 0.6, 17, color=INK2, lw=0.8, style="<|-|>")
        ax.text((x0 + x1) / 2, 19.2, lab, ha="center", va="bottom", fontsize=6.2, color=INK2, style="italic")

    x = cols[0]
    rq(ax, x + 3, 27, "RQ1a", "Can one common intensity\ndescribe regional differences?")
    rq(ax, x + 3, 15, "RQ1b", "How are the differences\ndistributed across roles and\ntiers?")
    x = cols[1]
    rq(ax, x + 3, 27, "RQ2", "How does a member's usual\ninformation work relate to\nteammates' and opponents'\nwork in the match?")
    x = cols[2]
    rq(ax, x + 3, 27, "RQ3", "How does relative maintenance\nbefore a contest relate to the\nnext objective, and when?")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig1_conceptual_framework.{ext}", dpi=600 if ext == "png" else None, facecolor="white")
    fig.savefig(OUT / "Fig1_conceptual_framework.tiff", dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    fig1()
