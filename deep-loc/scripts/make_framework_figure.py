"""
Draw a reference version of the framework overview figure (Fig. 1 of the paper).
"""

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
BLUE_FILL, ORANGE_FILL, AQUA_FILL, GRAY_FILL = '#dbe9fb', '#fde3d8', '#d6f1e6', '#eeedea'
INK, MUTED = '#0b0b0b', '#898781'

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 7.5,
                     'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Draw the framework overview figure')
    parser.add_argument('--output', type=str,
                       default=os.path.join(os.path.dirname(ROOT), 'paper', 'figures', 'framework_reference'),
                       help='Output path without extension (.pdf and .png are written)')
    return parser.parse_args()


def box(ax, x, y, w, h, text, fill, edge, fontsize=7.5, dashed=False, color=INK, weight='normal'):
    """Rounded box with centered text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', facecolor=fill,
                                edgecolor=edge, linewidth=0.9, linestyle='--' if dashed else '-'))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, color=color,
            fontweight=weight, linespacing=1.15)


def arrow(ax, start, end, color=INK, dashed=False):
    """Arrow between two points."""
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(
        arrowstyle='-|>', color=color, linewidth=0.9, linestyle='--' if dashed else '-',
        shrinkA=0, shrinkB=0, mutation_scale=7))


def stage_title(ax, x, w, text):
    """Title above a stage of the pipeline."""
    ax.text(x + w / 2, 43.2, text, ha='center', va='center', fontsize=8, fontweight='bold', color=INK)


def main():
    """Main function."""
    args = parse_args()
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.set_xlim(0, 160)
    ax.set_ylim(-5, 46)
    ax.axis('off')

    # Rows of the per-AP pipeline: AP 1, an undetected AP, AP Na
    rows = [(32.5, r'1', True), (20, r'\ell', False), (6.5, r'N_a', True)]
    row_h = 7

    # ---- Stage 1: deployment scenario
    stage_title(ax, 0, 27, 'Asynchronous APs')
    ax.add_patch(Rectangle((2, 5), 23, 34, facecolor='white', edgecolor=MUTED, linewidth=0.9))
    ax.plot([8, 8], [5, 24], color=MUTED, linewidth=1.2)  # partition
    device = (14, 19)
    aps = [((3.6, 37.2), True), ((23.4, 37.2), True), ((3.6, 6.8), False), ((23.4, 6.8), True)]
    for (x, y), detected in aps:
        color = INK if detected else MUTED
        ax.plot([x, device[0]], [y, device[1]], color=color, linewidth=0.7,
                linestyle='-' if detected else ':')
        ax.add_patch(Polygon([(x - 1.3, y - 1.1), (x + 1.3, y - 1.1), (x, y + 1.4)], closed=True,
                             facecolor=ORANGE if detected else GRAY_FILL, edgecolor=color, linewidth=0.7,
                             zorder=3))
    ax.add_patch(Circle(device, 1.2, facecolor=BLUE, edgecolor=INK, linewidth=0.7, zorder=3))
    ax.text(16, 19, r'device $\mathbf{x}$', ha='left', va='center', fontsize=6.5)
    ax.text(14.3, 8.6, 'no path: packet\nnot detected', ha='center', va='center', fontsize=6, color=MUTED,
            linespacing=1.1)
    ax.text(13.5, -0.2, r'known AP positions $\mathbf{p}_\ell$' '\n'
            r'independent clocks: SCO/CFO $\epsilon_\ell$,' '\n'
            r'phase noise $\theta_\ell[n]$, unknown delay $\tau_\ell$',
            ha='center', va='center', fontsize=6.5, linespacing=1.2)

    # ---- Stage 2: per-packet receiver processing
    stage_title(ax, 30, 20, 'Per-packet receiver')
    box(ax, 30, 6.5, 20, 33, 'Packet detection\nCFO correction\nSymbol timing\nChannel\nestimation\n'
        r'$\mathbf{H}_\ell[k]$' '\n\nIFFT\n' r'$\rightarrow$ CIR $\widehat{\mathbf{H}}_\ell[\tau]$',
        'white', MUTED, fontsize=7)
    arrow(ax, (25.5, 22.5), (30, 22.5))

    # ---- Stage 3: per-AP encoding
    stage_title(ax, 54, 54, 'Shared per-AP encoding')
    for y, index, detected in rows:
        edge = INK if detected else MUTED
        text_color = INK if detected else MUTED
        mid = y + row_h / 2
        arrow(ax, (50, mid), (54, mid), color=edge, dashed=not detected)
        # Complex CIR as real and imaginary halves
        ax.add_patch(Rectangle((54, y), 5.5, row_h, facecolor=BLUE_FILL if detected else GRAY_FILL,
                               edgecolor=edge, linewidth=0.8, linestyle='-' if detected else '--'))
        ax.add_patch(Rectangle((59.5, y), 5.5, row_h, facecolor=AQUA_FILL if detected else GRAY_FILL,
                               edgecolor=edge, linewidth=0.8, linestyle='-' if detected else '--'))
        ax.text(56.75, mid, 'Re', ha='center', va='center', fontsize=6.5, color=text_color)
        ax.text(62.25, mid, 'Im', ha='center', va='center', fontsize=6.5, color=text_color)
        ax.text(59.5, y + row_h + 1.4, rf'$\mathbf{{X}}_{{{index}}}$' + ('' if detected else ' (all zero)'),
                ha='center', va='center', fontsize=7, color=text_color)
        arrow(ax, (65, mid), (69, mid), color=edge, dashed=not detected)
        box(ax, 69, y, 18, row_h, 'CNN encoder\n' r'$f_\theta$ + pooling', BLUE_FILL if detected else GRAY_FILL,
            edge, fontsize=7, dashed=not detected, color=text_color)
        arrow(ax, (87, mid), (92.6, mid), color=edge, dashed=not detected)
        # Addition of the AP-position embedding
        ax.add_patch(Circle((94, mid), 1.4, facecolor='white', edgecolor=edge, linewidth=0.8,
                            linestyle='-' if detected else '--'))
        ax.text(94, mid, '+', ha='center', va='center', fontsize=8, color=text_color)
        box(ax, 88.5, y - 5.8, 11, 3.8, rf'$g_\phi(\mathbf{{p}}_{{{index}}})$',
            ORANGE_FILL if detected else GRAY_FILL, ORANGE if detected else MUTED, fontsize=6.5,
            dashed=not detected, color=text_color)
        arrow(ax, (94, y - 2), (94, mid - 1.4), color=ORANGE if detected else MUTED, dashed=not detected)
        arrow(ax, (95.4, mid), (108, mid), color=edge, dashed=not detected)
        ax.text(101.8, mid + 1.6, rf'$\mathbf{{e}}_{{{index}}}$', ha='center', va='center', fontsize=7,
                color=text_color)
        if not detected:
            ax.text(101.8, mid - 1.7, 'masked', ha='center', va='center', fontsize=6.5, color=text_color)
    ax.text(78, 17, 'shared weights', ha='center', va='center', fontsize=6.5, color=BLUE)
    ax.text(59.5, 3.9, r'$N_{\mathrm{tap}} \times 2M$', ha='center', va='center', fontsize=6.5)

    # ---- Stage 4: cross-AP fusion
    stage_title(ax, 108, 22, 'Cross-AP fusion')
    box(ax, 108, 6.5, 22, 33, 'Transformer\nencoder\n\nself-attention\nacross AP tokens\n\n'
        'no positional\nencoding\n\nundetected APs\nmasked', BLUE_FILL, BLUE, fontsize=7)

    # ---- Stage 5: aggregation and regression
    stage_title(ax, 133, 26, '3D position')
    arrow(ax, (130, 23), (134, 23))
    box(ax, 134, 17.5, 10.5, 11, 'Masked\nmean\n' r'$\mathbf{z}$', 'white', INK, fontsize=7)
    arrow(ax, (144.5, 23), (148, 23))
    box(ax, 148, 17.5, 10.5, 11, 'MLP\nhead\n' r'$\hat{\mathbf{x}} \in \mathbb{R}^3$', AQUA_FILL, AQUA, fontsize=7)
    ax.text(146.2, 34.5, 'invariant to the order\nof the APs, not to\ntheir positions', ha='center',
            va='center', fontsize=6.5, linespacing=1.15)
    ax.text(146.2, 10.5, 'variable number\nof available APs', ha='center', va='center', fontsize=6.5,
            linespacing=1.15)

    # The figure is drawn 9.6 in wide and printed at the 7.16 in text width,
    # so the text is enlarged to stay readable after scaling
    for text in ax.texts:
        text.set_fontsize(text.get_fontsize() * 1.2)

    fig.savefig(args.output + '.pdf')
    fig.savefig(args.output + '.png', dpi=300)
    plt.close(fig)
    print(f"Saved {args.output}.pdf and {args.output}.png")


if __name__ == '__main__':
    main()
