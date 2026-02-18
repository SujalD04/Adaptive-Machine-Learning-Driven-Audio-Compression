import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE = Path('c:/Users/sujal/adaptive_opus')
CSV = BASE / 'ood_test_results.csv'
OUTDIR = BASE / 'reports'
OUTDIR.mkdir(exist_ok=True)


def plot_mean_mos(df):
    means = df.groupby('controller')['mos'].mean()
    means = means.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))
    means.plot(kind='bar', ax=ax, color='C0')
    ax.set_ylabel('Mean MOS')
    ax.set_title('Mean MOS by Controller (OOD tests)')
    plt.tight_layout()
    out = OUTDIR / 'mean_mos_by_controller.png'
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_mos_vs_loss(df):
    import numpy as np
    controllers = df['controller'].unique()
    fig, ax = plt.subplots(figsize=(8,5))
    for c in controllers:
        sub = df[df['controller']==c]
        # compute mean at each loss
        grp = sub.groupby('packet_loss_perc')['mos'].mean().sort_index()
        ax.plot(grp.index.astype(float), grp.values, marker='o', label=c)
    ax.set_xlabel('Packet loss (%)')
    ax.set_ylabel('Mean MOS')
    ax.set_title('MOS vs Packet Loss by Controller')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    out = OUTDIR / 'mos_vs_loss.png'
    fig.savefig(out)
    plt.close(fig)
    return out


def top2_per_input(df):
    rows = []
    for inp, g in df.groupby('input_file'):
        best = g.sort_values('mos', ascending=False).head(2)
        for _, r in best.iterrows():
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    if not CSV.exists():
        print('No results CSV found at', CSV)
        return
    df = pd.read_csv(CSV)
    # convert mos to numeric (coerce)
    df['mos'] = pd.to_numeric(df['mos'], errors='coerce')
    # Plotting
    p1 = plot_mean_mos(df)
    p2 = plot_mos_vs_loss(df)

    top2 = top2_per_input(df)

    # Write report
    rpt = OUTDIR / 'OOD_REPORT.md'
    with open(rpt, 'w', encoding='utf-8') as f:
        f.write('# Out-of-Distribution Test Report\n\n')
        f.write('Summary of OOD sweep results. Generated plots below.\n\n')
        f.write('## Mean MOS by Controller\n')
        f.write(f'![Mean MOS]({p1.name})\n\n')
        f.write('## MOS vs Packet Loss\n')
        f.write(f'![MOS vs Loss]({p2.name})\n\n')
        f.write('## Top-2 runs per input file\n')
        f.write(top2[['input_file','controller','packet_loss_perc','mos','bitrate','frame_size','complexity','use_fec']].to_markdown(index=False))

    print('Report generated:', rpt)
    print('Plots:', p1, p2)


if __name__ == '__main__':
    main()
