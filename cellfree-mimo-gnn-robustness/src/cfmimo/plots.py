"""Rebuild figures from per_scenario.csv and training_history.csv."""
import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',default='results')
    p.add_argument('--figures',default='figures')
    args=p.parse_args()
    res, figs=Path(args.results),Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    with (res/'per_scenario.csv').open() as f:
        rows=list(csv.DictReader(f))
    with (res/'training_history.csv').open() as f:
        hist=list(csv.DictReader(f))
    methods=['uniform','gain_proportional','bipartite_gnn','iterative_reference']
    cases=['clean','25pct_ap_outage','bounded_random_csi_error','whitebox_fgsm_csi_error']
    data=defaultdict(list)
    for x in rows: data[(x['condition'],x['method'])].append(float(x['worst_user_se']))
    fig, ax=plt.subplots(figsize=(9.3,4.6))
    xpos=np.arange(len(cases)); width=.21
    labels={'clean':'Clean','25pct_ap_outage':'25% AP outage', 'bounded_random_csi_error':'Random CSI error', 'whitebox_fgsm_csi_error':'FGSM CSI error'}
    for i,method in enumerate(methods):
        means=[np.mean(data[(case,method)]) for case in cases]
        ax.bar(xpos+(i-1.5)*width,means,width,label={'uniform':'Equal per AP','gain_proportional':'Gain-proportional','bipartite_gnn':'Bipartite GNN','iterative_reference':'Iterative reference'}[method])
    ax.set_xticks(xpos,[labels[c] for c in cases]); ax.set_ylabel('Mean worst-user SE (bit/s/Hz)');
    ax.set_title('Toy-scale cell-free network · 16 APs / 4 UEs / 192 test layouts'); ax.legend(fontsize=8)
    ax.grid(axis='y',alpha=.23); fig.tight_layout(); fig.savefig(figs/'benchmark.png',dpi=190); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.4))
    for name,field in [('Training','train_loss'),('Validation','validation_loss')]:
        ax.plot([int(x['epoch']) for x in hist],[float(x[field]) for x in hist],label=name)
    ax.set_xlabel('Epoch');ax.set_ylabel('Negative soft-min SE objective');ax.set_title('Training history (training / held-out validation)');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(figs/'learning_curve.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.4))
    for method in methods:
        vals=np.sort(np.array(data[('clean',method)]));
        ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),label={'uniform':'Equal per AP','gain_proportional':'Gain-proportional','bipartite_gnn':'Bipartite GNN','iterative_reference':'Iterative reference'}[method])
    ax.set_xlabel('Worst-user SE (bit/s/Hz)');ax.set_ylabel('Empirical CDF'); ax.set_title('Clean test scenarios'); ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(figs/'cdf_clean.png',dpi=190);plt.close(fig)
    print('Wrote figures to',figs)

if __name__=='__main__':main()
