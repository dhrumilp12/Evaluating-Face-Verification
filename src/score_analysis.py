"""Offline ROC/DET and score summaries from validated saved verification scores."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from statistics import NormalDist
import subprocess
import time

import numpy as np

QUALITY = {
    'original': ('control', 0.0),
    'blur_sigma_1': ('blur', 1.0), 'blur_sigma_2': ('blur', 2.0), 'blur_sigma_3': ('blur', 3.0),
    'brightness_0.75': ('brightness', .75), 'brightness_0.5': ('brightness', .5),
    'brightness_0.25': ('brightness', .25),
}
LABELS = {'original': 'Original', 'resolution_80': 'Resolution 80 × 80',
          'resolution_40': 'Resolution 40 × 40', 'resolution_20': 'Resolution 20 × 20',
          **{k: f'Blur σ = {v[1]:g}' for k,v in QUALITY.items() if v[0]=='blur'},
          **{k: f'Brightness × {v[1]:.2f}' for k,v in QUALITY.items() if v[0]=='brightness'}}
INPUTS = ('baseline_summary.json', 'baseline_thresholds.json', 'baseline_scores.csv',
          'resolution_summary.json', 'resolution_predictions.csv',
          'quality_summary.json', 'quality_predictions.csv')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def empirical_curve(scores, labels):
    """Accept score >= threshold; group ties, include reject-all/accept-all endpoints."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels)
    if s.ndim != 1 or y.ndim != 1 or len(s)!=len(y) or not len(s):
        raise ValueError('Scores and labels must be nonempty matching 1D arrays')
    if not np.isfinite(s).all() or not np.isin(y,[0,1]).all() or set(y.tolist())!={0,1}:
        raise ValueError('Finite scores and both binary label classes are required')
    order = np.argsort(-s, kind='stable')
    ordered, labels_sorted = s[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(s)-1]
    tp = np.r_[0, np.cumsum(labels_sorted)[ends]]
    fp = np.r_[0, (1+ends)-np.cumsum(labels_sorted)[ends]]
    fmr, tpr = fp / (y==0).sum(), tp / (y==1).sum()
    fnmr = 1-tpr
    thresholds = np.r_[np.inf, ordered[ends]]
    auc = float(np.sum(np.diff(fmr)*(tpr[1:]+tpr[:-1])/2))
    d = fmr-fnmr
    hi = int(np.flatnonzero(d>=0)[0])
    if d[hi]==0:
        eer = float(fmr[hi])
    else:
        lo = hi-1
        weight = -d[lo]/(d[hi]-d[lo])
        eer = float(fmr[lo] + weight*(fmr[hi]-fmr[lo]))
    return {'threshold': thresholds, 'fmr': fmr, 'tpr': tpr, 'fnmr': fnmr,
            'auc': auc, 'eer_interpolated': eer}


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('pair_index','fold','label'):
            r[k]=int(r[k])
        if r['fold'] not in range(10) or r['label'] not in (0,1):
            raise ValueError('Invalid fold or label')
        r['cosine_similarity'] = float(r['cosine_similarity']) if r['cosine_similarity'] else None
        score=r['cosine_similarity']
        if r['status']=='scored':
            if score is None or not np.isfinite(score) or not -1<=score<=1:
                raise ValueError('Scored rows require finite cosine similarity in [-1,1]')
        elif score is not None:
            raise ValueError('Excluded rows must have no score')
    return rows


def fixed_metrics(rows, thresholds):
    folds=[]
    for fold in range(10):
        selected=[r for r in rows if r['fold']==fold and r['status']=='scored']
        y=np.array([r['label'] for r in selected])
        if set(y.tolist())!={0,1}:
            raise ValueError('Each fold needs both genuine and impostor scores')
        predicted=np.array([r['cosine_similarity']>=thresholds[str(fold)] for r in selected])
        fp=int(np.sum(predicted & (y==0)))
        fn=int(np.sum(~predicted & (y==1)))
        folds.append({'fold':fold, 'accuracy':float(np.mean(predicted==y)),
                      'fmr':fp/int(np.sum(y==0)), 'fnmr':fn/int(np.sum(y==1)),
                      'false_matches':fp, 'false_nonmatches':fn})
    out={f'mean_{k}':float(np.mean([f[k] for f in folds])) for k in ('accuracy','fmr','fnmr')}
    out.update({k:sum(f[k] for f in folds) for k in ('false_matches','false_nonmatches')})
    out['scored_pairs']=sum(r['status']=='scored' for r in rows)
    return out


def check_aggregate(rows, thresholds, expected, baseline=False):
    actual=fixed_metrics(rows,thresholds)
    for k in ('mean_accuracy','mean_fmr','mean_fnmr'):
        if not np.isclose(actual[k], expected[k], atol=1e-12, rtol=0):
            raise ValueError(f'Saved fixed-threshold metric mismatch: {k}')
    counts=expected['confusion_counts'] if baseline else expected
    for k in ('false_matches','false_nonmatches'):
        if actual[k]!=counts[k]: raise ValueError(f'Saved error count mismatch: {k}')
    if actual['scored_pairs']!=expected['scored_pairs']:
        raise ValueError('Saved scored-pair count mismatch')


def load_conditions(root):
    """Validate exported scores and original metadata without loading images/models."""
    p=Path(root)/'results/metrics'
    summaries={k:json.loads((p/f'{k}_summary.json').read_text()) for k in ('baseline','resolution','quality')}
    if any(s.get('status')!='completed' for s in summaries.values()):
        raise ValueError('Complete baseline, resolution, and quality experiments first')
    for family,names in {'baseline':('baseline_scores.csv','baseline_thresholds.json'),
                         'resolution':('resolution_predictions.csv',), 'quality':('quality_predictions.csv',)}.items():
        for name in names:
            if sha256(p/name)!=summaries[family]['output_sha256'][name]:
                raise ValueError(f'Input hash mismatch: {name}')
    baseline=summaries['baseline']
    saved=json.loads((p/'baseline_thresholds.json').read_text())
    thresholds={k:float(v) for k,v in saved['thresholds'].items()}
    if set(thresholds)!={str(n) for n in range(10)} or not np.isfinite(list(thresholds.values())).all():
        raise ValueError('Expected ten finite baseline fold thresholds')
    if saved['model_fingerprint']!=baseline['model_fingerprint']:
        raise ValueError('Baseline model records differ')
    for family in ('resolution','quality'):
        s=summaries[family]
        if s['model_fingerprint']!=baseline['model_fingerprint']:
            raise ValueError('Experiments used different model fingerprints')
        for key,name in (('baseline_summary_sha256','baseline_summary.json'),('baseline_thresholds_sha256','baseline_thresholds.json')):
            if s[key]!=sha256(p/name): raise ValueError('Experiments used different baseline records')
    original=read_rows(p/'baseline_scores.csv')
    index={r['pair_index']:r for r in original}
    if len(index)!=len(original) or len(original)!=baseline['requested_pairs']:
        raise ValueError('Baseline pair indices/count mismatch')
    if sorted(index)!=list(range(len(original))): raise ValueError('Baseline indices must be contiguous')
    eligible=sorted(r['pair_index'] for r in original if r['status']=='scored')
    if eligible!=sorted(saved['eligible_pair_indices']): raise ValueError('Baseline eligibility differs')
    check_aggregate(original,thresholds,baseline['aggregate'],baseline=True)
    groups={'original':sorted(original,key=lambda r:r['pair_index'])}
    for family in ('resolution','quality'):
        expected_keys={'160','80','40','20'} if family=='resolution' else set(QUALITY)
        key='resolution' if family=='resolution' else 'condition'
        rows=read_rows(p/f'{family}_predictions.csv')
        split={k:[] for k in expected_keys}
        for r in rows:
            if r[key] not in split: raise ValueError('Unexpected condition in predictions')
            split[r[key]].append(r)
        condition_summaries={str(r[key]):r for r in summaries[family]['conditions']}
        if set(condition_summaries)!=expected_keys or len(summaries[family]['conditions'])!=len(expected_keys):
            raise ValueError('Expected condition summaries are missing or duplicated')
        for condition,subset in split.items():
            if len(subset)!=len(original) or len({r['pair_index'] for r in subset})!=len(original):
                raise ValueError('Condition pair indices are missing or duplicated')
            subset.sort(key=lambda r:r['pair_index'])
            for r in subset:
                source=index.get(r['pair_index'])
                if source is None or any(r[k]!=source[k] for k in ('fold','reference','probe','label','status')):
                    raise ValueError('Condition changed pair metadata or eligibility')
                threshold=float(r['threshold'])
                if threshold!=thresholds[str(r['fold'])]: raise ValueError('Condition changed a baseline threshold')
                if family=='quality' and (r['family'],float(r['level']))!=QUALITY[condition]:
                    raise ValueError('Quality condition metadata differs')
                if r['status']=='scored':
                    prediction=int(r['cosine_similarity']>=threshold)
                    if int(r['prediction'])!=prediction or int(r['correct'])!=int(prediction==r['label']):
                        raise ValueError('Saved prediction differs from score/threshold')
                elif r['prediction'] or r['correct']:
                    raise ValueError('Excluded pair has a decision')
                if condition in ('160','original') and r['status']=='scored':
                    if not np.isclose(r['cosine_similarity'],source['cosine_similarity'],atol=1e-7,rtol=0):
                        raise ValueError('Original control differs from baseline scores')
            check_aggregate(subset,thresholds,condition_summaries[condition])
            if condition not in ('160','original'):
                groups[f'resolution_{condition}' if family=='resolution' else condition]=subset
    return {k:groups[k] for k in LABELS},thresholds


def analyze_groups(groups, thresholds):
    summaries,fold_rows,curve_rows,distribution_rows=[],[],[],[]
    for name,rows in groups.items():
        selected=[r for r in rows if r['status']=='scored']
        scores=np.array([r['cosine_similarity'] for r in selected])
        labels=np.array([r['label'] for r in selected])
        curve=empirical_curve(scores,labels)
        fixed=fixed_metrics(rows,thresholds)
        result={'condition':name, 'label':LABELS[name], 'requested_pairs':len(rows), **fixed,
                'excluded_pairs':len(rows)-len(selected), 'coverage':len(selected)/len(rows),
                'genuine_pairs':int(np.sum(labels==1)), 'impostor_pairs':int(np.sum(labels==0)),
                'pooled_auc':curve['auc'], 'pooled_eer_interpolated':curve['eer_interpolated']}
        local=[]
        for fold in range(10):
            subset=[r for r in selected if r['fold']==fold]
            c=empirical_curve([r['cosine_similarity'] for r in subset],[r['label'] for r in subset])
            record={'condition':name,'fold':fold,'scored_pairs':len(subset),'auc':c['auc'],'eer_interpolated':c['eer_interpolated']}
            local.append(record);fold_rows.append(record)
        for key in ('auc','eer_interpolated'):
            values=np.array([r[key] for r in local])
            result[f'mean_fold_{key}']=float(values.mean())
            result[f'sd_fold_{key}']=float(values.std(ddof=1))
        for label,kind in ((1,'genuine'),(0,'impostor')):
            s=scores[labels==label]
            record={'condition':name,'pair_type':kind,'count':len(s),'mean':float(s.mean()),
                    'sd':float(s.std(ddof=1)), 'p05':float(np.quantile(s,.05)),
                    'median':float(np.median(s)), 'p95':float(np.quantile(s,.95))}
            distribution_rows.append(record)
            result[f'mean_{kind}_score']=record['mean']
        summaries.append(result)
        for i,t in enumerate(curve['threshold']):
            curve_rows.append({'condition':name,'point':i,'threshold':None if np.isinf(t) else float(t),
                'decision_rule':'reject_all' if np.isinf(t) else 'score >= threshold',
                'fmr':float(curve['fmr'][i]),'tpr':float(curve['tpr'][i]),'fnmr':float(curve['fnmr'][i])})
    return summaries,fold_rows,curve_rows,distribution_rows


def make_plots(root, groups):
    import matplotlib.pyplot as plt
    figures=Path(root)/'results/figures'
    figures.mkdir(parents=True,exist_ok=True)
    families=[('Resolution',['original','resolution_80','resolution_40','resolution_20']),
              ('Blur',['original','blur_sigma_1','blur_sigma_2','blur_sigma_3']),
              ('Brightness',['original','brightness_0.75','brightness_0.5','brightness_0.25'])]
    curves={k:empirical_curve([r['cosine_similarity'] for r in v if r['status']=='scored'],
                             [r['label'] for r in v if r['status']=='scored']) for k,v in groups.items()}
    names=[]
    for plot_type in ('roc','det'):
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for ax,(family,conditions) in zip(axes,families):
            for name in conditions:
                c=curves[name]
                if plot_type=='roc':
                    ax.plot(c['fmr']*100,c['tpr']*100,label=LABELS[name],lw=1.5)
                else:
                    # True 0/1 rates are exported but omitted from probit rendering.
                    mask=(c['fmr']>0)&(c['fmr']<1)&(c['fnmr']>0)&(c['fnmr']<1)
                    normal=NormalDist()
                    x=[normal.inv_cdf(float(v)) for v in c['fmr'][mask]]
                    y=[normal.inv_cdf(float(v)) for v in c['fnmr'][mask]]
                    ax.plot(x,y,label=LABELS[name],lw=1.5)
            if plot_type=='roc':
                ax.set(xlim=(0,5),ylim=(75,100.1),xlabel='FMR (%)',ylabel='True match rate (%)')
            else:
                ticks=[.0001,.001,.01,.05,.2,.5]
                positions=[NormalDist().inv_cdf(v) for v in ticks]
                labels=[f'{100*v:g}' for v in ticks]
                ax.set_xticks(positions,labels);ax.set_yticks(positions,labels)
                ax.set(xlim=(positions[0],positions[-1]),ylim=(positions[0],positions[-1]),
                       xlabel='FMR (%) — probit scale',ylabel='FNMR (%) — probit scale')
            ax.set_title(family);ax.grid(alpha=.25);ax.legend(fontsize=8,loc='best')
        caption='Pooled descriptive ROC — low-FMR view (AUC uses full curve)' if plot_type=='roc' else 'Pooled descriptive DET — zero/one endpoints omitted from display'
        fig.suptitle(caption);fig.tight_layout()
        filename=f'analysis_{plot_type}.png';fig.savefig(figures/filename,dpi=160,bbox_inches='tight');plt.close(fig)
        names.append(filename)
    fig,axes=plt.subplots(5,2,figsize=(12,16),sharex=True,sharey=True)
    bins=np.linspace(-1,1,61)
    for ax,(name,rows) in zip(axes.flat,groups.items()):
        for label,kind,color in ((0,'Different person','#bd6947'),(1,'Same person','#28658a')):
            scores=[r['cosine_similarity'] for r in rows if r['status']=='scored' and r['label']==label]
            ax.hist(scores,bins=bins,density=True,histtype='step',lw=1.7,color=color,label=kind)
        ax.set_title(LABELS[name]);ax.set_xlabel('Cosine similarity');ax.set_ylabel('Density');ax.legend(fontsize=8)
    fig.suptitle('Score distributions — common bins and axes across conditions');fig.tight_layout()
    filename='analysis_score_distributions.png';fig.savefig(figures/filename,dpi=140,bbox_inches='tight');plt.close(fig)
    return names+[filename]


def run_analysis(root):
    root=Path(root).resolve();p=root/'results/metrics';start=time.perf_counter()
    report={'status':'started','run_at_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Descriptive pooled and per-fold ROC/AUC/interpolated EER from saved scores. No threshold calibration or model inference.',
            'eer_method':'Linear interpolation of adjacent empirical ROC points where FMR crosses FNMR; not a deployed threshold.',
            'det_display':'Normal quantiles of empirical FMR/FNMR; omit 0 and 1 endpoints only in plot; CSV rates unchanged.'}
    output=p/'analysis_summary.json'
    write_json(output,report)
    try:
        print('Validating saved scores, pair membership, thresholds, and control results ...',flush=True)
        groups,thresholds=load_conditions(root)
        report['input_sha256']={n:sha256(p/n) for n in INPUTS}
        report['source_sha256']=sha256(Path(__file__))
        report['python']=platform.python_version();report['numpy']=np.__version__
        import matplotlib
        report['matplotlib']=matplotlib.__version__
        try:
            report['git_head_before_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):report['git_head_before_commit']=None
        summary,folds,curves,distributions=analyze_groups(groups,thresholds)
        for name,rows in (('analysis_comparison.csv',summary),('analysis_folds.csv',folds),
                          ('analysis_curves.csv',curves),('analysis_score_statistics.csv',distributions)):
            write_csv(p/name,rows)
        figures=make_plots(root,groups)
        report['conditions']=summary
        report['output_sha256']={f'results/metrics/{n}':sha256(p/n) for n in
            ('analysis_comparison.csv','analysis_folds.csv','analysis_curves.csv','analysis_score_statistics.csv')}
        report['output_sha256'].update({f'results/figures/{n}':sha256(root/'results/figures'/n) for n in figures})
        report['status']='completed'
        for r in summary:
            print(f"{r['label']:24s} AUC {r['pooled_auc']:.6f} | interpolated EER {100*r['pooled_eer_interpolated']:.3f}%")
    except BaseException as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:
        report['elapsed_seconds']=round(time.perf_counter()-start,3);write_json(output,report)
    return report


if __name__=='__main__':
    run_analysis(Path(__file__).resolve().parents[1])
