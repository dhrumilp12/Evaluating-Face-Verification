"""Analytic ROC cases and structural validation of exported prediction files."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import score_analysis as a


class CurveTests(unittest.TestCase):
    def test_perfect_reversed_and_all_tied(self):
        for scores,auc,eer in (([.9,.8,.2,.1],1,0),([.1,.2,.8,.9],0,1),([.5]*4,.5,.5)):
            c=a.empirical_curve(scores,[1,1,0,0])
            self.assertAlmostEqual(c['auc'],auc)
            self.assertAlmostEqual(c['eer_interpolated'],eer)
            self.assertEqual((c['fmr'][0],c['tpr'][0]),(0,0))
            self.assertEqual((c['fmr'][-1],c['tpr'][-1]),(1,1))

    def test_tied_scores_permutation_and_pairwise_auc(self):
        scores=np.array([.8,.8,.4,.2,.2,.1]);labels=np.array([1,0,1,1,0,0])
        c=a.empirical_curve(scores,labels)
        differences=scores[labels==1,None]-scores[labels==0]
        expected=np.mean((differences>0)+.5*(differences==0))
        self.assertAlmostEqual(c['auc'],expected)
        for order in (np.arange(6)[::-1],np.array([2,0,5,3,1,4])):
            other=a.empirical_curve(scores[order],labels[order])
            for key in ('fmr','fnmr','threshold'):
                np.testing.assert_array_equal(c[key],other[key])
        for i,threshold in enumerate(c['threshold']):
            accepted=scores>=threshold
            self.assertAlmostEqual(c['fmr'][i],np.mean(accepted[labels==0]))
            self.assertAlmostEqual(c['fnmr'][i],np.mean(~accepted[labels==1]))

    def test_invalid_inputs(self):
        for scores,labels in (([],[]),([1],[1]),([1,0],[1,2]),([np.nan,0],[1,0]),([[1,0]],[[1,0]]),([1,0],[1])):
            with self.assertRaises(ValueError):a.empirical_curve(scores,labels)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.p=self.root/'results/metrics'
        self.thresholds={str(i):.5 for i in range(10)}
        self.rows=[]
        for fold in range(10):
            for label,score in ((1,.8),(0,.2)):
                self.rows.append(dict(pair_index=len(self.rows),fold=fold,reference='A',probe='B' if label else 'C',
                    label=label,status='scored',cosine_similarity=score))
        self.rows.append(dict(pair_index=20,fold=0,reference='missing',probe='B',label=1,status='preprocessing_failed',cosine_similarity=None))
        a.write_csv(self.p/'baseline_scores.csv',self.rows)
        saved={'model_fingerprint':{'synthetic':True},'thresholds':self.thresholds,'eligible_pair_indices':list(range(20))}
        a.write_json(self.p/'baseline_thresholds.json',saved)
        aggregate=a.fixed_metrics(self.rows,self.thresholds)
        aggregate['confusion_counts']={'false_matches':0,'false_nonmatches':0}
        baseline={'status':'completed','model_fingerprint':saved['model_fingerprint'],'requested_pairs':21,
            'aggregate':aggregate,'output_sha256':{n:a.sha256(self.p/n) for n in ('baseline_scores.csv','baseline_thresholds.json')}}
        a.write_json(self.p/'baseline_summary.json',baseline)
        for family,conditions in (('resolution',[160,80,40,20]),('quality',list(a.QUALITY))):
            predictions=[];summaries=[]
            key='resolution' if family=='resolution' else 'condition'
            for condition in conditions:
                for row in self.rows:
                    r=dict(row);score=r['cosine_similarity']
                    prediction=int(score>=.5) if score is not None else None
                    r.update({key:condition,'threshold':.5,'prediction':prediction,
                              'correct':int(prediction==r['label']) if prediction is not None else None})
                    if family=='quality':r.update(family=a.QUALITY[condition][0],level=a.QUALITY[condition][1])
                    predictions.append(r)
                summaries.append({key:condition,**a.fixed_metrics(self.rows,self.thresholds)})
            filename=f'{family}_predictions.csv';a.write_csv(self.p/filename,predictions)
            report={'status':'completed','model_fingerprint':saved['model_fingerprint'],
                'baseline_summary_sha256':a.sha256(self.p/'baseline_summary.json'),
                'baseline_thresholds_sha256':a.sha256(self.p/'baseline_thresholds.json'),
                'conditions':summaries,'output_sha256':{filename:a.sha256(self.p/filename)}}
            a.write_json(self.p/f'{family}_summary.json',report)

    def mutate_predictions(self,transform):
        filename='quality_predictions.csv'
        with (self.p/filename).open() as f:rows=list(csv.DictReader(f))
        transform(rows)
        a.write_csv(self.p/filename,rows)
        # Rehash to test structural checks independently of the file-integrity check.
        p=self.p/'quality_summary.json';s=json.loads(p.read_text())
        s['output_sha256'][filename]=a.sha256(self.p/filename);a.write_json(p,s)

    def test_full_loader_analysis_and_no_input_writes(self):
        before={n:a.sha256(self.p/n) for n in a.INPUTS}
        groups,thresholds=a.load_conditions(self.root)
        self.assertEqual(list(groups),list(a.LABELS))
        summaries,folds,curves,stats=a.analyze_groups(groups,thresholds)
        self.assertEqual((len(summaries),len(folds),len(stats)),(10,100,20))
        for s in summaries:
            self.assertEqual(s['scored_pairs'],20);self.assertEqual(s['excluded_pairs'],1)
            self.assertEqual(s['pooled_auc'],1);self.assertEqual(s['pooled_eer_interpolated'],0)
        self.assertEqual(before,{n:a.sha256(self.p/n) for n in a.INPUTS})
        self.assertTrue(all(r['threshold'] is None for r in curves if r['decision_rule']=='reject_all'))

    def test_corrupt_file_rejected(self):
        with (self.p/'quality_predictions.csv').open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):a.load_conditions(self.root)

    def test_changed_pair_rejected_even_with_updated_hash(self):
        self.mutate_predictions(lambda rows:rows[0].update(probe='wrong'))
        with self.assertRaisesRegex(ValueError,'pair metadata'):a.load_conditions(self.root)

    def test_changed_threshold_rejected(self):
        self.mutate_predictions(lambda rows:rows[0].update(threshold='.6'))
        with self.assertRaisesRegex(ValueError,'baseline threshold'):a.load_conditions(self.root)

    def test_missing_pair_rejected(self):
        self.mutate_predictions(lambda rows:rows.pop())
        with self.assertRaisesRegex(ValueError,'indices'):a.load_conditions(self.root)

    def test_control_score_change_rejected(self):
        self.mutate_predictions(lambda rows:rows[0].update(cosine_similarity='.7'))
        with self.assertRaisesRegex(ValueError,'control differs'):a.load_conditions(self.root)


if __name__=='__main__':unittest.main()
