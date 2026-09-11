"""Read-only project checks; optional JSON report. Does not run face inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import unquote

NOTEBOOKS = ('01_dataset_exploration.ipynb','02_face_embedding_pipeline.ipynb',
             '03_baseline_verification.ipynb','04_resolution_degradation.ipynb',
             '05_blur_brightness.ipynb','06_score_analysis.ipynb')
SOURCES = ('lfw_dataset.py','face_embedding.py','embedding_smoke_test.py','evaluation.py',
           'baseline_verification.py','degradation.py','resolution_experiment.py',
           'quality_degradation.py','quality_experiment.py','score_analysis.py')
TESTS = ('test_evaluation.py','test_face_embedding.py','test_resolution.py',
         'test_quality.py','test_score_analysis.py','test_repository_check.py')
REQUIRED = ('README.md','.gitignore','.gitattributes','requirements.txt','requirements-analysis.txt','LICENSE',
            'docs/dataset_research.md','docs/experiment_log.md','docs/reproducibility.md',
            'docs/repository_guide.md','scripts/check_repository.py',
            'results/metrics/dataset_summary.json','results/metrics/embedding_smoke_test.json',
            'results/metrics/embedding_smoke_pairs.csv','results/figures/lfw_identity_distribution.png',
            'results/figures/embedding_smoke_scores.png','results/figures/baseline_fold_metrics.png',
            'results/figures/resolution_comparison.png','results/figures/quality_comparison.png',
            'results/figures/quality_examples.png',
            *(f'src/{n}' for n in SOURCES),*(f'tests/{n}' for n in TESTS),
            *(f'notebooks/{n}' for n in NOTEBOOKS))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def pins(path):
    result={}
    for line in Path(path).read_text().splitlines():
        line=line.split('#',1)[0].strip()
        if not line:continue
        match=re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s]+)',line)
        if not match:raise ValueError(f'Expected an exact dependency pin: {line}')
        result[match[1]]=match[2]
    return result


def broken_links(root, document):
    root=Path(root).resolve();document=Path(document)
    text=re.sub(r'```.*?```','',document.read_text(encoding='utf-8'),flags=re.S)
    broken=[]
    for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)',text):
        if target.startswith(('#','http://','https://','mailto:')):continue
        target=unquote(target.split('#',1)[0])
        path=(document.parent/target).resolve()
        if not path.is_relative_to(root) or not path.exists():broken.append(target)
    return broken


def notebook_errors(path):
    notebook=json.loads(Path(path).read_text())
    if notebook.get('nbformat')!=4 or not isinstance(notebook.get('cells'),list):
        raise ValueError('Expected a version-4 notebook')
    code=[c for c in notebook['cells'] if c['cell_type']=='code']
    if not code:raise ValueError('Notebook contains no code cells')
    errors=[]
    for i,c in enumerate(code,1):
        if any(o.get('output_type')=='error' for o in c.get('outputs',[])):
            errors.append(f'code cell {i} has a saved error')
    unexecuted=sum(c.get('execution_count') is None for c in code)
    return errors,unexecuted,len(code)


def check(root, profile='full', skip_git=False):
    root=Path(root).resolve();checks=[]
    report={'run_at_utc':datetime.now(timezone.utc).isoformat(),'profile':profile,
            'git_check_requested':not skip_git,'python':platform.python_version(),
            'platform':platform.platform(),'checks':checks,
            'scope':'File organization, saved outputs, dependencies, and Git when requested. No new inference, network downloads, or archive creation.'}
    def add(name,status,detail):checks.append({'check':name,'status':status,'detail':detail})
    def perform(name,fn):
        try:detail=fn();add(name,'pass',detail)
        except Exception as e:add(name,'fail',f'{type(e).__name__}: {e}')
    def need(condition,message):
        if not condition:raise ValueError(message)
    def structure():
        missing=[n for n in REQUIRED if not (root/n).is_file()]
        need(not missing,'Missing: '+', '.join(missing))
        return f'{len(REQUIRED)} required code/document/notebook files present'
    perform('required_files',structure)
    for document in [root/'README.md',*sorted((root/'docs').glob('*.md'))]:
        def links(document=document):
            broken=broken_links(root,document);need(not broken,'Broken local targets: '+', '.join(broken))
            return 'Relative Markdown file targets exist; external URLs and section anchors are not checked'
        perform('links:'+document.relative_to(root).as_posix(),links)
    for name in NOTEBOOKS:
        def notebook(name=name):
            errors,unexecuted,count=notebook_errors(root/'notebooks'/name)
            need(not errors,'; '.join(errors))
            need(unexecuted==0,f'{unexecuted}/{count} code cells have no recorded execution count')
            return f'{count} code cells have execution counts and no saved errors; this does not establish freshness'
        perform('notebook:'+name,notebook)
    def requirement_files():
        full=pins(root/'requirements.txt');analysis=pins(root/'requirements-analysis.txt')
        need(set(analysis)=={'numpy','matplotlib'},'Analysis CLI must pin numpy and matplotlib only')
        need(all(full.get(k)==v for k,v in analysis.items()),'Analysis pins must match full project pins')
        return 'Analysis pins match full requirements; indirect dependencies are not locked'
    perform('requirements_consistency',requirement_files)
    if profile=='files':add('installed_dependencies','skipped','File-validation profile: installed versions are not compared to pins')
    else:
        requirement='requirements.txt' if profile=='full' else 'requirements-analysis.txt'
        def environment():
            expected=pins(root/requirement);actual={}
            for name,version in expected.items():
                actual[name]=importlib.metadata.version(name)
                need(actual[name]==version,f'{name}: installed {actual[name]}, required {version}')
            if profile=='full':need(platform.python_version()=='3.13.7','Full inference reference environment is Python 3.13.7; document and rerun the baseline for another environment')
            report['checked_versions']=actual
            return f'{len(actual)} installed direct dependencies match {requirement}'
        perform('installed_dependencies',environment)
    metric_dir=root/'results/metrics'
    def early_reports():
        dataset=json.loads((metric_dir/'dataset_summary.json').read_text())
        smoke=json.loads((metric_dir/'embedding_smoke_test.json').read_text())
        need(dataset.get('all_checks_passed') is True and bool(dataset.get('checks'))
             and all(dataset['checks'].values()),'Saved dataset checks are incomplete/failed')
        need(smoke.get('status')=='completed' and smoke.get('checks_passed') is True
             and bool(smoke.get('checks')) and all(smoke['checks'].values()),'Saved embedding smoke checks are incomplete/failed')
        return 'Historical dataset and embedding smoke reports record passing checks; no image revalidation performed'
    perform('initial_stage_reports',early_reports)
    for family in ('baseline','resolution','quality','analysis'):
        def outputs(family=family):
            summary_path=metric_dir/f'{family}_summary.json'
            summary=json.loads(summary_path.read_text())
            need(summary.get('status')=='completed','Summary is not completed')
            recorded=summary['output_sha256'];need(bool(recorded),'No recorded output hashes')
            for name,expected in recorded.items():
                path=((root if family=='analysis' else metric_dir)/name).resolve()
                need(path.is_relative_to(root),'Output path leaves repository')
                need(digest(path)==expected,f'Output hash mismatch: {name}')
            if family=='analysis':
                for name,expected in summary['input_sha256'].items():
                    path=(metric_dir/name).resolve();need(path.is_relative_to(root),'Input path leaves repository')
                    need(digest(path)==expected,f'Analysis input hash mismatch: {name}')
                need(digest(root/'src/score_analysis.py')==summary['source_sha256'],'Analysis source differs from its recorded run')
            return f'Completed summary; {len(recorded)} exported file hashes match'
        perform('outputs:'+family,outputs)
    def numerical_results():
        sys.path.insert(0,str(root/'src'))
        # Use the project's tested loader: no model imports, downloads, or writes.
        import score_analysis
        need(Path(score_analysis.__file__).resolve()==(root/'src/score_analysis.py').resolve(),'Another project score_analysis module is already loaded')
        groups,thresholds=score_analysis.load_conditions(root)
        computed,*_=score_analysis.analyze_groups(groups,thresholds)
        saved=json.loads((metric_dir/'analysis_summary.json').read_text())['conditions']
        need(len(computed)==len(saved)==10,'Expected ten analysis conditions')
        by_name={r['condition']:r for r in saved};need(len(by_name)==10,'Duplicate summary condition')
        import numpy as np
        for r in computed:
            expected=by_name[r['condition']]
            for key,value in r.items():
                if isinstance(value,(int,float)):
                    need(np.isclose(value,expected[key],atol=1e-12,rtol=0),f'Analysis metric mismatch: {r["condition"]}/{key}')
                else:need(value==expected[key],f'Analysis label mismatch: {key}')
        report['numerical_summary']={'conditions':10,'scored_pairs_per_condition':computed[0]['scored_pairs'],
            'baseline_false_matches':computed[0]['false_matches'],'baseline_false_nonmatches':computed[0]['false_nonmatches']}
        return 'Ten conditions reproduce fixed-threshold and AUC/EER summaries from saved scores'
    perform('saved_score_consistency',numerical_results)
    if skip_git:add('git_history','skipped','Explicit --skip-git: this file snapshot does not verify submission history')
    else:
        def git_history():
            def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.STDOUT).strip()
            need(Path(git('rev-parse','--show-toplevel')).resolve()==root,'Project root differs from Git root')
            count=int(git('rev-list','--count','HEAD'));need(count>0,'No commits')
            head=git('rev-parse','HEAD');git('fsck','--full')
            status=git('status','--porcelain=v1')
            report['git']={'head':head,'commit_count':count,'working_tree_clean':not bool(status)}
            tracked=git('ls-files','-z').split('\0')
            unwanted=[p for p in tracked if p.startswith(('data/','models/','.venv/','venv/')) or p.endswith('.zip')]
            need(not unwanted,'Large/local artifacts are tracked: '+', '.join(unwanted))
            return f'Git objects verified; {count} commits; working tree '+('clean' if not status else 'has changes (expected before this commit)')
        perform('git_history',git_history)
    report['passed']=sum(c['status']=='pass' for c in checks)
    report['failed']=sum(c['status']=='fail' for c in checks)
    report['skipped']=sum(c['status']=='skipped' for c in checks)
    report['status']='passed' if not report['failed'] else 'failed'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--profile',choices=('full','analysis','files'),default='full')
    parser.add_argument('--skip-git',action='store_true',help='For a file snapshot only; does not verify Git history')
    parser.add_argument('--output',type=Path,help='Optional JSON report; relative paths resolve inside project root')
    args=parser.parse_args();root=args.root.resolve()
    output=None
    if args.output:
        output=(root/args.output).resolve()
        if output!=root/'results/metrics/repository_check.json':
            parser.error('--output must resolve to results/metrics/repository_check.json (protects existing artifacts)')
    report=check(root,args.profile,args.skip_git)
    for c in report['checks']:print(f"{c['status'].upper():7s} {c['check']}: {c['detail']}")
    print(f"\nRepository check: {report['status'].upper()} | {report['passed']} passed | {report['failed']} failed | {report['skipped']} skipped")
    if output:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(f'Report: {output.relative_to(root)}')
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
