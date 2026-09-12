"""Package committed project files and the real .git directory; verify extraction."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.STDOUT).decode().strip()


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def snapshot(root):
    """Require a standalone, clean repository whose history is self-contained."""
    root=Path(root).resolve()
    if Path(git(root,'rev-parse','--show-toplevel')).resolve()!=root:
        raise ValueError('Run from the repository root')
    if not (root/'.git').is_dir() or (root/'.git').is_symlink():
        raise ValueError('A standalone repository with a real .git directory is required; linked worktrees are unsupported')
    if git(root,'rev-parse','--is-shallow-repository')!='false':
        raise ValueError('A complete Git history is required; this clone is shallow')
    if list((root/'.git/objects/pack').glob('*.promisor')):
        raise ValueError('Partial clones with promised objects are unsupported; use a complete repository')
    for name in ('alternates','http-alternates'):
        p=root/'.git/objects/info'/name
        if p.exists() and p.read_bytes().strip():
            raise ValueError('Git object alternates are unsupported; use a self-contained repository')
    if git(root,'status','--porcelain=v1','--untracked-files=all'):
        raise ValueError('Commit intended changes and resolve non-ignored untracked files before packaging; no files are automatically removed')
    git(root,'fsck','--full')
    paths=subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0')
    paths=[p for p in paths if p]
    for path in paths:
        pure=PurePosixPath(path)
        if pure.is_absolute() or '..' in pure.parts or not (root/path).is_file() or (root/path).is_symlink():
            raise ValueError(f'Unsupported tracked path: {path}')
        if path.startswith(('data/','models/','.venv/','venv/')) or path.endswith('.zip'):
            raise ValueError(f'Local/generated artifact is tracked: {path}')
    for parent,dirs,files in os.walk(root/'.git',followlinks=False):
        for name in dirs+files:
            p=Path(parent)/name
            if p.is_symlink():raise ValueError('Symlinks within .git are unsupported')
        for name in files:
            p=Path(parent)/name
            if not p.is_file() or name.endswith('.lock'):
                raise ValueError(f'Git is busy or contains an unsupported file: {p.relative_to(root)}')
            paths.append(p.relative_to(root).as_posix())
    return {'head':git(root,'rev-parse','HEAD'),'commit_count':int(git(root,'rev-list','--count','HEAD')),
            'paths':sorted(paths)}


def verify_archive(path, expected, member_hashes, prefix):
    """Extract to an isolated temporary directory and check bytes plus Git history."""
    with tempfile.TemporaryDirectory(prefix='biometrics-zip-check-') as temporary:
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:raise ValueError('ZIP CRC integrity check failed')
            if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate ZIP members')
            if set(z.namelist())!=set(member_hashes):raise ValueError('ZIP member inventory mismatch')
            for info in z.infolist():
                pure=PurePosixPath(info.filename)
                if pure.is_absolute() or '..' in pure.parts or not pure.parts or pure.parts[0]!=prefix:
                    raise ValueError('Unsafe or unexpected ZIP member')
                if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Symlink ZIP members are unsupported')
                target=Path(temporary).joinpath(*pure.parts)
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
                mode=(info.external_attr>>16)&0o777
                if mode:target.chmod(mode)
                if sha256(target)!=member_hashes[info.filename]:raise ValueError('Extracted file hash mismatch')
        extracted=Path(temporary)/prefix
        if not (extracted/'.git/HEAD').is_file():raise ValueError('Archive lacks .git/HEAD')
        if Path(git(extracted,'rev-parse','--show-toplevel')).resolve()!=extracted.resolve():
            raise ValueError('Extracted Git configuration points outside its working tree')
        git(extracted,'fsck','--full')
        if git(extracted,'rev-parse','HEAD')!=expected['head']:raise ValueError('Archived HEAD differs')
        if int(git(extracted,'rev-list','--count','HEAD'))!=expected['commit_count']:
            raise ValueError('Archived commit count differs')
        if git(extracted,'status','--porcelain=v1','--untracked-files=all'):
            raise ValueError('Extracted repository is not clean')
    return {'zip_crc':'passed','extracted_file_hashes':'passed','git_fsck':'passed',
            'head_matches':True,'commit_count_matches':True,'extracted_working_tree_clean':True}


def build_archive(root, output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.is_relative_to(root):raise ValueError('Save the submission ZIP outside the repository')
    if output.suffix.lower()!='.zip':raise ValueError('Output must end in .zip')
    receipt=output.with_suffix('.verification.json')
    if output.exists() or receipt.exists():raise FileExistsError('ZIP or verification receipt already exists; choose a new filename')
    if not output.parent.is_dir():raise ValueError('Output parent directory must exist')
    before=snapshot(root)
    prefix=root.name
    if not prefix or '\\' in prefix:raise ValueError('Unsupported repository directory name')
    hashes={f'{prefix}/{rel}':sha256(root/rel) for rel in before['paths']}
    fd,tempname=tempfile.mkstemp(prefix='.biometrics-package-',suffix='.zip',dir=output.parent)
    os.close(fd);temporary=Path(tempname)
    published=False
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,allowZip64=True,strict_timestamps=False) as z:
            for rel in before['paths']:z.write(root/rel,f'{prefix}/{rel}')
        after=snapshot(root)
        if before!=after or any(sha256(root/rel)!=hashes[f'{prefix}/{rel}'] for rel in before['paths']):
            raise ValueError('Repository changed during packaging; close Git writers and retry')
        verification=verify_archive(temporary,before,hashes,prefix)
        report={'status':'verified','created_at_utc':datetime.now(timezone.utc).isoformat(),
                'archive':output.name,'head':before['head'],'commit_count':before['commit_count'],
                'file_count':len(hashes),'git_files':sum('/.git/' in n for n in hashes),
                'archive_sha256':sha256(temporary),'archive_bytes':temporary.stat().st_size,
                'verification':verification,
                'scope':'All Git-tracked working files plus the actual complete .git directory. Ignored datasets, model caches and virtual environments are omitted.'}
        # Exclusive creation avoids replacing an archive produced by another process.
        with output.open('xb') as dst,temporary.open('rb') as src:
            published=True;shutil.copyfileobj(src,dst)
        if sha256(output)!=report['archive_sha256']:raise ValueError('Final archive copy differs')
        with receipt.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
        return report
    except BaseException:
        if published:output.unlink(missing_ok=True)
        raise
    finally:temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='New ZIP path outside the repository; defaults to a HEAD-specific file in its parent')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    # A clean state is checked before running validation or tests.
    state=snapshot(root)
    required=('docs/final_report.md','docs/submission.md','tests/test_package_submission.py')
    if any(not (root/p).is_file() for p in required):raise ValueError('Complete and commit the final report and submission files first')
    subprocess.run([sys.executable,str(root/'scripts/check_repository.py'),'--profile','full'],cwd=root,check=True)
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-q'],cwd=root,check=True)
    if snapshot(root)['head']!=state['head']:raise ValueError('HEAD changed during validation; rerun against the intended commit')
    output=args.output or root.parent/f'{root.name}_submission_{state["head"][:8]}.zip'
    result=build_archive(root,output)
    print('\nSubmission ZIP: VERIFIED')
    print('Archive:',Path(output).resolve())
    print('Verification receipt:',Path(output).resolve().with_suffix('.verification.json'))
    print('HEAD:',result['head'])
    print('Commits preserved:',result['commit_count'])
    print('Files in .git:',result['git_files'])
    print('SHA-256:',result['archive_sha256'])
    print('Extracted Git history and working tree: verified')


if __name__=='__main__':main()
