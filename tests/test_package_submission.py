"""Archive round trips with temporary fixture histories, never the user's history."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/package_submission.py'
spec=importlib.util.spec_from_file_location('package_submission',SCRIPT)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.base=Path(self.temporary.name);self.root=self.base/'fixture'
        self.root.mkdir()
        def git(*args):
            return subprocess.run(['git','-C',str(self.root),*args],check=True,capture_output=True,text=True)
        self.git=git
        git('init','-b','main')
        git('config','user.name','Archive Test Fixture')
        git('config','user.email','fixture@example.invalid')
        git('config','commit.gpgsign','false')
        git('config','core.autocrlf','false')
        (self.root/'.gitignore').write_text('data/\n.venv/\n')
        (self.root/'README.md').write_text('First fixture version\n')
        git('add','.');git('commit','-m','Create synthetic test fixture')
        (self.root/'README.md').write_text('Second fixture version\n')
        git('add','README.md');git('commit','-m','Update synthetic test fixture')
        (self.root/'data').mkdir();(self.root/'data/local-only.txt').write_text('not a submission file')

    def test_round_trip_preserves_history_and_omits_ignored_data(self):
        head=p.git(self.root,'rev-parse','HEAD')
        result=p.build_archive(self.root,self.base/'submission.zip')
        self.assertEqual(result['status'],'verified')
        self.assertEqual(result['head'],head)
        self.assertEqual(result['commit_count'],2)
        self.assertGreater(result['git_files'],0)
        self.assertTrue(all(v is True or v=='passed' for v in result['verification'].values()))
        with zipfile.ZipFile(self.base/'submission.zip') as z:
            self.assertIn('fixture/.git/HEAD',z.namelist())
            self.assertIn('fixture/README.md',z.namelist())
            self.assertFalse(any('/data/' in n for n in z.namelist()))
        self.assertTrue((self.base/'submission.verification.json').is_file())
        self.assertEqual(p.git(self.root,'rev-parse','HEAD'),head)
        self.assertEqual(p.git(self.root,'status','--porcelain'),'')

    def test_dirty_repository_rejected(self):
        (self.root/'README.md').write_text('not committed')
        with self.assertRaisesRegex(ValueError,'Commit intended changes'):
            p.build_archive(self.root,self.base/'dirty.zip')
        self.assertFalse((self.base/'dirty.zip').exists())

    def test_existing_output_and_inside_output_rejected(self):
        out=self.base/'existing.zip';out.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):p.build_archive(self.root,out)
        self.assertEqual(out.read_bytes(),b'preserve')
        with self.assertRaisesRegex(ValueError,'outside'):
            p.build_archive(self.root,self.root/'inside.zip')

    def test_external_git_objects_rejected(self):
        path=self.root/'.git/objects/info/alternates'
        path.write_text('/some/external/object/store\n')
        with self.assertRaisesRegex(ValueError,'alternates'):
            p.build_archive(self.root,self.base/'alternate.zip')

    def test_symlink_rejected(self):
        link=self.root/'outside-link'
        link.symlink_to(self.base/'not-in-repo')
        self.git('add','outside-link');self.git('commit','-m','Add unsupported fixture symlink')
        with self.assertRaisesRegex(ValueError,'Unsupported tracked path'):
            p.build_archive(self.root,self.base/'symlink.zip')


if __name__=='__main__':unittest.main()
