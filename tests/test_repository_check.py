"""Checks for repository-check diagnostics and protection of existing artifacts."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/check_repository.py'
spec=importlib.util.spec_from_file_location('repository_check',SCRIPT)
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)


class RepositoryCheckTests(unittest.TestCase):
    def test_exact_pins_and_comments(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'requirements.txt';p.write_text('# explanation\nnumpy==2.5.3 # version\n')
            self.assertEqual(checker.pins(p),{'numpy':'2.5.3'})
            p.write_text('numpy>=2\n')
            with self.assertRaises(ValueError):checker.pins(p)

    def test_local_links_and_code_examples(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'docs').mkdir();(root/'docs/real.md').write_text('present')
            p=root/'README.md'
            p.write_text('[ok](docs/real.md#section) [external](https://example.com) [anchor](#heading)\n'
                         '```markdown\n[example](not-a-file.md)\n```\n[broken](docs/missing.md)\n[escape](../outside.md)')
            self.assertEqual(checker.broken_links(root,p),['docs/missing.md','../outside.md'])

    def test_notebook_errors_and_execution_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'n.ipynb'
            n={'nbformat':4,'cells':[{'cell_type':'code','execution_count':1,'outputs':[]},
               {'cell_type':'code','execution_count':None,'outputs':[{'output_type':'error','ename':'ValueError'}]}]}
            p.write_text(json.dumps(n));errors,unexecuted,count=checker.notebook_errors(p)
            self.assertEqual((len(errors),unexecuted,count),(1,1,2))
            p.write_text('{broken')
            with self.assertRaises(ValueError):checker.notebook_errors(p)

    def test_missing_files_report_failures_without_writes(self):
        with tempfile.TemporaryDirectory() as d:
            report=checker.check(Path(d),profile='files',skip_git=True)
            self.assertEqual(report['status'],'failed')
            self.assertGreater(report['failed'],0)
            self.assertEqual(report['skipped'],2)
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_cli_refuses_to_overwrite_project_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'README.md';p.write_text('keep this')
            result=subprocess.run([sys.executable,str(SCRIPT),'--root',d,'--output','README.md'],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('protects existing artifacts',result.stderr)
            self.assertEqual(p.read_text(),'keep this')


if __name__=='__main__':unittest.main()
