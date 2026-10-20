"""Verify a snapshot; optionally restore only its follow-up plotting inputs."""
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def digest(data): return hashlib.sha256(data).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-followup', action='store_true')
    a = parser.parse_args()
    manifest = json.loads((ROOT/'PACKAGE_INDEX.json').read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
              and '.git' not in p.parts}
    expected = {x['path'] for x in manifest['files']} | {'PACKAGE_INDEX.json'}
    for row in manifest['files']:
        path = ROOT/row['path']
        assert path.is_file(), f'Missing: {row["path"]}'
        assert path.stat().st_size == row['bytes'], f'Size changed: {path}'
        assert digest(path.read_bytes()) == row['sha256'], f'Hash changed: {path}'
    extra = actual - expected
    if extra: print(f'{len(extra)} additional local files (for example restored evidence); not part of the upload snapshot.')
    archived = {}
    for row in manifest['followup_archive_entries']:
        archived.setdefault(row['archive'], []).append(row)
    for name, rows in archived.items():
        with ZipFile(ROOT/'archives'/name) as archive:
            assert archive.testzip() is None, f'Corrupt ZIP: {name}'
            assert set(archive.namelist()) == {r['path'] for r in rows}
            for row in rows:
                value = archive.read(row['path'])
                assert len(value) == row['bytes'] and digest(value) == row['sha256']
                if a.restore_followup:
                    target = (ROOT/row['path']).resolve()
                    assert target.is_relative_to(ROOT.resolve()), 'Unsafe archive path'
                    if target.exists():
                        assert digest(target.read_bytes()) == row['sha256'], f'Refusing to replace changed local file: {target}'
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(value)
    for row in manifest['legacy_archives']:
        with ZipFile(ROOT/row['path']) as archive:
            assert archive.testzip() is None
            for member in archive.namelist():
                assert (ROOT/member).resolve().is_relative_to(ROOT.resolve())
    assert len(expected) < 100
    assert all(row['bytes'] < 25*1024*1024 for row in manifest['files'])
    print(f'PASS: {len(expected)} upload files, hashes, archive contents and browser-upload sizes verified.')
    if a.restore_followup:
        print('Follow-up plotting inputs and saved image models restored. No training started.')

if __name__ == '__main__': main()
