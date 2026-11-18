"""Download official BSDS500; select a deterministic subset before scoring."""
from pathlib import Path
import hashlib
import json
import tarfile
import subprocess

ROOT = Path(__file__).resolve().parents[2]
URL = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz'

def main() -> None:
    target = ROOT / 'data' / 'bsds500'
    target.mkdir(parents=True, exist_ok=True)
    archive = target / 'BSR_bsds500.tgz'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        subprocess.run(['curl.exe', '-L', '--fail', '--retry', '2', '--connect-timeout', '20',
                        '--max-time', '600', '-C', '-', '-o', str(temporary), URL], check=True)
        temporary.replace(archive)
    records = []
    with tarfile.open(archive) as tar:
        for split, count in [('val', 5), ('test', 10)]:
            members = sorted((m for m in tar.getmembers() if m.isfile() and
                              f'/images/{split}/' in m.name and m.name.endswith('.jpg')),
                             key=lambda m: Path(m.name).name)
            if len(members) < count:
                raise ValueError(f'Not enough {split} images')
            for member in members[:count]:
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError(f'Cannot read {member.name}')
                data = stream.read()
                path = target / split / Path(member.name).name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(data)
                records.append(dict(split=split, id=path.stem, member=member.name,
                                    file=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(source=URL, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    selection='first filenames lexicographically before experiments', images=records)
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared {len(records)} images in team-project/data/bsds500')

if __name__ == '__main__':
    main()
