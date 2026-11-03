"""Download pinned, public CPU inference assets to a data-volume directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request

MODEL_REV = '23749fefcc72300e3a2ad315e1317431b06b590a'
ASSETS = [
    ('llama-b12057-bin-ubuntu-x64.tar.gz',
     'https://github.com/ggml-org/llama.cpp/releases/download/b12057/llama-b12057-bin-ubuntu-x64.tar.gz',
     22892921, '1bdfdac035014ff7b935aaf4a91d7bd5ea5018db9957e1a21b6567f189b8dbe7'),
    ('Qwen3-0.6B-Q8_0.gguf',
     f'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/{MODEL_REV}/Qwen3-0.6B-Q8_0.gguf',
     639446688, '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031'),
]


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    quota_root = Path('/mnt/data/ishaan')
    if not out.is_relative_to(quota_root):
        parser.error('large model assets must be on the data volume')
    used = int(subprocess.check_output(['du', '-sx', '-B1', str(quota_root)], text=True).split()[0])
    needed = sum(size for name, _, size, _ in ASSETS if not (out / name).exists())
    if used + needed + 300_000_000 > 100_000_000_000:
        raise RuntimeError('insufficient personal quota including extraction and training reserve')
    out.mkdir(parents=True, exist_ok=True)
    for name, url, size, expected in ASSETS:
        path = out / name
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.partial')
            print('Downloading', name, size, 'bytes', flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    if target.tell() > size:
                        raise ValueError('download larger than pinned asset')
            if partial.stat().st_size != size or sha256(partial) != expected:
                raise ValueError(f'checksum/size mismatch: {name}')
            partial.replace(path)
        if path.stat().st_size != size or sha256(path) != expected:
            raise ValueError(f'checksum/size mismatch: {name}')
        print('Verified', name, expected, flush=True)
    with tarfile.open(out / ASSETS[0][0]) as archive:
        archive.extractall(out / 'runtime', filter='data')
    (out / 'manifest.json').write_text(json.dumps(dict(model='Qwen/Qwen3-0.6B-GGUF',
        revision=MODEL_REV, runtime='llama.cpp b12057', assets=[dict(name=n, url=u, bytes=s, sha256=h)
        for n, u, s, h in ASSETS]), indent=2))
    print('Ready:', out, flush=True)


if __name__ == '__main__':
    main()
