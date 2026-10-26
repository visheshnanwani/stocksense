"""Give the entire Streamlit JS/CSS graph a release-specific cache namespace.

Run after pip install on the host. This avoids mixing browser-cached chunks
from earlier Streamlit builds with the current frontend. No runtime proxy or
changes to the dashboard UI are needed.
"""
import hashlib
from pathlib import Path
import re


MARKER = '<!-- stocksense-assets-v1 -->'


def prepare_assets(root: Path) -> int:
    index = root / 'index.html'
    if MARKER in index.read_text(encoding='utf-8'):
        return 0
    assets = sorted(p for p in root.rglob('*')
                    if p.suffix in {'.js', '.css'} and '.ss-' not in p.name)
    if not assets:
        raise RuntimeError('No Streamlit frontend assets found')
    digest = hashlib.sha256(b'stocksense-assets-v1')
    for path in assets:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    release = digest.hexdigest()[:16]
    names = {p.name: f'{p.stem}.ss-{release}{p.suffix}' for p in assets}
    if len(names) != len(assets):
        raise RuntimeError('Ambiguous frontend asset filenames')
    pattern = re.compile('|'.join(re.escape(name) for name in names))
    documents = assets + list(root.rglob('*.html')) + list(root.rglob('*.json'))
    for path in documents:
        content = path.read_text(encoding='utf-8')
        content = pattern.sub(lambda match: names[match.group()], content)
        if path == index:
            content = MARKER + '\n' + content
        target = path.with_name(names[path.name]) if path in assets else path
        target.write_text(content, encoding='utf-8')
    # Keep original chunks available for already-open browser sessions.
    return len(assets)


if __name__ == '__main__':
    import streamlit
    root = Path(streamlit.__file__).parent / 'static'
    print(f'Prepared {prepare_assets(root)} Streamlit frontend assets')
