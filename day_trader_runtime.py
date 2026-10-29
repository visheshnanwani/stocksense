"""Isolated, bounded execution of the vendored MIT day-trader tools."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import tempfile
from threading import BoundedSemaphore

ROOT = Path(__file__).resolve().parent / 'vendor' / 'day_trader'
TOOLS = {
    'Scanner': ('trader-scanner', 'scan.py'),
    'News': ('trader-news', 'news.py'),
    'Entry timer': ('trader-entry', 'entry.py'),
    'Exit planner': ('trader-exit', 'exit.py'),
    'Trade simulation': ('trader-timemachine', 'timemachine.py'),
    'Budget': ('trader-budget', 'size.py'),
    'Risk check': ('trader-risk', 'check.py'),
    'Broker setup': ('trader-broker', 'ticket.py'),
    'Alerts': ('trader-alerts', 'alerts.py'),
    'Earnings': ('trader-analyst', 'facts.py'),
    'Congress disclosures': ('trader-pelosi', 'ptr.py'),
}
_slots = BoundedSemaphore(2)


def symbols(text, limit=20):
    values = list(dict.fromkeys(re.split(r'[\s,]+', text.strip().upper())))
    if not values or len(values) > limit or any(not re.fullmatch(r'[A-Z]{1,5}(?:[.-][A-Z])?', x) for x in values):
        raise ValueError(f'Enter 1–{limit} US stock symbols, separated by commas. NSE/BSE symbols are not supported by these tools yet.')
    return values


class Workspace:
    def __init__(self):
        self._temp = tempfile.TemporaryDirectory(prefix='stocksense-trader-')
        self.path = Path(self._temp.name)
        (self.path / 'trader').mkdir()

    def read(self, name):
        path = self.path / 'trader' / name
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def write(self, name, value):
        path = self.path / 'trader' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding='utf-8')

    def invalidate(self, *names):
        for name in names:
            (self.path / 'trader' / name).unlink(missing_ok=True)

    def run(self, tool, args, sec_identity='', timeout=120):
        folder, script = TOOLS[tool]
        if tool == 'Broker setup' and args != ['show']:
            raise ValueError('The dashboard only previews broker tickets; it cannot approve or place orders.')
        if not _slots.acquire(blocking=False):
            return 2, 'Other analysis jobs are running. Please try again shortly.'
        env = {k: v for k, v in os.environ.items() if k in
               {'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'HOME', 'LANG', 'SSL_CERT_FILE'}}
        env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        if sec_identity:
            env['SEC_USER_AGENT'] = sec_identity
        try:
            result = subprocess.run([sys.executable, str(ROOT / folder / script), *map(str, args)],
                                    cwd=self.path, env=env, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace', timeout=timeout,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            output = '\n'.join(x for x in [result.stdout.strip(), result.stderr.strip()] if x)
            if sec_identity:
                output = output.replace(sec_identity, '[SEC identity]')
            return result.returncode, output[-60000:] or 'No output returned.'
        except subprocess.TimeoutExpired:
            return 2, 'Request timed out. Try fewer symbols or retry when the data provider is available.'
        finally:
            _slots.release()
