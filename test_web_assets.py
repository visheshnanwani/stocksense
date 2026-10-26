import tempfile
from pathlib import Path
import unittest

from prepare_web_assets import prepare_assets


class WebAssetsTests(unittest.TestCase):
    def test_all_imports_and_preloads_use_same_namespace(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'index.html').write_text('<script src="main.js"></script><link href="theme.css">')
            (root / 'main.js').write_text('import {n} from "./chunk.js"; import("./lazy.js");')
            (root / 'chunk.js').write_text('export const n=1;')
            (root / 'lazy.js').write_text('import {n} from "./chunk.js";')
            (root / 'theme.css').write_text('body{color:white}')
            self.assertEqual(prepare_assets(root), 4)
            new_main = next(root.glob('main.ss-*.js'))
            self.assertIn(new_main.name, (root / 'index.html').read_text())
            self.assertIn(next(root.glob('chunk.ss-*.js')).name, new_main.read_text())
            self.assertIn(next(root.glob('lazy.ss-*.js')).name, new_main.read_text())
            self.assertEqual((root / 'chunk.js').read_text(), 'export const n=1;')
            self.assertEqual(prepare_assets(root), 0)


if __name__ == '__main__':
    unittest.main()
