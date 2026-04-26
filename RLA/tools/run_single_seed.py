"""Run one evaluation config seed directly.

Usage from paper/:
    python3 ../tools/run_single_seed.py exp/leo/<ds>/<variant>-evaluation 2
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

if __name__ == '__main__':
    cwd = Path.cwd()
    assert (cwd / 'pixi.toml').exists(), 'Run from paper/'
    sys.path.append(str(cwd))

import lib
from bin.model import main as model_main


def main() -> None:
    eval_dir = Path(sys.argv[1]).resolve()
    seed = int(sys.argv[2])
    config = lib.load_config(eval_dir / '0')
    config['seed'] = seed
    output = eval_dir / str(seed)
    if output.exists():
        shutil.rmtree(output)
    cfg_path = output.with_suffix('.toml')
    if seed > 0:
        cfg_path.unlink(missing_ok=True)
        lib.dump_config(output, config)
    model_main(config, output, force=True)


if __name__ == '__main__':
    lib.configure_libraries()
    main()
