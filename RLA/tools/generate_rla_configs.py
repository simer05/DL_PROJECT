"""Generate RLA evaluation config templates for the TabReD sweep.

For each (dataset, variant) pair, write
``paper/exp/rla/<dataset>/<variant>-evaluation/0.toml``,
inheriting all hyperparameters from the corresponding tuned baseline TabM
config and overriding only the model section to add the RLA flags.

Variants
--------
* ``baseline``                    — plain TabM, rla_rank=1 (sanity row).
* ``rla_first_r{1,2,4,8}``        — RLA-first at rank ∈ {1,2,4,8}.
* ``rla_uniform_r{1,2,4,8}``      — RLA-uniform at rank ∈ {1,2,4,8}.
* ``rla_additive_first_r4``       — additive LoRA-style defensive row.

Run from the repo root:
    python tools/generate_rla_configs.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER = REPO_ROOT / 'paper'
OUTPUT_ROOT = PAPER / 'exp' / 'rla'

DEFAULT_DATASETS = [
    'homesite-insurance',
    'ecom-offers',
    'sberbank-housing',
    'cooking-time',
    'delivery-eta',
]

# Three template families. Key = base prefix used in the variant name;
# value = relative path under paper/exp/ to the per-dataset template root.
TEMPLATE_FAMILIES = {
    '': PAPER / 'exp' / 'tabm' / 'tabred',  # plain TabM
    'plr_': PAPER / 'exp' / 'tabm-piecewiselinear' / 'tabred',
    'mini_plr_': PAPER / 'exp' / 'tabm-mini-piecewiselinear' / 'tabred',
}

VARIANTS: dict[str, dict] = {}
# Plain TabM variants (kept for backward compatibility with the v1 sweep).
for r in (1, 2, 4, 8):
    VARIANTS[f'rla_first_r{r}'] = {'rla_rank': r, 'rla_first_only': True}
    VARIANTS[f'rla_uniform_r{r}'] = {'rla_rank': r, 'rla_first_only': False}
VARIANTS['baseline'] = {'rla_rank': 1, 'rla_first_only': False}
VARIANTS['rla_additive_first_r4'] = {
    'rla_rank': 4,
    'rla_first_only': True,
    'rla_additive': True,
}

# v2 variants: PLR + mini-PLR baselines with base-preserving init.
def _add_v2_variant(name: str, **overrides) -> None:
    VARIANTS[name] = overrides

# PLR family — BF16 AMP variants (paper default).
_add_v2_variant(
    'baseline_plr', rla_rank=1, rla_first_only=False, amp=True
)
_add_v2_variant(
    'rla_plr_first_r2_basepreserve',
    rla_rank=2, rla_first_only=True, rla_init='base_preserving', amp=True,
)
# PLR family — FP32 variants for fair comparison and stability on high-rank.
_add_v2_variant(
    'baseline_plr_fp32',
    rla_rank=1, rla_first_only=False, amp=False,
)
_add_v2_variant(
    'rla_plr_first_r2_basepreserve_fp32',
    rla_rank=2, rla_first_only=True, rla_init='base_preserving', amp=False,
)
_add_v2_variant(
    'rla_plr_first_r4_basepreserve_fp32',
    rla_rank=4, rla_first_only=True, rla_init='base_preserving', amp=False,
)
_add_v2_variant(
    'rla_plr_uniform_r4_basepreserve_fp32',
    rla_rank=4, rla_first_only=False, rla_init='base_preserving', amp=False,
)
_add_v2_variant(
    'rla_plr_uniform_r8_basepreserve_fp32',
    rla_rank=8, rla_first_only=False, rla_init='base_preserving', amp=False,
)
# BF16 high-rank variants kept for the smoke matrix that decides the policy.
_add_v2_variant(
    'rla_plr_first_r4_basepreserve_bf16',
    rla_rank=4, rla_first_only=True, rla_init='base_preserving', amp=True,
)
_add_v2_variant(
    'rla_plr_uniform_r4_basepreserve_bf16',
    rla_rank=4, rla_first_only=False, rla_init='base_preserving', amp=True,
)
# Mini-PLR family (informational only — tabm-mini does not plumb RLA flags).
_add_v2_variant('baseline_mini_plr', rla_rank=1, rla_first_only=False, amp=True)
_add_v2_variant(
    'rla_mini_plr_first_r2_basepreserve',
    rla_rank=2, rla_first_only=True, rla_init='base_preserving', amp=True,
)
_add_v2_variant(
    'rla_mini_plr_first_r4_basepreserve',
    rla_rank=4, rla_first_only=True, rla_init='base_preserving', amp=True,
)
# Backwards-compat aliases (old variant names → bf16 equivalents).
_add_v2_variant(
    'rla_plr_first_r4_basepreserve',
    rla_rank=4, rla_first_only=True, rla_init='base_preserving', amp=True,
)
_add_v2_variant(
    'rla_plr_uniform_r4_basepreserve',
    rla_rank=4, rla_first_only=False, rla_init='base_preserving', amp=True,
)
_add_v2_variant(
    'rla_plr_uniform_r8_basepreserve',
    rla_rank=8, rla_first_only=False, rla_init='base_preserving', amp=True,
)


def _resolve_template(variant: str, dataset: str) -> Path | None:
    """Pick the template file for a given variant name.

    Suffixes such as ``_fp32`` / ``_bf16`` are routing hints only and do
    not change the template family. The template is determined purely by
    ``mini_plr`` / ``plr`` membership in the variant name.
    """
    if 'mini_plr' in variant or variant.startswith('baseline_mini_plr'):
        root = TEMPLATE_FAMILIES['mini_plr_']
    elif 'plr' in variant or variant.startswith('baseline_plr'):
        root = TEMPLATE_FAMILIES['plr_']
    else:
        root = TEMPLATE_FAMILIES['']
    p = root / dataset / '0-evaluation' / '0.toml'
    return p if p.exists() else None


def _read(path: Path) -> str:
    return path.read_text()


def _override_top_level(template: str, key: str, value) -> str:
    """Replace or insert a top-level scalar key-value at the top of the toml.

    Used to make amp / batch_size / etc. an *explicit* per-variant override
    instead of a manually-edited setting that can be silently restored on
    regeneration.
    """
    lines = template.splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        # Stop overwriting once we enter any section.
        if stripped.startswith('[') and stripped.endswith(']'):
            if not found:
                out.append(_render_kv(key, value))
                found = True
            out.append(line)
            continue
        if (not found) and stripped.startswith(f'{key} ') and '=' in stripped:
            out.append(_render_kv(key, value))
            found = True
            continue
        out.append(line)
    if not found:
        # No section yet — append at end.
        out.append(_render_kv(key, value))
    text = '\n'.join(out)
    return text if text.endswith('\n') else text + '\n'


def _render_kv(key: str, val) -> str:
    if isinstance(val, bool):
        return f'{key} = {"true" if val else "false"}'
    if isinstance(val, str):
        return f'{key} = "{val}"'
    return f'{key} = {val}'


def _inject_rla_block(template: str, overrides: dict) -> str:
    """Insert RLA flags into the top-level [model] section of the template.

    Top-level keys (anything not under ``[model]`` or sub-tables) such as
    ``amp`` are applied via ``_override_top_level``; the rest go inside
    ``[model]``.

    The model-level flags must be emitted BEFORE any sub-section header
    such as ``[model.backbone]`` or ``[model.num_embeddings]``, otherwise
    TOML parses them as keys of the sub-section instead of the [model]
    table.
    """
    # Top-level overrides (currently only `amp`).
    text = template
    top_level_keys = {'amp'}
    model_overrides = {}
    for k, v in overrides.items():
        if k in top_level_keys:
            text = _override_top_level(text, k, v)
        else:
            model_overrides[k] = v

    if not model_overrides:
        return text if text.endswith('\n') else text + '\n'

    lines = text.splitlines()
    out: list[str] = []
    in_model_top = False
    inserted = False
    for line in lines:
        stripped = line.strip()
        if stripped == '[model]':
            in_model_top = True
            out.append(line)
            continue
        # Any new section header ends the top-level [model] block. If we
        # are about to leave that block, emit the overrides first.
        if in_model_top and stripped.startswith('[') and stripped.endswith(']'):
            if not inserted:
                if out and out[-1].strip() != '':
                    out.append('')
                out.extend(_render_overrides(model_overrides))
                out.append('')
                inserted = True
            in_model_top = False
        out.append(line)
    if in_model_top and not inserted:
        if out and out[-1].strip() != '':
            out.append('')
        out.extend(_render_overrides(model_overrides))
        inserted = True
    if not inserted:
        raise RuntimeError('No [model] section found in template')
    text = '\n'.join(out)
    return text if text.endswith('\n') else text + '\n'


def _render_overrides(overrides: dict) -> list[str]:
    rows = []
    for key, val in overrides.items():
        if isinstance(val, bool):
            rows.append(f'{key} = {"true" if val else "false"}')
        elif isinstance(val, (int, float)):
            rows.append(f'{key} = {val}')
        elif isinstance(val, str):
            rows.append(f'{key} = "{val}"')
        else:
            rows.append(f'{key} = {val}')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=DEFAULT_DATASETS,
        help='TabReD datasets to generate configs for.',
    )
    parser.add_argument(
        '--variants',
        nargs='+',
        default=sorted(VARIANTS.keys()),
        help='Subset of variants to emit (default: all).',
    )
    args = parser.parse_args()

    n_written = 0
    for ds in args.datasets:
        for variant in args.variants:
            if variant not in VARIANTS:
                print(f'WARN unknown variant {variant}; skipping')
                continue
            template_path = _resolve_template(variant, ds)
            if template_path is None:
                print(f'SKIP {ds}/{variant}: no matching template')
                continue
            template = _read(template_path)
            overrides = VARIANTS[variant]
            new_toml = _inject_rla_block(template, overrides)
            out_dir = OUTPUT_ROOT / ds / f'{variant}-evaluation'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / '0.toml'
            out_path.write_text(new_toml)
            n_written += 1
            print(f'OK  {out_path.relative_to(REPO_ROOT)}')
    print(f'\nWrote {n_written} configs.')


if __name__ == '__main__':
    main()
