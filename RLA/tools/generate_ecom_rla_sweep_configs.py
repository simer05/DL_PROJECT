"""Generate ecom-offers-only RLA sweep configs."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER = REPO_ROOT / 'paper'
RLA_ROOT = PAPER / 'exp' / 'rla'
DATASET = 'ecom-offers'


def render(key: str, value) -> str:
    if isinstance(value, bool):
        return f'{key} = {"true" if value else "false"}'
    if isinstance(value, str):
        return f'{key} = "{value}"'
    return f'{key} = {value}'


def override_top_level(text: str, overrides: dict) -> str:
    keys = set(overrides)
    lines = text.splitlines()
    out: list[str] = []
    inserted = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']') and not inserted:
            out.extend(render(k, v) for k, v in overrides.items())
            inserted = True
        if not inserted and any(stripped.startswith(f'{k} ') for k in keys):
            continue
        out.append(line)
    if not inserted:
        out.extend(render(k, v) for k, v in overrides.items())
    text = '\n'.join(out)
    return text if text.endswith('\n') else text + '\n'


def override_section(text: str, section: str, overrides: dict) -> str:
    keys = set(overrides)
    lines = text.splitlines()
    out: list[str] = []
    in_section = False
    inserted = False
    header = f'[{section}]'
    for line in lines:
        stripped = line.strip()
        if stripped == header:
            in_section = True
            inserted = False
            out.append(line)
            continue
        if in_section and stripped.startswith('[') and stripped.endswith(']'):
            if not inserted:
                if out and out[-1].strip():
                    out.append('')
                out.extend(render(k, v) for k, v in overrides.items())
                out.append('')
                inserted = True
            in_section = False
        if in_section and any(stripped.startswith(f'{k} ') for k in keys):
            continue
        out.append(line)
    if in_section and not inserted:
        if out and out[-1].strip():
            out.append('')
        out.extend(render(k, v) for k, v in overrides.items())
    text = '\n'.join(out)
    return text if text.endswith('\n') else text + '\n'


def noise_tag(x: float) -> str:
    return {1e-6: 'n1e6', 1e-5: 'n1e5', 1e-4: 'n1e4'}[x]


def lr_tag(x: float) -> str:
    return {0.5: 'lr050', 0.75: 'lr075', 1.0: 'lr100', 1.25: 'lr125', 1.5: 'lr150'}[x]


def template_path(family: str) -> Path:
    if family == 'plr':
        return PAPER / 'exp' / 'tabm-piecewiselinear' / 'tabred' / DATASET / '0-evaluation' / '0.toml'
    if family == 'plr_k64':
        return RLA_ROOT / DATASET / 'baseline_plr_k64-evaluation' / '0.toml'
    if family == 'plr_fp32':
        return RLA_ROOT / DATASET / 'baseline_plr_fp32-evaluation' / '0.toml'
    raise ValueError(family)


def specs() -> list[dict]:
    rows = []
    noises = [1e-5, 1e-6, 1e-4]
    for family in ['plr', 'plr_k64']:
        for noise in noises:
            for lr_mult in [0.5, 0.75, 1.25, 1.5]:
                rows.append(
                    {
                        'rank': 2,
                        'first_only': True,
                        'family': family,
                        'amp': True,
                        'noise': noise,
                        'lr_mult': lr_mult,
                    }
                )
    for rank in [4]:
        for first_only in [True, False]:
            for family, amp in [('plr', True), ('plr_fp32', False), ('plr_k64', True)]:
                for noise in noises:
                    rows.append(
                        {
                            'rank': rank,
                            'first_only': first_only,
                            'family': family,
                            'amp': amp,
                            'noise': noise,
                            'lr_mult': 1.0,
                        }
                    )
    return rows


def variant_name(spec: dict) -> str:
    mode = 'first' if spec['first_only'] else 'uniform'
    precision = 'bf16' if spec['amp'] else 'fp32'
    family = 'k64' if spec['family'] == 'plr_k64' else 'k32'
    return (
        f"rla_ecom_sweep_{family}_{mode}_r{spec['rank']}_{precision}_"
        f"{noise_tag(spec['noise'])}_{lr_tag(spec['lr_mult'])}"
    )


def main() -> None:
    n = 0
    for spec in specs():
        template = template_path(spec['family'])
        if not template.exists():
            print(f'SKIP missing {template.relative_to(REPO_ROOT)}')
            continue
        text = template.read_text()
        import tomllib

        base_config = tomllib.loads(text)
        base_lr = float(base_config['optimizer']['lr'])
        text = override_top_level(text, {'amp': spec['amp']})
        text = override_section(
            text,
            'optimizer',
            {'lr': base_lr * spec['lr_mult']},
        )
        text = override_section(
            text,
            'model',
            {
                'rla_rank': spec['rank'],
                'rla_first_only': spec['first_only'],
                'rla_init': 'base_preserving',
                'rla_base_preserve_noise': spec['noise'],
            },
        )
        out_dir = RLA_ROOT / DATASET / f'{variant_name(spec)}-evaluation'
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / '0.toml').write_text(text)
        n += 1
        print(f'OK  {(out_dir / "0.toml").relative_to(REPO_ROOT)}')
    print(f'Wrote {n} ecom RLA sweep configs.')


if __name__ == '__main__':
    main()
