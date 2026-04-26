"""Generate targeted RLA validation-smoke configs."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER = REPO_ROOT / 'paper'
OUTPUT_ROOT = PAPER / 'exp' / 'rla'


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


def override_model(text: str, overrides: dict) -> str:
    keys = set(overrides)
    lines = text.splitlines()
    out: list[str] = []
    in_model = False
    inserted = False
    for line in lines:
        stripped = line.strip()
        if stripped == '[model]':
            in_model = True
            inserted = False
            out.append(line)
            continue
        if in_model and stripped.startswith('[') and stripped.endswith(']'):
            if not inserted:
                if out and out[-1].strip():
                    out.append('')
                out.extend(render(k, v) for k, v in overrides.items())
                out.append('')
                inserted = True
            in_model = False
        if in_model and any(stripped.startswith(f'{k} ') for k in keys):
            continue
        out.append(line)
    if in_model and not inserted:
        if out and out[-1].strip():
            out.append('')
        out.extend(render(k, v) for k, v in overrides.items())
    text = '\n'.join(out)
    return text if text.endswith('\n') else text + '\n'


def noise_tag(x: float) -> str:
    return {0.0: 'noise0', 1e-6: 'noise1e6', 1e-5: 'noise1e5', 1e-4: 'noise1e4'}[x]


def lr_tag(x: float) -> str:
    return {0.5: 'lr05', 2.0: 'lr2', 4.0: 'lr4'}[x]


def template_for(dataset: str, k64: bool) -> Path:
    if k64:
        return OUTPUT_ROOT / dataset / 'baseline_plr_k64-evaluation' / '0.toml'
    return PAPER / 'exp' / 'tabm-piecewiselinear' / 'tabred' / dataset / '0-evaluation' / '0.toml'


def add(
    specs: list[tuple[str, str, dict]],
    dataset: str,
    *,
    rank: int,
    first_only: bool,
    precision: str,
    noise: float,
    lr_multiplier: float,
    k64: bool = False,
) -> None:
    family = 'first' if first_only else 'uniform'
    variant = (
        f'rla_followup_plr_r{rank}_{family}_{precision}_{noise_tag(noise)}_'
        f'{lr_tag(lr_multiplier)}_warm10'
    )
    if k64:
        variant += '_k64'
    specs.append(
        (
            dataset,
            variant,
            {
                'rank': rank,
                'first_only': first_only,
                'precision': precision,
                'noise': noise,
                'lr_multiplier': lr_multiplier,
                'k64': k64,
            },
        )
    )


def build_specs() -> list[tuple[str, str, dict]]:
    specs: list[tuple[str, str, dict]] = []
    lrs = [0.5, 2.0, 4.0]
    for noise in [0.0, 1e-6, 1e-5]:
        for lr in lrs:
            add(specs, 'sberbank-housing', rank=4, first_only=True, precision='fp32', noise=noise, lr_multiplier=lr)
    for rank in [2, 4]:
        for noise in [0.0, 1e-6, 1e-5]:
            for lr in lrs:
                add(specs, 'cooking-time', rank=rank, first_only=True, precision='fp32', noise=noise, lr_multiplier=lr)
    for k64 in [False, True]:
        for noise in [1e-6, 1e-5]:
            for lr in lrs:
                add(specs, 'homesite-insurance', rank=2, first_only=True, precision='bf16', noise=noise, lr_multiplier=lr, k64=k64)
    for noise in [1e-6, 1e-5]:
        for lr in lrs:
            add(specs, 'delivery-eta', rank=2, first_only=True, precision='bf16', noise=noise, lr_multiplier=lr)
            add(specs, 'delivery-eta', rank=4, first_only=True, precision='fp32', noise=noise, lr_multiplier=lr)
    for rank in [2, 4]:
        for precision in ['bf16', 'fp32']:
            for noise in [1e-6, 1e-5]:
                for lr in lrs:
                    add(specs, 'ecom-offers', rank=rank, first_only=True, precision=precision, noise=noise, lr_multiplier=lr)
    return specs


def main() -> None:
    n = 0
    for dataset, variant, spec in build_specs():
        template_path = template_for(dataset, spec['k64'])
        if not template_path.exists():
            print(f'SKIP {dataset}/{variant}: missing {template_path.relative_to(REPO_ROOT)}')
            continue
        text = template_path.read_text()
        text = override_top_level(
            text,
            {
                'amp': spec['precision'] == 'bf16',
                'rla_adapter_lr_multiplier': spec['lr_multiplier'],
                'rla_extra_paths_freeze_fraction': 0.1,
            },
        )
        text = override_model(
            text,
            {
                'rla_rank': spec['rank'],
                'rla_first_only': spec['first_only'],
                'rla_init': 'base_preserving',
                'rla_base_preserve_noise': spec['noise'],
            },
        )
        out_dir = OUTPUT_ROOT / dataset / f'{variant}-evaluation'
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / '0.toml').write_text(text)
        print(f'OK  {(out_dir / "0.toml").relative_to(REPO_ROOT)}')
        n += 1
    print(f'Wrote {n} RLA follow-up configs.')


if __name__ == '__main__':
    main()
