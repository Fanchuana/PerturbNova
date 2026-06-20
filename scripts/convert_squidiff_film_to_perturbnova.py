#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

ROOT = Path('/work/home/cryoem666/xyf/temp/pycharm/PerturbNova')
SQUIDIFF_ROOT = Path('/work/home/cryoem666/xyf/temp/pycharm/Squidiff')
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
if str(SQUIDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(SQUIDIFF_ROOT))

from perturbnova.models import build_model as build_pn_model
from perturbnova.data import StateDataArtifacts
from perturbnova.utils.checkpoint import load_checkpoint_payload

KEY_REPLACEMENTS = [
    ('encoder.perturb_embed.', 'encoder.perturbation_embedding.'),
    ('encoder.cell_type_embed.', 'encoder.cell_type_embedding.'),
    ('encoder.pert_encoder.', 'encoder.perturbation_encoder.'),
    ('encoder.cell_base_generator.', 'encoder.base_generator.'),
    ('encoder.perturb_projection.', 'encoder.perturbation_projection.'),
    ('mlp_blocks.', 'blocks.'),
    ('time_dense.', 'time_projection.'),
]


def convert_key(key: str) -> str:
    new_key = key
    for old, new in KEY_REPLACEMENTS:
        new_key = new_key.replace(old, new)
    return new_key


def build_pn_from_template(template_ckpt: Path):
    payload = load_checkpoint_payload(template_ckpt, map_location='cpu')
    art = StateDataArtifacts.from_dict(payload['dataset_artifacts'])
    pn_model = build_pn_model(payload['config']['model'], art.to_dict())
    return payload, pn_model, art


def convert_state_dict(
    squidiff_state: dict[str, torch.Tensor],
    pn_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    converted = {k: v.detach().clone() for k, v in pn_state.items()}
    used_source_keys: set[str] = set()
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for src_key, src_value in squidiff_state.items():
        dst_key = convert_key(src_key)
        if dst_key not in converted:
            continue
        src_tensor = src_value.detach().clone()
        dst_tensor = converted[dst_key]

        # Squidiff older checkpoints may not include the extra null perturb row.
        if dst_key == 'encoder.perturbation_embedding.weight' and src_tensor.shape[0] + 1 == dst_tensor.shape[0]:
            new_tensor = dst_tensor.clone()
            new_tensor.zero_()
            new_tensor[: src_tensor.shape[0]] = src_tensor
            converted[dst_key] = new_tensor
            used_source_keys.add(src_key)
            continue

        # Older Squidiff checkpoints may only learn one half of delta_generator output.
        if dst_key in {'encoder.delta_generator.2.weight', 'encoder.delta_generator.2.bias'} and src_tensor.shape[0] * 2 == dst_tensor.shape[0]:
            new_tensor = torch.zeros_like(dst_tensor)
            new_tensor[: src_tensor.shape[0]] = src_tensor
            converted[dst_key] = new_tensor
            used_source_keys.add(src_key)
            continue

        if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
            shape_mismatches.append((dst_key, tuple(dst_tensor.shape), tuple(src_tensor.shape)))
            continue

        converted[dst_key] = src_tensor
        used_source_keys.add(src_key)

    missing_keys = [k for k in converted.keys() if k not in {convert_key(x) for x in used_source_keys}]
    unused_source_keys = [k for k in squidiff_state.keys() if k not in used_source_keys]
    return converted, missing_keys, unused_source_keys, shape_mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert Squidiff FiLM checkpoint to PerturbNova state dict / checkpoint.')
    parser.add_argument('--squidiff-model', required=True, help='Path to Squidiff raw model.pt')
    parser.add_argument('--template-checkpoint', default='', help='PerturbNova checkpoint to use as template for config/artifacts and as target architecture')
    parser.add_argument('--output', required=True, help='Output path (.pt)')
    args = parser.parse_args()

    squidiff_state = torch.load(args.squidiff_model, map_location='cpu')
    if not isinstance(squidiff_state, dict):
        raise TypeError('Squidiff checkpoint must be a raw state_dict dictionary.')

    if not args.template_checkpoint:
        raise ValueError('--template-checkpoint is required for now, so we can build the exact PerturbNova target architecture.')

    payload, pn_model, art = build_pn_from_template(Path(args.template_checkpoint))
    converted, missing, unused_source, shape_mismatches = convert_state_dict(squidiff_state, pn_model.state_dict())

    # Validate final converted weights can load.
    load_result = pn_model.load_state_dict(converted, strict=False)
    print('Missing target keys after conversion:', load_result.missing_keys)
    print('Unexpected target keys after conversion:', load_result.unexpected_keys)
    print('Unused source keys:', unused_source[:20], 'count=', len(unused_source))
    print('Shape mismatches handled/skipped:', shape_mismatches[:20], 'count=', len(shape_mismatches))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write a PerturbNova-style checkpoint payload.
    new_payload = dict(payload)
    new_payload['model'] = {k: v.detach().cpu() for k, v in converted.items()}
    new_payload['ema'] = {}
    torch.save(new_payload, out_path)
    print(f'Saved converted checkpoint to: {out_path}')
    print('NOTE: this is a best-effort conversion. For no-control checkpoints, control-attention weights remain as PerturbNova init.')
    print('NOTE: for older Squidiff delta_generator output shape, only the first half is populated; second half is zeroed.')


if __name__ == '__main__':
    main()
