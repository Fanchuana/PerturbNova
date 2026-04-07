from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from perturbnova.config import load_dataset_config
from perturbnova.data import StateDataArtifacts, build_inference_loader
from perturbnova.utils.distributed import DistributedContext


class InferenceSplitTests(unittest.TestCase):
    def test_inference_can_slice_test_subset_from_full_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "full_dataset.h5ad"
            config_path = root / "data.toml"
            ad.settings.allow_write_nullable_strings = True

            adata = ad.AnnData(
                X=np.asarray(
                    [
                        [0.0, 0.1],
                        [1.0, 1.1],
                        [2.0, 2.1],
                        [3.0, 3.1],
                        [4.0, 4.1],
                    ],
                    dtype=np.float32,
                )
            )
            adata.obs["gene"] = ["ctrl", "val_gene", "test_gene", "train_gene", "other_gene"]
            adata.obs["gem_group"] = ["batch1"] * adata.n_obs
            adata.obs["cell_line"] = ["hepg2", "hepg2", "hepg2", "hepg2", "k562"]
            adata.write_h5ad(data_path)

            config_path.write_text(
                '\n'.join(
                    [
                        "[data]",
                        'dataset_name = "toy_replogle"',
                        'task = "state_perturbation"',
                        f'data_path = "{data_path}"',
                        "use_hvg = false",
                        "",
                        "[data.feature_space]",
                        'source = "X"',
                        'key = ""',
                        "",
                        "[data.obs_keys]",
                        'perturbation = "gene"',
                        'batch = "gem_group"',
                        'cell_type = "cell_line"',
                        "",
                        "[data.control]",
                        'label = "ctrl"',
                        "",
                        "[data.split]",
                        'mode = "fewshot_holdout"',
                        'target_cell_type = "hepg2"',
                        'val_perturbations = ["val_gene"]',
                        'test_perturbations = ["test_gene"]',
                    ]
                ),
                encoding="utf-8",
            )

            train_dataset_config = load_dataset_config(config_path)
            artifacts = StateDataArtifacts(
                feature_dim=2,
                raw_feature_dim=2,
                output_dim=2,
                obs_keys={
                    "perturbation": "gene",
                    "batch": "gem_group",
                    "cell_type": "cell_line",
                },
                feature_space={"source": "X", "key": ""},
                use_hvg=False,
                hvg_key="highly_variable",
                condition_sizes={
                    "perturbation": 5,
                    "batch": 1,
                    "cell_type": 2,
                },
                vocab={
                    "perturbation": {
                        "ctrl": 0,
                        "other_gene": 1,
                        "test_gene": 2,
                        "train_gene": 3,
                        "val_gene": 4,
                    },
                    "batch": {"batch1": 0},
                    "cell_type": {"hepg2": 0, "k562": 1},
                },
                default_labels={
                    "perturbation": "ctrl",
                    "batch": "batch1",
                    "cell_type": "hepg2",
                },
                control_label="ctrl",
            )
            infer_config = {
                "input": {
                    "data_path": "",
                    "reference_data_path": "",
                    "split": {
                        "subset": "test",
                        "dataset_config_path": str(config_path),
                    },
                    "obs_keys": {
                        "perturbation": "",
                        "batch": "",
                        "cell_type": "",
                    },
                    "defaults": {
                        "perturbation": "",
                        "batch": "",
                        "cell_type": "",
                    },
                },
                "control": {
                    "enabled": False,
                    "label": "ctrl",
                    "samples_per_query": 4,
                },
                "sampling": {
                    "batch_size": 8,
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                },
            }
            context = DistributedContext(
                rank=0,
                world_size=1,
                local_rank=0,
                device=torch.device("cpu"),
                backend="gloo",
            )
            target_adata, loader, sampler = build_inference_loader(
                infer_config=infer_config,
                train_dataset_config=train_dataset_config,
                artifacts=artifacts,
                distributed_context=context,
            )

            self.assertEqual(target_adata.n_obs, 2)
            self.assertEqual(target_adata.obs["gene"].tolist(), ["ctrl", "test_gene"])
            self.assertEqual(len(loader.dataset), 2)
            self.assertIsNone(sampler)

    def test_inference_can_slice_zeroshot_test_subset_from_full_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "full_dataset.h5ad"
            config_path = root / "data.toml"
            ad.settings.allow_write_nullable_strings = True

            adata = ad.AnnData(
                X=np.asarray(
                    [
                        [0.0, 0.1],
                        [1.0, 1.1],
                        [2.0, 2.1],
                        [3.0, 3.1],
                    ],
                    dtype=np.float32,
                )
            )
            adata.obs["gene"] = ["ctrl", "train_gene", "target_gene", "ctrl"]
            adata.obs["gem_group"] = ["batch1"] * adata.n_obs
            adata.obs["cell_line"] = ["k562", "k562", "hepg2", "hepg2"]
            adata.write_h5ad(data_path)

            config_path.write_text(
                '\n'.join(
                    [
                        'task = "state_perturbation"',
                        "",
                        "[datasets]",
                        f'toy = "{data_path}"',
                        "",
                        "[training]",
                        'toy = "train"',
                        "",
                        "[zeroshot]",
                        '"toy.hepg2" = "test"',
                        "",
                        "[fewshot]",
                        "",
                        "[data]",
                        "use_hvg = false",
                        "",
                        "[data.obs_keys]",
                        'perturbation = "gene"',
                        'batch = "gem_group"',
                        'cell_type = "cell_line"',
                        "",
                        "[data.control]",
                        'label = "ctrl"',
                        "",
                        "[data.feature_space]",
                        'source = "X"',
                        'key = ""',
                    ]
                ),
                encoding="utf-8",
            )

            train_dataset_config = load_dataset_config(config_path)
            artifacts = StateDataArtifacts(
                feature_dim=2,
                raw_feature_dim=2,
                output_dim=2,
                obs_keys={
                    "perturbation": "gene",
                    "batch": "gem_group",
                    "cell_type": "cell_line",
                },
                feature_space={"source": "X", "key": ""},
                use_hvg=False,
                hvg_key="highly_variable",
                condition_sizes={
                    "perturbation": 3,
                    "batch": 1,
                    "cell_type": 2,
                },
                vocab={
                    "perturbation": {
                        "ctrl": 0,
                        "target_gene": 1,
                        "train_gene": 2,
                    },
                    "batch": {"batch1": 0},
                    "cell_type": {"hepg2": 0, "k562": 1},
                },
                default_labels={
                    "perturbation": "ctrl",
                    "batch": "batch1",
                    "cell_type": "hepg2",
                },
                control_label="ctrl",
            )
            infer_config = {
                "input": {
                    "data_path": "",
                    "reference_data_path": "",
                    "split": {
                        "subset": "test",
                        "dataset_config_path": str(config_path),
                    },
                    "obs_keys": {
                        "perturbation": "",
                        "batch": "",
                        "cell_type": "",
                    },
                    "defaults": {
                        "perturbation": "",
                        "batch": "",
                        "cell_type": "",
                    },
                },
                "control": {
                    "enabled": False,
                    "label": "ctrl",
                    "samples_per_query": 4,
                },
                "sampling": {
                    "batch_size": 8,
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                },
            }
            context = DistributedContext(
                rank=0,
                world_size=1,
                local_rank=0,
                device=torch.device("cpu"),
                backend="gloo",
            )
            target_adata, loader, sampler = build_inference_loader(
                infer_config=infer_config,
                train_dataset_config=train_dataset_config,
                artifacts=artifacts,
                distributed_context=context,
            )

            self.assertEqual(target_adata.n_obs, 2)
            self.assertEqual(target_adata.obs["cell_line"].tolist(), ["hepg2", "hepg2"])
            self.assertEqual(target_adata.obs["gene"].tolist(), ["target_gene", "ctrl"])
            self.assertEqual(len(loader.dataset), 2)
            self.assertIsNone(sampler)


if __name__ == "__main__":
    unittest.main()
