from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from perturbnova.config import load_infer_config, load_train_config


class TrainConfigTests(unittest.TestCase):
    def test_state_style_dataset_config_normalizes_fewshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "run.toml").write_text(
                '\n'.join(
                    [
                        'task = "state_perturbation"',
                        "",
                        "[datasets]",
                        'toy = "/tmp/toy.h5ad"',
                        "",
                        "[training]",
                        'toy = "train"',
                        "",
                        "[zeroshot]",
                        "",
                        "[fewshot]",
                        '[fewshot."toy.hepg2"]',
                        'val = ["g1"]',
                        'test = ["g2"]',
                    ]
                ),
                encoding="utf-8",
            )

            config = load_train_config(root / "run.toml")

            self.assertEqual(config["dataset"]["name"], "state_perturbation")
            self.assertEqual(config["dataset"]["source_name"], "toy")
            self.assertEqual(config["dataset"]["data_path"], "/tmp/toy.h5ad")
            self.assertEqual(config["dataset"]["split"]["mode"], "fewshot_holdout")
            self.assertEqual(config["dataset"]["split"]["target_cell_type"], "hepg2")
            self.assertEqual(config["dataset"]["split"]["val_perturbations"], ["g1"])
            self.assertEqual(config["dataset"]["split"]["test_perturbations"], ["g2"])

    def test_state_style_dataset_config_normalizes_zeroshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "run.toml").write_text(
                '\n'.join(
                    [
                        'task = "state_perturbation"',
                        "",
                        "[datasets]",
                        'toy = "/tmp/toy.h5ad"',
                        "",
                        "[training]",
                        'toy = "train"',
                        "",
                        "[zeroshot]",
                        '"toy.jurkat" = "test"',
                        "",
                        "[fewshot]",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_train_config(root / "run.toml")

            self.assertEqual(config["dataset"]["split"]["mode"], "zeroshot_holdout")
            self.assertEqual(config["dataset"]["split"]["target_cell_type"], "jurkat")
            self.assertEqual(config["dataset"]["split"]["zeroshot_role"], "test")

    def test_task_alias_populates_dataset_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "run.toml").write_text(
                '\n'.join(
                    [
                        "[data]",
                        'task = "state_perturbation"',
                        'data_path = "/tmp/replogle_subset.h5ad"',
                    ]
                ),
                encoding="utf-8",
            )

            config = load_train_config(root / "run.toml")

            self.assertEqual(config["dataset"]["name"], "state_perturbation")

    def test_data_alias_merges_before_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "data.toml").write_text(
                '\n'.join(
                    [
                        "[data]",
                        'data_path = "/tmp/replogle_subset.h5ad"',
                        "",
                        "[data.split]",
                        'target_cell_type = "hepg2"',
                    ]
                ),
                encoding="utf-8",
            )
            (root / "split.toml").write_text(
                '\n'.join(
                    [
                        "[dataset.split]",
                        'target_cell_type = "jurkat"',
                    ]
                ),
                encoding="utf-8",
            )
            (root / "run.toml").write_text(
                '\n'.join(
                    [
                        'base_configs = ["./data.toml", "./split.toml"]',
                        "",
                        "[training]",
                        'stage = "stage2"',
                    ]
                ),
                encoding="utf-8",
            )

            config = load_train_config(root / "run.toml")

            self.assertEqual(config["dataset"]["data_path"], "/tmp/replogle_subset.h5ad")
            self.assertEqual(config["dataset"]["split"]["target_cell_type"], "jurkat")
            self.assertEqual(config["training"]["mode"], "diffusion_only")
            self.assertTrue(config["vae"]["freeze"])

    def test_stage2_unfreezes_vae_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "run.toml").write_text(
                '\n'.join(
                    [
                        "[dataset]",
                        'data_path = "/tmp/replogle_subset.h5ad"',
                        "",
                        "[training]",
                        'stage = "stage2"',
                        "unfreeze_vae_in_stage2 = true",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_train_config(root / "run.toml")

            self.assertEqual(config["training"]["mode"], "joint")
            self.assertFalse(config["vae"]["freeze"])

    def test_objective_loss_weights_normalize_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "run.toml").write_text(
                '\n'.join(
                    [
                        "[dataset]",
                        'data_path = "/tmp/replogle_subset.h5ad"',
                        "",
                        "[objective]",
                        'control_usage = "loss_only"',
                        "xstart_mse_weight = 0.1",
                        "",
                        "[objective.loss_weights]",
                        "effect_batch_mse = 0.2",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_train_config(root / "run.toml")

            self.assertEqual(config["objective"]["control_usage"], "loss_only")
            self.assertEqual(config["objective"]["loss_weights"]["diffusion_mse"], 1.0)
            self.assertEqual(config["objective"]["loss_weights"]["xstart_mse"], 0.1)
            self.assertEqual(config["objective"]["loss_weights"]["effect_batch_mse"], 0.2)

    def test_all_replogle_stage_run_configs_load(self) -> None:
        roots = [
            Path("/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/configs/replogle_fewshot/runs"),
            Path("/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/configs/replogle_zeroshot/runs"),
        ]
        stage_files = []
        for root in roots:
            stage_files.extend(sorted(root.glob("*/*.toml")))
        stage_files = [
            path
            for path in stage_files
            if path.name in {"stage1.toml", "stage2.toml", "stage2_unfrozen_vae.toml", "stage2_unfrozen_vae_v1.toml"}
        ]
        self.assertTrue(stage_files)
        for path in stage_files:
            config = load_train_config(path)
            self.assertTrue(config["dataset"]["data_path"])
            self.assertIn(config["training"]["mode"], {"vae_only", "diffusion_only", "joint"})


class InferConfigTests(unittest.TestCase):
    def test_squidiff_profile_alias_normalizes_to_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "infer.toml").write_text(
                "\n".join(
                    [
                        "[checkpoint]",
                        "path = \"/tmp/ckpt.pt\"",
                        "",
                        "[input]",
                        "data_path = \"/tmp/data.h5ad\"",
                        "",
                        "[cell_eval]",
                        "profile = \"squidiff\"",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_infer_config(root / "infer.toml")

            self.assertEqual(config["cell_eval"]["profile"], "full")

    def test_diffusion_overrides_load_for_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "infer.toml").write_text(
                "\n".join(
                    [
                        "[checkpoint]",
                        "path = \"/tmp/ckpt.pt\"",
                        "",
                        "[input]",
                        "data_path = \"/tmp/data.h5ad\"",
                        "",
                        "[diffusion]",
                        "timestep_respacing = \"ddim50\"",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_infer_config(root / "infer.toml")

            self.assertEqual(config["diffusion"]["timestep_respacing"], "ddim50")

    def test_base_infer_relative_dataset_config_path_resolves_from_base_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_dir = root / "base"
            child_dir = root / "child"
            data_dir = root / "data"
            base_dir.mkdir()
            child_dir.mkdir()
            data_dir.mkdir()

            (data_dir / "dataset.toml").write_text(
                "\n".join(
                    [
                        "[dataset]",
                        'data_path = "/tmp/data.h5ad"',
                    ]
                ),
                encoding="utf-8",
            )
            (base_dir / "infer_base.toml").write_text(
                "\n".join(
                    [
                        "[checkpoint]",
                        'path = "/tmp/ckpt.pt"',
                        "",
                        "[input.split]",
                        'subset = "test"',
                        'dataset_config_path = "../data/dataset.toml"',
                    ]
                ),
                encoding="utf-8",
            )
            (child_dir / "infer.toml").write_text(
                '\n'.join(
                    [
                        'base_configs = ["../base/infer_base.toml"]',
                    ]
                ),
                encoding="utf-8",
            )

            config = load_infer_config(child_dir / "infer.toml")

            self.assertEqual(
                config["input"]["split"]["dataset_config_path"],
                str((data_dir / "dataset.toml").resolve()),
            )


    def test_all_replogle_infer_configs_load(self) -> None:
        roots = [
            Path("/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/configs/replogle_fewshot/inference"),
            Path("/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/configs/replogle_zeroshot/inference"),
        ]
        infer_files = []
        for root in roots:
            infer_files.extend(sorted(root.glob("*/*.toml")))
        infer_files = [path for path in infer_files if path.name in {"infer.toml", "infer_v1.toml"}]
        self.assertTrue(infer_files)
        for path in infer_files:
            config = load_infer_config(path)
            self.assertTrue(config["checkpoint"]["path"])
            self.assertEqual(config["input"]["split"]["subset"], "test")
            self.assertEqual(config["cell_eval"]["profile"], "full")

    def test_split_subset_allows_data_path_to_be_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "run.toml").write_text(
                '\n'.join(
                    [
                        "[checkpoint]",
                        'path = "/tmp/model.pt"',
                        "",
                        "[input.split]",
                        'subset = "test"',
                    ]
                ),
                encoding="utf-8",
            )

            config = load_infer_config(root / "run.toml")

            self.assertEqual(config["input"]["split"]["subset"], "test")
            self.assertEqual(config["input"]["data_path"], "")

    def test_invalid_infer_split_subset_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "run.toml").write_text(
                '\n'.join(
                    [
                        "[checkpoint]",
                        'path = "/tmp/model.pt"',
                        "",
                        "[input]",
                        'data_path = "/tmp/input.h5ad"',
                        "",
                        "[input.split]",
                        'subset = "holdout"',
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "`input.split.subset`"):
                load_infer_config(root / "run.toml")


if __name__ == "__main__":
    unittest.main()
