"""GPU-free checks for the pinned upstream 0.2.12 offline launch profile."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PREPARE = Path(__file__).resolve().parents[1] / "packaging" / "offline_iq3xxs_0212" / "prepare.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location("upstream_0212_prepare", PREPARE)
assert spec and spec.loader
prepare_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_mod)


class OfflineProfile(unittest.TestCase):
    def test_config_uses_original_engine_mtp_and_256k(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pack = root / "packs" / "iq3_xxs"
            (pack / "tokenizer").mkdir(parents=True)
            (pack / "native_experts.txt").write_text("ready")
            (pack / "tokenizer" / "vocab.json").write_text("{}")
            with (mock.patch.object(prepare_mod, "ROOT", root), mock.patch.object(prepare_mod, "PACK", pack),
                  mock.patch.object(prepare_mod, "MTP", root / "mtp" / "rt"),
                  mock.patch.object(prepare_mod, "MMPROJ", root / "models" / "mmproj.gguf"),
                  mock.patch.object(prepare_mod, "SHARDS", [root / "models" / "first.gguf", root / "models" / "second.gguf"]),
                  mock.patch.object(prepare_mod, "CONFIG", root / "strata-iq3_xxs.json"),
                  mock.patch.object(prepare_mod, "check_driver"), mock.patch.object(prepare_mod, "check_files")):
                config = prepare_mod.prepare("mtp")
            saved = json.loads((root / "strata-iq3_xxs.json").read_text())
            self.assertEqual(config, saved)
            args = config["args"]
            self.assertEqual(args[args.index("--max-context") + 1], "262144")
            self.assertEqual(args[args.index("--kv-resident") + 1], "32768")
            self.assertEqual(args[args.index("--kv") + 1], "int8")
            self.assertEqual(args[args.index("--prefill") + 1], "auto")
            self.assertEqual(config["drafting"], "mtp")
            self.assertIn("--mtp", args)
            self.assertNotIn("--draft", args)
            self.assertEqual(config["host"], "127.0.0.1")
            self.assertNotIn("auto_prefix_min_chars", config)

    def test_driver_floor_comes_from_original_engine_manifest(self):
        query = mock.Mock(returncode=0, stdout="NVIDIA RTX A6000, 580.65, 49140 MiB\n")
        with mock.patch.object(prepare_mod.subprocess, "run", return_value=query):
            self.assertIn("RTX A6000", prepare_mod.check_driver())
        query.stdout = "NVIDIA RTX A6000, 580.64, 49140 MiB\n"
        with mock.patch.object(prepare_mod.subprocess, "run", return_value=query):
            with self.assertRaisesRegex(RuntimeError, "580.65"):
                prepare_mod.check_driver()

    def test_dflash2_choice_checks_the_addon_and_selects_one_drafter(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pack = root / "dflash2" / "test.pack"
            pack.parent.mkdir()
            pack.write_bytes(b"verified drafter")
            with (mock.patch.object(prepare_mod, "DRAFT_PACK", pack),
                  mock.patch.object(prepare_mod, "DRAFT_BYTES", pack.stat().st_size),
                  mock.patch.object(prepare_mod, "DRAFT_SHA256", hashlib.sha256(pack.read_bytes()).hexdigest())):
                prepare_mod.check_dflash_pack()
                pack.write_bytes(b"changed drafter!")
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    prepare_mod.check_dflash_pack()
            with (mock.patch.object(prepare_mod, "ROOT", root),
                  mock.patch.object(prepare_mod, "PACK", root / "packs" / "iq3_xxs"),
                  mock.patch.object(prepare_mod, "MTP", root / "mtp" / "rt"),
                  mock.patch.object(prepare_mod, "MMPROJ", root / "models" / "mmproj.gguf"),
                  mock.patch.object(prepare_mod, "SHARDS", [root / "models" / "first.gguf", root / "models" / "second.gguf"]),
                  mock.patch.object(prepare_mod, "CONFIG", root / "config.json"),
                  mock.patch.object(prepare_mod, "check_driver"),
                  mock.patch.object(prepare_mod, "check_files") as checked):
                (root / "packs" / "iq3_xxs" / "tokenizer").mkdir(parents=True)
                (root / "packs" / "iq3_xxs" / "native_experts.txt").write_text("ready")
                (root / "packs" / "iq3_xxs" / "tokenizer" / "vocab.json").write_text("{}")
                config = prepare_mod.prepare()
            checked.assert_called_once_with("dflash2")
            self.assertEqual(config["drafting"], "dflash2")
            self.assertIn("--mtp", config["args"])  # original server keeps the MTP path for its draft vocabulary

    def test_dflash2_base_needs_no_mtp_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            engine = root / "engine"
            engine.mkdir()
            for name in ("strata.exe", "strata-vision.exe", "cublas64_13.dll", "cublasLt64_13.dll"):
                (engine / name).touch()
            (engine / "BUILD.json").write_text('{"version":"0.1.58","archs":[86]}')
            data = root / "data"
            data.mkdir()
            for name in ("expert-profile.bin", "draft_vocab.bin"):
                (data / name).touch()
            models = root / "models"
            models.mkdir()
            mmproj = models / "mmproj.gguf"
            shard1, shard2 = models / "first.gguf", models / "second.gguf"
            for path, size in ((mmproj, 500_000_001), (shard1, 1_000_000_001), (shard2, 1_000_000_001)):
                with path.open("wb") as stream:
                    stream.truncate(size)  # sparse files; the reader below supplies only metadata
            first_tensor = mock.Mock(offset=0, expected_bytes=lambda: 1, type_name="IQ3_XXS")
            first_tensor.name = "expert"
            second_tensor = mock.Mock(offset=0, expected_bytes=lambda: 1, type_name="F16")
            second_tensor.name = "per_layer_token_embd.weight"
            first = mock.Mock(data_start=0, tensors=[first_tensor])
            second = mock.Mock(data_start=0, tensors=[second_tensor])
            with (mock.patch.object(prepare_mod, "ROOT", root),
                  mock.patch.object(prepare_mod, "MMPROJ", mmproj),
                  mock.patch.object(prepare_mod, "MTP", root / "mtp" / "rt"),
                  mock.patch.object(prepare_mod, "SHARDS", [shard1, shard2]),
                  mock.patch.object(prepare_mod, "GGUFFile", side_effect=[first, second]),
                  mock.patch.object(prepare_mod, "check_dflash_pack") as pack_check):
                prepare_mod.check_files("dflash2")
                pack_check.assert_called_once()
                with self.assertRaisesRegex(RuntimeError, "MTP 파일이 없습니다"):
                    prepare_mod.check_files("mtp")


if __name__ == "__main__":
    unittest.main()
