"""Target/draft model packages (LLiMa 0.6.0 speculative decoding).

A paired package, such as a DFlash or Gemma 4 MTP build, keeps each model in
its own subfolder with its own ``devkit/`` and ``elf_files/``. The draft's
``vlm_config.json`` says ``is_draft``. The catalog reads metadata from the
target, counts accelerator memory and load stages across both, and loads the
package from its root.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from shared.config import classify_model_dir, model_dir_complete, model_runtime_dirs


def _component(root: Path, name: str, *, draft: bool, elfs: dict[str, int],
               vision: bool = False) -> Path:
    d = root / name
    (d / "devkit").mkdir(parents=True)
    (d / "elf_files").mkdir()
    cfg = {"lm_cfg": {"speculative_decoding_cfg": {"is_draft": draft}},
           "pipeline_cfg": {"max_num_tokens": 4096 if not draft else 1024}}
    if vision:
        cfg["vision_cfg"] = {"image_size": {"height": 448, "width": 448}}
    (d / "devkit" / "vlm_config.json").write_text(json.dumps(cfg))
    for elf, size in elfs.items():
        (d / "elf_files" / elf).write_bytes(b"x" * size)
    return d


class PairedPackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.pkg = self.tmp / "gemma-4-E2B-it-mtp"
        self.pkg.mkdir()
        # Sorted by name the draft comes first, so order must come from is_draft.
        self.draft = _component(self.pkg, "assistant", draft=True,
                                elfs={"d_stage0_mla.elf": 10})
        self.target = _component(self.pkg, "target", draft=False,
                                 elfs={"t_stage0_mla.elf": 300, "t_stage1_mla.elf": 200})
        (self.pkg / ".neat-complete").write_text("ok\n")

    def test_the_target_comes_before_its_draft(self):
        self.assertEqual(model_runtime_dirs(self.pkg), [self.target, self.draft])

    def test_a_flat_model_is_its_own_component(self):
        flat = _component(self.tmp, "flat", draft=False, elfs={"s_mla.elf": 1})
        self.assertEqual(model_runtime_dirs(flat), [flat])

    def test_the_package_is_listed_and_loaded_from_its_root(self):
        info = classify_model_dir(self.pkg)
        self.assertEqual(info["name"], "gemma-4-E2B-it-mtp")
        self.assertEqual(info["path"], self.pkg)
        self.assertEqual(info["type"], "chat")

    def test_completeness_checks_every_component(self):
        (self.pkg / ".neat-complete").unlink()     # judge from the files alone
        self.assertEqual(model_dir_complete(self.pkg), (True, ""))
        (self.draft / "elf_files" / "d_stage0_mla.elf").write_bytes(b"")
        ok, _why = model_dir_complete(self.pkg)
        self.assertFalse(ok)

    def test_a_folder_without_a_model_is_not_listed(self):
        (self.tmp / "notes").mkdir()
        self.assertIsNone(classify_model_dir(self.tmp / "notes"))


class PairedPackageManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        pkg = self.tmp / "pair"
        pkg.mkdir()
        _component(pkg, "draft", draft=True, elfs={"d_mla.elf": 10})
        _component(pkg, "target", draft=False, elfs={"a_mla.elf": 300, "b_mla.elf": 200})
        (pkg / ".neat-complete").write_text("ok\n")
        self.pkg = pkg

    def test_memory_stages_and_context_cover_the_pair(self):
        from server.model_manager import ModelManager
        manager = ModelManager.__new__(ModelManager)
        manager._context_cache = {}
        self.assertEqual(manager._elf_bytes(self.pkg), 510)
        self.assertEqual(manager._count_elf_stages(self.pkg), 3)
        self.assertEqual(manager._context_length(self.pkg), 4096)   # the target's


if __name__ == "__main__":
    unittest.main()
