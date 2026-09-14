"""Source-level release contract; does not claim CUDA runtime validation."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ExperimentalDefaultsTest(unittest.TestCase):
    def test_experimental_modes_are_opt_in(self):
        source = (ROOT / "apps/DensifyPointCloud/DensifyPointCloud.cpp").read_text()
        for option in (
            "patch-match-cuda-apd", "patch-match-cuda-dvp-epipolar-family",
            "patch-match-cuda-dvp-depth-edge-mode", "patch-match-cuda-dvp-visibility-mode",
            "patch-match-cuda-dvp-visible-normal-mode",
        ):
            with self.subTest(option=option):
                declaration = re.search(r'\("' + re.escape(option) + r'"[^\n]+', source)
                self.assertIsNotNone(declaration)
                self.assertIn("->default_value(0)", declaration.group())

    def test_observer_build_is_opt_in(self):
        source = (ROOT / "CMakeLists.txt").read_text()
        self.assertRegex(source, r'OPTION\(OpenMVS_DMAP_INSTRUMENTATION\s+"[^"\n]*"\s+OFF\)')

    def test_help_does_not_claim_full_paper_reproduction(self):
        for path in ("apps/DensifyPointCloud/DensifyPointCloud.cpp", "libs/MVS/DepthMap.cpp"):
            with self.subTest(path=path):
                source = (ROOT / path).read_text()
                self.assertNotIn("full paper mechanics", source)
                self.assertIn("1 - adaptive support, 2 - deformation-only ablation", source)


if __name__ == "__main__":
    unittest.main()
