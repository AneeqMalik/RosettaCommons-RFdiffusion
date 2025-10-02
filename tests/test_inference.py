import os
import tempfile
import unittest
from unittest.mock import patch

from src import inference


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_dir = tempfile.TemporaryDirectory()
        self.model_dir = tempfile.TemporaryDirectory()
        self.output_dir = tempfile.TemporaryDirectory()

        scripts_dir = os.path.join(self.repo_dir.name, "scripts")
        os.makedirs(scripts_dir, exist_ok=True)
        self.script_path = os.path.join(scripts_dir, "run_inference.py")
        with open(self.script_path, "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env python\nprint('stub')\n")

        os.environ["RF_DIFFUSION_HOME"] = self.repo_dir.name
        os.environ["RF_OUTPUT_DIR"] = self.output_dir.name

        self.context = inference.load_model(self.model_dir.name)

    def tearDown(self) -> None:
        os.environ.pop("RF_DIFFUSION_HOME", None)
        os.environ.pop("RF_OUTPUT_DIR", None)
        self.repo_dir.cleanup()
        self.model_dir.cleanup()
        self.output_dir.cleanup()

    def test_predict_builds_command_and_collects_outputs(self) -> None:
        body = {
            "contigs": "[150-150]",
            "num_designs": 1,
            "hydra_overrides": ["diffuser.T=5"],
            "inline_outputs": True,
            "run_name": "unit_test",
        }

        def fake_run(*args, **kwargs):
            cmd = args[0]
            self.assertIn("contigmap.contigs=[150-150]", cmd)
            self.assertIn("diffuser.T=5", cmd)

            prefix_arg = next((a for a in cmd if a.startswith("inference.output_prefix=")), None)
            self.assertIsNotNone(prefix_arg)
            output_prefix = prefix_arg.split("=", 1)[1]
            os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
            with open(f"{output_prefix}_0.pdb", "w", encoding="utf-8") as handle:
                handle.write("HEADER test\nEND\n")

            class Result:
                returncode = 0
                stdout = "ok"
                stderr = ""

            return Result()

        with patch("subprocess.run", side_effect=fake_run) as mocked_run:
            response = inference.predict(body, self.context)

        self.assertTrue(mocked_run.called)
        self.assertEqual(response["status"], "ok")
        self.assertIn("base64", response["outputs"][0])
        self.assertEqual(response["outputs"][0]["filename"], "unit_test_0.pdb")

    def test_invalid_payload_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            inference.predict("not-a-dict", self.context)


if __name__ == "__main__":
    unittest.main()
