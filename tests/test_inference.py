"""CPU-only checks; execute from an exported checkout with unittest discover -s tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from examples.make_demo import make_demo
from henets import _engine, infer


class PortableInference(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        # OpenSlide detects the actual Aperio TIFF rather than requiring .svs.
        self.source = self.root/"synthetic.tiff"
        make_demo(self.source)
        self.cfg = self.root/"config.json"
        self.data = dict(samples=[dict(sample_id="demo", he_path="synthetic.tiff")],
                         output_directory="runs", weights_root="weights", ctranspath_root="encoder")
        self.cfg.write_text(json.dumps(self.data), encoding="utf-8")

    def test_real_calibrated_preview_and_grid(self):
        folder = infer.preview(self.cfg, "demo")
        slide = json.loads((folder/"slide.json").read_text(encoding="utf-8"))
        self.assertEqual(slide["analysis_shape"], [1024,1024])
        self.assertEqual(slide["header"]["mpp_x"], 0.325)
        self.assertEqual(slide["header"]["reader"], "OpenSlide")
        self.assertEqual(len(_engine.load_csv(folder/"grid.csv")), 4)
        self.assertEqual(infer.preview(self.cfg, "demo"), folder)
        self.assertTrue((folder/"tissue_selection.png").is_file())
        self.assertFalse((folder.parent/"metadata.json").exists())

    def test_reject_invalid_id_and_changed_geometry(self):
        for sid,conf in [("../escape",{}), ("demo",{"analysis_mpp":0.65})]:
            self.data.update(conf)
            self.cfg.write_text(json.dumps(self.data),encoding="utf-8")
            with self.assertRaises(ValueError):
                infer.load_config(self.cfg,sid)

    def test_reject_wrong_external_code_before_loading_checkpoints(self):
        directory=self.root/"encoder/path2space/frozen/ctrans";directory.mkdir(parents=True)
        (directory/"__init__.py").write_text("raise Exception('must not execute')", encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"encoder source mismatch"):
            infer.verify_models(infer.load_config(self.cfg,"demo"))

    def test_unsupported_multifile_and_vendor_sdk_formats_are_rejected(self):
        for suffix in (".mrxs",".vms",".vmu",".kfb",".png"):
            source=self.root/("multi"+suffix);source.write_bytes(b"fixture")
            self.data["samples"][0]["he_path"]=source.name
            self.cfg.write_text(json.dumps(self.data),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"single-file"):
                infer.load_config(self.cfg,"demo")

    def test_cpu_rejects_reference_inference_instead_of_switching_metrics(self):
        with patch("torch.cuda.is_available",return_value=False):
            with self.assertRaisesRegex(RuntimeError,"CUDA"):
                infer.run(self.cfg,"demo")

    def test_reference_score_and_spatial_centers(self):
        np.testing.assert_allclose(_engine.he_nets_score(np.array([.25, -1, 3]),np.array([.64, .4, 1])), [40,0,100])
        folder=infer.preview(self.cfg,"demo")
        info=json.loads((folder/"slide.json").read_text(encoding="utf-8"))
        with _engine.OpenSlideReader() as reader:
            ctx=_engine.read_context(reader,info,0,0,512)
            native=reader.read_region(self.source,144,144,224,224)
        np.testing.assert_array_equal(ctx[400:624,400:624], native[...,::-1])
        self.assertEqual(ctx.shape,(1024,1024,3))


if __name__=="__main__":
    unittest.main()
