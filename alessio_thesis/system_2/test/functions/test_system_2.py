import numpy as np
import torch
import sys
import os
from torch.utils.data import DataLoader
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from system_2.utils.inference_utils import run_inference


class TestRunInference(unittest.TestCase):

    model = None
    device = "cpu"

    def _make_case_test(case_idx, vector_h_cloud, correct_label):
        def _test(self):
            preds, probs = run_inference(self.model, vector_h_cloud,
                                        device=self.device, return_probs=True)
            pred = preds[0] 
            prob_vec = probs[0]
            self.assertEqual(
                pred, correct_label,
                msg=f"[case {case_idx}] predicted={pred}, expected={correct_label}, prob={prob_vec[pred]:.4f}"
            )
        _test.__name__ = f"test_eval_new_data_{case_idx}"
        return _test
