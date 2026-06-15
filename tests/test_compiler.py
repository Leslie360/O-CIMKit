import unittest
import os
import torch
from core.compiler import BionicCoDesignCompiler

class TestCompiler(unittest.TestCase):
    def test_compiler_compilation(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        profile_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
        if not os.path.exists(profile_path):
            return
            
        compiler = BionicCoDesignCompiler(profile_path)
        settings = compiler.compile()
        
        self.assertIn("suggested_lr", settings)
        self.assertIn("quantizer_mode", settings)
        self.assertIn("layer_recommendation", settings)

    def test_model_synthesis(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        profile_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
        if not os.path.exists(profile_path):
            return
            
        compiler = BionicCoDesignCompiler(profile_path)
        model_ff = compiler.synthesize_model("feedforward", in_features=10, out_features=2)
        self.assertIsNotNone(model_ff)
        
        x = torch.randn(4, 10)
        out = model_ff(x)
        self.assertEqual(out.shape, (4, 2))
