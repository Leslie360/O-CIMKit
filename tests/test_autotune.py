import unittest
from core.autotune import AutoTuner

class TestAutotune(unittest.TestCase):
    def test_autotuner(self):
        # Simple target function to maximize
        def mock_eval(spectral_radius, input_scale, leaking_rate, ridge_alpha):
            score = -((spectral_radius - 0.8)**2 + (input_scale - 0.5)**2 + (leaking_rate - 0.3)**2 + (ridge_alpha - 0.1)**2)
            return float(score)
            
        tuner = AutoTuner(target_accuracy_fn=mock_eval, n_trials=10)
        best_params, best_score = tuner.tune()
        
        self.assertIn("spectral_radius", best_params)
        self.assertIn("input_scale", best_params)
        self.assertIsNotNone(best_score)
