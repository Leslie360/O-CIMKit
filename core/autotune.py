import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import logging

class AutoTuner:
    """
    AutoTuner optimizes hyperparameters of organic reservoirs for custom device profiles.
    Tunes spectral radius, input scaling, leaking rate, and regularization ridge parameters.
    """
    def __init__(self, target_accuracy_fn, n_trials=30):
        """
        Args:
            target_accuracy_fn (callable): Function taking (spectral_radius, input_scale, leaking_rate, ridge_alpha) 
                                           and returning accuracy metric.
            n_trials (int): Number of optimization trials.
        """
        self.target_fn = target_accuracy_fn
        self.n_trials = n_trials
        
    def tune(self):
        print("🔍 Starting AutoTuner hyperparameter search...")
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                spectral_radius = trial.suggest_float("spectral_radius", 0.5, 1.2)
                input_scale = trial.suggest_float("input_scale", 0.1, 2.0)
                leaking_rate = trial.suggest_float("leaking_rate", 0.01, 1.0)
                ridge_alpha = trial.suggest_float("ridge_alpha", 1e-5, 1e2, log=True)
                
                return self.target_fn(spectral_radius, input_scale, leaking_rate, ridge_alpha)
                
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials)
            
            print("🏆 Tuning completed successfully using Optuna!")
            print(f"  Best accuracy: {study.best_value:.4f}")
            print("  Best hyperparameters:")
            for k, v in study.best_params.items():
                print(f"    - {k}: {v}")
            return study.best_params, study.best_value
            
        except ImportError:
            print("⚠️ Optuna is not installed. Falling back to robust Grid Search...")
            # Fallback to smart grid search
            spectral_radii = [0.7, 0.9, 1.1]
            input_scales = [0.3, 0.8, 1.5]
            leaking_rates = [0.1, 0.5, 0.8]
            ridge_alphas = [1e-3, 1.0]
            
            best_val = -1.0
            best_params = {}
            
            count = 0
            total_searches = len(spectral_radii) * len(input_scales) * len(leaking_rates) * len(ridge_alphas)
            
            for sr in spectral_radii:
                for iscale in input_scales:
                    for lr in leaking_rates:
                        for ra in ridge_alphas:
                            count += 1
                            val = self.target_fn(sr, iscale, lr, ra)
                            if val > best_val:
                                best_val = val
                                best_params = {
                                    "spectral_radius": sr,
                                    "input_scale": iscale,
                                    "leaking_rate": lr,
                                    "ridge_alpha": ra
                                }
                            if count % 10 == 0:
                                print(f"  Grid Search progress: {count}/{total_searches} trials completed.")
                                
            print("🏆 Tuning completed successfully using Grid Search!")
            print(f"  Best accuracy: {best_val:.4f}")
            print("  Best hyperparameters:")
            for k, v in best_params.items():
                print(f"    - {k}: {v}")
            return best_params, best_val
