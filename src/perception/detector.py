# AV Overconfidence Death Zone Mapper
# Vamshikrishna Gadde | MS Robotics ASU

import numpy as np
from config import *

class AVPerceptionSystem:
    """
    Simulates an AV perception system's
    confidence scoring and actual accuracy
    under progressive sensor degradation.

    Key insight: confidence scores are trained
    on clean data. Under degradation the model
    does not know it is failing — it stays
    confident while accuracy collapses.
    This is the overconfidence death zone.
    """

    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)

    def get_true_accuracy(self, condition,
                          degradation_level):
        curves = {
            'school_zone_occlusion': {
                'drop_start': 0.3,
                'drop_rate': 1.8,
                'floor': 0.08
            },
            'construction_zone_confusion': {
                'drop_start': 0.25,
                'drop_rate': 1.6,
                'floor': 0.10
            },
            'exposure_loss_glare': {
                'drop_start': 0.2,
                'drop_rate': 2.0,
                'floor': 0.05
            },
            'rain_plus_crowd': {
                'drop_start': 0.35,
                'drop_rate': 1.4,
                'floor': 0.12
            },
            'blind_curve_occlusion': {
                'drop_start': 0.4,
                'drop_rate': 1.5,
                'floor': 0.07
            }
        }

        c = curves[condition]
        if degradation_level < c['drop_start']:
            base_accuracy = 0.97
        else:
            excess = (degradation_level
                     - c['drop_start'])
            base_accuracy = max(
                c['floor'],
                0.97 * np.exp(
                    -c['drop_rate'] * excess
                )
            )

        noise = self.rng.normal(0, 0.02)
        return float(np.clip(
            base_accuracy + noise, 0.0, 1.0
        ))

    def get_reported_confidence(self, condition,
                                degradation_level):
        curves = {
            'school_zone_occlusion': {
                'drop_start': 0.55,
                'drop_rate': 0.8,
                'floor': 0.65
            },
            'construction_zone_confusion': {
                'drop_start': 0.50,
                'drop_rate': 0.7,
                'floor': 0.60
            },
            'exposure_loss_glare': {
                'drop_start': 0.45,
                'drop_rate': 0.9,
                'floor': 0.58
            },
            'rain_plus_crowd': {
                'drop_start': 0.60,
                'drop_rate': 0.6,
                'floor': 0.62
            },
            'blind_curve_occlusion': {
                'drop_start': 0.65,
                'drop_rate': 0.7,
                'floor': 0.60
            }
        }

        c = curves[condition]
        if degradation_level < c['drop_start']:
            base_conf = 0.96
        else:
            excess = (degradation_level
                     - c['drop_start'])
            base_conf = max(
                c['floor'],
                0.96 * np.exp(
                    -c['drop_rate'] * excess
                )
            )

        noise = self.rng.normal(0, 0.015)
        return float(np.clip(
            base_conf + noise, 0.0, 1.0
        ))

    def run_trial(self, condition,
                  degradation_level):
        accuracy = self.get_true_accuracy(
            condition, degradation_level
        )
        confidence = self.get_reported_confidence(
            condition, degradation_level
        )

        in_death_zone = (
            confidence > CONFIDENCE_THRESHOLD
            and accuracy < ACCURACY_THRESHOLD
        )

        overconfidence_gap = max(
            0, confidence - accuracy
        )

        return {
            'accuracy': accuracy,
            'confidence': confidence,
            'in_death_zone': in_death_zone,
            'overconfidence_gap': overconfidence_gap
        }

    def run_monte_carlo(self, condition,
                        degradation_level,
                        n_trials=TRIALS_PER_LEVEL):
        results = [
            self.run_trial(
                condition, degradation_level
            )
            for _ in range(n_trials)
        ]

        return {
            'accuracy': np.mean(
                [r['accuracy'] for r in results]
            ),
            'confidence': np.mean(
                [r['confidence'] for r in results]
            ),
            'death_zone_rate': np.mean(
                [r['in_death_zone'] for r in results]
            ),
            'overconfidence_gap': np.mean(
                [r['overconfidence_gap']
                 for r in results]
            ),
            'accuracy_std': np.std(
                [r['accuracy'] for r in results]
            ),
            'confidence_std': np.std(
                [r['confidence'] for r in results]
            )
        }