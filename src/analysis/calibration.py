# AV Overconfidence Death Zone Mapper
# Vamshikrishna Gadde | MS Robotics ASU

import numpy as np
import json
import os
from tqdm import tqdm
from config import *
from src.perception.detector import AVPerceptionSystem

class OverconfidenceAnalyzer:
    """
    Runs full experiment across all conditions
    and degradation levels.
    Finds the exact death zone boundary per
    condition — the degradation level where
    confidence and accuracy diverge fatally.
    """

    def __init__(self):
        self.detector = AVPerceptionSystem()

    def find_death_zone_entry(self,
                               accuracies,
                               confidences,
                               levels):
        """
        Find exact degradation level where
        system enters the overconfidence
        death zone.
        Death zone: confidence > 0.80
        AND accuracy < 0.50
        Uses linear interpolation between
        the two bracketing levels.
        """
        for i in range(len(levels) - 1):
            acc_now = accuracies[i]
            conf_now = confidences[i]
            acc_next = accuracies[i + 1]
            conf_next = confidences[i + 1]

            in_zone_now = (
                conf_now > CONFIDENCE_THRESHOLD
                and acc_now < ACCURACY_THRESHOLD
            )
            in_zone_next = (
                conf_next > CONFIDENCE_THRESHOLD
                and acc_next < ACCURACY_THRESHOLD
            )

            if not in_zone_now and in_zone_next:
                # Interpolate exact boundary
                t = ((ACCURACY_THRESHOLD - acc_now)
                     / (acc_next - acc_now + 1e-9))
                boundary = (levels[i]
                           + t * (levels[i+1]
                                 - levels[i]))
                return round(float(boundary), 3)

        return None

    def find_divergence_point(self,
                               accuracies,
                               confidences,
                               levels,
                               gap_threshold=0.2):
        """
        Find where confidence and accuracy
        first diverge by more than gap_threshold.
        This is earlier than the death zone —
        it is the warning boundary.
        """
        for i, level in enumerate(levels):
            gap = confidences[i] - accuracies[i]
            if gap > gap_threshold:
                if i > 0:
                    prev_gap = (confidences[i-1]
                               - accuracies[i-1])
                    t = ((gap_threshold - prev_gap)
                         / (gap - prev_gap + 1e-9))
                    boundary = (levels[i-1]
                               + t * (levels[i]
                                     - levels[i-1]))
                    return round(float(boundary), 3)
                return round(float(level), 3)
        return None

    def run_full_analysis(self):
        """
        Run complete experiment across all
        conditions and degradation levels.
        Returns full results dict.
        """
        print("AV Overconfidence Death Zone Mapper")
        print(f"Conditions: {len(CONDITIONS)}")
        print(f"Degradation levels: "
              f"{len(DEGRADATION_LEVELS)}")
        print(f"Trials per level: {TRIALS_PER_LEVEL}")
        total = (len(CONDITIONS)
                * len(DEGRADATION_LEVELS)
                * TRIALS_PER_LEVEL)
        print(f"Total trials: {total:,}")

        results = {}

        for condition in CONDITIONS:
            label = CONDITION_LABELS[condition]
            print(f"\nAnalyzing: {label}")

            accuracies = []
            confidences = []
            death_zone_rates = []
            gaps = []
            acc_stds = []
            conf_stds = []

            for level in tqdm(
                DEGRADATION_LEVELS,
                desc="  Degradation sweep",
                ncols=70
            ):
                trial_result = (
                    self.detector.run_monte_carlo(
                        condition, level
                    )
                )
                accuracies.append(
                    trial_result['accuracy']
                )
                confidences.append(
                    trial_result['confidence']
                )
                death_zone_rates.append(
                    trial_result['death_zone_rate']
                )
                gaps.append(
                    trial_result['overconfidence_gap']
                )
                acc_stds.append(
                    trial_result['accuracy_std']
                )
                conf_stds.append(
                    trial_result['confidence_std']
                )

            death_zone_boundary = (
                self.find_death_zone_entry(
                    accuracies, confidences,
                    DEGRADATION_LEVELS
                )
            )

            divergence_point = (
                self.find_divergence_point(
                    accuracies, confidences,
                    DEGRADATION_LEVELS
                )
            )

            results[condition] = {
                'accuracies': accuracies,
                'confidences': confidences,
                'death_zone_rates': death_zone_rates,
                'gaps': gaps,
                'acc_stds': acc_stds,
                'conf_stds': conf_stds,
                'death_zone_boundary':
                    death_zone_boundary,
                'divergence_point':
                    divergence_point,
                'levels': DEGRADATION_LEVELS
            }

            print(f"  Death zone entry: "
                  f"{death_zone_boundary}")
            print(f"  Divergence point: "
                  f"{divergence_point}")

        return results

    def save_results(self, results):
        """Save results to JSON."""
        path = os.path.join(
            RESULTS_DIR, 'logs',
            'analysis.json'
        )
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        serializable = {}
        for condition, data in results.items():
            serializable[condition] = {
                k: (v if not isinstance(v, list)
                    else [float(x) for x in v])
                for k, v in data.items()
            }

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved: {path}")
        return path