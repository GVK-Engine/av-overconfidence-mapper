# AV Overconfidence Death Zone Mapper
# Vamshikrishna Gadde | MS Robotics ASU
# September 2026 — grounded in real incidents

CONDITIONS = [
    'school_zone_occlusion',
    'construction_zone_confusion',
    'exposure_loss_glare',
    'rain_plus_crowd',
    'blind_curve_occlusion'
]

CONDITION_LABELS = {
    'school_zone_occlusion':       'School Zone Occlusion',
    'construction_zone_confusion': 'Construction Zone',
    'exposure_loss_glare':         'Exposure Loss + Glare',
    'rain_plus_crowd':             'Rain + Crowd',
    'blind_curve_occlusion':       'Blind Curve'
}

CONDITION_INCIDENTS = {
    'school_zone_occlusion':
        'Waymo child strike Jan 2026',
    'construction_zone_confusion':
        'Waymo 3871 vehicle recall 2026',
    'exposure_loss_glare':
        'Motional nuReasoning edge case',
    'rain_plus_crowd':
        'Motional nuReasoning edge case',
    'blind_curve_occlusion':
        'Waymo stopped school bus 2026'
}

DEGRADATION_LEVELS = [
    0.0, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]

TRIALS_PER_LEVEL = 200

CONFIDENCE_THRESHOLD = 0.80
ACCURACY_THRESHOLD = 0.50

RESULTS_DIR = "D:/av-overconfidence-mapper/results"
CHART_DPI = 150
CHART_BG = "#0d0d0d"
CHART_TEXT = "white"
TITLE_NAME = "Vamshikrishna Gadde | MS Robotics ASU"
SEED = 42

INCIDENT_DEGRADATION_ESTIMATES = {
    'Waymo Child Strike Jan 2026':
        ('school_zone_occlusion', 0.65),
    'Waymo Construction Recall 2026':
        ('construction_zone_confusion', 0.55),
    'Waymo School Bus 2026':
        ('blind_curve_occlusion', 0.70),
    'Motional Edge Case Dataset':
        ('exposure_loss_glare', 0.60)
}