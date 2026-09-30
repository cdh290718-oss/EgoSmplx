"""Stable names for the current algorithm and the reproducible archived route."""
CURRENT = 'mano_guided_original_seam_20260929_v1'
LEGACY = 'session_hand6_full_pipeline_20260926_v1'
SUPPORTED = (CURRENT, LEGACY)


def selected(config):
    profile = config.get('pipeline_profile', CURRENT)
    if profile not in SUPPORTED:
        raise ValueError('Unsupported pipeline_profile: ' + str(profile))
    return profile
