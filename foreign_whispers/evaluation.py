"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }

def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    align_report: dict | None = None,
) -> dict:
    """Multi-dimensional dubbing quality scorecard (improved).

    Returns normalized scores in [0,1] for:
        - timing_accuracy
        - naturalness
        - semantic_fidelity
        - intelligibility
        - overall
    """

    import statistics as _stats
    from difflib import SequenceMatcher

    if not metrics or not aligned:
        return {
            "timing_accuracy": 0.0,
            "naturalness": 0.0,
            "semantic_fidelity": 0.0,
            "intelligibility": 0.0,
            "overall": 0.0,
        }

    report = align_report or clip_evaluation_report(metrics, aligned)

    # ── 1. TIMING ACCURACY ────────────────────────────────────────────
    max_error = 3.0  # seconds worst-case normalization

    error_score = max(0.0, 1.0 - report["mean_abs_duration_error_s"] / max_error)

    severe_penalty = report["pct_severe_stretch"] / 100.0
    drift_penalty = min(report["total_cumulative_drift_s"] / 5.0, 1.0)

    timing_accuracy = max(
        0.0,
        error_score * (1.0 - 0.5 * severe_penalty) * (1.0 - 0.5 * drift_penalty)
    )

    # ── 2. NATURALNESS ────────────────────────────────────────────────
    # Use ACTUAL playback rate (aligned duration, not source)
    rates = []
    for m, a in zip(metrics, aligned):
        duration = a.scheduled_end - a.scheduled_start
        if duration > 0:
            rates.append(len(m.translated_text) / duration)

    if len(rates) > 1:
        mean_rate = _stats.mean(rates)
        std_rate = _stats.stdev(rates)
        cv = std_rate / mean_rate if mean_rate > 0 else 1.0

        # Penalize high variance in speaking rate
        naturalness = max(0.0, 1.0 - min(cv, 1.0))
    else:
        naturalness = 1.0

    # ── 3. SEMANTIC FIDELITY ──────────────────────────────────────────
    # Lightweight proxy: normalized string similarity
    sim_scores = []

    for m in metrics:
        src = (m.source_text or "").lower()
        tgt = (m.translated_text or "").lower()

        if src and tgt:
            sim = SequenceMatcher(None, src, tgt).ratio()
            sim_scores.append(sim)

    semantic_fidelity = _stats.mean(sim_scores) if sim_scores else 0.0

    # ── 4. INTELLIGIBILITY ────────────────────────────────────────────
    # Proxy: penalize extreme speaking rates + long segments
    intelligibility_scores = []

    for m, a in zip(metrics, aligned):
        duration = a.scheduled_end - a.scheduled_start

        if duration <= 0:
            continue

        rate = len(m.translated_text) / duration

        # Ideal speaking rate range (chars/sec)
        if 8 <= rate <= 18:
            score = 1.0
        elif 5 <= rate < 8 or 18 < rate <= 25:
            score = 0.7
        else:
            score = 0.3

        # Penalize very long segments (harder to understand)
        if duration > 6:
            score *= 0.8

        intelligibility_scores.append(score)

    intelligibility = _stats.mean(intelligibility_scores) if intelligibility_scores else 0.0

    # ── 5. OVERALL ────────────────────────────────────────────────────
    overall = (
        0.35 * timing_accuracy +
        0.25 * naturalness +
        0.25 * semantic_fidelity +
        0.15 * intelligibility
    )

    return {
        "timing_accuracy": round(timing_accuracy, 3),
        "naturalness": round(naturalness, 3),
        "semantic_fidelity": round(semantic_fidelity, 3),
        "intelligibility": round(intelligibility, 3),
        "overall": round(overall, 3),
    }