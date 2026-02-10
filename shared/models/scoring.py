"""
FIV 1.0 Scoring Engine — Deterministic confidence calculation.

Implements the penalty-based scoring system from LLM_Signature_System_Design_v4.0.md.
Each metric (M1–M7) is individually evaluated by the LLM, then the scoring engine
applies veto checks, penalties, and bonuses to produce a final confidence & decision.
"""
from typing import Optional
from .metrics import (
    MetricsResult,
    MetricThresholds,
    VetoResult,
    ScoringEntry,
    SignatureMatchScore,
)


def calculate_confidence_fiv1(
    metrics: MetricsResult,
    thresholds: Optional[MetricThresholds] = None,
    llm_model: str = "",
    processing_time_ms: int = 0,
) -> SignatureMatchScore:
    """
    FIV 1.0 Confidence Aggregation (Deterministic).

    Applies:
      1. Veto checks (M1, M3, M5 kill-switches)
      2. Penalty calculation per metric
      3. Confidence score = 100 − penalties
      4. Decision: APPROVE / FLAG / REJECT

    Args:
        metrics: Fully populated MetricsResult from LLM analysis.
        thresholds: Optional overrides; defaults to design-doc values.
        llm_model: The Gemini model name used (for audit).
        processing_time_ms: Wall-clock time of the full pipeline.

    Returns:
        SignatureMatchScore with complete breakdown.
    """
    cfg = thresholds or MetricThresholds()
    score = 100.0
    penalties: list[ScoringEntry] = []
    bonuses: list[ScoringEntry] = []

    # ------------------------------------------------------------------
    # 1. VETO LOGIC (Kill-switches) — immediate REJECT, score = 0
    # ------------------------------------------------------------------

    # M1 veto: shape mismatch
    if metrics.m1_global_form.aspect_ratio_delta > cfg.m1_veto:
        return _build_result(
            metrics=metrics,
            veto=VetoResult(
                vetoed=True,
                veto_metric="M1",
                veto_reason=f"Shape mismatch: aspect_ratio_delta={metrics.m1_global_form.aspect_ratio_delta:.3f} > veto={cfg.m1_veto}",
            ),
            final_confidence=0.0,
            decision="REJECT",
            decision_reason="VETO: Shape mismatch (M1)",
            llm_model=llm_model,
            processing_time_ms=processing_time_ms,
            penalties=penalties,
            bonuses=bonuses,
            cfg=cfg,
        )

    # M3 veto: slant reversal
    if metrics.m3_slant_angle.slant_delta_degrees > cfg.m3_veto:
        return _build_result(
            metrics=metrics,
            veto=VetoResult(
                vetoed=True,
                veto_metric="M3",
                veto_reason=f"Slant reversal: slant_delta={metrics.m3_slant_angle.slant_delta_degrees:.1f}° > veto={cfg.m3_veto}°",
            ),
            final_confidence=0.0,
            decision="REJECT",
            decision_reason="VETO: Slant reversal (M3)",
            llm_model=llm_model,
            processing_time_ms=processing_time_ms,
            penalties=penalties,
            bonuses=bonuses,
            cfg=cfg,
        )

    # M5 veto: terminal strokes completely missing
    if metrics.m5_terminal_strokes.match_status == "COMPLETE_MISMATCH":
        return _build_result(
            metrics=metrics,
            veto=VetoResult(
                vetoed=True,
                veto_metric="M5",
                veto_reason="Terminal strokes completely missing in questioned signature",
            ),
            final_confidence=0.0,
            decision="REJECT",
            decision_reason="VETO: Terminal strokes missing (M5)",
            llm_model=llm_model,
            processing_time_ms=processing_time_ms,
            penalties=penalties,
            bonuses=bonuses,
            cfg=cfg,
        )

    # ------------------------------------------------------------------
    # 2. PENALTY CALCULATION
    # ------------------------------------------------------------------

    # M1 Global Form
    m1_delta = metrics.m1_global_form.aspect_ratio_delta
    if m1_delta > cfg.m1_tolerance:
        overshoot = m1_delta - cfg.m1_tolerance
        penalty = min(overshoot * cfg.m1_weight, 40.0)
        score -= penalty
        penalties.append(ScoringEntry(
            metric="M1", label="M1_Shape",
            points=-penalty,
            reason=f"delta={m1_delta:.3f} exceeds tolerance={cfg.m1_tolerance}",
        ))

    # M2 Visual Tremor
    if metrics.m2_line_quality.tremor_detected:
        score -= cfg.m2_penalty
        penalties.append(ScoringEntry(
            metric="M2", label="M2_Tremor",
            points=-cfg.m2_penalty,
            reason="Tremor / hesitation detected in questioned signature",
        ))

    # M3 Slant
    m3_delta = metrics.m3_slant_angle.slant_delta_degrees
    if m3_delta > cfg.m3_tolerance:
        overshoot = m3_delta - cfg.m3_tolerance
        penalty = min(overshoot * cfg.m3_weight, 30.0)
        score -= penalty
        penalties.append(ScoringEntry(
            metric="M3", label="M3_Slant",
            points=-penalty,
            reason=f"delta={m3_delta:.1f}° exceeds tolerance={cfg.m3_tolerance}°",
        ))

    # M4 Baseline
    m4_drift = metrics.m4_baseline_stability.drift_delta
    if m4_drift > cfg.m4_tolerance:
        penalty = min(m4_drift * cfg.m4_weight, 20.0)
        score -= penalty
        penalties.append(ScoringEntry(
            metric="M4", label="M4_Baseline",
            points=-penalty,
            reason=f"drift_delta={m4_drift:.3f} exceeds tolerance={cfg.m4_tolerance}",
        ))

    # M5 Terminal Strokes (partial match)
    if metrics.m5_terminal_strokes.match_status == "PARTIAL_MATCH":
        score -= cfg.m5_penalty
        penalties.append(ScoringEntry(
            metric="M5", label="M5_Markers",
            points=-cfg.m5_penalty,
            reason="Only partial match on terminal stroke markers",
        ))

    # M6 Spacing Density
    m6_delta = metrics.m6_spacing_density.density_delta
    if m6_delta > cfg.m6_tolerance:
        penalty = min(m6_delta * cfg.m6_weight, 15.0)
        score -= penalty
        penalties.append(ScoringEntry(
            metric="M6", label="M6_Density",
            points=-penalty,
            reason=f"density_delta={m6_delta:.3f} exceeds tolerance={cfg.m6_tolerance}",
        ))

    # M7 Pressure
    m7_delta = metrics.m7_pressure_inference.pressure_delta
    if m7_delta > cfg.m7_tolerance:
        penalty = min(m7_delta * cfg.m7_weight, 20.0)
        score -= penalty
        penalties.append(ScoringEntry(
            metric="M7", label="M7_Pressure",
            points=-penalty,
            reason=f"pressure_delta={m7_delta:.1f} exceeds tolerance={cfg.m7_tolerance}",
        ))

    # ------------------------------------------------------------------
    # 3. CLAMP & DECIDE
    # ------------------------------------------------------------------
    final_score = max(0.0, min(100.0, score))

    if final_score >= cfg.approve_threshold:
        decision = "APPROVE"
    elif final_score >= cfg.flag_min_threshold:
        decision = "FLAG"
    else:
        decision = "REJECT"

    confidence_band = (
        "HIGH" if final_score >= cfg.approve_threshold
        else ("AMBIGUOUS" if final_score >= cfg.flag_min_threshold else "LOW")
    )

    total_penalties = sum(abs(p.points) for p in penalties)
    decision_reason = (
        f"Score {final_score:.1f}/100 after {len(penalties)} penalties totalling -{total_penalties:.1f} points"
        if penalties
        else f"Score {final_score:.1f}/100 — all metrics within thresholds"
    )

    return _build_result(
        metrics=metrics,
        veto=VetoResult(vetoed=False),
        final_confidence=round(final_score, 1),
        decision=decision,
        decision_reason=decision_reason,
        llm_model=llm_model,
        processing_time_ms=processing_time_ms,
        penalties=penalties,
        bonuses=bonuses,
        cfg=cfg,
        confidence_band=confidence_band,
    )


# ============================================
# Helper
# ============================================
def _build_result(
    *,
    metrics: MetricsResult,
    veto: VetoResult,
    final_confidence: float,
    decision: str,
    decision_reason: str,
    llm_model: str,
    processing_time_ms: int,
    penalties: list[ScoringEntry],
    bonuses: list[ScoringEntry],
    cfg: MetricThresholds,
    confidence_band: str = "LOW",
) -> SignatureMatchScore:
    return SignatureMatchScore(
        fiv_version="1.0",
        llm_model=llm_model,
        metrics=metrics,
        veto=veto,
        base_score=100.0,
        penalties=penalties,
        bonuses=bonuses,
        final_confidence=final_confidence,
        decision=decision,
        confidence_band=confidence_band,
        decision_reason=decision_reason,
        processing_time_ms=processing_time_ms,
        audit_summary={
            "M1_delta": metrics.m1_global_form.aspect_ratio_delta,
            "M2_tremor": metrics.m2_line_quality.tremor_detected,
            "M3_delta_deg": metrics.m3_slant_angle.slant_delta_degrees,
            "M4_drift": metrics.m4_baseline_stability.drift_delta,
            "M5_status": metrics.m5_terminal_strokes.match_status,
            "M6_delta": metrics.m6_spacing_density.density_delta,
            "M7_delta": metrics.m7_pressure_inference.pressure_delta,
            "penalties_count": len(penalties),
            "total_penalty_points": sum(abs(p.points) for p in penalties),
            "veto_triggered": veto.vetoed,
            "veto_metric": veto.veto_metric,
        },
    )
