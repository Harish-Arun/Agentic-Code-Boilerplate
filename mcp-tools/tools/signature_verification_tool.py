"""
Signature Verification Tool — M1–M7 metrics-driven signature verification.

Business logic (called by agent orchestrator via MCP):
  1. Calls Gemini Vision with M1–M7 metric prompt
  2. Parses individual metric scores from LLM response
  3. Runs deterministic FIV 1.0 scoring engine (veto + penalties)
  4. Returns full verification result with per-metric breakdown
"""
from typing import Dict, Any, Optional
from fastmcp import FastMCP
import os
import json
import base64

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import AppConfig
from adapters import get_gemini_adapter
from models.metrics import (
    M1GlobalForm, M2LineQuality, M3SlantAngle, M4BaselineStability,
    M5TerminalStrokes, M6SpacingDensity, M7PressureInference,
    MetricsResult, MetricThresholds,
)
from models.scoring import calculate_confidence_fiv1


def register_signature_verification_tools(mcp: FastMCP, config: AppConfig):
    """Register signature verification business-logic tools with the MCP server."""

    @mcp.tool()
    async def verify_signature(
        signature_path: Optional[str] = None,
        reference_path: Optional[str] = None,
        signature_blob: Optional[str] = None,
        reference_blob: Optional[str] = None,
        signature_mime_type: str = "image/png",
        reference_mime_type: str = "image/png"
    ) -> Dict[str, Any]:
        """
        Verify a signature against a reference using M1-M7 metrics + FIV 1.0 scoring.

        Accepts either file paths or base64-encoded blobs (as returned by ISV system).
        Blob inputs take priority over file paths if both are provided.

        Pipeline:
        1. Sends signature + reference to Gemini Vision with M1-M7 prompt
        2. Parses the 7 individual metric results
        3. Runs the deterministic FIV 1.0 scoring engine
        4. Returns APPROVE / FLAG / REJECT decision with full breakdown

        Args:
            signature_path: Path to questioned signature image (fallback if no blob)
            reference_path: Path to reference signature image (fallback if no blob)
            signature_blob: Base64-encoded questioned signature blob (preferred)
            reference_blob: Base64-encoded reference signature blob (preferred)
            signature_mime_type: MIME type of signature blob (default: image/png)
            reference_mime_type: MIME type of reference blob (default: image/png)

        Returns:
            Full verification result including metrics, scoring, and decision
        """
        # Resolve signature input: blob takes priority
        if signature_blob:
            sig_input = (base64.b64decode(signature_blob), signature_mime_type)
        elif signature_path and os.path.exists(signature_path):
            sig_input = signature_path
        else:
            return {
                "success": False,
                "error": "No valid signature provided (need signature_blob or signature_path)",
                "verification": None,
                "metrics_score": None
            }

        # Resolve reference input: blob takes priority
        ref_input = None
        if reference_blob:
            ref_input = (base64.b64decode(reference_blob), reference_mime_type)
        elif reference_path and os.path.exists(reference_path):
            ref_input = reference_path

        try:
            gemini = get_gemini_adapter(config)

            # Load prompts from business config
            system_prompt = None
            user_prompt = None
            
            if hasattr(config, 'business') and hasattr(config.business, 'prompts'):
                prompts_cfg = config.business.prompts
                if hasattr(prompts_cfg, 'signature_verification') and hasattr(prompts_cfg.signature_verification, 'system'):
                    system_prompt = prompts_cfg.signature_verification.system
                    user_prompt = prompts_cfg.signature_verification.user

            # Call Gemini with M1–M7 prompt
            if ref_input:
                raw_result = await gemini.verify_signatures(
                    sig_input,
                    ref_input,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            else:
                raw_result = await _analyze_single_signature(gemini, sig_input)

            # Parse LLM output into MetricsResult
            metrics = _parse_metrics_response(raw_result)

            # Debug: Print raw metrics
            print("\n🔍 M1–M7 Metrics from Gemini:")
            print(f"   M1 aspect_ratio_delta: {metrics.m1_global_form.aspect_ratio_delta}")
            print(f"   M2 tremor_detected: {metrics.m2_line_quality.tremor_detected}")
            print(f"   M3 slant_delta_degrees: {metrics.m3_slant_angle.slant_delta_degrees}")
            print(f"   M4 drift_delta: {metrics.m4_baseline_stability.drift_delta}")
            print(f"   M5 match_status: {metrics.m5_terminal_strokes.match_status}")
            print(f"   M6 density_delta: {metrics.m6_spacing_density.density_delta}")
            print(f"   M7 pressure_delta: {metrics.m7_pressure_inference.pressure_delta}")
            print()

            # Load thresholds from business config
            thresholds = _load_thresholds(config)

            # Run deterministic FIV 1.0 scoring engine
            match_score = calculate_confidence_fiv1(
                metrics=metrics,
                thresholds=thresholds,
                llm_model=gemini.model,
            )

            # Build legacy-compatible verification result
            m = match_score.metrics
            verification = {
                "match": match_score.decision == "APPROVE",
                "confidence": match_score.final_confidence / 100.0,
                "reasoning": (
                    f"[FIV {match_score.fiv_version}] {match_score.decision_reason} | "
                    f"M1={m.m1_global_form.status} M2={m.m2_line_quality.status} "
                    f"M3={m.m3_slant_angle.status} M4={m.m4_baseline_stability.status} "
                    f"M5={m.m5_terminal_strokes.status} M6={m.m6_spacing_density.status} "
                    f"M7={m.m7_pressure_inference.status}"
                ),
                "similarity_factors": {
                    "overall_shape": {
                        "score": m.m1_global_form.score,
                        "delta": m.m1_global_form.aspect_ratio_delta,
                        "status": m.m1_global_form.status,
                    },
                    "stroke_patterns": {
                        "score": m.m3_slant_angle.score,
                        "delta_deg": m.m3_slant_angle.slant_delta_degrees,
                        "status": m.m3_slant_angle.status,
                    },
                    "pressure_consistency": {
                        "score": m.m7_pressure_inference.score,
                        "delta": m.m7_pressure_inference.pressure_delta,
                        "status": m.m7_pressure_inference.status,
                    },
                    "unique_characteristics": {
                        "score": m.m5_terminal_strokes.score,
                        "match": m.m5_terminal_strokes.match_status,
                        "status": m.m5_terminal_strokes.status,
                    },
                },
                "risk_indicators": (
                    [p.reason for p in match_score.penalties] +
                    ([match_score.veto.veto_reason] if match_score.veto.vetoed else [])
                ),
                "recommendation": match_score.decision,
            }

            # Serialize metrics_score for transport
            metrics_score_dict = match_score.model_dump()

            return {
                "success": True,
                "verification": verification,
                "metrics_score": metrics_score_dict,
                "model_used": gemini.model,
                "decision": match_score.decision,
                "final_confidence": match_score.final_confidence,
                "veto_triggered": match_score.veto.vetoed,
            }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()[:1000],
                "verification": None,
                "metrics_score": None,
                "model_used": config.llm.gemini.model
            }


# ============================================
# Helpers (business logic moved from agents)
# ============================================

def _parse_metrics_response(raw: dict) -> MetricsResult:
    """Parse Gemini's M1–M7 JSON response into MetricsResult."""

    def _safe(d: dict, key: str, default=0.0):
        return d.get(key, default)

    m1 = raw.get("M1_global_form", {})
    m2 = raw.get("M2_line_quality", {})
    m3 = raw.get("M3_slant_angle", {})
    m4 = raw.get("M4_baseline_stability", {})
    m5 = raw.get("M5_terminal_strokes", {})
    m6 = raw.get("M6_spacing_density", {})
    m7 = raw.get("M7_pressure_inference", {})

    return MetricsResult(
        m1_global_form=M1GlobalForm(
            aspect_ratio_reference=float(_safe(m1, "aspect_ratio_reference")),
            aspect_ratio_questioned=float(_safe(m1, "aspect_ratio_questioned")),
            aspect_ratio_delta=float(_safe(m1, "aspect_ratio_delta")),
            score=float(_safe(m1, "score")),
            status=str(m1.get("status", "PENDING")).split("|")[0].strip(),
            notes=str(m1.get("notes", "")),
        ),
        m2_line_quality=M2LineQuality(
            tremor_detected=bool(m2.get("tremor_detected", False)),
            hesitation_marks=int(_safe(m2, "hesitation_marks", 0)),
            quality_score_reference=float(_safe(m2, "quality_score_reference")),
            quality_score_questioned=float(_safe(m2, "quality_score_questioned")),
            score=float(_safe(m2, "score")),
            status=str(m2.get("status", "PENDING")).split("|")[0].strip(),
            notes=str(m2.get("notes", "")),
        ),
        m3_slant_angle=M3SlantAngle(
            slant_angle_reference=float(_safe(m3, "slant_angle_reference")),
            slant_angle_questioned=float(_safe(m3, "slant_angle_questioned")),
            slant_delta_degrees=float(_safe(m3, "slant_delta_degrees")),
            score=float(_safe(m3, "score")),
            status=str(m3.get("status", "PENDING")).split("|")[0].strip(),
            notes=str(m3.get("notes", "")),
        ),
        m4_baseline_stability=M4BaselineStability(
            drift_reference=float(_safe(m4, "drift_reference")),
            drift_questioned=float(_safe(m4, "drift_questioned")),
            drift_delta=float(_safe(m4, "drift_delta")),
            slope_variance_reference=float(_safe(m4, "slope_variance_reference")),
            slope_variance_questioned=float(_safe(m4, "slope_variance_questioned")),
            score=float(_safe(m4, "score")),
            status=str(m4.get("status", "PENDING")).split("|")[0].strip(),
            notes=str(m4.get("notes", "")),
        ),
        m5_terminal_strokes=M5TerminalStrokes(
            match_status=str(m5.get("match_status", "PENDING")).split("|")[0].strip(),
            markers_reference=m5.get("markers_reference", []) if isinstance(m5.get("markers_reference"), list) else [],
            markers_questioned=m5.get("markers_questioned", []) if isinstance(m5.get("markers_questioned"), list) else [],
            markers_matched=m5.get("markers_matched", []) if isinstance(m5.get("markers_matched"), list) else [],
            marker_confidence=float(_safe(m5, "marker_confidence")),
            score=float(_safe(m5, "score")),
            status=str(m5.get("status", "PENDING")).split("|")[0].strip(),
            notes=str(m5.get("notes", "")),
        ),
        m6_spacing_density=M6SpacingDensity(
            density_reference=float(_safe(m6, "density_reference")),
            density_questioned=float(_safe(m6, "density_questioned")),
            density_delta=float(_safe(m6, "density_delta")),
            score=float(_safe(m6, "score")),
            status=str(m6.get("status", "PENDING")).split("|")[0].strip(),
            notes=str(m6.get("notes", "")),
        ),
        m7_pressure_inference=M7PressureInference(
            pressure_mean_reference=float(_safe(m7, "pressure_mean_reference")),
            pressure_mean_questioned=float(_safe(m7, "pressure_mean_questioned")),
            pressure_delta=float(_safe(m7, "pressure_delta")),
            variance_pct_reference=float(_safe(m7, "variance_pct_reference")),
            variance_pct_questioned=float(_safe(m7, "variance_pct_questioned")),
            score=float(_safe(m7, "score")),
            status=str(m7.get("status", "PENDING")).split("|")[0].strip(),
            notes=str(m7.get("notes", "")),
        ),
    )


def _load_thresholds(config: AppConfig) -> MetricThresholds:
    """Load metric thresholds from business config if available."""
    try:
        biz = config.business.metric_thresholds
        return MetricThresholds(
            m1_tolerance=biz.m1_tolerance,
            m1_veto=biz.m1_veto,
            m3_tolerance=biz.m3_tolerance,
            m3_veto=biz.m3_veto,
        )
    except Exception:
        return MetricThresholds()


async def _analyze_single_signature(gemini, signature_input) -> dict:
    """Analyze a single signature without reference (basic authenticity check).

    Args:
        gemini: GeminiRestAdapter instance
        signature_input: File path (str) or (bytes, mime_type) tuple

    Returns the same M1–M7 JSON structure with _reference values estimated
    from typical signature characteristics and _questioned from actual observation.
    """
    prompt = """You are a Forensic Document Examiner. You have ONE signature image to analyze.
There is NO reference signature available for comparison.

Analyze this signature for authenticity indicators. For each metric, compare the signature
against itself (self-consistency) and general expectations for a genuine signature.

Provide the same M1–M7 metrics structure but evaluate quality/authenticity rather than comparison.
Set _reference values to estimated "typical" values and _questioned to actual observed values.

=== M1: GLOBAL FORM === Report aspect ratio; compare to typical 1.5–3.0 range.
=== M2: LINE QUALITY === Look for tremor, hesitation, blobbing, pen lifts.
=== M3: SLANT ANGLE === Report angle; consistency within the signature.
=== M4: BASELINE STABILITY === How stable is the writing line?
=== M5: TERMINAL STROKES === Are there distinctive personal quirks present?
=== M6: SPACING & DENSITY === Is ink density normal?
=== M7: PRESSURE INFERENCE === Is pressure consistent throughout?

Return ONLY valid JSON in the same M1–M7 schema."""

    schema = {
        "M1_global_form": {"aspect_ratio_reference": 0.0, "aspect_ratio_questioned": 0.0, "aspect_ratio_delta": 0.0, "score": 0, "status": "string", "notes": "string"},
        "M2_line_quality": {"tremor_detected": False, "hesitation_marks": 0, "quality_score_reference": 0, "quality_score_questioned": 0, "score": 0, "status": "string", "notes": "string"},
        "M3_slant_angle": {"slant_angle_reference": 0.0, "slant_angle_questioned": 0.0, "slant_delta_degrees": 0.0, "score": 0, "status": "string", "notes": "string"},
        "M4_baseline_stability": {"drift_reference": 0.0, "drift_questioned": 0.0, "drift_delta": 0.0, "slope_variance_reference": 0.0, "slope_variance_questioned": 0.0, "score": 0, "status": "string", "notes": "string"},
        "M5_terminal_strokes": {"match_status": "string", "markers_reference": [], "markers_questioned": [], "markers_matched": [], "marker_confidence": 0.0, "score": 0, "status": "string", "notes": "string"},
        "M6_spacing_density": {"density_reference": 0.0, "density_questioned": 0.0, "density_delta": 0.0, "score": 0, "status": "string", "notes": "string"},
        "M7_pressure_inference": {"pressure_mean_reference": 0.0, "pressure_mean_questioned": 0.0, "pressure_delta": 0.0, "variance_pct_reference": 0.0, "variance_pct_questioned": 0.0, "score": 0, "status": "string", "notes": "string"},
    }

    return await gemini.generate_structured(
        prompt,
        schema,
        system_prompt="You are a forensic signature analyst. Provide quantitative assessment.",
        files=[signature_input],
    )
