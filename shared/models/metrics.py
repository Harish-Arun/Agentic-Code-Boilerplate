"""
Signature Verification Metrics - M1 to M7 (FIV 1.0 Scope).

Each metric is individually computed by the LLM (Gemini Vision) and returned
with its own score, raw values, execution method, and status.

Based on LLM_Signature_System_Design_v4.0.md:
  M1: Global Form (aspect ratio delta)
  M2: Line Quality (tremor / hesitation)
  M3: Slant Angle (PCA-derived angle delta)
  M4: Baseline Stability (drift / slope variance)
  M5: Terminal Strokes (personal quirks match)
  M6: Spacing & Density (ink density delta)
  M7: Pressure Inference (grayscale intensity delta)
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ============================================
# Individual Metric Models
# ============================================
class M1GlobalForm(BaseModel):
    """M1: Global Form — bounding box aspect ratio comparison."""
    aspect_ratio_reference: float = Field(0.0, description="Aspect ratio of reference signature")
    aspect_ratio_questioned: float = Field(0.0, description="Aspect ratio of questioned signature")
    aspect_ratio_delta: float = Field(0.0, description="Absolute difference between the two (0–2.0)")
    score: float = Field(0.0, ge=0, le=100, description="Metric score (100=perfect match)")
    status: str = Field("PENDING", description="PASS | WARNING | FAIL | VETO")
    execution: str = "llm"
    notes: str = ""


class M2LineQuality(BaseModel):
    """M2: Line Quality — smoothness, tremor, pen control."""
    tremor_detected: bool = Field(False, description="Whether suspicious tremor/hesitation is detected")
    hesitation_marks: int = Field(0, description="Count of hesitation marks found")
    quality_score_reference: float = Field(0.0, ge=0, le=100, description="Line quality of reference (0=shaky, 100=smooth)")
    quality_score_questioned: float = Field(0.0, ge=0, le=100, description="Line quality of questioned")
    score: float = Field(0.0, ge=0, le=100, description="Metric score")
    status: str = Field("PENDING", description="PASS | WARNING | FAIL")
    execution: str = "llm_vision"
    notes: str = ""


class M3SlantAngle(BaseModel):
    """M3: Slant Angle — dominant stroke angle comparison via PCA."""
    slant_angle_reference: float = Field(0.0, description="Dominant slant angle of reference (degrees)")
    slant_angle_questioned: float = Field(0.0, description="Dominant slant angle of questioned (degrees)")
    slant_delta_degrees: float = Field(0.0, description="Absolute difference in degrees (0–180)")
    score: float = Field(0.0, ge=0, le=100, description="Metric score")
    status: str = Field("PENDING", description="PASS | WARNING | FAIL | VETO")
    execution: str = "llm"
    notes: str = ""


class M4BaselineStability(BaseModel):
    """M4: Baseline Stability — writing baseline drift and slope variance."""
    drift_reference: float = Field(0.0, description="Baseline drift of reference (normalised)")
    drift_questioned: float = Field(0.0, description="Baseline drift of questioned (normalised)")
    drift_delta: float = Field(0.0, description="Absolute difference in drift")
    slope_variance_reference: float = Field(0.0, description="Slope variance of reference")
    slope_variance_questioned: float = Field(0.0, description="Slope variance of questioned")
    score: float = Field(0.0, ge=0, le=100, description="Metric score")
    status: str = Field("PENDING", description="PASS | WARNING | FAIL")
    execution: str = "llm"
    notes: str = ""


class M5TerminalStrokes(BaseModel):
    """M5: Terminal Strokes — distinctive hooks, loops, flourishes (personal quirks)."""
    match_status: str = Field("PENDING", description="MATCH | PARTIAL_MATCH | COMPLETE_MISMATCH")
    markers_reference: List[str] = Field(default_factory=list, description="Quirks found in reference")
    markers_questioned: List[str] = Field(default_factory=list, description="Quirks found in questioned")
    markers_matched: List[str] = Field(default_factory=list, description="Quirks that matched")
    marker_confidence: float = Field(0.0, ge=0, le=1.0, description="Confidence in marker matching")
    score: float = Field(0.0, ge=0, le=100, description="Metric score")
    status: str = Field("PENDING", description="PASS | WARNING | FAIL | VETO")
    execution: str = "llm_vision"
    notes: str = ""


class M6SpacingDensity(BaseModel):
    """M6: Spacing & Density — ink density ratio comparison."""
    density_reference: float = Field(0.0, description="Ink density ratio of reference (0–1)")
    density_questioned: float = Field(0.0, description="Ink density ratio of questioned (0–1)")
    density_delta: float = Field(0.0, description="Absolute difference (0–1)")
    score: float = Field(0.0, ge=0, le=100, description="Metric score")
    status: str = Field("PENDING", description="PASS | WARNING | FAIL")
    execution: str = "llm"
    notes: str = ""


class M7PressureInference(BaseModel):
    """M7: Pressure Inference — grayscale intensity analysis as proxy for pen pressure."""
    pressure_mean_reference: float = Field(0.0, description="Mean grayscale intensity of reference (0–255)")
    pressure_mean_questioned: float = Field(0.0, description="Mean grayscale intensity of questioned (0–255)")
    pressure_delta: float = Field(0.0, description="Absolute difference in mean pressure")
    variance_pct_reference: float = Field(0.0, description="Pressure variance percentage of reference")
    variance_pct_questioned: float = Field(0.0, description="Pressure variance percentage of questioned")
    score: float = Field(0.0, ge=0, le=100, description="Metric score")
    status: str = Field("PENDING", description="PASS | WARNING | FAIL")
    execution: str = "llm"
    notes: str = ""


# ============================================
# Aggregated Metrics Result
# ============================================
class MetricsResult(BaseModel):
    """Complete M1–M7 metrics output from signature verification."""
    m1_global_form: M1GlobalForm = Field(default_factory=M1GlobalForm)
    m2_line_quality: M2LineQuality = Field(default_factory=M2LineQuality)
    m3_slant_angle: M3SlantAngle = Field(default_factory=M3SlantAngle)
    m4_baseline_stability: M4BaselineStability = Field(default_factory=M4BaselineStability)
    m5_terminal_strokes: M5TerminalStrokes = Field(default_factory=M5TerminalStrokes)
    m6_spacing_density: M6SpacingDensity = Field(default_factory=M6SpacingDensity)
    m7_pressure_inference: M7PressureInference = Field(default_factory=M7PressureInference)


# ============================================
# Veto Check Result
# ============================================
class VetoResult(BaseModel):
    """Result of veto (kill-switch) checks."""
    vetoed: bool = False
    veto_metric: Optional[str] = None
    veto_reason: Optional[str] = None


# ============================================
# Penalty / Bonus Entry
# ============================================
class ScoringEntry(BaseModel):
    """A single penalty or bonus applied during scoring."""
    metric: str
    label: str
    points: float  # negative = penalty, positive = bonus
    reason: str = ""


# ============================================
# Final Scoring Output
# ============================================
class SignatureMatchScore(BaseModel):
    """
    Full FIV 1.0 scoring output with per-metric breakdown.

    This is the primary output of the signature verification pipeline.
    """
    # Identity
    fiv_version: str = "1.0"
    llm_model: str = ""
    
    # Metrics breakdown
    metrics: MetricsResult = Field(default_factory=MetricsResult)
    
    # Veto check
    veto: VetoResult = Field(default_factory=VetoResult)
    
    # Scoring
    base_score: float = Field(100.0, description="Starting score before penalties")
    penalties: List[ScoringEntry] = Field(default_factory=list)
    bonuses: List[ScoringEntry] = Field(default_factory=list)
    final_confidence: float = Field(0.0, ge=0, le=100)
    
    # Decision
    decision: str = Field("PENDING", description="APPROVE | FLAG | REJECT")
    confidence_band: str = Field("", description="HIGH | AMBIGUOUS | LOW")
    decision_reason: str = ""
    
    # Audit helpers
    processing_time_ms: int = 0
    audit_summary: Dict[str, Any] = Field(default_factory=dict)


# ============================================
# Metric Threshold Configuration
# ============================================
class MetricThresholds(BaseModel):
    """FIV 1.0 configurable thresholds for scoring engine."""
    # M1 Global Form
    m1_tolerance: float = 0.10
    m1_veto: float = 0.50
    m1_weight: float = 200.0

    # M2 Line Quality
    m2_penalty: float = 15.0

    # M3 Slant Angle
    m3_tolerance: float = 5.0
    m3_veto: float = 45.0
    m3_weight: float = 1.5

    # M4 Baseline Stability
    m4_tolerance: float = 0.05
    m4_weight: float = 100.0

    # M5 Terminal Strokes
    m5_penalty: float = 25.0

    # M6 Spacing Density
    m6_tolerance: float = 0.08
    m6_weight: float = 50.0

    # M7 Pressure
    m7_tolerance: float = 15.0
    m7_weight: float = 100.0

    # Decision thresholds
    approve_threshold: float = 80.0
    flag_min_threshold: float = 60.0
