import React from 'react'

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

export const renderMetric = (code, name, metricData) => {
    if (!metricData) return null

    const score = typeof metricData === 'object' ? (metricData.score || 0) : metricData
    const usesLegacyScale = score <= 5
    const scorePercent = usesLegacyScale ? (score / 5) * 100 : score
    const scoreLabel = usesLegacyScale ? `${score}/5` : `${score.toFixed(1)}/100`
    const status = metricData.status || ''
    const notes = metricData.notes || ''
    const execution = metricData.execution || ''

    const scoreColor = scorePercent >= 80 ? 'var(--color-success)' :
        scorePercent >= 60 ? 'var(--color-warning)' :
            'var(--color-error)'

    return (
        <details key={code} style={{
            border: '1px solid rgba(0,0,0,0.15)',
            borderRadius: 'var(--radius-sm)',
            marginBottom: 'var(--spacing-sm)',
            background: 'rgba(0,0,0,0.06)',  /* dark-mode safe — no white */
            color: '#1f2937'                  /* always dark text */
        }}>
            <summary style={{
                padding: 'var(--spacing-sm)',
                cursor: 'pointer',
                fontWeight: 600,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                listStyle: 'none',
                color: '#1f2937'
            }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
                    <span style={{ color: score >= 3 ? '#059669' : '#d97706' }}>
                        {score >= 3 ? '\u2713' : '\u26A0'} {code}
                    </span>
                    <span style={{ fontWeight: 400, color: '#4b5563' }}>{name}</span>
                </span>
                <span style={{
                    background: scoreColor,
                    color: 'white',
                    padding: '2px 8px',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.75rem'
                }}>
                    {scoreLabel} ({scorePercent.toFixed(0)}%)
                </span>
            </summary>
            {(status || notes) && (
                <div style={{ padding: 'var(--spacing-sm)', fontSize: '0.8rem', borderTop: '1px solid rgba(0,0,0,0.1)', color: '#1f2937' }}>
                    {status && (
                        <div style={{ marginBottom: 'var(--spacing-xs)' }}>
                            <strong style={{ color: '#374151' }}>Status: </strong>
                            <span style={{ color: scoreColor, fontWeight: 600 }}>{status}</span>
                        </div>
                    )}
                    {execution && (
                        <div style={{ marginBottom: 'var(--spacing-xs)' }}>
                            <strong style={{ color: '#374151' }}>Execution: </strong>
                            <span style={{ color: '#6b7280' }}>
                                {execution === 'llm' ? 'AI Analysis' :
                                    execution === 'llm_vision' ? 'Visual Analysis' : execution}
                            </span>
                        </div>
                    )}
                    {notes && (
                        <div style={{
                            marginTop: 'var(--spacing-sm)',
                            padding: 'var(--spacing-sm)',
                            background: 'rgba(0,0,0,0.05)',
                            borderRadius: 'var(--radius-sm)',
                            fontSize: '0.75rem',
                            lineHeight: 1.5,
                            color: '#374151'
                        }}>
                            <strong style={{ color: '#111827' }}>Analysis: </strong>{notes}
                        </div>
                    )}
                </div>
            )}
        </details>
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// SignatureResultDetail — the full rich result card, exported for reuse
// ─────────────────────────────────────────────────────────────────────────────

export function SignatureResultDetail({ result }) {
    if (!result) return null

    return (
        <div className="card" style={{
            background: result.match
                ? 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)'
                : 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
            border: `3px solid ${result.match ? '#10b981' : '#ef4444'}`,
            borderRadius: '12px',
            boxShadow: result.match
                ? '0 4px 12px rgba(16, 185, 129, 0.2)'
                : '0 4px 12px rgba(239, 68, 68, 0.2)'
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)', marginBottom: 'var(--spacing-md)' }}>
                <div style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    background: result.match ? '#10b981' : '#ef4444',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: `0 4px 8px ${result.match ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`
                }}>
                    {result.match ? (
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                            <polyline points="20 6 9 17 4 12" />
                        </svg>
                    ) : (
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                            <line x1="18" y1="6" x2="6" y2="18" />
                            <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                    )}
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{
                        fontSize: '1rem',
                        fontWeight: 700,
                        color: result.match ? '#065f46' : '#991b1b'
                    }}>
                        {result.match ? 'Signature Match' : 'Signature Mismatch'}
                        {result.scoring_details?.decision && (
                            <span style={{
                                marginLeft: '8px',
                                fontSize: '0.7rem',
                                padding: '2px 8px',
                                borderRadius: '4px',
                                background: result.scoring_details.decision === 'APPROVE' ? '#059669' :
                                    result.scoring_details.decision === 'FLAG' ? '#d97706' : '#dc2626',
                                color: 'white',
                                fontWeight: 700,
                                verticalAlign: 'middle'
                            }}>
                                {result.scoring_details.decision}
                            </span>
                        )}
                    </div>
                    <div style={{
                        fontSize: '0.875rem',
                        fontWeight: 600,
                        color: result.match ? '#047857' : '#dc2626'
                    }}>
                        Confidence: {(Number(result.confidence || 0) * 100).toFixed(0)}%
                        {(result.reference_signer_name || result.reference_id || result.referenceSigner) && (
                            <span style={{ marginLeft: '12px', fontWeight: 400, color: '#6b7280', fontSize: '0.8rem' }}>
                                vs <strong>{result.reference_signer_name || result.referenceSigner || result.reference_id}</strong>
                                {result.reference_signer_name && result.reference_id && (
                                    <span style={{ opacity: 0.7 }}> ({result.reference_id})</span>
                                )}
                            </span>
                        )}
                    </div>
                </div>
            </div>

            {Number(result.confidence || 0) === 0 && (
                <div style={{
                    fontSize: '0.8rem',
                    color: '#92400e',
                    background: 'rgba(245,158,11,0.15)',
                    border: '1px solid rgba(245,158,11,0.4)',
                    padding: 'var(--spacing-sm) var(--spacing-md)',
                    borderRadius: '6px',
                    marginBottom: 'var(--spacing-md)'
                }}>
                    ⚠️ AI analysis returned 0% confidence — metrics could not be computed. Use manual judgment.
                </div>
            )}

            {(result.reasoning || result.recommendation) && (
                <div style={{
                    fontSize: '0.875rem',
                    color: '#374151',
                    background: 'rgba(255,255,255,0.6)',
                    padding: 'var(--spacing-md)',
                    borderRadius: '8px',
                    lineHeight: '1.6',
                    marginBottom: 'var(--spacing-md)'
                }}>
                    {result.reasoning
                        ? <><strong style={{ color: '#1f2937' }}>Analysis:</strong> {result.reasoning}</>
                        : <span style={{ color: '#6b7280', fontStyle: 'italic' }}>No reasoning provided by AI. Decision: <strong>{result.recommendation || 'N/A'}</strong></span>
                    }
                </div>
            )}

            {/* Signature Images */}
            {(result.signature_blob || result.reference_signatures?.length > 0 || result.reference_blob) && (
                <div style={{ marginTop: 'var(--spacing-lg)', paddingTop: 'var(--spacing-lg)', borderTop: '2px solid #e5e7eb', marginBottom: 'var(--spacing-md)' }}>
                    <h4 style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: 'var(--spacing-md)', color: '#1f2937' }}>
                        Signature Comparison
                    </h4>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-md)' }}>
                        {result.signature_blob && (
                            <div>
                                <div style={{
                                    fontSize: '0.75rem', fontWeight: 700, color: '#5b21b6',
                                    marginBottom: 'var(--spacing-xs)', padding: '4px 10px',
                                    background: 'rgba(91, 33, 182, 0.1)', borderRadius: 'var(--radius-full)', display: 'inline-block'
                                }}>
                                    Questioned
                                </div>
                                <div style={{
                                    border: '2px solid #5b21b6', borderRadius: 'var(--radius-lg)',
                                    padding: 'var(--spacing-md)', background: '#f5f3ff',
                                    display: 'flex', justifyContent: 'center', minHeight: '140px', alignItems: 'center'
                                }}>
                                    <img
                                        src={`data:${result.blob_mime_type || 'image/png'};base64,${result.signature_blob}`}
                                        alt="Questioned Signature"
                                        style={{ maxWidth: '100%', maxHeight: '180px', objectFit: 'contain' }}
                                    />
                                </div>
                            </div>
                        )}
                        {(() => {
                            const refSigs = result.reference_signatures || []
                            if (refSigs.length === 0 && result.reference_blob) {
                                refSigs.push({
                                    blob: result.reference_blob,
                                    mime_type: result.blob_mime_type || 'image/png',
                                    customer_id: result.reference_signature_id
                                })
                            }
                            if (refSigs.length === 0) return null
                            const ref = refSigs[0]
                            return (
                                <div>
                                    <div style={{
                                        fontSize: '0.75rem', fontWeight: 700, color: '#047857',
                                        marginBottom: 'var(--spacing-xs)', padding: '4px 10px',
                                        background: 'rgba(4, 120, 87, 0.1)', borderRadius: 'var(--radius-full)', display: 'inline-block'
                                    }}>
                                        Reference
                                    </div>
                                    <div style={{
                                        border: '2px solid #047857', borderRadius: 'var(--radius-lg)',
                                        padding: 'var(--spacing-md)', background: '#ecfdf5',
                                        display: 'flex', justifyContent: 'center', minHeight: '140px', alignItems: 'center'
                                    }}>
                                        <img
                                            src={`data:${ref.mime_type};base64,${ref.blob}`}
                                            alt="Reference Signature"
                                            style={{ maxWidth: '100%', maxHeight: '180px', objectFit: 'contain' }}
                                        />
                                    </div>
                                </div>
                            )
                        })()}
                    </div>
                </div>
            )}

            {/* M1-M7 Metrics */}
            {result.metrics && (
                <div style={{ marginTop: 'var(--spacing-lg)', paddingTop: 'var(--spacing-md)', borderTop: '1px solid rgba(0,0,0,0.15)', marginBottom: 'var(--spacing-md)' }}>
                    <h4 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: 'var(--spacing-md)', color: '#1f2937' }}>
                        M1–M7 Metrics Analysis
                    </h4>
                    <div style={{ display: 'grid', gap: 'var(--spacing-sm)' }}>
                        {renderMetric('M1', 'Global Form', result.metrics.m1_global_form)}
                        {renderMetric('M2', 'Line Quality', result.metrics.m2_line_quality)}
                        {renderMetric('M3', 'Slant Angle', result.metrics.m3_slant_angle)}
                        {renderMetric('M4', 'Baseline Stability', result.metrics.m4_baseline_stability)}
                        {renderMetric('M5', 'Terminal Strokes', result.metrics.m5_terminal_strokes)}
                        {renderMetric('M6', 'Spacing Density', result.metrics.m6_spacing_density)}
                        {renderMetric('M7', 'Pressure Inference', result.metrics.m7_pressure_inference)}
                    </div>
                </div>
            )}

            {/* FIV Scoring */}
            {result.scoring_details && (
                <div style={{ marginTop: 'var(--spacing-lg)', paddingTop: 'var(--spacing-md)', borderTop: '1px solid rgba(0,0,0,0.15)' }}>
                    <h4 style={{ fontSize: '0.875rem', fontWeight: 700, marginBottom: 'var(--spacing-md)', color: '#1f2937' }}>
                        FIV {result.scoring_details.fiv_version} Scoring
                    </h4>
                    {/* Explicit color: '#111827' on every cell so dark-mode inherited white text doesn't bleed through */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-sm)', fontSize: '0.875rem', color: '#111827' }}>
                        <div style={{ padding: 'var(--spacing-sm)', background: 'rgba(0,0,0,0.06)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(0,0,0,0.1)' }}>
                            <span style={{ fontWeight: 600, color: '#374151' }}>Vetoed</span>
                            <div style={{ marginTop: 2, fontWeight: 700, color: result.scoring_details.vetoed ? '#dc2626' : '#059669' }}>
                                {result.scoring_details.vetoed != null ? (result.scoring_details.vetoed ? 'Yes' : 'No') : '—'}
                            </div>
                        </div>
                        <div style={{ padding: 'var(--spacing-sm)', background: 'rgba(0,0,0,0.06)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(0,0,0,0.1)' }}>
                            <span style={{ fontWeight: 600, color: '#374151' }}>Base Score</span>
                            <div style={{ marginTop: 2, fontWeight: 700, color: '#1d4ed8' }}>
                                {result.scoring_details.base_score != null ? result.scoring_details.base_score.toFixed(1) : '—'}
                            </div>
                        </div>
                        <div style={{ padding: 'var(--spacing-sm)', background: 'rgba(0,0,0,0.06)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(0,0,0,0.1)' }}>
                            <span style={{ fontWeight: 600, color: '#374151' }}>Penalties</span>
                            <div style={{ marginTop: 2, fontWeight: 700, color: '#dc2626' }}>
                                {result.scoring_details.penalties_applied != null ? `-${result.scoring_details.penalties_applied.toFixed(1)}` : '—'}
                            </div>
                        </div>
                        <div style={{ padding: 'var(--spacing-sm)', background: 'rgba(0,0,0,0.06)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(0,0,0,0.1)' }}>
                            <span style={{ fontWeight: 600, color: '#374151' }}>Final Score</span>
                            <div style={{ marginTop: 2, fontWeight: 800, color: '#1d4ed8', fontSize: '1.1rem' }}>
                                {result.scoring_details.final_score != null ? result.scoring_details.final_score.toFixed(1) : '—'}
                            </div>
                        </div>
                        <div style={{ padding: 'var(--spacing-sm)', background: 'rgba(0,0,0,0.06)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(0,0,0,0.1)', gridColumn: '1 / -1', display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                            <div>
                                <span style={{ fontWeight: 600, color: '#374151' }}>Decision</span>
                                <div style={{ marginTop: 2, fontWeight: 700, color: '#111827', fontSize: '0.95rem' }}>
                                    {result.scoring_details.decision || 'UNKNOWN'}
                                </div>
                            </div>
                            {result.scoring_details.confidence_band && (
                                <span style={{
                                    padding: '4px 12px', borderRadius: 'var(--radius-sm)',
                                    background: result.scoring_details.confidence_band === 'HIGH' ? '#059669' :
                                        result.scoring_details.confidence_band === 'MEDIUM' ? '#d97706' : '#dc2626',
                                    color: '#fff', fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.05em'
                                }}>
                                    {result.scoring_details.confidence_band}
                                </span>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// SignatureTiles — tile grid + selected detail
// ─────────────────────────────────────────────────────────────────────────────

function SignatureTiles({
    signatureDetections = [],
    signatureResults = [],
    selectedIndex = 0,
    onSelectSignature,
    compact = false
}) {
    const activeResult = signatureResults[selectedIndex] || signatureResults[0] || null

    if (signatureDetections.length === 0 && signatureResults.length === 0) {
        return (
            <div style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem', padding: 'var(--spacing-md)' }}>
                No signatures detected.
            </div>
        )
    }

    return (
        <div>
            {/* Tile Grid */}
            <div style={{ display: 'flex', gap: 'var(--spacing-md)', flexWrap: 'wrap', marginBottom: 'var(--spacing-lg)' }}>
                {(signatureDetections.length > 0 ? signatureDetections : signatureResults).map((sig, idx) => {
                    const result = signatureResults[idx]
                    const blob = result?.signature_blob || sig?.image_blob || sig?.signature_blob
                    const mimeType = result?.blob_mime_type || sig?.blob_mime_type || 'image/png'
                    const isActive = idx === selectedIndex

                    return (
                        <div
                            key={idx}
                            className={`signature-tile ${isActive ? 'signature-tile-active' : ''}`}
                            onClick={() => onSelectSignature && onSelectSignature(idx)}
                            style={{ width: compact ? '90px' : '140px' }}
                        >
                            {/* Image area with S-label badge overlaid */}
                            <div className="signature-tile-image-wrap">
                                <span className="signature-tile-label">S{idx + 1}</span>
                                {blob ? (
                                    <img
                                        src={`data:${mimeType};base64,${blob}`}
                                        alt={`Signature ${idx + 1}`}
                                        className="signature-tile-image"
                                    />
                                ) : (
                                    <span style={{ fontSize: '0.7rem', color: '#9ca3af' }}>No image</span>
                                )}
                            </div>

                            {/* Status strip — only shown when a result exists */}
                            {result && (
                                <div style={{
                                    padding: '4px 6px',
                                    background: result.match ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.12)',
                                    borderTop: `1px solid ${result.match ? 'rgba(16,185,129,0.35)' : 'rgba(239,68,68,0.35)'}`,
                                    fontSize: '0.6rem',
                                    fontWeight: 700,
                                    textAlign: 'center',
                                    color: result.match ? '#10b981' : '#ef4444',
                                    letterSpacing: '0.04em',
                                    textTransform: 'uppercase'
                                }}>
                                    {result.match ? '✓ Match' : '✗ Mismatch'}
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>

            {/* Selected detail (full card) */}
            {activeResult && !compact && <SignatureResultDetail result={activeResult} />}

            {/* Compact summary */}
            {activeResult && compact && (
                <div style={{
                    padding: 'var(--spacing-md)',
                    background: activeResult.match ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                    borderRadius: 'var(--radius-md)',
                    border: `1px solid ${activeResult.match ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                    fontSize: '0.875rem'
                }}>
                    <strong style={{ color: activeResult.match ? 'var(--color-success)' : 'var(--color-error)' }}>
                        {activeResult.match ? 'Signature Match' : 'Signature Mismatch'}
                    </strong>
                    {' '} - Confidence: {(Number(activeResult.confidence || 0) * 100).toFixed(0)}%
                    {activeResult.scoring_details?.decision && (
                        <span style={{ marginLeft: 'var(--spacing-sm)', color: 'var(--color-text-muted)' }}>
                            ({activeResult.scoring_details.decision})
                        </span>
                    )}
                </div>
            )}
        </div>
    )
}

export default SignatureTiles
