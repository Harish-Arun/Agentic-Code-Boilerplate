import React, { useState } from 'react'
import { api } from '../services/api'
import { SignatureResultDetail } from './SignatureTiles'

function SignatureValidationTab({
    signatureDetections = [],
    signatureResults = [],
    extractedData = {},
    documentId
}) {
    const [isvResponse, setIsvResponse] = useState(null)
    const [isvLoading, setIsvLoading] = useState(false)
    const [isvError, setIsvError] = useState(null)
    const [selectedRefSig, setSelectedRefSig] = useState({})
    const [validationResults, setValidationResults] = useState({})
    const [validating, setValidating] = useState({})

    const accountNumber = extractedData?.debtor_account?.value || extractedData?.creditor_account?.value || ''
    const sortCode = extractedData?.debtor_sort_code?.value || extractedData?.creditor_sort_code?.value || ''

    const handleFetchReferences = async () => {
        setIsvLoading(true)
        setIsvError(null)
        try {
            const response = await api.lookupISV(accountNumber, sortCode)
            setIsvResponse(response)
        } catch (err) {
            setIsvError('Failed to fetch reference signatures from ISV. ' + err.message)
        }
        setIsvLoading(false)
    }

    const handleValidate = async (tileIndex) => {
        const refIdx = selectedRefSig[tileIndex]
        if (refIdx === undefined || !isvResponse?.signatories?.[refIdx]) return

        const signatory = isvResponse.signatories[refIdx]
        setValidating(prev => ({ ...prev, [tileIndex]: true }))
        try {
            const result = await api.verifySignature(
                documentId,
                tileIndex,
                signatory.signatureGif,
                signatory.sigId,
                signatory.signatureMimeType || 'image/gif'
            )
            setValidationResults(prev => ({
                ...prev,
                [tileIndex]: {
                    success: true,
                    referenceSigner: signatory.signerName,
                    match: result.match,
                    confidence: result.confidence,
                    reasoning: result.reasoning,
                    recommendation: result.recommendation,
                    metrics: result.metrics || {},
                    scoring_details: result.scoring_details || {},
                    risk_indicators: result.risk_indicators || [],
                    signature_blob: result.signature_blob,
                    reference_blob: result.reference_blob,
                    blob_mime_type: result.blob_mime_type || 'image/png',
                }
            }))
        } catch (err) {
            setValidationResults(prev => ({
                ...prev,
                [tileIndex]: { success: false, message: 'Authentication failed: ' + err.message }
            }))
        }
        setValidating(prev => ({ ...prev, [tileIndex]: false }))
    }

    const tiles = signatureDetections.length > 0 ? signatureDetections : signatureResults

    return (
        <div>
            {/* ISV Lookup */}
            <div className="card" style={{ marginBottom: 'var(--spacing-lg)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)', marginBottom: 'var(--spacing-md)' }}>
                    <div style={{ flex: 1 }}>
                        <label style={{ fontSize: '0.75rem' }}>Account Number</label>
                        <div className="field-readonly">{accountNumber || 'N/A'}</div>
                    </div>
                    <div style={{ flex: 1 }}>
                        <label style={{ fontSize: '0.75rem' }}>Sort Code</label>
                        <div className="field-readonly">{sortCode || 'N/A'}</div>
                    </div>
                    <div style={{ alignSelf: 'flex-end' }}>
                        <button
                            className="btn btn-primary"
                            onClick={handleFetchReferences}
                            disabled={isvLoading || (!accountNumber && !sortCode)}
                        >
                            {isvLoading && <span className="spinner" style={{ width: 14, height: 14 }}></span>}
                            Fetch References
                        </button>
                    </div>
                </div>
                {isvError && <div style={{ color: 'var(--color-error)', fontSize: '0.85rem' }}>{isvError}</div>}
            </div>

            {/* Freeform signing rule */}
            {isvResponse?.freeformRule && (
                <div className="card" style={{ marginBottom: 'var(--spacing-lg)', background: 'var(--color-bg-tertiary)' }}>
                    <label style={{ fontSize: '0.75rem', marginBottom: 'var(--spacing-xs)' }}>Signing Rule (Freeform)</label>
                    <div style={{
                        padding: 'var(--spacing-md)', background: 'var(--color-bg-secondary)',
                        borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)',
                        fontSize: '0.875rem', lineHeight: 1.6
                    }}>
                        {isvResponse.freeformRule}
                    </div>
                </div>
            )}

            {/* Per-tile cards */}
            {tiles.length === 0 ? (
                <div style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem', padding: 'var(--spacing-md)' }}>
                    No signatures detected on this document.
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-xl)' }}>
                    {tiles.map((sig, idx) => {
                        const result = signatureResults[idx]
                        const tileValidation = validationResults[idx]
                        const blob = tileValidation?.signature_blob || result?.signature_blob || sig?.image_blob || sig?.signature_blob
                        const mimeType = result?.blob_mime_type || sig?.blob_mime_type || 'image/png'
                        const refIdx = selectedRefSig[idx]

                        return (
                            <div key={idx} className="section-card">
                                <div className="section-card-header">
                                    <span>Signature {idx + 1} (S{idx + 1})</span>
                                    {tileValidation?.success && (
                                        <span style={{
                                            fontSize: '0.75rem',
                                            color: tileValidation.match ? 'var(--color-success)' : 'var(--color-error)',
                                            fontWeight: 600
                                        }}>
                                            {tileValidation.match ? 'Match' : 'Mismatch'} — {(Number(tileValidation.confidence || 0) * 100).toFixed(0)}%
                                        </span>
                                    )}
                                </div>
                                <div className="section-card-body">
                                    {/* Questioned vs Reference side-by-side */}
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-lg)' }}>
                                        <div>
                                            <div style={{
                                                fontSize: '0.75rem', fontWeight: 700, color: '#7c3aed',
                                                marginBottom: 'var(--spacing-xs)', padding: '4px 10px',
                                                background: 'rgba(124,58,237,0.1)', borderRadius: 'var(--radius-full)', display: 'inline-block'
                                            }}>
                                                Questioned Signature
                                            </div>
                                            <div style={{
                                                border: '2px solid #7c3aed', borderRadius: 'var(--radius-lg)',
                                                padding: 'var(--spacing-md)', background: '#faf5ff',
                                                display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200
                                            }}>
                                                {blob
                                                    ? <img src={`data:${mimeType};base64,${blob}`} alt={`S${idx + 1}`} style={{ maxWidth: '100%', maxHeight: 240, objectFit: 'contain' }} />
                                                    : <span style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>No image available</span>
                                                }
                                            </div>
                                        </div>

                                        <div>
                                            <div style={{
                                                fontSize: '0.75rem', fontWeight: 700, color: '#059669',
                                                marginBottom: 'var(--spacing-xs)', padding: '4px 10px',
                                                background: 'rgba(5,150,105,0.1)', borderRadius: 'var(--radius-full)', display: 'inline-block'
                                            }}>
                                                Reference Signature
                                            </div>
                                            {isvResponse?.signatories?.length > 0 ? (
                                                <>
                                                    <select
                                                        value={refIdx ?? ''}
                                                        onChange={e => setSelectedRefSig(prev => ({ ...prev, [idx]: Number(e.target.value) }))}
                                                        style={{ marginBottom: 'var(--spacing-sm)', width: '100%' }}
                                                    >
                                                        <option value="">Select reference signatory...</option>
                                                        {isvResponse.signatories.map((s, sIdx) => (
                                                            <option key={sIdx} value={sIdx}>{s.signerName} ({s.sigId})</option>
                                                        ))}
                                                    </select>
                                                    <div style={{
                                                        border: '2px solid #059669', borderRadius: 'var(--radius-lg)',
                                                        padding: 'var(--spacing-md)', background: '#f0fdf4',
                                                        display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200
                                                    }}>
                                                        {refIdx !== undefined && isvResponse.signatories[refIdx]?.signatureGif
                                                            ? <img
                                                                src={`data:${isvResponse.signatories[refIdx].signatureMimeType || 'image/gif'};base64,${isvResponse.signatories[refIdx].signatureGif}`}
                                                                alt={`Ref ${isvResponse.signatories[refIdx].signerName}`}
                                                                style={{ maxWidth: '100%', maxHeight: 240, objectFit: 'contain' }}
                                                            />
                                                            : <span style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>
                                                                {refIdx !== undefined ? 'No signature image' : 'Select a signatory above'}
                                                            </span>
                                                        }
                                                    </div>
                                                </>
                                            ) : (
                                                <div style={{
                                                    border: '2px dashed var(--color-border)', borderRadius: 'var(--radius-lg)',
                                                    display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200,
                                                    color: 'var(--color-text-muted)', fontSize: '0.85rem'
                                                }}>
                                                    Click "Fetch References" to load ISV signatures
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Authenticate button */}
                                    {isvResponse?.signatories?.length > 0 && (
                                        <div style={{ marginTop: 'var(--spacing-md)' }}>
                                            <button
                                                className="btn btn-primary"
                                                onClick={() => handleValidate(idx)}
                                                disabled={refIdx === undefined || validating[idx]}
                                            >
                                                {validating[idx] && <span className="spinner" style={{ width: 14, height: 14 }}></span>}
                                                {validating[idx] ? 'Authenticating...' : 'Authenticate Signature'}
                                            </button>
                                        </div>
                                    )}

                                    {/* Error */}
                                    {tileValidation && !tileValidation.success && (
                                        <div style={{
                                            marginTop: 'var(--spacing-md)', padding: 'var(--spacing-md)',
                                            background: 'rgba(239,68,68,0.1)', borderRadius: 'var(--radius-md)',
                                            border: '1px solid rgba(239,68,68,0.3)', color: 'var(--color-error)', fontSize: '0.875rem'
                                        }}>
                                            {tileValidation.message}
                                        </div>
                                    )}

                                    {/* Shared full result card */}
                                    {tileValidation?.success && (
                                        <div style={{ marginTop: 'var(--spacing-lg)' }}>
                                            <SignatureResultDetail result={tileValidation} />
                                        </div>
                                    )}
                                </div>
                            </div>
                        )
                    })}
                </div>
            )}
        </div>
    )
}

export default SignatureValidationTab
