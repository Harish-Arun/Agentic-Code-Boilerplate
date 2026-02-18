import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../services/api'
import PdfHighlightViewer from '../components/PdfHighlightViewer'
import ExtractionFields from '../components/ExtractionFields'
import SignatureTiles from '../components/SignatureTiles'
import FeedbackSection from '../components/FeedbackSection'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function VerifierView() {
    const { id } = useParams()
    const navigate = useNavigate()
    const [document, setDocument] = useState(null)
    const [initialLoading, setInitialLoading] = useState(true)
    const [selectedSigIndex, setSelectedSigIndex] = useState(0)
    const [row1Expanded, setRow1Expanded] = useState(true)
    const [row2Expanded, setRow2Expanded] = useState(true)
    const [feedback, setFeedback] = useState(null)
    const [feedbackComment, setFeedbackComment] = useState('')
    const [acting, setActing] = useState(false)

    useEffect(() => {
        fetchDocument(true)
        // Poll every 10s — silent refresh (no spinner, no UI disruption)
        const interval = setInterval(() => fetchDocument(false), 10000)
        return () => clearInterval(interval)
    }, [id])

    const fetchDocument = async (isInitial = false) => {
        if (isInitial) setInitialLoading(true)
        try {
            const doc = await api.getDocument(id)
            setDocument(doc)
        } catch (err) {
            console.warn('API unavailable')
            if (isInitial) setDocument({ id, status: 'VERIFIED', extracted_data: {}, signature_result: {}, feedback: {} })
        }
        if (isInitial) setInitialLoading(false)
    }

    const handleDecision = async (decision) => {
        setActing(true)
        try {
            // Resume the LangGraph workflow past the verifier_review interrupt.
            // The graph's verifier_review_node reads decision + feedback from modifications,
            // then the completion_node sets current_step to "complete" or "rejected".
            // The api-service then updates the document status to CONFIRMED or EXTRACTED.
            await api.resumeDocument(id, 'verifier', {
                decision,   // "accept" | "reject"
                feedback: feedbackComment,
                thumbs: feedback,
            })
            navigate('/documents')
        } catch (err) {
            console.error('Failed to submit verifier decision:', err)
            // Fallback: update status directly so the UI doesn't get stuck
            await api.updateDocumentStatus(id, decision === 'accept' ? 'CONFIRMED' : 'EXTRACTED').catch(() => {})
            navigate('/documents')
        }
        setActing(false)
    }

    if (initialLoading) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
                <div className="spinner"></div>
            </div>
        )
    }

    const extractedFieldEntries = Object.entries(document.extracted_data || {}).filter(([key, data]) => (
        key !== 'additional_fields' && data && typeof data === 'object' && 'value' in data && 'confidence' in data
    ))

    const additionalFieldsData = document?.extracted_data?.additional_fields || null

    const signatureResults = Array.isArray(document?.signature_result?.all_verifications)
        ? document.signature_result.all_verifications
        : (document?.signature_result && Object.keys(document.signature_result).length > 0 ? [document.signature_result] : [])

    // Merge manual per-tile authentication results on top of AI results
    // manual_verification_results is keyed by sig index (string) from the authenticator's work
    const manualResults = document?.signature_result?.manual_verification_results || {}
    const mergedSignatureResults = signatureResults.map((r, idx) =>
        manualResults[String(idx)] ? { ...r, ...manualResults[String(idx)], _source: 'manual' } : r
    )
    // If no AI results but manual results exist, build from manual only
    const finalSignatureResults = mergedSignatureResults.length > 0
        ? mergedSignatureResults
        : Object.entries(manualResults).map(([idx, r]) => ({ ...r, _source: 'manual' }))

    const signatureDetections = Array.isArray(document?.signature_result?.detections)
        ? document.signature_result.detections
        : []

    // Previous phase feedbacks
    const keyerFeedback = document?.feedback
    const authFeedback = document?.feedback?.auth_feedback

    const fileName = document?.raw_file_path?.split(/[\\/]/).pop()
    const documentUrl = fileName ? `${API_BASE_URL}/static/uploads/${fileName}` : ''

    return (
        <div className="animate-fadeIn">
            {/* Page Header */}
            <div className="page-header">
                <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                        <button
                            onClick={() => navigate('/documents')}
                            className="btn btn-secondary"
                            style={{ padding: '6px' }}
                        >
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="19" y1="12" x2="5" y2="12" />
                                <polyline points="12 19 5 12 12 5" />
                            </svg>
                        </button>
                        <h1>Verifier - Final Review</h1>
                        <span className={`badge badge-${document.status?.toLowerCase()}`}>
                            {document.status}
                        </span>
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)', marginTop: 'var(--spacing-xs)' }}>
                        {document.id} &bull; Review all phases and make final decision
                    </p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--spacing-md)' }}>
                    <button
                        className="btn btn-danger"
                        onClick={() => handleDecision('reject')}
                        disabled={acting}
                    >
                        {acting && <span className="spinner" style={{ width: 16, height: 16 }}></span>}
                        Reject
                    </button>
                    <button
                        className="btn btn-success"
                        onClick={() => handleDecision('accept')}
                        disabled={acting}
                    >
                        {acting && <span className="spinner" style={{ width: 16, height: 16 }}></span>}
                        Verify
                    </button>
                </div>
            </div>

            {/* Row 1: PDF Preview (left) + Extraction Fields (right) */}
            <div className="section-card" style={{ marginBottom: 'var(--spacing-lg)' }}>
                <div className="section-card-header" onClick={() => setRow1Expanded(!row1Expanded)} style={{ cursor: 'pointer' }}>
                    <span>Document Preview &amp; Extracted Fields</span>
                    <span style={{ fontSize: '1.2rem', transition: 'transform 0.2s', transform: row1Expanded ? 'rotate(180deg)' : 'rotate(0deg)' }}>&#9660;</span>
                </div>
            {row1Expanded && <div className="section-card-body" style={{ padding: 0 }}>
            <div className="split-view" style={{ margin: 0 }}>
                <div className="split-panel">
                    <div className="panel-header">Document Preview</div>
                    <div className="panel-content" style={{ display: 'flex', flexDirection: 'column', minHeight: '560px', background: 'var(--color-bg-tertiary)' }}>
                        <PdfHighlightViewer
                            fileUrl={documentUrl}
                            activeField={null}
                            activeSignature={null}
                            focusPage={1}
                        />
                        {documentUrl && (
                            <div style={{ marginTop: 'var(--spacing-md)', textAlign: 'center' }}>
                                <a href={documentUrl} target="_blank" rel="noreferrer"
                                    style={{ fontSize: '0.875rem', color: 'var(--color-primary)' }}>
                                    Open in new tab
                                </a>
                            </div>
                        )}
                    </div>
                </div>

                <div className="split-panel">
                    <div className="panel-header">Extracted Payment Fields</div>
                    <div className="panel-content">
                        {keyerFeedback?.thumbs && (
                            <div style={{ marginBottom: 'var(--spacing-md)', fontSize: '0.8rem', color: keyerFeedback.thumbs === 'up' ? 'var(--color-success)' : 'var(--color-error)' }}>
                                Keyer marked: {keyerFeedback.thumbs === 'up' ? 'Looks Good' : 'Needs Correction'}
                                {keyerFeedback.comment && <span style={{ color: 'var(--color-text-muted)', marginLeft: 8 }}>&mdash; {keyerFeedback.comment}</span>}
                            </div>
                        )}
                        <ExtractionFields
                            extractedFields={extractedFieldEntries}
                            additionalFieldsData={additionalFieldsData}
                            editable={false}
                        />

                        {/* Detected Signature Crops */}
                        {signatureDetections.length > 0 && (
                            <div style={{ marginTop: 'var(--spacing-xl)' }}>
                                <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginBottom: 'var(--spacing-sm)' }}>
                                    Detected Signatures ({signatureDetections.length})
                                </div>
                                <div style={{ display: 'flex', gap: 'var(--spacing-md)', flexWrap: 'wrap' }}>
                                    {signatureDetections.map((sig, idx) => (
                                        <div key={idx} style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', marginBottom: 4 }}>S{idx + 1}</div>
                                            {sig.image_blob ? (
                                                <img
                                                    src={`data:${sig.blob_mime_type || 'image/png'};base64,${sig.image_blob}`}
                                                    alt={`Signature ${idx + 1}`}
                                                    style={{ width: 120, height: 'auto', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-sm)', background: '#fff' }}
                                                />
                                            ) : (
                                                <div style={{ width: 120, height: 60, background: 'var(--color-bg-secondary)', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>No image</div>
                                            )}
                                            {sig.signer_name && <div style={{ fontSize: '0.7rem', marginTop: 4, color: 'var(--color-text-secondary)' }}>{sig.signer_name}</div>}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div></div>}</div>

            {/* Row 2: Authentication Results (full width) */}
            <div className="section-card" style={{ marginBottom: 'var(--spacing-lg)' }}>
                <div className="section-card-header" onClick={() => setRow2Expanded(!row2Expanded)} style={{ cursor: 'pointer' }}>
                    <span>Authentication Phase (Authenticator Output)</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                    {authFeedback?.thumbs && (
                        <span style={{ fontSize: '0.75rem', color: authFeedback.thumbs === 'up' ? 'var(--color-success)' : 'var(--color-error)' }}>
                            Auth: {authFeedback.thumbs === 'up' ? 'Looks Good' : 'Needs Correction'}
                        </span>
                    )}
                    <span style={{ fontSize: '1.2rem', transition: 'transform 0.2s', transform: row2Expanded ? 'rotate(180deg)' : 'rotate(0deg)' }}>&#9660;</span>
                    </div>
                </div>
                {row2Expanded && <div className="section-card-body">
                    <SignatureTiles
                        signatureDetections={signatureDetections}
                        signatureResults={finalSignatureResults}
                        selectedIndex={selectedSigIndex}
                        onSelectSignature={setSelectedSigIndex}
                        compact={false}
                    />
                    {authFeedback?.comment && (
                        <div style={{ marginTop: 'var(--spacing-lg)', padding: 'var(--spacing-md)', background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-md)', fontSize: '0.85rem' }}>
                            <strong style={{ color: 'var(--color-text-secondary)' }}>Authenticator Feedback:</strong>
                            <div style={{ marginTop: 'var(--spacing-xs)', color: 'var(--color-text-muted)' }}>{authFeedback.comment}</div>
                        </div>
                    )}
                </div>}
            </div>

            {/* Verifier Feedback */}
            <div className="section-card">
                <div className="section-card-body">
                    <FeedbackSection
                        feedback={feedback}
                        onFeedbackChange={setFeedback}
                        comment={feedbackComment}
                        onCommentChange={setFeedbackComment}
                    />
                </div>
            </div>
        </div>
    )
}

export default VerifierView
