import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../services/api'
import PdfHighlightViewer from '../components/PdfHighlightViewer'
import ExtractionFields from '../components/ExtractionFields'
import FeedbackSection from '../components/FeedbackSection'
import SignatureValidationTab from '../components/SignatureValidationTab'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function AuthenticatorView() {
    const { id } = useParams()
    const navigate = useNavigate()
    const [document, setDocument] = useState(null)
    const [initialLoading, setInitialLoading] = useState(true)
    const [activeTab, setActiveTab] = useState('signature_validation')
    const [row1Expanded, setRow1Expanded] = useState(true)
    const [row2Expanded, setRow2Expanded] = useState(true)
    const [feedback, setFeedback] = useState(null)
    const [feedbackComment, setFeedbackComment] = useState('')
    const [proceeding, setProceeding] = useState(false)

    useEffect(() => {
        fetchDocument(true)
        // Poll every 10s — silent refresh (no spinner, no remount of child components)
        const interval = setInterval(() => fetchDocument(false), 10000)
        return () => clearInterval(interval)
    }, [id])

    const fetchDocument = async (isInitial = false) => {
        if (isInitial) setInitialLoading(true)
        try {
            const doc = await api.getDocument(id)
            setDocument(doc)
            if (doc.feedback?.auth_feedback) {
                setFeedback(doc.feedback.auth_feedback.thumbs || null)
                setFeedbackComment(doc.feedback.auth_feedback.comment || '')
            }
        } catch (err) {
            console.warn('API unavailable')
            if (isInitial) setDocument({ id, status: 'AUTHENTICATED', extracted_data: {}, signature_result: {} })
        }
        if (isInitial) setInitialLoading(false)
    }

    const handleProceed = async () => {
        setProceeding(true)
        try {
            // Save feedback into DB immediately
            const existingFeedback = document.feedback || {}
            await api.updateDocument(id, {
                feedback: {
                    ...existingFeedback,
                    auth_feedback: { phase: 'authenticator', thumbs: feedback, comment: feedbackComment }
                }
            })

            // Resume the LangGraph workflow past auth_review.
            // The graph runs completion_node and ends — doc moves to VERIFIED.
            await api.resumeDocument(id, 'authenticator', {
                auth_feedback: { phase: 'authenticator', thumbs: feedback, comment: feedbackComment }
            })

            navigate('/documents')
        } catch (err) {
            console.error('Failed to proceed:', err)
            navigate('/documents')
        }
        setProceeding(false)
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

    const signatureDetections = Array.isArray(document?.signature_result?.detections)
        ? document.signature_result.detections
        : []

    const TABS = [
        { id: 'account_check', label: 'Account Check', wip: true },
        { id: 'duplicate_check', label: 'Duplicate Check', wip: true },
        { id: 'confirmation_payee', label: 'Confirmation of Payee', wip: true },
        { id: 'signature_validation', label: 'Signature Authentication', wip: false },
        { id: 'fraud_check', label: 'Fraud Check', wip: true }
    ]

    const renderTabContent = () => {
        const currentTab = TABS.find(t => t.id === activeTab)
        if (currentTab?.wip) {
            return (
                <div className="wip-placeholder">
                    <div className="wip-placeholder-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: 'var(--color-text-muted)' }}>
                            <circle cx="12" cy="12" r="10" />
                            <path d="M12 6v6l4 2" />
                        </svg>
                    </div>
                    <div className="wip-placeholder-title">{currentTab.label}</div>
                    <div className="wip-placeholder-text">This feature is currently under development and will be available soon.</div>
                </div>
            )
        }
        if (activeTab === 'signature_validation') {
            return (
                <SignatureValidationTab
                    signatureDetections={signatureDetections}
                    signatureResults={signatureResults}
                    extractedData={document.extracted_data || {}}
                    documentId={id}
                />
            )
        }
        return null
    }

    const keyerFeedback = document?.feedback
    const fileName = document?.raw_file_path?.split(/[\\/]/).pop()
    const documentUrl = fileName ? `${API_BASE_URL}/static/uploads/${fileName}` : ''

    return (
        <div className="animate-fadeIn">
            {/* Page Header */}
            <div className="page-header">
                <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                        <button onClick={() => navigate('/documents')} className="btn btn-secondary" style={{ padding: '6px' }}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="19" y1="12" x2="5" y2="12" />
                                <polyline points="12 19 5 12 12 5" />
                            </svg>
                        </button>
                        <h1>Authenticator - Validation</h1>
                        <span className={`badge badge-${document.status?.toLowerCase()}`}>{document.status}</span>
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)', marginTop: 'var(--spacing-xs)' }}>
                        {document.id} &bull; Perform authentication checks
                    </p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--spacing-md)' }}>
                    <button className="btn btn-primary" onClick={handleProceed} disabled={proceeding}>
                        {proceeding && <span className="spinner" style={{ width: 16, height: 16 }}></span>}
                        Proceed
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

                    </div>
                </div>
            </div></div>}</div>

            {/* Detected Signature Crops — full-width, between extraction and auth tabs */}
            {signatureDetections.length > 0 && (
                <div className="section-card" style={{ marginBottom: 'var(--spacing-lg)' }}>
                    <div className="section-card-header">
                        <span>Detected Signatures ({signatureDetections.length})</span>
                    </div>
                    <div className="section-card-body">
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 'var(--spacing-lg)' }}>
                            {signatureDetections.map((sig, idx) => (
                                <div key={idx} style={{
                                    border: '2px solid var(--color-border-light)',
                                    borderRadius: 'var(--radius-lg)',
                                    background: 'var(--color-bg-card)',
                                    overflow: 'hidden'
                                }}>
                                    <div style={{
                                        padding: '8px 12px',
                                        background: 'var(--color-bg-tertiary)',
                                        borderBottom: '1px solid var(--color-border)',
                                        fontSize: '0.8rem', fontWeight: 700,
                                        color: 'var(--color-text-secondary)',
                                        display: 'flex', alignItems: 'center', gap: 8
                                    }}>
                                        <span style={{
                                            background: 'var(--color-primary)', color: '#fff',
                                            borderRadius: '4px', padding: '1px 8px', fontSize: '0.7rem', fontWeight: 800
                                        }}>S{idx + 1}</span>
                                        {sig.signer_name || 'Detected Signature'}
                                    </div>
                                    <div style={{
                                        padding: 'var(--spacing-md)', background: '#fff',
                                        display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 180
                                    }}>
                                        {sig.image_blob ? (
                                            <img
                                                src={`data:${sig.blob_mime_type || 'image/png'};base64,${sig.image_blob}`}
                                                alt={`Signature ${idx + 1}`}
                                                style={{ maxWidth: '100%', maxHeight: 220, objectFit: 'contain' }}
                                            />
                                        ) : (
                                            <span style={{ color: '#9ca3af', fontSize: '0.85rem' }}>No image available</span>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Row 2: Authentication Tabs (full width) */}
            <div className="section-card" style={{ marginBottom: 'var(--spacing-lg)' }}>
                <div className="section-card-header" onClick={() => setRow2Expanded(!row2Expanded)} style={{ cursor: 'pointer' }}>
                    <span>Authentication Checks</span>
                    <span style={{ fontSize: '1.2rem', transition: 'transform 0.2s', transform: row2Expanded ? 'rotate(180deg)' : 'rotate(0deg)' }}>&#9660;</span>
                </div>
            {row2Expanded && <><div className="tab-bar">
                    {TABS.map(tab => (
                        <button
                            key={tab.id}
                            className={`tab-btn ${activeTab === tab.id ? 'tab-btn-active' : ''}`}
                            onClick={() => setActiveTab(tab.id)}
                        >
                            {tab.label}
                            {tab.wip && (
                                <span style={{ marginLeft: 'var(--spacing-xs)', fontSize: '0.65rem', padding: '1px 6px', background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-full)', color: 'var(--color-text-muted)' }}>
                                    WIP
                                </span>
                            )}
                        </button>
                    ))}
                </div>
                <div className="section-card-body">
                    {renderTabContent()}
                </div>
            </>}</div>

            {/* Feedback */}
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

export default AuthenticatorView
