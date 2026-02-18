import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../services/api'
import PdfHighlightViewer from '../components/PdfHighlightViewer'
import ExtractionFields from '../components/ExtractionFields'
import SignatureTiles from '../components/SignatureTiles'
import FeedbackSection from '../components/FeedbackSection'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function KeyerView() {
    const { id } = useParams()
    const navigate = useNavigate()
    const [document, setDocument] = useState(null)
    const [loading, setLoading] = useState(true)
    const [editedFields, setEditedFields] = useState({})
    const [activeField, setActiveField] = useState(null)
    const [selectedSigIndex, setSelectedSigIndex] = useState(0)
    const [feedback, setFeedback] = useState(null)
    const [feedbackComment, setFeedbackComment] = useState('')
    const [proceeding, setProceeding] = useState(false)

    useEffect(() => {
        fetchDocument()
    }, [id])

    const fetchDocument = async () => {
        setLoading(true)
        try {
            const doc = await api.getDocument(id)
            setDocument(doc)
            // Load existing feedback if any
            if (doc.feedback) {
                setFeedback(doc.feedback.thumbs || null)
                setFeedbackComment(doc.feedback.comment || '')
            }
        } catch (err) {
            console.warn('API unavailable')
            setDocument({ id, status: 'EXTRACTED', extracted_data: {}, signature_result: {} })
        }
        setLoading(false)
    }

    const handleFieldChange = (field, value) => {
        setEditedFields(prev => ({ ...prev, [field]: value }))
        // Auto-set feedback to thumbs-down when a field is manually edited
        setFeedback('down')
    }

    const handleProceed = async () => {
        setProceeding(true)
        try {
            // Build merged extracted data with source updates for any keyer edits
            const mergedData = { ...(document.extracted_data || {}) }
            for (const [field, value] of Object.entries(editedFields)) {
                if (mergedData[field] && typeof mergedData[field] === 'object') {
                    mergedData[field] = { ...mergedData[field], value, source: 'human' }
                }
            }

            // Save feedback into DB immediately so it persists
            await api.updateDocument(id, {
                feedback: {
                    phase: 'keyer',
                    thumbs: feedback,
                    comment: feedbackComment,
                    edited_fields: Object.keys(editedFields)
                }
            })

            // Resume the LangGraph workflow — passes field corrections through
            // the keyer_review interrupt and runs verification in the background.
            // The agents service transitions the doc to AUTHENTICATED when done.
            await api.resumeDocument(id, 'keyer', {
                extracted_data: mergedData,
                feedback: {
                    phase: 'keyer',
                    thumbs: feedback,
                    comment: feedbackComment,
                    edited_fields: Object.keys(editedFields)
                }
            })

            navigate('/documents')
        } catch (err) {
            console.error('Failed to proceed:', err)
            navigate('/documents')
        }
        setProceeding(false)
    }

    if (loading) {
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

    // Keyer should only see signature detections, not verification results
    // Verification results are shown in Authenticator phase
    const signatureDetections = Array.isArray(document?.signature_result?.detections)
        ? document.signature_result.detections
        : []

    const activeSignatureDetection = signatureDetections[selectedSigIndex] || signatureDetections[0] || null

    const fileName = document?.raw_file_path?.split('/').pop()
    const documentUrl = fileName ? `${API_BASE_URL}/static/uploads/${fileName}` : ''

    const isReturned = document?.feedback?.returned_from === 'verifier'

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
                        <h1>Keyer - Extraction Review</h1>
                        <span className={`badge badge-${document.status?.toLowerCase()}`}>
                            {document.status}
                        </span>
                        {isReturned && (
                            <span className="badge badge-returned">Returned for Correction</span>
                        )}
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)', marginTop: 'var(--spacing-xs)' }}>
                        {document.id} &bull; Review and verify extracted payment fields
                    </p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--spacing-md)' }}>
                    <button
                        className="btn btn-primary"
                        onClick={handleProceed}
                        disabled={proceeding}
                    >
                        {proceeding && <span className="spinner" style={{ width: 16, height: 16 }}></span>}
                        Proceed
                    </button>
                </div>
            </div>

            {/* Split View */}
            <div className="split-view">
                {/* Left Panel - PDF Viewer */}
                <div className="split-panel">
                    <div className="panel-header">Document Preview</div>
                    <div className="panel-content" style={{
                        display: 'flex',
                        flexDirection: 'column',
                        minHeight: '560px',
                        background: 'var(--color-bg-tertiary)'
                    }}>
                        <PdfHighlightViewer
                            fileUrl={documentUrl}
                            activeField={activeField}
                            activeSignature={activeSignatureDetection}
                            focusPage={activeField?.data?.bounding_box?.page || activeSignatureDetection?.bounding_box?.page || 1}
                        />
                        <div style={{ marginTop: 'var(--spacing-md)', textAlign: 'center' }}>
                            <a href={documentUrl} target="_blank" rel="noreferrer"
                                style={{ fontSize: '0.875rem', color: 'var(--color-primary)' }}>
                                Open in new tab
                            </a>
                        </div>
                    </div>
                </div>

                {/* Right Panel - Extraction + Signatures + Feedback */}
                <div className="split-panel">
                    <div className="panel-header">Extracted Payment Fields</div>
                    <div className="panel-content">
                        {/* Extraction Fields (editable) */}
                        <ExtractionFields
                            extractedFields={extractedFieldEntries}
                            additionalFieldsData={additionalFieldsData}
                            editable={true}
                            editedFields={editedFields}
                            onFieldChange={handleFieldChange}
                            onFieldFocus={setActiveField}
                            activeField={activeField}
                        />

                        {/* Signature Tiles - Detection only for Keyer */}
                        {signatureDetections.length > 0 && (
                            <div style={{ marginTop: 'var(--spacing-xl)', paddingTop: 'var(--spacing-lg)', borderTop: '1px solid var(--color-border)' }}>
                                <h3 style={{ marginBottom: 'var(--spacing-md)' }}>Detected Signatures</h3>
                                <p style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)', marginBottom: 'var(--spacing-md)' }}>
                                    Signature locations detected in the document. Validation will be performed in the Authenticator phase.
                                </p>
                                <SignatureTiles
                                    signatureDetections={signatureDetections}
                                    signatureResults={[]}
                                    selectedIndex={selectedSigIndex}
                                    onSelectSignature={setSelectedSigIndex}
                                />
                            </div>
                        )}

                        {/* Feedback */}
                        <FeedbackSection
                            feedback={feedback}
                            onFeedbackChange={setFeedback}
                            comment={feedbackComment}
                            onCommentChange={setFeedbackComment}
                        />
                    </div>
                </div>
            </div>
        </div>
    )
}

export default KeyerView
