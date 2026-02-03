import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../services/api'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Mock document data
const MOCK_DOCUMENT = {
    id: 'DOC-001',
    source: 'manual',
    uploaded_by: 'john.smith',
    status: 'VERIFIED',
    raw_file_path: '/uploads/payment_instruction_001.pdf',
    created_at: '2026-01-29T10:30:00Z',
    updated_at: '2026-01-29T11:45:00Z',
    extracted_data: {
        creditor_name: { value: 'ACME Corporation Ltd', confidence: 0.95, source: 'ai' },
        creditor_account: { value: 'GB29 NWBK 6016 1331 9268 19', confidence: 0.92, source: 'ai' },
        debtor_name: { value: 'John Smith', confidence: 0.94, source: 'ai' },
        debtor_account: { value: 'GB82 WEST 1234 5698 7654 32', confidence: 0.91, source: 'ai' },
        amount: { value: '15,000.00', confidence: 0.98, source: 'ai' },
        currency: { value: 'GBP', confidence: 0.99, source: 'ai' },
        payment_type: { value: 'CHAPS', confidence: 0.88, source: 'ai' },
        payment_date: { value: '29/01/2026', confidence: 0.85, source: 'ai' }
    },
    signature_result: {
        match: true,
        confidence: 0.87,
        reasoning: 'The signature shows consistent stroke patterns and pressure distribution. The overall shape and flow match the reference signature with high confidence.'
    }
}

function DocumentReview() {
    const { id } = useParams()
    const navigate = useNavigate()
    const [document, setDocument] = useState(null)
    const [loading, setLoading] = useState(true)
    const [editedFields, setEditedFields] = useState({})
    const [processing, setProcessing] = useState(false)

    useEffect(() => {
        fetchDocument()
    }, [id])

    const fetchDocument = async () => {
        setLoading(true)
        try {
            const doc = await api.getDocument(id)
            setDocument(doc)
        } catch (err) {
            console.log('Using mock data')
            setDocument({ ...MOCK_DOCUMENT, id })
        }
        setLoading(false)
    }

    const handleFieldChange = (field, value) => {
        setEditedFields(prev => ({
            ...prev,
            [field]: value
        }))
    }

    const handleReprocess = async (step = 'all') => {
        setProcessing(true)
        try {
            await api.rerunProcessing(id, step)
            fetchDocument()
        } catch (err) {
            // Mock update
            setDocument(prev => ({ ...prev, status: 'PROCESSING' }))
            setTimeout(() => {
                setDocument(prev => ({ ...prev, status: 'VERIFIED' }))
                setProcessing(false)
            }, 2000)
        }
        setProcessing(false)
    }

    const handleApprove = async () => {
        try {
            await api.updateDocumentStatus(id, 'CONFIRMED')
            navigate('/documents')
        } catch (err) {
            navigate('/documents')
        }
    }

    const handleReject = async () => {
        try {
            await api.updateDocumentStatus(id, 'REJECTED')
            navigate('/documents')
        } catch (err) {
            navigate('/documents')
        }
    }

    const getConfidenceClass = (confidence) => {
        if (confidence >= 0.9) return 'confidence-high'
        if (confidence >= 0.7) return 'confidence-medium'
        return 'confidence-low'
    }

    const getConfidenceLabel = (confidence) => {
        if (confidence >= 0.9) return 'High'
        if (confidence >= 0.7) return 'Medium'
        return 'Low'
    }

    if (loading) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
                <div className="spinner"></div>
            </div>
        )
    }

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
                        <h1>Document Review</h1>
                        <span className={`badge badge-${document.status.toLowerCase()}`}>
                            {document.status}
                        </span>
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)', marginTop: 'var(--spacing-xs)' }}>
                        {document.id} • Uploaded by {document.uploaded_by}
                    </p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--spacing-md)' }}>
                    <button
                        className="btn btn-secondary"
                        onClick={() => handleReprocess('extraction')}
                        disabled={processing}
                    >
                        {processing ? <span className="spinner" style={{ width: 16, height: 16 }}></span> : null}
                        Re-run Extraction
                    </button>
                    <button
                        className="btn btn-secondary"
                        onClick={() => handleReprocess('signature')}
                        disabled={processing}
                    >
                        Re-run Signature
                    </button>
                    <button
                        className="btn btn-danger"
                        onClick={handleReject}
                    >
                        Reject
                    </button>
                    <button
                        className="btn btn-success"
                        onClick={handleApprove}
                    >
                        Approve & Confirm
                    </button>
                </div>
            </div>

            {/* Split View */}
            <div className="split-view">
                {/* Left Panel - PDF Viewer (Placeholder) */}
                <div className="split-panel">
                    <div className="panel-header">
                        Document Preview
                    </div>
                    <div className="panel-content" style={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        minHeight: '500px',
                        background: 'var(--color-bg-tertiary)'
                    }}>
                        <iframe
                            src={`${API_BASE_URL}/static/uploads/${document.raw_file_path.split('/').pop()}`}
                            style={{
                                width: '100%',
                                height: '600px',
                                border: 'none',
                                borderRadius: 'var(--radius-md)'
                            }}
                            title="Document Preview"
                        />
                        <div style={{ marginTop: 'var(--spacing-md)', textAlign: 'center' }}>
                            <a
                                href={`${API_BASE_URL}/static/uploads/${document.raw_file_path.split('/').pop()}`}
                                target="_blank"
                                rel="noreferrer"
                                style={{ fontSize: '0.875rem', color: 'var(--color-primary)' }}
                            >
                                Open in new tab
                            </a>
                        </div>
                    </div>
                </div>

                {/* Right Panel - Extracted Fields */}
                <div className="split-panel">
                    <div className="panel-header">
                        Extracted Payment Fields
                    </div>
                    <div className="panel-content">
                        {/* Extracted Fields Form */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-lg)' }}>
                            {Object.entries(document.extracted_data || {}).map(([field, data]) => (
                                <div key={field}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-xs)' }}>
                                        <label style={{ textTransform: 'capitalize' }}>
                                            {field.replace(/_/g, ' ')}
                                        </label>
                                        <div className={`confidence ${getConfidenceClass(data.confidence)}`}>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
                                                {getConfidenceLabel(data.confidence)}
                                            </span>
                                            <div className="confidence-bar">
                                                <div
                                                    className="confidence-fill"
                                                    style={{ width: `${data.confidence * 100}%` }}
                                                ></div>
                                            </div>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                                                {(data.confidence * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    </div>
                                    <input
                                        type="text"
                                        value={editedFields[field] ?? data.value}
                                        onChange={(e) => handleFieldChange(field, e.target.value)}
                                    />
                                    <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: 'var(--spacing-xs)' }}>
                                        Source: {data.source === 'ai' ? '🤖 AI Extracted' : '✏️ Manual Edit'}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Signature Verification Result */}
                        {document.signature_result && (
                            <div style={{ marginTop: 'var(--spacing-xl)', paddingTop: 'var(--spacing-lg)', borderTop: '1px solid var(--color-border)' }}>
                                <h3 style={{ marginBottom: 'var(--spacing-md)' }}>Signature Verification</h3>
                                <div className="card" style={{
                                    background: document.signature_result.match
                                        ? 'var(--color-success-light)'
                                        : 'var(--color-error-light)',
                                    border: 'none'
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)', marginBottom: 'var(--spacing-md)' }}>
                                        {document.signature_result.match ? (
                                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--color-success)" strokeWidth="2">
                                                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                                                <polyline points="22 4 12 14.01 9 11.01" />
                                            </svg>
                                        ) : (
                                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--color-error)" strokeWidth="2">
                                                <circle cx="12" cy="12" r="10" />
                                                <line x1="15" y1="9" x2="9" y2="15" />
                                                <line x1="9" y1="9" x2="15" y2="15" />
                                            </svg>
                                        )}
                                        <div>
                                            <div style={{
                                                fontWeight: 600,
                                                color: document.signature_result.match ? 'var(--color-success)' : 'var(--color-error)'
                                            }}>
                                                {document.signature_result.match ? 'Signature Match' : 'Signature Mismatch'}
                                            </div>
                                            <div style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                                                Confidence: {(document.signature_result.confidence * 100).toFixed(0)}%
                                            </div>
                                        </div>
                                    </div>
                                    <div style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                                        <strong>Analysis:</strong> {document.signature_result.reasoning}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default DocumentReview
