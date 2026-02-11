import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../services/api'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Placeholder document used when API hasn't processed the document yet
const EMPTY_DOCUMENT = {
    id: '',
    source: 'pending',
    uploaded_by: '',
    status: 'INGESTED',
    raw_file_path: '',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    extracted_data: {},
    signature_result: {}
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
            console.warn('API unavailable, showing empty document state')
            setDocument({ ...EMPTY_DOCUMENT, id })
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
            // Reprocess request failed — show processing state and re-fetch after delay
            setDocument(prev => ({ ...prev, status: 'PROCESSING' }))
            setTimeout(() => {
                fetchDocument()
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

    const renderMetric = (code, name, metricData) => {
        if (!metricData) return null
        
        // Handle both old format (score only) and new format (full object)
        const score = typeof metricData === 'object' ? (metricData.score || 0) : metricData
        const scorePercent = (score / 5) * 100
        const status = metricData.status || ''
        const notes = metricData.notes || ''
        const execution = metricData.execution || ''
        
        const scoreColor = score >= 4 ? 'var(--color-success)' : 
                          score >= 3 ? 'var(--color-warning)' : 
                          'var(--color-error)'
        
        return (
            <details key={code} style={{
                border: '1px solid rgba(0,0,0,0.1)',
                borderRadius: 'var(--radius-sm)',
                marginBottom: 'var(--spacing-sm)',
                background: 'rgba(255,255,255,0.7)'
            }}>
                <summary style={{
                    padding: 'var(--spacing-sm)',
                    cursor: 'pointer',
                    fontWeight: 600,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    listStyle: 'none'
                }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
                        <span style={{ color: score >= 3 ? 'var(--color-success)' : 'var(--color-warning)' }}>
                            {score >= 3 ? '✓' : '⚠'} {code}
                        </span>
                        <span style={{ fontWeight: 400, color: 'var(--color-text-secondary)' }}>{name}</span>
                    </span>
                    <span style={{ 
                        background: scoreColor,
                        color: 'white',
                        padding: '2px 8px',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '0.75rem'
                    }}>
                        {score}/5 ({scorePercent.toFixed(0)}%)
                    </span>
                </summary>
                {(status || notes) && (
                    <div style={{ padding: 'var(--spacing-sm)', fontSize: '0.8rem', borderTop: '1px solid rgba(0,0,0,0.1)' }}>
                        {status && (
                            <div style={{ marginBottom: 'var(--spacing-xs)' }}>
                                <strong style={{ color: 'var(--color-text)' }}>Status:</strong>
                                <span style={{ marginLeft: 'var(--spacing-xs)', color: scoreColor }}>{status}</span>
                            </div>
                        )}
                        {execution && (
                            <div style={{ marginBottom: 'var(--spacing-xs)' }}>
                                <strong style={{ color: 'var(--color-text)' }}>Execution:</strong>
                                <span style={{ marginLeft: 'var(--spacing-xs)', color: 'var(--color-text-muted)' }}>
                                    {execution === 'llm' ? '🤖 AI Analysis' : 
                                     execution === 'llm_vision' ? '👁️ Visual Analysis' : 
                                     execution}
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
                                color: 'var(--color-text-secondary)'
                            }}>
                                <strong>Analysis:</strong><br/>
                                {notes}
                            </div>
                        )}
                    </div>
                )}
            </details>
        )
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
                            <div style={{ marginTop: 'var(--spacing-xl)', paddingTop: 'var(--spacing-lg)', borderTop: '2px solid #e5e7eb' }}>
                                <h3 style={{ 
                                    marginBottom: 'var(--spacing-lg)',
                                    fontSize: '1.25rem',
                                    fontWeight: 700,
                                    color: '#ffffff',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '10px'
                                }}>
                                    <span style={{ fontSize: '1.5rem' }}>🔐</span>
                                    Signature Verification
                                </h3>
                                <div className="card" style={{
                                    background: document.signature_result.match
                                        ? 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)'
                                        : 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
                                    border: `3px solid ${document.signature_result.match ? '#10b981' : '#ef4444'}`,
                                    borderRadius: '12px',
                                    boxShadow: document.signature_result.match
                                        ? '0 4px 12px rgba(16, 185, 129, 0.2)'
                                        : '0 4px 12px rgba(239, 68, 68, 0.2)'
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)', marginBottom: 'var(--spacing-md)' }}>
                                        {document.signature_result.match ? (
                                            <div style={{ 
                                                width: '48px', 
                                                height: '48px', 
                                                borderRadius: '50%', 
                                                background: '#10b981',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                boxShadow: '0 4px 8px rgba(16, 185, 129, 0.3)'
                                            }}>
                                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                                                    <polyline points="20 6 9 17 4 12" />
                                                </svg>
                                            </div>
                                        ) : (
                                            <div style={{ 
                                                width: '48px', 
                                                height: '48px', 
                                                borderRadius: '50%', 
                                                background: '#ef4444',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                boxShadow: '0 4px 8px rgba(239, 68, 68, 0.3)'
                                            }}>
                                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                                                    <line x1="18" y1="6" x2="6" y2="18" />
                                                    <line x1="6" y1="6" x2="18" y2="18" />
                                                </svg>
                                            </div>
                                        )}
                                        <div>
                                            <div style={{
                                                fontSize: '1.125rem',
                                                fontWeight: 700,
                                                color: document.signature_result.match ? '#065f46' : '#991b1b',
                                                marginBottom: '4px'
                                            }}>
                                                {document.signature_result.match ? 'Signature Match' : 'Signature Mismatch'}
                                            </div>
                                            <div style={{ 
                                                fontSize: '0.875rem', 
                                                fontWeight: 600,
                                                color: document.signature_result.match ? '#047857' : '#dc2626',
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: '6px'
                                            }}>
                                                <span>Confidence:</span>
                                                <span style={{ 
                                                    fontSize: '1rem',
                                                    padding: '2px 8px',
                                                    background: 'rgba(255,255,255,0.5)',
                                                    borderRadius: '6px'
                                                }}>
                                                    {(document.signature_result.confidence * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                    <div style={{ 
                                        fontSize: '0.875rem', 
                                        color: '#374151',
                                        background: 'rgba(255,255,255,0.6)',
                                        padding: 'var(--spacing-md)',
                                        borderRadius: '8px',
                                        lineHeight: '1.6',
                                        marginBottom: 'var(--spacing-md)'
                                    }}>
                                        <strong style={{ color: '#1f2937' }}>Analysis:</strong> {document.signature_result.reasoning}
                                    </div>
                                    
                                    {/* Signature Images */}
                                    {(document.signature_result.signature_blob || document.signature_result.reference_blob) && (
                                        <div style={{ marginTop: 'var(--spacing-lg)', paddingTop: 'var(--spacing-lg)', borderTop: '2px solid #e5e7eb' }}>
                                            <h4 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 'var(--spacing-lg)', color: '#1f2937', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                <span style={{ fontSize: '1.25rem' }}>✍️</span>
                                                Signature Comparison
                                            </h4>
                                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-lg)' }}>
                                                {document.signature_result.signature_blob && (
                                                    <div>
                                                        <div style={{ 
                                                            display: 'inline-flex',
                                                            alignItems: 'center',
                                                            gap: '6px',
                                                            padding: '6px 14px',
                                                            background: 'linear-gradient(135deg, #5b21b6 0%, #6b21a8 100%)',
                                                            borderRadius: '20px',
                                                            fontSize: '0.8rem',
                                                            fontWeight: 700,
                                                            color: 'white',
                                                            marginBottom: 'var(--spacing-sm)',
                                                            boxShadow: '0 2px 6px rgba(91, 33, 182, 0.3)',
                                                            textShadow: '0 1px 2px rgba(0,0,0,0.2)'
                                                        }}>
                                                            <span>🔍</span>
                                                            Questioned Signature
                                                        </div>
                                                        <div style={{ 
                                                            border: '3px solid #5b21b6',
                                                            borderRadius: '12px',
                                                            padding: 'var(--spacing-lg)',
                                                            background: 'linear-gradient(to bottom, #f5f3ff, #ffffff)',
                                                            display: 'flex',
                                                            justifyContent: 'center',
                                                            alignItems: 'center',
                                                            minHeight: '180px',
                                                            boxShadow: '0 4px 12px rgba(91, 33, 182, 0.15)',
                                                            transition: 'transform 0.2s, box-shadow 0.2s',
                                                            cursor: 'pointer'
                                                        }}
                                                        onMouseEnter={(e) => {
                                                            e.currentTarget.style.transform = 'translateY(-2px)';
                                                            e.currentTarget.style.boxShadow = '0 8px 20px rgba(91, 33, 182, 0.25)';
                                                        }}
                                                        onMouseLeave={(e) => {
                                                            e.currentTarget.style.transform = 'translateY(0)';
                                                            e.currentTarget.style.boxShadow = '0 4px 12px rgba(91, 33, 182, 0.15)';
                                                        }}>
                                                            <img 
                                                                src={`data:${document.signature_result.blob_mime_type || 'image/png'};base64,${document.signature_result.signature_blob}`}
                                                                alt="Questioned Signature"
                                                                style={{ maxWidth: '100%', maxHeight: '220px', objectFit: 'contain', filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))' }}
                                                            />
                                                        </div>
                                                    </div>
                                                )}
                                                {document.signature_result.reference_blob && (
                                                    <div>
                                                        <div style={{ 
                                                            display: 'inline-flex',
                                                            alignItems: 'center',
                                                            gap: '6px',
                                                            padding: '6px 14px',
                                                            background: 'linear-gradient(135deg, #047857 0%, #065f46 100%)',
                                                            borderRadius: '20px',
                                                            fontSize: '0.8rem',
                                                            fontWeight: 700,
                                                            color: 'white',
                                                            marginBottom: 'var(--spacing-sm)',
                                                            boxShadow: '0 2px 6px rgba(4, 120, 87, 0.3)',
                                                            textShadow: '0 1px 2px rgba(0,0,0,0.2)'
                                                        }}>
                                                            <span>✓</span>
                                                            Reference Signature
                                                        </div>
                                                        <div style={{ 
                                                            border: '3px solid #047857',
                                                            borderRadius: '12px',
                                                            padding: 'var(--spacing-lg)',
                                                            background: 'linear-gradient(to bottom, #ecfdf5, #ffffff)',
                                                            display: 'flex',
                                                            justifyContent: 'center',
                                                            alignItems: 'center',
                                                            minHeight: '180px',
                                                            boxShadow: '0 4px 12px rgba(4, 120, 87, 0.15)',
                                                            transition: 'transform 0.2s, box-shadow 0.2s',
                                                            cursor: 'pointer'
                                                        }}
                                                        onMouseEnter={(e) => {
                                                            e.currentTarget.style.transform = 'translateY(-2px)';
                                                            e.currentTarget.style.boxShadow = '0 8px 20px rgba(4, 120, 87, 0.25)';
                                                        }}
                                                        onMouseLeave={(e) => {
                                                            e.currentTarget.style.transform = 'translateY(0)';
                                                            e.currentTarget.style.boxShadow = '0 4px 12px rgba(4, 120, 87, 0.15)';
                                                        }}>
                                                            <img 
                                                                src={`data:${document.signature_result.blob_mime_type || 'image/png'};base64,${document.signature_result.reference_blob}`}
                                                                alt="Reference Signature"
                                                                style={{ maxWidth: '100%', maxHeight: '220px', objectFit: 'contain', filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))' }}
                                                            />
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                    
                                    {/* M1-M7 Metrics Breakdown */}
                                    {document.signature_result.metrics && (
                                        <div style={{ marginTop: 'var(--spacing-lg)', paddingTop: 'var(--spacing-md)', borderTop: '1px solid rgba(0,0,0,0.1)' }}>
                                            <h4 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: 'var(--spacing-md)', color: 'var(--color-text)' }}>
                                                📊 M1-M7 Metrics Analysis
                                            </h4>
                                            <div style={{ display: 'grid', gap: 'var(--spacing-sm)' }}>
                                                {renderMetric('M1', 'Global Form', document.signature_result.metrics.m1_global_form)}
                                                {renderMetric('M2', 'Line Quality', document.signature_result.metrics.m2_line_quality)}
                                                {renderMetric('M3', 'Slant Angle', document.signature_result.metrics.m3_slant_angle)}
                                                {renderMetric('M4', 'Baseline Stability', document.signature_result.metrics.m4_baseline_stability)}
                                                {renderMetric('M5', 'Terminal Strokes', document.signature_result.metrics.m5_terminal_strokes)}
                                                {renderMetric('M6', 'Spacing Density', document.signature_result.metrics.m6_spacing_density)}
                                                {renderMetric('M7', 'Pressure Inference', document.signature_result.metrics.m7_pressure_inference)}
                                            </div>
                                            
                                            {/* Scoring Details */}
                                            {document.signature_result.scoring_details && (
                                                <div style={{ marginTop: 'var(--spacing-lg)', paddingTop: 'var(--spacing-md)', borderTop: '1px solid rgba(0,0,0,0.1)' }}>
                                                    <h4 style={{ fontSize: '0.875rem', fontWeight: 700, marginBottom: 'var(--spacing-md)', color: '#1f2937' }}>
                                                        📈 FIV {document.signature_result.scoring_details.fiv_version} Scoring Details
                                                    </h4>
                                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-sm)', fontSize: '0.875rem' }}>
                                                        <div style={{ padding: 'var(--spacing-sm)', background: '#ffffff', borderRadius: 'var(--radius-sm)', border: '1px solid #e5e7eb' }}>
                                                            <strong style={{ color: '#1f2937' }}>Vetoed:</strong> 
                                                            <span style={{ 
                                                                marginLeft: 'var(--spacing-xs)', 
                                                                color: document.signature_result.scoring_details.vetoed ? '#dc2626' : '#059669',
                                                                fontWeight: 600
                                                            }}>
                                                                {document.signature_result.scoring_details.vetoed ? 'Yes' : 'No'}
                                                            </span>
                                                            {document.signature_result.scoring_details.veto_metric && (
                                                                <span style={{ marginLeft: 'var(--spacing-xs)', color: '#6b7280' }}>
                                                                    ({document.signature_result.scoring_details.veto_metric})
                                                                </span>
                                                            )}
                                                        </div>
                                                        <div style={{ padding: 'var(--spacing-sm)', background: '#ffffff', borderRadius: 'var(--radius-sm)', border: '1px solid #e5e7eb' }}>
                                                            <strong style={{ color: '#1f2937' }}>Base Score:</strong> 
                                                            <span style={{ marginLeft: 'var(--spacing-xs)', color: '#1f2937', fontWeight: 600 }}>
                                                                {document.signature_result.scoring_details.base_score != null ? document.signature_result.scoring_details.base_score.toFixed(1) : 'N/A'}
                                                            </span>
                                                        </div>
                                                        <div style={{ padding: 'var(--spacing-sm)', background: '#ffffff', borderRadius: 'var(--radius-sm)', border: '1px solid #e5e7eb' }}>
                                                            <strong style={{ color: '#1f2937' }}>Penalties:</strong> 
                                                            <span style={{ marginLeft: 'var(--spacing-xs)', color: '#dc2626', fontWeight: 600 }}>
                                                                -{document.signature_result.scoring_details.penalties_applied != null ? document.signature_result.scoring_details.penalties_applied.toFixed(1) : '0.0'}
                                                            </span>
                                                        </div>
                                                        <div style={{ padding: 'var(--spacing-sm)', background: '#ffffff', borderRadius: 'var(--radius-sm)', border: '1px solid #e5e7eb' }}>
                                                            <strong style={{ color: '#1f2937' }}>Final Score:</strong> 
                                                            <span style={{ marginLeft: 'var(--spacing-xs)', fontWeight: 700, color: '#2563eb', fontSize: '1rem' }}>
                                                                {document.signature_result.scoring_details.final_score != null ? document.signature_result.scoring_details.final_score.toFixed(1) : '0.0'}
                                                            </span>
                                                        </div>
                                                        <div style={{ padding: 'var(--spacing-sm)', background: '#ffffff', borderRadius: 'var(--radius-sm)', border: '1px solid #e5e7eb', gridColumn: '1 / -1' }}>
                                                            <strong style={{ color: '#1f2937' }}>Decision:</strong> 
                                                            <span style={{ marginLeft: 'var(--spacing-xs)', fontWeight: 700, color: '#1f2937', fontSize: '0.95rem' }}>
                                                                {document.signature_result.scoring_details.decision || 'UNKNOWN'} 
                                                            </span>
                                                            <span style={{ 
                                                                marginLeft: 'var(--spacing-xs)',
                                                                padding: '3px 8px',
                                                                borderRadius: 'var(--radius-sm)',
                                                                background: document.signature_result.scoring_details.confidence_band === 'HIGH' ? '#059669' :
                                                                           document.signature_result.scoring_details.confidence_band === 'MEDIUM' ? '#d97706' : '#dc2626',
                                                                color: 'white',
                                                                fontSize: '0.75rem',
                                                                fontWeight: 700
                                                            }}>
                                                                {document.signature_result.scoring_details.confidence_band}
                                                            </span>
                                                        </div>
                                                        {document.signature_result.scoring_details.veto_reason && (
                                                            <div style={{ padding: 'var(--spacing-sm)', background: '#fee2e2', borderRadius: 'var(--radius-sm)', gridColumn: '1 / -1', border: '1px solid #fecaca' }}>
                                                                <strong style={{ color: '#991b1b' }}>Veto Reason:</strong><br/>
                                                                <span style={{ color: '#1f2937' }}>
                                                                    {document.signature_result.scoring_details.veto_reason}
                                                                </span>
                                                            </div>
                                                        )}
                                                        {document.signature_result.scoring_details.llm_model && (
                                                            <div style={{ padding: 'var(--spacing-sm)', background: 'rgba(255,255,255,0.5)', borderRadius: 'var(--radius-sm)', gridColumn: '1 / -1', fontSize: '0.7rem' }}>
                                                                <strong>Analysis Model:</strong> {document.signature_result.scoring_details.llm_model}
                                                                {document.signature_result.scoring_details.processing_time_ms > 0 && (
                                                                    <span style={{ marginLeft: 'var(--spacing-md)', color: 'var(--color-text-muted)' }}>
                                                                        Time: {document.signature_result.scoring_details.processing_time_ms}ms
                                                                    </span>
                                                                )}
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}
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
