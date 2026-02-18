import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import { useRole } from '../contexts/RoleContext'

// Role-to-queue status mapping
const ROLE_QUEUE_STATUS = {
    keyer:         'PENDING_KEYER',
    authenticator: 'PENDING_AUTH',
    verifier:      'PENDING_VERIFIER'
}

const ROLE_QUEUE_LABELS = {
    keyer:         'Keyer Queue - Extraction Review',
    authenticator: 'Authenticator Queue - Signature Authentication',
    verifier:      'Verifier Queue - Final Review'
}

// Fallback sample data shown when API is unreachable
const SAMPLE_DOCUMENTS = [
    {
        id: 'DOC-001',
        source: 'manual',
        uploaded_by: 'john.smith',
        status: 'VERIFIED',
        raw_file_path: '/uploads/payment_instruction_001.pdf',
        created_at: '2026-01-29T10:30:00Z',
        updated_at: '2026-01-29T11:45:00Z',
        feedback: {}
    },
    {
        id: 'DOC-002',
        source: 'network_drive',
        uploaded_by: 'service_account',
        status: 'PROCESSING',
        raw_file_path: '/uploads/payment_instruction_002.pdf',
        created_at: '2026-01-29T11:00:00Z',
        updated_at: '2026-01-29T11:00:00Z',
        feedback: {}
    },
    {
        id: 'DOC-003',
        source: 'manual',
        uploaded_by: 'jane.doe',
        status: 'EXTRACTED',
        raw_file_path: '/uploads/payment_instruction_003.pdf',
        created_at: '2026-01-29T11:30:00Z',
        updated_at: '2026-01-29T11:30:00Z',
        feedback: {}
    },
    {
        id: 'DOC-004',
        source: 'manual',
        uploaded_by: 'john.smith',
        status: 'AUTHENTICATED',
        raw_file_path: '/uploads/payment_instruction_004.pdf',
        created_at: '2026-01-28T09:00:00Z',
        updated_at: '2026-01-28T10:15:00Z',
        feedback: {}
    },
    {
        id: 'DOC-005',
        source: 'network_drive',
        uploaded_by: 'service_account',
        status: 'CONFIRMED',
        raw_file_path: '/uploads/payment_instruction_005.pdf',
        created_at: '2026-01-27T14:00:00Z',
        updated_at: '2026-01-27T16:30:00Z',
        feedback: {}
    }
]

function DocumentList() {
    const { role } = useRole()
    const [documents, setDocuments] = useState([])
    const [loading, setLoading] = useState(true)
    const [statusFilter, setStatusFilter] = useState('')

    // Set default filter based on role
    useEffect(() => {
        setStatusFilter(ROLE_QUEUE_STATUS[role] || '')
    }, [role])

    useEffect(() => {
        fetchDocuments()
        // Auto-refresh every 8 seconds so status changes appear without manual reload
        const interval = setInterval(fetchDocuments, 8000)
        return () => clearInterval(interval)
    }, [statusFilter])

    const fetchDocuments = async () => {
        // Use silent refresh (no loading spinner after first load)
        if (documents.length === 0) setLoading(true)
        try {
            const response = await api.getDocuments(statusFilter)
            setDocuments(response.documents || SAMPLE_DOCUMENTS)
        } catch (err) {
            console.warn('API unavailable, showing sample data:', err.message)
            if (documents.length === 0) {
                setDocuments(SAMPLE_DOCUMENTS.filter(d =>
                    !statusFilter || d.status === statusFilter
                ))
            }
        }
        setLoading(false)
    }

    const handleProcess = async (docId) => {
        try {
            await api.processDocument(docId)
            fetchDocuments()
        } catch (err) {
            setDocuments(prev => prev.map(d =>
                d.id === docId ? { ...d, status: 'PROCESSING' } : d
            ))
        }
    }

    const handleFileUpload = async (event) => {
        const file = event.target.files[0]
        if (!file) return
        event.target.value = null  // reset so the same file can be re-uploaded

        try {
            await api.uploadDocument(file)
            // Switch to "All Statuses" so the newly uploaded INGESTED doc is visible
            setStatusFilter('')
            // fetchDocuments will fire automatically via the statusFilter useEffect
        } catch (err) {
            console.error('Upload failed:', err)
            const newDoc = {
                id: `DOC-LOCAL-${Date.now()}`,
                source: 'manual',
                uploaded_by: 'current_user',
                status: 'INGESTED',
                raw_file_path: `/uploads/${file.name}`,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                feedback: {}
            }
            setDocuments(prev => [newDoc, ...prev])
        }
    }

    const getStatusBadgeClass = (status) => {
        const statusMap = {
            'INGESTED':         'badge-ingested',
            'PROCESSING':       'badge-processing',
            'PENDING_KEYER':    'badge-pending-keyer',
            'PENDING_AUTH':     'badge-pending-auth',
            'PENDING_VERIFIER': 'badge-pending-verifier',
            'CONFIRMED':        'badge-confirmed',
            'REJECTED':         'badge-rejected',
            // legacy
            'EXTRACTED':        'badge-pending-keyer',
            'AUTHENTICATED':    'badge-pending-auth',
            'VERIFIED':         'badge-pending-verifier',
            'AWAITING_APPROVAL':'badge-processing',
            'APPROVED':         'badge-confirmed',
            'DISPATCHED':       'badge-confirmed',
        }
        return statusMap[status] || 'badge-ingested'
    }

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleString('en-GB', {
            day: '2-digit',
            month: 'short',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        })
    }

    const getFileName = (path) => {
        return path.split('/').pop()
    }

    const queueStatus = ROLE_QUEUE_STATUS[role]
    const inQueueCount = documents.filter(d => d.status === queueStatus).length
    const processingCount = documents.filter(d => d.status === 'PROCESSING').length
    const completedCount = documents.filter(d => ['CONFIRMED', 'APPROVED', 'DISPATCHED'].includes(d.status)).length

    return (
        <div className="animate-fadeIn">
            {/* Page Header */}
            <div className="page-header">
                <div>
                    <h1>My Queue</h1>
                    <p style={{ color: 'var(--color-text-secondary)', marginTop: 'var(--spacing-xs)' }}>
                        {ROLE_QUEUE_LABELS[role]}
                    </p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--spacing-md)' }}>
                    <select
                        value={statusFilter}
                        onChange={(e) => setStatusFilter(e.target.value)}
                        style={{ width: '180px' }}
                    >
                        <option value="">All Statuses</option>
                        <option value="INGESTED">Ingested</option>
                        <option value="PROCESSING">Processing</option>
                        <option value="PENDING_KEYER">Pending Keyer</option>
                        <option value="PENDING_AUTH">Pending Authenticator</option>
                        <option value="PENDING_VERIFIER">Pending Verifier</option>
                        <option value="CONFIRMED">Confirmed</option>
                        <option value="REJECTED">Rejected</option>
                    </select>
                    <button
                        className="btn btn-primary"
                        onClick={() => document.getElementById('file-upload').click()}
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="12" y1="5" x2="12" y2="19" />
                            <line x1="5" y1="12" x2="19" y2="12" />
                        </svg>
                        Upload Document
                    </button>
                    <input
                        type="file"
                        id="file-upload"
                        style={{ display: 'none' }}
                        accept=".pdf,.png,.jpg,.jpeg"
                        onChange={handleFileUpload}
                    />
                </div>
            </div>

            {/* Stats Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--spacing-md)', marginBottom: 'var(--spacing-xl)' }}>
                {[
                    { label: 'Total', value: documents.length, color: 'var(--color-text)' },
                    { label: 'In My Queue', value: inQueueCount, color: 'var(--color-primary)' },
                    { label: 'Processing', value: processingCount, color: 'var(--color-warning)' },
                    { label: 'Completed', value: completedCount, color: 'var(--color-success)' }
                ].map((stat, i) => (
                    <div key={i} className="card" style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '2rem', fontWeight: '700', color: stat.color }}>{stat.value}</div>
                        <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.875rem' }}>{stat.label}</div>
                    </div>
                ))}
            </div>

            {/* Documents Table */}
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                {loading ? (
                    <div style={{ padding: 'var(--spacing-xl)', textAlign: 'center' }}>
                        <div className="spinner" style={{ margin: '0 auto' }}></div>
                        <div style={{ marginTop: 'var(--spacing-md)', color: 'var(--color-text-secondary)' }}>
                            Loading documents...
                        </div>
                    </div>
                ) : (
                    <table className="table">
                        <thead>
                            <tr>
                                <th>Document ID</th>
                                <th>File Name</th>
                                <th>Source</th>
                                <th>Uploaded By</th>
                                <th>Status</th>
                                <th>Last Updated</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {documents.map(doc => {
                                const isReturned = doc.feedback?.returned_from === 'verifier'
                                return (
                                    <tr key={doc.id}>
                                        <td>
                                            <Link to={`/documents/${doc.id}`} style={{ fontWeight: 500 }}>
                                                {doc.id}
                                            </Link>
                                        </td>
                                        <td style={{ color: 'var(--color-text-secondary)' }}>
                                            {getFileName(doc.raw_file_path)}
                                        </td>
                                        <td>
                                            <span style={{
                                                textTransform: 'capitalize',
                                                color: doc.source === 'manual' ? 'var(--color-info)' : 'var(--color-text-secondary)'
                                            }}>
                                                {doc.source.replace('_', ' ')}
                                            </span>
                                        </td>
                                        <td>{doc.uploaded_by}</td>
                                        <td>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-xs)' }}>
                                                <span className={`badge ${getStatusBadgeClass(doc.status)}`}>
                                                    {doc.status === 'PROCESSING' && (
                                                        <span className="spinner" style={{ width: 12, height: 12, marginRight: 4 }}></span>
                                                    )}
                                                    {doc.status}
                                                </span>
                                                {isReturned && (
                                                    <span className="badge badge-returned" style={{ fontSize: '0.65rem' }}>
                                                        Returned
                                                    </span>
                                                )}
                                            </div>
                                        </td>
                                        <td style={{ color: 'var(--color-text-secondary)' }}>
                                            {formatDate(doc.updated_at)}
                                        </td>
                                        <td>
                                            <div style={{ display: 'flex', gap: 'var(--spacing-sm)' }}>
                                                {doc.status === 'INGESTED' && (
                                                    <button
                                                        className="btn btn-primary"
                                                        onClick={() => handleProcess(doc.id)}
                                                        style={{ padding: '4px 12px', fontSize: '0.75rem' }}
                                                    >
                                                        Process
                                                    </button>
                                                )}
                                                <Link
                                                    to={`/documents/${doc.id}`}
                                                    className="btn btn-secondary"
                                                    style={{ padding: '4px 12px', fontSize: '0.75rem' }}
                                                >
                                                    {doc.status === queueStatus ? 'Review' : 'View'}
                                                </Link>
                                            </div>
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    )
}

export default DocumentList
