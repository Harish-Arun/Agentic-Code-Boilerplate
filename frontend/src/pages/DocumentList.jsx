import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import { useRole } from '../contexts/RoleContext'

// What each role's primary action status is (the HITL pause point for that role)
const ROLE_PRIMARY_STATUS = {
    keyer:         'PENDING_KEYER',
    authenticator: 'PENDING_AUTH',
    verifier:      'PENDING_VERIFIER',
}

// Full set of statuses each role is permitted to see — enforced client-side too
//
// Keyer:         INGESTED      - trigger AI processing
//                PROCESSING    - monitor AI extraction progress
//                EXTRACTED     - legacy alias for PENDING_KEYER (old DB docs)
//                PENDING_KEYER - main HITL queue (graph paused; keyer reviews & corrects)
//                REJECTED      - legacy rejected docs returned for rework
//
// Authenticator: PENDING_AUTH  - main HITL queue (graph paused; authenticate signatures)
//                AUTHENTICATED - legacy alias for PENDING_AUTH (old DB docs)
//
// Verifier:      PENDING_VERIFIER - main HITL queue (graph paused; final sign-off)
//                CONFIRMED        - verifier accepted (visible for tracking)
const ROLE_ALLOWED_STATUSES = {
    keyer:         ['INGESTED', 'PROCESSING', 'EXTRACTED', 'PENDING_KEYER', 'REJECTED'],
    authenticator: ['PENDING_AUTH', 'AUTHENTICATED'],
    verifier:      ['PENDING_VERIFIER', 'CONFIRMED'],
}

// Human-readable filter options per role
const ROLE_FILTER_OPTIONS = {
    keyer: [
        { value: 'PENDING_KEYER', label: 'Pending Keyer Review' },
        { value: 'INGESTED',      label: 'Ingested' },
        { value: 'PROCESSING',    label: 'Processing' },
        { value: 'EXTRACTED',     label: 'Extracted (legacy)' },
        { value: 'REJECTED',      label: 'Returned / Rejected' },
        { value: '',              label: 'All (my phase)' },
    ],
    authenticator: [
        { value: 'PENDING_AUTH',  label: 'Pending Authentication' },
        { value: 'AUTHENTICATED', label: 'Authenticated (legacy)' },
        { value: '',              label: 'All (my phase)' },
    ],
    verifier: [
        { value: 'PENDING_VERIFIER', label: 'Pending Verification' },
        { value: 'CONFIRMED',        label: 'Confirmed' },
        { value: '',                 label: 'All (my phase)' },
    ],
}

const ROLE_QUEUE_LABELS = {
    keyer:         'Keyer Queue — Data Extraction & Review',
    authenticator: 'Authenticator Queue — Signature Authentication',
    verifier:      'Verifier Queue — Final Verification & Approval',
}

// Fallback sample data shown when API is unreachable
const SAMPLE_DOCUMENTS = [
    {
        id: 'DOC-001',
        source: 'manual',
        uploaded_by: 'john.smith',
        status: 'PENDING_KEYER',
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
        status: 'INGESTED',
        raw_file_path: '/uploads/payment_instruction_003.pdf',
        created_at: '2026-01-29T11:30:00Z',
        updated_at: '2026-01-29T11:30:00Z',
        feedback: {}
    },
    {
        id: 'DOC-004',
        source: 'manual',
        uploaded_by: 'john.smith',
        status: 'PENDING_AUTH',
        raw_file_path: '/uploads/payment_instruction_004.pdf',
        created_at: '2026-01-28T09:00:00Z',
        updated_at: '2026-01-28T10:15:00Z',
        feedback: {}
    },
    {
        id: 'DOC-005',
        source: 'network_drive',
        uploaded_by: 'service_account',
        status: 'PENDING_VERIFIER',
        raw_file_path: '/uploads/payment_instruction_005.pdf',
        created_at: '2026-01-27T14:00:00Z',
        updated_at: '2026-01-27T16:30:00Z',
        feedback: {}
    },
    {
        id: 'DOC-006',
        source: 'manual',
        uploaded_by: 'jane.doe',
        status: 'CONFIRMED',
        raw_file_path: '/uploads/payment_instruction_006.pdf',
        created_at: '2026-01-27T14:00:00Z',
        updated_at: '2026-01-27T17:00:00Z',
        feedback: {}
    }
]

function DocumentList() {
    const { role } = useRole()
    const [allDocuments, setAllDocuments] = useState([])
    const [loading, setLoading] = useState(true)
    const [statusFilter, setStatusFilter] = useState(() => ROLE_PRIMARY_STATUS[role] || '')

    // When role changes: clear stale docs (shows spinner) and reset filter.
    // The filter change then triggers the fetch effect below — no race condition.
    useEffect(() => {
        setAllDocuments([])
        setStatusFilter(ROLE_PRIMARY_STATUS[role] || '')
    }, [role])

    // Fetch whenever the status filter changes (including after a role switch)
    useEffect(() => {
        fetchDocuments()
        const interval = setInterval(fetchDocuments, 8000)
        return () => clearInterval(interval)
    }, [statusFilter])

    const fetchDocuments = async () => {
        if (allDocuments.length === 0) setLoading(true)
        try {
            // Fetch with the UI filter (or no filter to get all allowed statuses at once)
            const response = await api.getDocuments(statusFilter)
            setAllDocuments(response.documents || SAMPLE_DOCUMENTS)
        } catch (err) {
            console.warn('API unavailable, showing sample data:', err.message)
            if (allDocuments.length === 0) {
                setAllDocuments(SAMPLE_DOCUMENTS.filter(d =>
                    !statusFilter || d.status === statusFilter
                ))
            }
        }
        setLoading(false)
    }

    // Hard client-side enforcement: never show docs outside this role's allowed set
    const allowed = ROLE_ALLOWED_STATUSES[role] || []
    const documents = allDocuments.filter(d =>
        allowed.length === 0 || allowed.includes(d.status)
    )

    const handleProcess = async (docId) => {
        try {
            await api.processDocument(docId)
            fetchDocuments()
        } catch (err) {
            setAllDocuments(prev => prev.map(d =>
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
            setAllDocuments(prev => [newDoc, ...prev])
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

    const queueStatus = ROLE_PRIMARY_STATUS[role]
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
                        style={{ width: '220px' }}
                    >
                        {(ROLE_FILTER_OPTIONS[role] || [{ value: '', label: 'All Statuses' }]).map(opt => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
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
