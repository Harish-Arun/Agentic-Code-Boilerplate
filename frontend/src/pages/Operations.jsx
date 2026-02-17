import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'

function Operations() {
    const [operations, setOperations] = useState([])
    const [loading, setLoading] = useState(true)
    const [statusFilter, setStatusFilter] = useState('')

    useEffect(() => {
        fetchOperations()
    }, [statusFilter])

    const fetchOperations = async () => {
        setLoading(true)
        try {
            const response = await api.listOperations(statusFilter || null)
            setOperations(response.operations || [])
        } catch (error) {
            console.error('Failed to load operations:', error)
            setOperations([])
        }
        setLoading(false)
    }

    const getStatusBadgeClass = (status) => {
        const statusMap = {
            INGESTED: 'badge-ingested',
            PROCESSING: 'badge-processing',
            EXTRACTED: 'badge-extracted',
            AUTHENTICATED: 'badge-verified',
            VERIFIED: 'badge-verified',
            AWAITING_APPROVAL: 'badge-processing',
            REVIEW_PENDING: 'badge-verified',
            REVIEWED: 'badge-verified',
            CONFIRMED: 'badge-confirmed',
            APPROVED: 'badge-confirmed',
            DISPATCHED: 'badge-confirmed',
            REJECTED: 'badge-rejected'
        }
        return statusMap[status] || 'badge-ingested'
    }

    const formatDate = (dateString) => {
        if (!dateString) return 'N/A'
        return new Date(dateString).toLocaleString('en-GB', {
            day: '2-digit',
            month: 'short',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        })
    }

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <div>
                    <h1>Software Operations</h1>
                    <p style={{ color: 'var(--color-text-secondary)', marginTop: 'var(--spacing-xs)' }}>
                        Operation-level status transition events for audit and traceability
                    </p>
                </div>
                <div style={{ width: '220px' }}>
                    <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                        <option value="">All Statuses</option>
                        <option value="INGESTED">Ingested</option>
                        <option value="PROCESSING">Processing</option>
                        <option value="EXTRACTED">Extracted</option>
                        <option value="VERIFIED">Verified</option>
                        <option value="AWAITING_APPROVAL">Awaiting Approval</option>
                        <option value="CONFIRMED">Confirmed</option>
                        <option value="APPROVED">Approved</option>
                        <option value="DISPATCHED">Dispatched</option>
                        <option value="REJECTED">Rejected</option>
                    </select>
                </div>
            </div>

            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                {loading ? (
                    <div style={{ padding: 'var(--spacing-xl)', textAlign: 'center' }}>
                        <div className="spinner" style={{ margin: '0 auto' }}></div>
                        <div style={{ marginTop: 'var(--spacing-md)', color: 'var(--color-text-secondary)' }}>
                            Loading operations...
                        </div>
                    </div>
                ) : (
                    <table className="table">
                        <thead>
                            <tr>
                                <th>Document</th>
                                <th>Operation</th>
                                <th>From</th>
                                <th>To</th>
                                <th>Reason</th>
                                <th>Changed By</th>
                                <th>Timestamp</th>
                            </tr>
                        </thead>
                        <tbody>
                            {operations.length === 0 ? (
                                <tr>
                                    <td colSpan="7" style={{ textAlign: 'center', color: 'var(--color-text-secondary)' }}>
                                        No software operation events found
                                    </td>
                                </tr>
                            ) : (
                                operations.map((op) => (
                                    <tr key={op.id}>
                                        <td>
                                            <Link to={`/documents/${op.document_id}`} style={{ fontWeight: 500 }}>
                                                {op.document_id}
                                            </Link>
                                        </td>
                                        <td>{op.operation || 'STATUS_TRANSITION'}</td>
                                        <td>{op.from_status || 'START'}</td>
                                        <td>
                                            <span className={`badge ${getStatusBadgeClass(op.to_status)}`}>
                                                {op.to_status || 'UNKNOWN'}
                                            </span>
                                        </td>
                                        <td>{op.reason || 'N/A'}</td>
                                        <td>{op.changed_by || 'system'}</td>
                                        <td>{formatDate(op.changed_at)}</td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    )
}

export default Operations
