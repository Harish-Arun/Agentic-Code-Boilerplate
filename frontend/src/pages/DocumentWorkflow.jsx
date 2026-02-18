import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useRole } from '../contexts/RoleContext'
import { api } from '../services/api'
import KeyerView from './KeyerView'
import AuthenticatorView from './AuthenticatorView'
import VerifierView from './VerifierView'

// Map document status → which view renders it
const STATUS_VIEW = {
    PENDING_KEYER:    'keyer',
    PENDING_AUTH:     'authenticator',
    PENDING_VERIFIER: 'verifier',
    CONFIRMED:        'verifier',   // read-only final view
    REJECTED:         'keyer',      // returned to keyer
    // legacy aliases
    EXTRACTED:        'keyer',
    AUTHENTICATED:    'authenticator',
    VERIFIED:         'verifier',
}

function DocumentWorkflow() {
    const { role } = useRole()
    const { id } = useParams()
    const [docStatus, setDocStatus] = useState(null)

    useEffect(() => {
        if (!id) return
        api.getDocument(id)
            .then(doc => setDocStatus(doc.status))
            .catch(() => setDocStatus(null))
    }, [id])

    // Determine which view to render:
    // - If doc status maps to a view, show that view regardless of active role
    //   (so a verifier looking at a PENDING_KEYER doc still sees the keyer read-only view)
    // - Fall back to role-based routing while status is loading or unknown
    const effectiveView = (docStatus && STATUS_VIEW[docStatus]) || role

    switch (effectiveView) {
        case 'keyer':
            return <KeyerView />
        case 'authenticator':
            return <AuthenticatorView />
        case 'verifier':
            return <VerifierView />
        default:
            return <KeyerView />
    }
}

export default DocumentWorkflow
