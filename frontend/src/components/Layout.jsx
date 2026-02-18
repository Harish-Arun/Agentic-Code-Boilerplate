import React from 'react'
import { NavLink } from 'react-router-dom'
import { useRole } from '../contexts/RoleContext'

const ROLE_LABELS = {
    keyer: 'Keyer',
    authenticator: 'Authenticator',
    verifier: 'Verifier'
}

const ROLE_ICONS = {
    keyer: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
        </svg>
    ),
    authenticator: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        </svg>
    ),
    verifier: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
        </svg>
    )
}

function Layout({ children }) {
    const { role, setRole } = useRole()

    return (
        <div className="app-layout">
            {/* Sidebar */}
            <aside className="sidebar">
                <div className="logo">
                    <div className="logo-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                            <path d="M12 2L2 7l10 5 10-5-10-5z" />
                            <path d="M2 17l10 5 10-5" />
                            <path d="M2 12l10 5 10-5" />
                        </svg>
                    </div>
                    <span>MMP-AI</span>
                </div>

                {/* Role Selector */}
                <div className="role-selector">
                    <label className="role-selector-label">Active Role</label>
                    <div className="role-selector-buttons">
                        {Object.entries(ROLE_LABELS).map(([key, label]) => (
                            <button
                                key={key}
                                className={`role-btn ${role === key ? 'role-btn-active' : ''}`}
                                onClick={() => setRole(key)}
                                title={label}
                            >
                                {ROLE_ICONS[key]}
                                <span>{label}</span>
                            </button>
                        ))}
                    </div>
                </div>

                <nav>
                    <ul className="nav-menu">
                        <li className="nav-item">
                            <NavLink
                                to="/documents"
                                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                    <polyline points="14 2 14 8 20 8" />
                                </svg>
                                My Queue
                            </NavLink>
                        </li>
                        <li className="nav-item">
                            <NavLink
                                to="/operations"
                                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <rect x="3" y="4" width="18" height="16" rx="2" ry="2" />
                                    <line x1="3" y1="10" x2="21" y2="10" />
                                    <line x1="8" y1="14" x2="8" y2="20" />
                                </svg>
                                Operations
                            </NavLink>
                        </li>
                    </ul>
                </nav>

                <div style={{ marginTop: 'auto', paddingTop: 'var(--spacing-xl)', borderTop: '1px solid var(--color-border)' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                        <div>Environment: Development</div>
                        <div>Version: 1.0.0</div>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="main-content">
                {children}
            </main>
        </div>
    )
}

export default Layout
