import React from 'react'
import { NavLink } from 'react-router-dom'

function Layout({ children }) {
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
                                Documents
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
