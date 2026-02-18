import React from 'react'
import ReactDOM from 'react-dom/client'
import { RoleProvider } from './contexts/RoleContext'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <RoleProvider>
            <App />
        </RoleProvider>
    </React.StrictMode>
)
