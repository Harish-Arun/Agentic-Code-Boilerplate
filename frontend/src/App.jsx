import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import DocumentList from './pages/DocumentList'
import DocumentReview from './pages/DocumentReview'

function App() {
    return (
        <BrowserRouter>
            <Layout>
                <Routes>
                    <Route path="/" element={<Navigate to="/documents" replace />} />
                    <Route path="/documents" element={<DocumentList />} />
                    <Route path="/documents/:id" element={<DocumentReview />} />
                </Routes>
            </Layout>
        </BrowserRouter>
    )
}

export default App
