import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import DocumentList from './pages/DocumentList'
import DocumentReview from './pages/DocumentReview'
import Operations from './pages/Operations'

function App() {
    return (
        <BrowserRouter>
            <Layout>
                <Routes>
                    <Route path="/" element={<Navigate to="/documents" replace />} />
                    <Route path="/documents" element={<DocumentList />} />
                    <Route path="/documents/:id" element={<DocumentReview />} />
                    <Route path="/operations" element={<Operations />} />
                </Routes>
            </Layout>
        </BrowserRouter>
    )
}

export default App
