import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import DocumentList from './pages/DocumentList'
import DocumentWorkflow from './pages/DocumentWorkflow'
import Operations from './pages/Operations'

function App() {
    return (
        <BrowserRouter>
            <Layout>
                <Routes>
                    <Route path="/" element={<Navigate to="/documents" replace />} />
                    <Route path="/documents" element={<DocumentList />} />
                    <Route path="/documents/:id" element={<DocumentWorkflow />} />
                    <Route path="/operations" element={<Operations />} />
                </Routes>
            </Layout>
        </BrowserRouter>
    )
}

export default App
