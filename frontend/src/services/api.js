/**
 * API Service - Handles all API calls to the backend.
 * Configurable via environment variables.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

class ApiService {
    constructor(baseUrl) {
        this.baseUrl = baseUrl
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`

        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        }

        try {
            const response = await fetch(url, config)

            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`)
            }

            // Handle empty responses
            const text = await response.text()
            return text ? JSON.parse(text) : null
        } catch (error) {
            console.error(`API Request failed: ${endpoint}`, error)
            throw error
        }
    }

    // ============================================
    // Document Endpoints
    // ============================================

    async getDocuments(status = null) {
        let url = '/documents';
        if (status) {
            url += `?status=${status}`;
        }
        return this.request(url);
    }

    async getDocument(id) {
        return this.request(`/documents/${id}`)
    }

    async uploadDocument(file) {
        const formData = new FormData()
        formData.append('file', file)

        // Remove content-type header to let browser set boundary for multipart/form-data
        return this.request('/documents/upload', {
            method: 'POST',
            body: formData,
            headers: {}
        })
    }

    async createDocument(data) {
        return this.request('/documents', {
            method: 'POST',
            body: JSON.stringify(data)
        })
    }

    async updateDocument(id, data) {
        return this.request(`/documents/${id}`, {
            method: 'PATCH',
            body: JSON.stringify(data)
        })
    }

    async updateDocumentStatus(id, status) {
        return this.request(`/documents/${id}/status?status=${status}`, {
            method: 'PATCH'
        })
    }

    async deleteDocument(id) {
        return this.request(`/documents/${id}`, {
            method: 'DELETE'
        })
    }

    // ============================================
    // Processing Endpoints
    // ============================================

    async processDocument(documentId) {
        return this.request('/process/document', {
            method: 'POST',
            body: JSON.stringify({
                document_id: documentId,
                run_extraction: true,
                run_signature_verification: true
            })
        })
    }

    async getProcessingStatus(documentId) {
        return this.request(`/process/status/${documentId}`)
    }

    async rerunProcessing(documentId, step = 'all') {
        return this.request(`/process/rerun/${documentId}?step=${step}`, {
            method: 'POST'
        })
    }

    // ============================================
    // Health Check
    // ============================================

    async healthCheck() {
        return this.request('/health')
    }
}

export const api = new ApiService(API_BASE_URL)
export default api
