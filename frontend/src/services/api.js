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

        const isFormData = options.body instanceof FormData
        const headers = {
            ...(options.headers || {})
        }

        // Default to JSON, but NEVER force Content-Type for multipart/form-data
        if (!isFormData) {
            const hasContentType =
                Object.keys(headers).some(k => k.toLowerCase() === 'content-type')
            if (!hasContentType) {
                headers['Content-Type'] = 'application/json'
            }
        } else {
            // Let the browser set the multipart boundary
            for (const key of Object.keys(headers)) {
                if (key.toLowerCase() === 'content-type') {
                    delete headers[key]
                }
            }
        }

        const config = {
            ...options,
            headers
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

        // Let the browser set boundary for multipart/form-data
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
            })
        })
    }

    async resumeDocument(documentId, humanRole, modifications = {}) {
        return this.request('/process/resume', {
            method: 'POST',
            body: JSON.stringify({
                document_id: documentId,
                human_role: humanRole,
                modifications,
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

    async getDocumentStatusHistory(documentId, limit = 100, offset = 0) {
        return this.request(`/documents/${documentId}/status-history?limit=${limit}&offset=${offset}`)
    }

    async getDocumentOperation(documentId) {
        return this.request(`/documents/${documentId}/operation`)
    }

    async listOperations(status = null, limit = 100, offset = 0) {
        let url = `/documents/operations/list?limit=${limit}&offset=${offset}`
        if (status) {
            url += `&status=${status}`
        }
        return this.request(url)
    }

    // ============================================
    // Signature Authentication Endpoints
    // ============================================

    async verifySignature(documentId, signatureIndex, referenceBlob, referenceId, referenceMimeType = 'image/png', referenceSignerName = '') {
        return this.request('/process/verify', {
            method: 'POST',
            body: JSON.stringify({
                document_id: documentId,
                signature_index: signatureIndex,
                reference_blob: referenceBlob,
                reference_id: referenceId,
                reference_mime_type: referenceMimeType,
                reference_signer_name: referenceSignerName,
            })
        })
    }

    // ============================================
    // ISV (Signature Reference) Endpoints
    // ============================================

    async lookupISV(accountNumber, sortCode) {
        return this.request('/documents/isv/lookup', {
            method: 'POST',
            body: JSON.stringify({
                account_number: accountNumber,
                sort_code: sortCode
            })
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
