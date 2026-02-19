import React, { useEffect, useMemo, useRef, useState } from 'react'
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist'
import pdfWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url'

GlobalWorkerOptions.workerSrc = pdfWorker

function normalizeText(value) {
    return String(value || '')
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .trim()
}

function buildSearchTerms(activeField) {
    if (!activeField || !activeField.data) return []

    const terms = []
    const value = activeField.data.value
    const location = activeField.data.location

    if (value !== null && value !== undefined && String(value).trim().length > 0) {
        terms.push(normalizeText(value))
    }

    if (location && String(location).trim().length > 0) {
        const compactLocation = normalizeText(location)
        if (compactLocation.length >= 4) {
            terms.push(compactLocation)
            compactLocation.split(',').map(part => part.trim()).filter(part => part.length >= 4).forEach(part => terms.push(part))
        }
    }

    return Array.from(new Set(terms)).filter(Boolean)
}

function toRectFromTransform(item, viewport) {
    const [a, b, c, d, e, f] = item.transform || [1, 0, 0, 1, 0, 0]
    const fontHeight = Math.hypot(b, d) || item.height || 10
    const itemWidth = item.width || Math.hypot(a, c) || 10

    const x = e * viewport.scale
    const y = viewport.height - (f * viewport.scale)
    const height = fontHeight * viewport.scale
    const width = itemWidth * viewport.scale

    return {
        left: Math.max(0, x),
        top: Math.max(0, y - height),
        width: Math.max(2, width),
        height: Math.max(2, height)
    }
}

export default function PdfHighlightViewer({
    fileUrl,
    activeField,
    activeSignature,
    focusPage,
    onPageChange
}) {
    const containerRef = useRef(null)
    const canvasRef = useRef(null)
    const renderTaskRef = useRef(null)   // track active PDF.js render task for proper cancellation

    const [pdfDoc, setPdfDoc] = useState(null)
    const [pageNumber, setPageNumber] = useState(1)
    const [numPages, setNumPages] = useState(1)
    const [pageSize, setPageSize] = useState({ width: 0, height: 0 })
    const [textHighlights, setTextHighlights] = useState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        if (!fileUrl) {
            setPdfDoc(null)
            return
        }

        let cancelled = false
        const load = async () => {
            try {
                setLoading(true)
                setError('')
                const task = getDocument(fileUrl)
                const doc = await task.promise
                if (cancelled) return
                setPdfDoc(doc)
                setNumPages(doc.numPages || 1)
                setPageNumber(1)
            } catch (err) {
                if (!cancelled) {
                    setError('Unable to render PDF preview')
                    setPdfDoc(null)
                }
            } finally {
                if (!cancelled) setLoading(false)
            }
        }

        load()
        return () => {
            cancelled = true
        }
    }, [fileUrl])

    useEffect(() => {
        if (!focusPage || !numPages) return
        const nextPage = Math.max(1, Math.min(numPages, Number(focusPage) || 1))
        setPageNumber(nextPage)
    }, [focusPage, numPages])

    const searchTerms = useMemo(() => buildSearchTerms(activeField), [activeField])

    useEffect(() => {
        if (!pdfDoc || !canvasRef.current || !containerRef.current) return

        let cancelled = false

        const renderPage = async () => {
            // Cancel any in-progress PDF.js render before starting a new one.
            // Without this, two renders on the same canvas cause a PDF.js error
            // ("There is already a pending render task") caught as "Failed to render".
            if (renderTaskRef.current) {
                renderTaskRef.current.cancel()
                renderTaskRef.current = null
            }

            try {
                setLoading(true)

                const page = await pdfDoc.getPage(pageNumber)
                if (cancelled) return

                const baseViewport = page.getViewport({ scale: 1 })
                const containerWidth = Math.max(320, containerRef.current.clientWidth - 8)
                const scale = Math.max(0.5, containerWidth / baseViewport.width)
                const viewport = page.getViewport({ scale })

                const canvas = canvasRef.current
                const context = canvas.getContext('2d')
                canvas.width = viewport.width
                canvas.height = viewport.height

                const task = page.render({ canvasContext: context, viewport })
                renderTaskRef.current = task
                await task.promise
                renderTaskRef.current = null
                if (cancelled) return

                setPageSize({ width: viewport.width, height: viewport.height })

                if (searchTerms.length === 0 || activeField?.data?.bounding_box) {
                    setTextHighlights([])
                } else {
                    const textContent = await page.getTextContent()
                    if (cancelled) return

                    const matches = []
                    for (const item of textContent.items || []) {
                        const itemText = normalizeText(item.str)
                        if (!itemText) continue

                        const isMatch = searchTerms.some(term => term && (itemText.includes(term) || term.includes(itemText)))
                        if (!isMatch) continue
                        matches.push(toRectFromTransform(item, viewport))
                    }
                    setTextHighlights(matches.slice(0, 12))
                }
            } catch (err) {
                // RenderingCancelledException is expected when a new render preempts this one
                if (err?.name === 'RenderingCancelledException') return
                if (!cancelled) {
                    setError('Failed to render selected page')
                }
            } finally {
                if (!cancelled) {
                    setLoading(false)
                    if (onPageChange) onPageChange(pageNumber)
                }
            }
        }

        renderPage()
        return () => {
            cancelled = true
            // Also cancel any in-flight PDF.js task on unmount / deps change
            if (renderTaskRef.current) {
                renderTaskRef.current.cancel()
                renderTaskRef.current = null
            }
        }
    }, [pdfDoc, pageNumber, searchTerms, activeField, onPageChange])

    const fieldBox = activeField?.data?.bounding_box && Number(activeField.data.bounding_box.page || 1) === pageNumber
        ? activeField.data.bounding_box
        : null

    const signatureBox = activeSignature?.bounding_box && Number(activeSignature.bounding_box.page || activeSignature.page || 1) === pageNumber
        ? activeSignature.bounding_box
        : null

    const toOverlayStyle = (box) => {
        if (!box) {
            return { left: '0%', top: '0%', width: '0%', height: '0%', display: 'none' }
        }

        const x1Norm = Math.max(0, Math.min(1, Number(box?.x1 || 0)))
        const y1Norm = Math.max(0, Math.min(1, Number(box?.y1 || 0)))
        const x2Norm = Math.max(0, Math.min(1, Number(box?.x2 || 0)))
        const y2Norm = Math.max(0, Math.min(1, Number(box?.y2 || 0)))

        const left = Math.min(x1Norm, x2Norm) * 100
        const top = Math.min(y1Norm, y2Norm) * 100
        const width = Math.abs(x2Norm - x1Norm) * 100
        const height = Math.abs(y2Norm - y1Norm) * 100

        return {
            left: `${left}%`,
            top: `${top}%`,
            width: `${width}%`,
            height: `${height}%`
        }
    }

    return (
        <div className="pdf-viewer-shell">
            <div className="pdf-toolbar">
                <button className="btn btn-secondary" onClick={() => setPageNumber(prev => Math.max(1, prev - 1))} disabled={pageNumber <= 1 || loading}>
                    Prev
                </button>
                <span className="pdf-page-label">Page {pageNumber} / {numPages}</span>
                <button className="btn btn-secondary" onClick={() => setPageNumber(prev => Math.min(numPages, prev + 1))} disabled={pageNumber >= numPages || loading}>
                    Next
                </button>
            </div>

            <div className="pdf-canvas-wrap" ref={containerRef}>
                {error && <div className="pdf-error">{error}</div>}
                {!error && (
                    <div className="pdf-canvas-stack">
                        <canvas ref={canvasRef} className="pdf-canvas" />

                        {fieldBox && <div className="pdf-overlay pdf-overlay-field" style={toOverlayStyle(fieldBox)} />}
                        {signatureBox && <div className="pdf-overlay pdf-overlay-signature" style={toOverlayStyle(signatureBox)} />}

                        {!fieldBox && pageSize.width > 0 && textHighlights.map((rect, idx) => (
                            <div
                                key={`text-h-${idx}`}
                                className="pdf-overlay pdf-overlay-field"
                                style={{
                                    left: `${(rect.left / pageSize.width) * 100}%`,
                                    top: `${(rect.top / pageSize.height) * 100}%`,
                                    width: `${(rect.width / pageSize.width) * 100}%`,
                                    height: `${(rect.height / pageSize.height) * 100}%`
                                }}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}