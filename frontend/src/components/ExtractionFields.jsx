import React, { useState } from 'react'

const getConfidenceClass = (confidence) => {
    if (confidence >= 0.9) return 'confidence-high'
    if (confidence >= 0.7) return 'confidence-medium'
    return 'confidence-low'
}

const getConfidenceLabel = (confidence) => {
    if (confidence >= 0.9) return 'High'
    if (confidence >= 0.7) return 'Medium'
    return 'Low'
}

function ExtractionFields({
    extractedFields = [],
    additionalFieldsData = null,
    editable = false,
    editedFields = {},
    onFieldChange,
    onFieldFocus,
    activeField
}) {
    const [additionalFieldsExpanded, setAdditionalFieldsExpanded] = useState(false)

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-lg)' }}>
            {extractedFields.map(([field, data]) => {
                const isEdited = editedFields[field] !== undefined && editedFields[field] !== data.value
                const displaySource = isEdited ? 'human' : (data.source || 'ai')

                return (
                    <div
                        key={field}
                        onClick={() => onFieldFocus && onFieldFocus({ field, data })}
                        style={{
                            border: activeField?.field === field ? '1px solid var(--color-primary)' : '1px solid transparent',
                            borderRadius: 'var(--radius-md)',
                            padding: 'var(--spacing-sm)',
                            cursor: 'pointer'
                        }}
                    >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-xs)' }}>
                            <label style={{ textTransform: 'capitalize' }}>
                                {field.replace(/_/g, ' ')}
                            </label>
                            <div className={`confidence ${getConfidenceClass(data.confidence)}`}>
                                <span style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
                                    {getConfidenceLabel(data.confidence)}
                                </span>
                                <div className="confidence-bar">
                                    <div
                                        className="confidence-fill"
                                        style={{ width: `${data.confidence * 100}%` }}
                                    ></div>
                                </div>
                                <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                                    {(data.confidence * 100).toFixed(0)}%
                                </span>
                            </div>
                        </div>

                        {editable ? (
                            <input
                                type="text"
                                value={editedFields[field] ?? data.value}
                                onChange={(e) => onFieldChange && onFieldChange(field, e.target.value)}
                                onFocus={() => onFieldFocus && onFieldFocus({ field, data })}
                            />
                        ) : (
                            <div className="field-readonly">
                                {editedFields[field] ?? data.value}
                            </div>
                        )}

                        <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: 'var(--spacing-xs)' }}>
                            Source: {displaySource === 'ai' ? 'AI Extracted' : 'Manual Edit'}
                        </div>
                    </div>
                )
            })}

            {/* Additional Fields - Catch-All */}
            {additionalFieldsData && (
                <div style={{ marginTop: 'var(--spacing-xl)', paddingTop: 'var(--spacing-lg)', borderTop: '1px solid var(--color-border)' }}>
                    <div
                        onClick={() => setAdditionalFieldsExpanded(!additionalFieldsExpanded)}
                        style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            cursor: 'pointer',
                            marginBottom: additionalFieldsExpanded ? 'var(--spacing-md)' : '0'
                        }}
                    >
                        <h3>Additional Fields</h3>
                        <span style={{ fontSize: '1.5rem', transition: 'transform 0.2s', transform: additionalFieldsExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}>
                            ▼
                        </span>
                    </div>

                    {additionalFieldsExpanded && (
                        <div className="card" style={{ marginTop: 'var(--spacing-sm)' }}>
                            <pre style={{
                                background: '#1e293b',
                                color: '#e2e8f0',
                                padding: 'var(--spacing-md)',
                                borderRadius: 'var(--radius-md)',
                                fontSize: '0.75rem',
                                overflow: 'auto',
                                maxHeight: '400px',
                                fontFamily: 'monospace',
                                lineHeight: '1.5'
                            }}>
                                {JSON.stringify(additionalFieldsData, null, 2)}
                            </pre>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default ExtractionFields
