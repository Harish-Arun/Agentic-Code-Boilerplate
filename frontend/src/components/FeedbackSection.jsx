import React from 'react'

function FeedbackSection({
    feedback = null,
    onFeedbackChange,
    comment = '',
    onCommentChange,
    readOnly = false
}) {
    return (
        <div className="feedback-section">
            <h4 style={{ marginBottom: 'var(--spacing-md)', fontSize: '0.9rem', fontWeight: 600 }}>
                Feedback
            </h4>

            <div className="feedback-buttons">
                <button
                    className={`feedback-btn ${feedback === 'up' ? 'selected-up' : ''}`}
                    onClick={() => !readOnly && onFeedbackChange && onFeedbackChange(feedback === 'up' ? null : 'up')}
                    disabled={readOnly}
                    style={readOnly ? { opacity: 0.6, cursor: 'default' } : {}}
                >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill={feedback === 'up' ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2">
                        <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" />
                        <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
                    </svg>
                    Looks Good
                </button>
                <button
                    className={`feedback-btn ${feedback === 'down' ? 'selected-down' : ''}`}
                    onClick={() => !readOnly && onFeedbackChange && onFeedbackChange(feedback === 'down' ? null : 'down')}
                    disabled={readOnly}
                    style={readOnly ? { opacity: 0.6, cursor: 'default' } : {}}
                >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill={feedback === 'down' ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2">
                        <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" />
                        <path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3" />
                    </svg>
                    Needs Correction
                </button>
            </div>

            <textarea
                className="feedback-comment"
                placeholder={readOnly ? 'No feedback comments' : 'Add feedback comments...'}
                value={comment}
                onChange={(e) => !readOnly && onCommentChange && onCommentChange(e.target.value)}
                readOnly={readOnly}
                style={readOnly ? { opacity: 0.7, cursor: 'default' } : {}}
            />
        </div>
    )
}

export default FeedbackSection
