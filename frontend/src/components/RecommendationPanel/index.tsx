import React from 'react';

const RecommendationPanel: React.FC<{ recommendations: string[]; onSelect: (item: string) => void }> = ({ recommendations, onSelect }) => {
    return (
        <div className="recommendation-panel">
            <h2>Recommended Items</h2>
            <ul>
                {recommendations.map((item, index) => (
                    <li key={index} onClick={() => onSelect(item)}>
                        {item}
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default RecommendationPanel;