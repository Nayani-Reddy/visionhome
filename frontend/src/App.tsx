import React from 'react';
import UploadDropzone from './components/UploadDropzone';
import CanvasEditor from './components/CanvasEditor';
import ItemGenerator from './components/ItemGenerator';
import DayNightSlider from './components/DayNightSlider';
import RecommendationPanel from './components/RecommendationPanel';
import './styles/globals.css';

const App: React.FC = () => {
    return (
        <div className="app-container">
            <h1>VisionHome</h1>
            <UploadDropzone />
            <CanvasEditor />
            <ItemGenerator />
            <DayNightSlider />
            <RecommendationPanel />
        </div>
    );
};

export default App;