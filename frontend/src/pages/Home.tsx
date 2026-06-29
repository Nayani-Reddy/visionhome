import React from 'react';
import UploadDropzone from '../components/UploadDropzone';
import CanvasEditor from '../components/CanvasEditor';
import ItemGenerator from '../components/ItemGenerator';
import DayNightSlider from '../components/DayNightSlider';
import RecommendationPanel from '../components/RecommendationPanel';

const Home: React.FC = () => {
    return (
        <div className="home-container">
            <h1>Welcome to VisionHome</h1>
            <UploadDropzone />
            <CanvasEditor />
            <ItemGenerator />
            <DayNightSlider />
            <RecommendationPanel />
        </div>
    );
};

export default Home;