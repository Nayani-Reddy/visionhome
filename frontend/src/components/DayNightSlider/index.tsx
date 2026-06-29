import React, { useState } from 'react';
import './styles.css';

const DayNightSlider = () => {
    const [isDay, setIsDay] = useState(true);

    const toggleDayNight = () => {
        setIsDay(!isDay);
    };

    return (
        <div className={`day-night-slider ${isDay ? 'day' : 'night'}`}>
            <label className="switch">
                <input type="checkbox" checked={isDay} onChange={toggleDayNight} />
                <span className="slider round"></span>
            </label>
            <div className="label">
                {isDay ? 'Day' : 'Night'}
            </div>
        </div>
    );
};

export default DayNightSlider;