import React, { useRef, useEffect, useState } from 'react';
import './styles.css';

const CanvasEditor = () => {
  const canvasRef = useRef<any>(null); // 'any' used to avoid TS issues with Fabric
  const [canvas, setCanvas] = useState<any>(null);
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const [dayNight, setDayNight] = useState<boolean>(true);

  useEffect(() => {
    // Import Fabric dynamically to avoid ESLint import order issues
    const fabric = require('fabric').fabric;

    const initCanvas = new fabric.Canvas('canvas', {
      backgroundColor: dayNight ? '#ffffff' : '#000000',
    });

    setCanvas(initCanvas);

    return () => {
      initCanvas.dispose();
    };
  }, [dayNight]);

  const addItemToCanvas = () => {
    if (!canvas || !selectedItem) return;
    const fabric = require('fabric').fabric;

    fabric.Image.fromURL(selectedItem, (img: any) => {
      img.set({
        left: canvas.width / 2,
        top: canvas.height / 2,
        selectable: true,
      });
      canvas.add(img);
      canvas.renderAll();
    });
  };

  const toggleDayNight = () => {
    setDayNight(!dayNight);
    if (canvas) {
      canvas.setBackgroundColor(
        dayNight ? '#000000' : '#ffffff',
        canvas.renderAll.bind(canvas)
      );
    }
  };

  return (
    <div className="canvas-editor">
      <canvas id="canvas" width={800} height={600}></canvas>
      <div className="controls">
        <input
          type="text"
          placeholder="Enter image URL"
          onChange={(e) => setSelectedItem(e.target.value)}
        />
        <button onClick={addItemToCanvas}>Add Item</button>
        <button onClick={toggleDayNight}>
          {dayNight ? 'Switch to Night' : 'Switch to Day'}
        </button>
      </div>
    </div>
  );
};

export default CanvasEditor;
