import React, { useState } from 'react';

const ItemGenerator: React.FC = () => {
    const [itemDescription, setItemDescription] = useState('');
    const [generatedImages, setGeneratedImages] = useState<string[]>([]);
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [placementCoordinates, setPlacementCoordinates] = useState<{ x: number; y: number } | null>(null);

    const handleDescriptionChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setItemDescription(event.target.value);
    };

    const handleGenerateImages = async () => {
        // Call the backend API to generate images based on the item description
        const response = await fetch('/api/generate-images', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ description: itemDescription }),
        });
        const data = await response.json();
        setGeneratedImages(data.images);
    };

    const handleImageSelect = (image: string) => {
        setSelectedImage(image);
    };

    const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        setPlacementCoordinates({ x, y });
    };

    return (
        <div>
            <input
                type="text"
                value={itemDescription}
                onChange={handleDescriptionChange}
                placeholder="Describe the item (e.g., luxury green velvet sofa)"
            />
            <button onClick={handleGenerateImages}>Generate Images</button>
            <div>
                {generatedImages.map((image, index) => (
                    <img
                        key={index}
                        src={image}
                        alt={`Generated item ${index}`}
                        onClick={() => handleImageSelect(image)}
                        style={{ cursor: 'pointer', margin: '5px' }}
                    />
                ))}
            </div>
            <canvas onClick={handleCanvasClick} style={{ border: '1px solid black', width: '100%', height: '400px' }}>
                {/* Canvas rendering logic will go here */}
            </canvas>
            {selectedImage && placementCoordinates && (
                <img
                    src={selectedImage}
                    alt="Selected item"
                    style={{
                        position: 'absolute',
                        left: placementCoordinates.x,
                        top: placementCoordinates.y,
                        transform: 'translate(-50%, -50%)',
                    }}
                />
            )}
        </div>
    );
};

export default ItemGenerator;