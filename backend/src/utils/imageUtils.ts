// This file contains utility functions for image processing tasks.

import { createCanvas, loadImage } from 'canvas';

// Function to overlay an image onto a base image at a specified position
export const overlayImage = async (baseImagePath: string, overlayImagePath: string, position: { x: number; y: number }): Promise<Buffer> => {
    const baseImage = await loadImage(baseImagePath);
    const overlayImage = await loadImage(overlayImagePath);

    const canvas = createCanvas(baseImage.width, baseImage.height);
    const ctx = canvas.getContext('2d');

    // Draw the base image
    ctx.drawImage(baseImage, 0, 0);
    // Draw the overlay image at the specified position
    ctx.drawImage(overlayImage, position.x, position.y);

    return canvas.toBuffer();
};

// Function to generate a daytime or nighttime effect on an image
export const applyDayNightEffect = (imageBuffer: Buffer, isDaytime: boolean): Buffer => {
    const canvas = createCanvas(1, 1);
    const ctx = canvas.getContext('2d');
    const img = loadImage(imageBuffer);

    ctx.drawImage(img, 0, 0);

    if (!isDaytime) {
        // Apply a dark overlay for nighttime effect
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)'; // 50% black overlay
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    return canvas.toBuffer();
};