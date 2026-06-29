import { Request, Response } from 'express';
import { generateImages, placeItem } from '../services/aiService';

export const generateItemImages = async (req: Request, res: Response) => {
    try {
        const { description, location } = req.body;
        const images = await generateImages(description);
        res.status(200).json(images);
    } catch (error) {
        res.status(500).json({ message: 'Error generating images', error });
    }
};

export const addItemToImage = async (req: Request, res: Response) => {
    try {
        const { image, itemImage, location } = req.body;
        const updatedImage = await placeItem(image, itemImage, location);
        res.status(200).json(updatedImage);
    } catch (error) {
        res.status(500).json({ message: 'Error placing item on image', error });
    }
};