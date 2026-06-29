import axios from 'axios';
import { Item } from '../models/item';

const AI_API_URL = process.env.AI_API_URL || 'http://localhost:5000/api/ai';

export const generateImages = async (description: string, location: string): Promise<string[]> => {
    try {
        const response = await axios.post(`${AI_API_URL}/generate`, {
            description,
            location,
        });
        return response.data.images; // Assuming the API returns an array of image URLs
    } catch (error) {
        console.error('Error generating images:', error);
        throw new Error('Failed to generate images');
    }
};

export const placeItemOnImage = async (imageUrl: string, item: Item, position: { x: number; y: number }): Promise<string> => {
    try {
        const response = await axios.post(`${AI_API_URL}/place`, {
            imageUrl,
            item,
            position,
        });
        return response.data.updatedImageUrl; // Assuming the API returns the updated image URL
    } catch (error) {
        console.error('Error placing item on image:', error);
        throw new Error('Failed to place item on image');
    }
};