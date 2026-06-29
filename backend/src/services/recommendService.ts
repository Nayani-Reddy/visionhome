import axios from 'axios';
import { Item } from '../models/item';

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:5000';

export const fetchRecommendations = async (userInput: string): Promise<Item[]> => {
    try {
        const response = await axios.post(`${API_BASE_URL}/recommendations`, { input: userInput });
        return response.data;
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        throw new Error('Could not fetch recommendations');
    }
};