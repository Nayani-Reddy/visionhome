import { Request, Response } from 'express';
import { getRecommendations } from '../services/recommendService';

export const recommendItems = async (req: Request, res: Response) => {
    try {
        const { userInput } = req.body;
        const recommendations = await getRecommendations(userInput);
        res.status(200).json(recommendations);
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
};