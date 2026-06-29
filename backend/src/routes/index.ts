import express from 'express';
import uploadController from '../controllers/uploadController';
import aiController from '../controllers/aiController';
import recommendController from '../controllers/recommendController';

const router = express.Router();

// Route for uploading images
router.post('/upload', uploadController.uploadImage);

// Route for generating images based on user input
router.post('/generate', aiController.generateImage);

// Route for getting item recommendations
router.get('/recommendations', recommendController.getRecommendations);

export default router;