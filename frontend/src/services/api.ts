import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api'; // Adjust the base URL as needed

// Function to upload an image
export const uploadImage = async (formData) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error uploading image:', error);
        throw error;
    }
};

// Function to generate related images based on user input
export const generateImages = async (description) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/generate`, { description });
        return response.data;
    } catch (error) {
        console.error('Error generating images:', error);
        throw error;
    }
};

// Function to get item recommendations
export const getRecommendations = async (selectedItems) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/recommend`, { items: selectedItems });
        return response.data;
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        throw error;
    }
};