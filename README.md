# VisionHome Project

VisionHome is an innovative application that allows users to visualize and design their living spaces by uploading images of empty rooms and selecting furniture or items to place within those spaces. The application utilizes AI to generate related images and provides recommendations for real-world items available for purchase.

## Features

- **Image Upload**: Users can upload images of empty rooms such as living rooms, bedrooms, kitchens, and gardens.
- **Item Selection**: Users can describe the furniture or items they wish to add (e.g., luxury green velvet sofa) and specify where to place them in the image.
- **AI Image Generation**: The application generates related images based on user input, allowing users to choose their preferred item.
- **Canvas Editor**: Users can visualize their designs on a canvas where they can place selected items.
- **Day/Night Slider**: A feature that allows users to toggle between day and night views of their designs, making the visualization more dynamic.
- **Recommendation Panel**: The application recommends real-world items from platforms like Amazon based on user selections.

## Project Structure

The project is divided into two main parts: the frontend and the backend.

### Frontend

- **React Application**: Built using TypeScript, the frontend handles user interactions and visualizations.
- **Components**: Includes components for uploading images, editing the canvas, generating items, and displaying recommendations.
- **Services**: Contains API calls to the backend for image processing and recommendations.

### Backend

- **Express Server**: The backend is built using Node.js and Express, handling image uploads and AI functionalities.
- **Controllers**: Manages the logic for uploading images, generating AI images, and providing recommendations.
- **Services**: Contains functions for interacting with AI models and fetching real-world item recommendations.

## Setup Instructions

1. **Clone the Repository**: 
   ```
   git clone <repository-url>
   cd visionhome
   ```

2. **Install Dependencies**:
   - For the frontend:
     ```
     cd frontend
     npm install
     ```
   - For the backend:
     ```
     cd backend
     npm install
     ```

3. **Environment Variables**:
   - Copy the `.env.example` file to `.env` and fill in the required variables.

4. **Run the Application**:
   - To run the frontend:
     ```
     cd frontend
     npm start
     ```
   - To run the backend:
     ```
     cd backend
     npm run start
     ```

5. **Docker Setup** (optional):
   - If you prefer to run the application using Docker, use the following command:
     ```
     docker-compose up
     ```

## Usage

- Open the application in your web browser.
- Upload an image of an empty room.
- Describe the items you want to add and select their placement.
- View the generated images and choose your preferred option.
- Use the day/night slider to see your design in different lighting.
- Check the recommendation panel for real-world items you can purchase.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.