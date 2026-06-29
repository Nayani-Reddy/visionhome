FROM node:14

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package.json ./
COPY package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the frontend
RUN cd frontend && npm install && npm run build

# Expose the port the app runs on
EXPOSE 3000

# Start the application
CMD ["node", "backend/src/server.js"]