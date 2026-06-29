import { Request, Response } from 'express';
import multer from 'multer';
import path from 'path';

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/'); // Directory to save uploaded images
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`); // Unique filename
  },
});

const upload = multer({ storage });

// Controller to handle image uploads
export const uploadImage = (req: Request, res: Response) => {
  upload.single('image')(req, res, (err) => {
    if (err) {
      return res.status(500).json({ message: 'Image upload failed', error: err });
    }
    return res.status(200).json({ message: 'Image uploaded successfully', file: req.file });
  });
};