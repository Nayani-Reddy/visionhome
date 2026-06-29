import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const UploadDropzone: React.FC<{ onDrop: (acceptedFiles: File[]) => void }> = ({ onDrop }) => {
    const onDropCallback = useCallback((acceptedFiles: File[]) => {
        onDrop(acceptedFiles);
    }, [onDrop]);

    const { getRootProps, getInputProps } = useDropzone({
        onDrop: onDropCallback,
        accept: 'image/*',
    });

    return (
        <div {...getRootProps({ className: 'dropzone' })}>
            <input {...getInputProps()} />
            <p>Drag 'n' drop some files here, or click to select files</p>
        </div>
    );
};

export default UploadDropzone;