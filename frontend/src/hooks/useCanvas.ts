import { useEffect, useRef, useState } from 'react';

const useCanvas = (initialImage: string | null) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const [context, setContext] = useState<CanvasRenderingContext2D | null>(null);
    const [items, setItems] = useState<{ image: string; position: { x: number; y: number } }[]>([]);

    useEffect(() => {
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) {
                setContext(ctx);
            }
        }
    }, [canvasRef]);

    useEffect(() => {
        if (context && initialImage) {
            const img = new Image();
            img.src = initialImage;
            img.onload = () => {
                context.clearRect(0, 0, canvasRef.current!.width, canvasRef.current!.height);
                context.drawImage(img, 0, 0);
                items.forEach(item => {
                    const itemImg = new Image();
                    itemImg.src = item.image;
                    itemImg.onload = () => {
                        context.drawImage(itemImg, item.position.x, item.position.y);
                    };
                });
            };
        }
    }, [context, initialImage, items]);

    const addItem = (image: string, position: { x: number; y: number }) => {
        setItems(prevItems => [...prevItems, { image, position }]);
    };

    const setCanvasRef = (ref: HTMLCanvasElement) => {
        canvasRef.current = ref;
    };

    return { setCanvasRef, addItem };
};

export default useCanvas;