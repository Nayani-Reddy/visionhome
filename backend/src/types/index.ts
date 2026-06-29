export interface Item {
    id: string;
    name: string;
    description: string;
    imageUrl: string;
    category: string;
    price: number;
}

export interface Room {
    id: string;
    type: 'livingRoom' | 'bedroom' | 'kitchen' | 'garden' | 'drawingRoom';
    imageUrl: string;
    items: Item[];
}

export interface UserInput {
    roomType: 'livingRoom' | 'bedroom' | 'kitchen' | 'garden' | 'drawingRoom';
    desiredItems: string[];
    placementCoordinates: { x: number; y: number }[];
}

export interface Recommendation {
    itemId: string;
    itemName: string;
    itemImage: string;
    itemLink: string;
}