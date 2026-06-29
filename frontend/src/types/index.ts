export interface FurnitureItem {
    id: string;
    name: string;
    imageUrl: string;
    dimensions: {
        width: number;
        height: number;
        depth: number;
    };
    category: string;
}

export interface Room {
    id: string;
    type: 'livingRoom' | 'bedroom' | 'kitchen' | 'garden' | 'drawingRoom';
    imageUrl: string;
    furniture: FurnitureItem[];
}

export interface Recommendation {
    itemId: string;
    itemName: string;
    itemImage: string;
    itemLink: string;
}

export interface UserInput {
    roomType: 'livingRoom' | 'bedroom' | 'kitchen' | 'garden' | 'drawingRoom';
    desiredItems: string[];
    placementCoordinates: {
        x: number;
        y: number;
    };
}