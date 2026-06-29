export interface Item {
    id: string;
    name: string;
    description: string;
    imageUrl: string;
    category: string;
    dimensions: {
        width: number;
        height: number;
        depth: number;
    };
    price: number;
    available: boolean;
}

export interface Recommendation {
    itemId: string;
    recommendedItems: Item[];
}