import numpy as np
from typing import List, Dict

class FashionRecommender:
    def __init__(self):
        self.occasions = {
            'casual': {
                'male': [
                    {'top': 'T-shirt', 'bottom': 'Jeans', 'shoes': 'Sneakers'},
                    {'top': 'Polo Shirt', 'bottom': 'Chinos', 'shoes': 'Loafers'},
                    {'top': 'Hoodie', 'bottom': 'Joggers', 'shoes': 'Running Shoes'}
                ],
                'female': [
                    {'top': 'T-shirt', 'bottom': 'Jeans', 'shoes': 'Sneakers'},
                    {'top': 'Blouse', 'bottom': 'Skirt', 'shoes': 'Flats'},
                    {'top': 'Sweater', 'bottom': 'Leggings', 'shoes': 'Boots'}
                ]
            },
            'formal': {
                'male': [
                    {'top': 'Suit Jacket', 'bottom': 'Suit Pants', 'shoes': 'Oxford Shoes'},
                    {'top': 'Blazer', 'bottom': 'Dress Pants', 'shoes': 'Derby Shoes'},
                    {'top': 'Tuxedo', 'bottom': 'Tuxedo Pants', 'shoes': 'Patent Leather Shoes'}
                ],
                'female': [
                    {'top': 'Blazer', 'bottom': 'Pencil Skirt', 'shoes': 'Pumps'},
                    {'top': 'Evening Gown', 'bottom': 'None', 'shoes': 'Heels'},
                    {'top': 'Dress Shirt', 'bottom': 'Slacks', 'shoes': 'Flats'}
                ]
            },
            'party': {
                'male': [
                    {'top': 'Button-up Shirt', 'bottom': 'Dark Jeans', 'shoes': 'Chelsea Boots'},
                    {'top': 'Polo Shirt', 'bottom': 'Chinos', 'shoes': 'Sneakers'},
                    {'top': 'Henley', 'bottom': 'Jeans', 'shoes': 'Loafers'}
                ],
                'female': [
                    {'top': 'Crop Top', 'bottom': 'High-waisted Jeans', 'shoes': 'Heels'},
                    {'top': 'Dress', 'bottom': 'None', 'shoes': 'Sandals'},
                    {'top': 'Blouse', 'bottom': 'Skirt', 'shoes': 'Boots'}
                ]
            }
        }
        
    def get_recommendations(self, gender: str, occasion: str, num_recommendations: int = 3) -> List[Dict]:
        """
        Get fashion recommendations based on gender and occasion.
        
        Args:
            gender (str): 'male' or 'female'
            occasion (str): 'casual', 'formal', or 'party'
            num_recommendations (int): Number of outfit recommendations to return
            
        Returns:
            List[Dict]: List of outfit recommendations
        """
        if occasion not in self.occasions:
            raise ValueError(f"Invalid occasion. Must be one of {list(self.occasions.keys())}")
            
        if gender not in self.occasions[occasion]:
            raise ValueError(f"Invalid gender. Must be one of {list(self.occasions[occasion].keys())}")
            
        recommendations = self.occasions[occasion][gender]
        return recommendations[:num_recommendations]
    
    def get_all_occasions(self) -> List[str]:
        """Get list of all available occasions."""
        return list(self.occasions.keys())
    
    def get_outfit_details(self, outfit: Dict) -> str:
        """Convert outfit dictionary to readable string."""
        return f"Top: {outfit['top']}, Bottom: {outfit['bottom']}, Shoes: {outfit['shoes']}" 