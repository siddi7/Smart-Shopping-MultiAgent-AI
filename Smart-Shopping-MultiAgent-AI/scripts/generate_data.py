"""
Synthetic Data Generator for Smart Shopping Multi-Agent AI System
Creates realistic e-commerce data for demonstration and testing
"""

import random
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from faker import Faker

from src.database.models import (
    Customer, Product, CustomerInteraction, Order, OrderItem,
    RecommendationHistory, AgentMemory, RealTimeMetrics,
    DatabaseManager, get_database_manager
)
from config.settings import settings

fake = Faker()

class SyntheticDataGenerator:
    """Generate realistic synthetic data for the Smart Shopping system"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.product_categories = settings.PRODUCT_CATEGORIES
        self.customer_segments = settings.CUSTOMER_SEGMENTS
        
        # Brand lists for different categories
        self.brands = {
            "electronics": ["Apple", "Samsung", "Sony", "LG", "HP", "Dell", "Microsoft", "Google"],
            "fashion": ["Nike", "Adidas", "Zara", "H&M", "Gucci", "Prada", "Calvin Klein", "Tommy Hilfiger"],
            "home_garden": ["IKEA", "Home Depot", "Wayfair", "West Elm", "Pottery Barn", "Crate & Barrel"],
            "books": ["Penguin", "Random House", "HarperCollins", "Simon & Schuster", "Macmillan"],
            "sports_outdoors": ["Nike", "Adidas", "Under Armour", "Patagonia", "North Face", "Columbia"],
            "health_beauty": ["L'Oreal", "Maybelline", "Clinique", "Estee Lauder", "Neutrogena", "Olay"],
            "automotive": ["Bosch", "Michelin", "Castrol", "3M", "Chemical Guys", "Meguiar's"],
            "toys_games": ["Lego", "Mattel", "Hasbro", "Fisher-Price", "Nerf", "Monopoly"],
            "grocery": ["Organic Valley", "Whole Foods", "Kirkland", "Great Value", "Simply Organic"],
            "jewelry": ["Tiffany & Co", "Cartier", "Pandora", "Kay Jewelers", "Zales", "Blue Nile"]
        }
    
    def generate_customers(self, num_customers: int = 1000) -> List[Customer]:
        """Generate synthetic customer data"""
        customers = []
        
        for _ in range(num_customers):
            customer_id = str(uuid.uuid4())
            registration_date = fake.date_time_between(start_date='-2y', end_date='now')
            
            # Generate realistic customer attributes
            age = random.randint(18, 75)
            gender = random.choice(['male', 'female', 'other'])
            
            # Generate spending patterns based on age and other factors
            if age < 25:
                avg_spending = random.uniform(20, 100)
                segment = random.choice(['tech_enthusiast', 'price_sensitive', 'fashion_lover'])
            elif age < 40:
                avg_spending = random.uniform(50, 300)
                segment = random.choice(['frequent_buyer', 'tech_enthusiast', 'health_conscious'])
            elif age < 60:
                avg_spending = random.uniform(100, 500)
                segment = random.choice(['luxury_shopper', 'frequent_buyer', 'eco_friendly'])
            else:
                avg_spending = random.uniform(80, 200)
                segment = random.choice(['frequent_buyer', 'health_conscious'])
            
            total_orders = random.randint(0, 50)
            total_spent = total_orders * avg_spending * random.uniform(0.5, 2.0)
            avg_order_value = total_spent / max(total_orders, 1)
            
            # Generate preferences based on segment
            preferences = self._generate_customer_preferences(segment, age, gender)
            
            customer = Customer(
                customer_id=customer_id,
                email=fake.email(),
                first_name=fake.first_name(),
                last_name=fake.last_name(),
                age=age,
                gender=gender,
                location=f"{fake.city()}, {fake.state()}",
                registration_date=registration_date,
                last_active=fake.date_time_between(start_date=registration_date, end_date='now'),
                total_orders=total_orders,
                total_spent=total_spent,
                avg_order_value=avg_order_value,
                lifetime_value=total_spent * random.uniform(1.2, 2.0),  # Projected LTV
                churn_probability=random.uniform(0.0, 0.8),
                satisfaction_score=random.uniform(3.0, 5.0),
                customer_segment=segment,
                preferences=preferences,
                data_consent=random.choice([True, True, True, False]),  # 75% consent rate
                marketing_consent=random.choice([True, False])
            )
            
            customers.append(customer)
        
        return customers
    
    def _generate_customer_preferences(self, segment: str, age: int, gender: str) -> Dict[str, Any]:
        """Generate customer preferences based on segment and demographics"""
        base_preferences = {
            "preferred_categories": [],
            "preferred_brands": [],
            "price_range": {"min": 0, "max": 1000},
            "shopping_frequency": "monthly",
            "preferred_delivery": "standard",
            "communication_channel": "email"
        }
        
        # Segment-based preferences
        if segment == "tech_enthusiast":
            base_preferences["preferred_categories"] = ["electronics", "toys_games"]
            base_preferences["preferred_brands"] = ["Apple", "Samsung", "Google", "Microsoft"]
            base_preferences["price_range"] = {"min": 100, "max": 2000}
        elif segment == "fashion_lover":
            base_preferences["preferred_categories"] = ["fashion", "jewelry", "health_beauty"]
            base_preferences["preferred_brands"] = ["Nike", "Zara", "Gucci", "L'Oreal"]
            base_preferences["price_range"] = {"min": 20, "max": 500}
        elif segment == "price_sensitive":
            base_preferences["preferred_categories"] = ["grocery", "home_garden"]
            base_preferences["price_range"] = {"min": 10, "max": 100}
            base_preferences["shopping_frequency"] = "weekly"
        elif segment == "luxury_shopper":
            base_preferences["preferred_categories"] = ["jewelry", "fashion", "automotive"]
            base_preferences["preferred_brands"] = ["Gucci", "Tiffany & Co", "Cartier"]
            base_preferences["price_range"] = {"min": 200, "max": 5000}
            base_preferences["preferred_delivery"] = "express"
        
        # Age-based adjustments
        if age < 30:
            base_preferences["communication_channel"] = random.choice(["email", "sms", "app_notification"])
            base_preferences["shopping_frequency"] = "weekly"
        elif age > 50:
            base_preferences["communication_channel"] = "email"
            base_preferences["preferred_delivery"] = "standard"
        
        return base_preferences
    
    def generate_products(self, num_products: int = 2000) -> List[Product]:
        """Generate synthetic product data"""
        products = []
        
        for _ in range(num_products):
            product_id = str(uuid.uuid4())
            category = random.choice(self.product_categories)
            brand = random.choice(self.brands.get(category, ["Generic Brand"]))
            
            # Generate category-specific product details
            name, description, price = self._generate_product_details(category, brand)
            
            # Generate realistic metrics
            view_count = random.randint(0, 10000)
            purchase_count = int(view_count * random.uniform(0.01, 0.1))  # 1-10% conversion
            popularity_score = min(view_count / 1000, 1.0)
            
            product = Product(
                product_id=product_id,
                name=name,
                description=description,
                category=category,
                subcategory=self._get_subcategory(category),
                brand=brand,
                price=price,
                original_price=price * random.uniform(1.0, 1.3),  # Some discounted items
                discount_percentage=random.uniform(0, 30) if random.random() < 0.3 else 0,
                stock_quantity=random.randint(0, 500),
                availability_status="in_stock" if random.random() < 0.8 else "out_of_stock",
                color=random.choice(["Black", "White", "Blue", "Red", "Green", "Gray", "Silver"]) if random.random() < 0.6 else None,
                size=random.choice(["XS", "S", "M", "L", "XL", "XXL"]) if category == "fashion" else None,
                weight=random.uniform(0.1, 10.0),
                material=self._get_material(category),
                popularity_score=popularity_score,
                rating=random.uniform(3.0, 5.0),
                num_reviews=random.randint(0, 1000),
                view_count=view_count,
                purchase_count=purchase_count,
                created_date=fake.date_time_between(start_date='-1y', end_date='now'),
                is_active=random.choice([True, True, True, False])  # 75% active
            )
            
            products.append(product)
        
        return products
    
    def _generate_product_details(self, category: str, brand: str) -> tuple:
        """Generate product name, description, and price based on category"""
        if category == "electronics":
            products = [
                ("Smartphone", "High-performance smartphone with advanced camera", (200, 1200)),
                ("Laptop", "Powerful laptop for work and gaming", (500, 2500)),
                ("Headphones", "Premium wireless headphones with noise cancellation", (50, 400)),
                ("Tablet", "Versatile tablet for productivity and entertainment", (150, 800)),
                ("Smart Watch", "Feature-rich smartwatch with health monitoring", (100, 600))
            ]
        elif category == "fashion":
            products = [
                ("T-Shirt", "Comfortable cotton t-shirt with modern fit", (15, 80)),
                ("Jeans", "Premium denim jeans with contemporary styling", (40, 200)),
                ("Sneakers", "Athletic sneakers with superior comfort and style", (60, 250)),
                ("Dress", "Elegant dress perfect for any occasion", (30, 300)),
                ("Jacket", "Stylish jacket for layering and warmth", (50, 400))
            ]
        elif category == "home_garden":
            products = [
                ("Coffee Maker", "Programmable coffee maker with multiple settings", (30, 200)),
                ("Garden Tools Set", "Complete set of essential gardening tools", (25, 150)),
                ("Throw Pillow", "Decorative throw pillow for home comfort", (10, 60)),
                ("Plant Pot", "Ceramic plant pot with drainage system", (15, 80)),
                ("Bookshelf", "Modern bookshelf with multiple storage compartments", (80, 400))
            ]
        else:
            # Generic products for other categories
            products = [
                (f"{category.title()} Item", f"High-quality {category} product from {brand}", (20, 300))
            ]
        
        product_info = random.choice(products)
        name = f"{brand} {product_info[0]}"
        description = f"{product_info[1]}. Made by {brand} with attention to quality and detail."
        price = random.uniform(product_info[2][0], product_info[2][1])
        
        return name, description, round(price, 2)
    
    def _get_subcategory(self, category: str) -> str:
        """Get subcategory based on main category"""
        subcategories = {
            "electronics": ["smartphones", "laptops", "tablets", "accessories", "gaming"],
            "fashion": ["clothing", "shoes", "accessories", "bags", "jewelry"],
            "home_garden": ["furniture", "decor", "kitchen", "garden", "tools"],
            "books": ["fiction", "non-fiction", "educational", "children", "comics"],
            "sports_outdoors": ["fitness", "outdoor", "team_sports", "water_sports", "cycling"],
            "health_beauty": ["skincare", "makeup", "haircare", "supplements", "personal_care"],
            "automotive": ["parts", "accessories", "tools", "care_products", "electronics"],
            "toys_games": ["action_figures", "board_games", "educational", "outdoor_toys", "puzzles"],
            "grocery": ["organic", "snacks", "beverages", "pantry", "frozen"],
            "jewelry": ["rings", "necklaces", "earrings", "bracelets", "watches"]
        }
        
        return random.choice(subcategories.get(category, ["general"]))
    
    def _get_material(self, category: str) -> str:
        """Get material based on category"""
        materials = {
            "electronics": ["Plastic", "Metal", "Glass", "Silicon"],
            "fashion": ["Cotton", "Polyester", "Denim", "Leather", "Wool"],
            "home_garden": ["Wood", "Metal", "Ceramic", "Plastic", "Glass"],
            "jewelry": ["Gold", "Silver", "Platinum", "Stainless Steel"]
        }
        
        return random.choice(materials.get(category, ["Mixed Materials"]))
    
    def generate_interactions(self, customers: List[Customer], products: List[Product], 
                            num_interactions: int = 10000) -> List[CustomerInteraction]:
        """Generate realistic customer interactions"""
        interactions = []
        interaction_types = ["view", "click", "add_to_cart", "purchase", "review"]
        
        for _ in range(num_interactions):
            customer = random.choice(customers)
            product = random.choice(products)
            
            # Generate realistic interaction patterns based on customer segment
            interaction_weights = self._get_interaction_weights(customer.customer_segment)
            interaction_type = random.choices(interaction_types, weights=interaction_weights)[0]
            
            interaction = CustomerInteraction(
                interaction_id=str(uuid.uuid4()),
                customer_id=customer.customer_id,
                product_id=product.product_id,
                interaction_type=interaction_type,
                timestamp=fake.date_time_between(start_date='-6m', end_date='now'),
                session_id=str(uuid.uuid4()),
                page_url=f"/products/{product.product_id}",
                referrer=random.choice(["/", "/search", "/category", "/recommendations"]),
                duration_seconds=random.randint(5, 600),
                scroll_depth=random.uniform(0.1, 1.0),
                device_type=random.choice(["desktop", "mobile", "tablet"]),
                browser=random.choice(["Chrome", "Firefox", "Safari", "Edge"]),
                search_query=self._generate_search_query(product) if random.random() < 0.3 else None,
                sentiment_score=random.uniform(-1.0, 1.0),
                engagement_score=random.uniform(0.0, 1.0)
            )
            
            interactions.append(interaction)
        
        return interactions
    
    def _get_interaction_weights(self, segment: str) -> List[float]:
        """Get interaction type weights based on customer segment"""
        base_weights = [50, 20, 15, 10, 5]  # view, click, add_to_cart, purchase, review
        
        if segment == "frequent_buyer":
            return [40, 25, 20, 15, 10]  # More likely to purchase
        elif segment == "price_sensitive":
            return [60, 25, 10, 3, 2]   # Mostly browsing
        elif segment == "luxury_shopper":
            return [30, 20, 25, 20, 5]  # High conversion rate
        else:
            return base_weights
    
    def _generate_search_query(self, product: Product) -> str:
        """Generate realistic search query"""
        queries = [
            product.name.split()[0],  # Brand or first word
            product.category,
            f"{product.category} {product.brand}",
            f"best {product.category}",
            f"cheap {product.category}",
            f"{product.brand} products"
        ]
        return random.choice(queries)
    
    def generate_orders(self, customers: List[Customer], products: List[Product], 
                       interactions: List[CustomerInteraction]) -> List[Order]:
        """Generate order data based on purchase interactions"""
        orders = []
        order_items = []
        
        # Find purchase interactions
        purchase_interactions = [i for i in interactions if i.interaction_type == "purchase"]
        
        # Group by customer and session to create realistic orders
        customer_sessions = {}
        for interaction in purchase_interactions:
            key = f"{interaction.customer_id}_{interaction.session_id}"
            if key not in customer_sessions:
                customer_sessions[key] = []
            customer_sessions[key].append(interaction)
        
        for session_purchases in customer_sessions.values():
            if not session_purchases:
                continue
                
            customer_id = session_purchases[0].customer_id
            order_id = str(uuid.uuid4())
            order_date = session_purchases[0].timestamp
            
            # Calculate order totals
            total_amount = 0
            for interaction in session_purchases:
                product = next((p for p in products if p.product_id == interaction.product_id), None)
                if product:
                    quantity = random.randint(1, 3)
                    unit_price = product.price
                    item_total = unit_price * quantity
                    total_amount += item_total
                    
                    # Create order item
                    order_item = OrderItem(
                        item_id=str(uuid.uuid4()),
                        order_id=order_id,
                        product_id=product.product_id,
                        quantity=quantity,
                        unit_price=unit_price,
                        total_price=item_total
                    )
                    order_items.append(order_item)
            
            # Calculate additional costs
            discount_amount = total_amount * random.uniform(0, 0.2) if random.random() < 0.3 else 0
            tax_amount = (total_amount - discount_amount) * 0.08  # 8% tax
            shipping_cost = 0 if total_amount > 50 else random.uniform(5, 15)
            
            order = Order(
                order_id=order_id,
                customer_id=customer_id,
                order_date=order_date,
                status=random.choice(["completed", "pending", "shipped", "delivered"]),
                total_amount=total_amount + tax_amount + shipping_cost - discount_amount,
                discount_amount=discount_amount,
                tax_amount=tax_amount,
                shipping_cost=shipping_cost,
                shipping_address={
                    "street": fake.street_address(),
                    "city": fake.city(),
                    "state": fake.state(),
                    "zip": fake.zipcode()
                },
                estimated_delivery=order_date + timedelta(days=random.randint(3, 10)),
                predicted_satisfaction=random.uniform(3.0, 5.0),
                churn_risk_flag=random.choice([True, False]),
                recommendation_influence=random.uniform(0.0, 0.8)
            )
            
            orders.append(order)
        
        return orders, order_items
    
    def save_all_data(self, customers: List[Customer], products: List[Product], 
                     interactions: List[CustomerInteraction], orders: List[Order], 
                     order_items: List[OrderItem]):
        """Save all generated data to database"""
        session = self.db_manager.get_session()
        
        try:
            # Add all data to session
            session.add_all(customers)
            session.add_all(products)
            session.add_all(interactions)
            session.add_all(orders)
            session.add_all(order_items)
            
            # Commit all changes
            session.commit()
            
            print(f"Successfully saved:")
            print(f"  {len(customers)} customers")
            print(f"  {len(products)} products")
            print(f"  {len(interactions)} interactions")
            print(f"  {len(orders)} orders")
            print(f"  {len(order_items)} order items")
            
        except Exception as e:
            session.rollback()
            print(f"Error saving data: {str(e)}")
            raise
        finally:
            session.close()

def main():
    """Generate and save synthetic data"""
    print("Generating synthetic data for Smart Shopping Multi-Agent AI System...")
    
    # Initialize database
    db_manager = get_database_manager()
    db_manager.create_tables()
    
    # Initialize data generator
    generator = SyntheticDataGenerator(db_manager)
    
    # Generate data
    print("Generating customers...")
    customers = generator.generate_customers(1000)
    
    print("Generating products...")
    products = generator.generate_products(2000)
    
    print("Generating interactions...")
    interactions = generator.generate_interactions(customers, products, 10000)
    
    print("Generating orders...")
    orders, order_items = generator.generate_orders(customers, products, interactions)
    
    # Save to database
    print("Saving data to database...")
    generator.save_all_data(customers, products, interactions, orders, order_items)
    
    print("Synthetic data generation completed successfully!")

if __name__ == "__main__":
    main()