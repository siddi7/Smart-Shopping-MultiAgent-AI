"""
Advanced SQLite Database Schema for Smart Shopping Multi-Agent AI System
Supports long-term memory, customer behavior tracking, and real-time analytics
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, 
    ForeignKey, JSON, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import json
from typing import Dict, Any, List, Optional

Base = declarative_base()

class Customer(Base):
    """Enhanced customer model with behavioral analytics"""
    __tablename__ = "customers"
    
    customer_id = Column(String(50), primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    age = Column(Integer)
    gender = Column(String(20))
    location = Column(String(200))
    registration_date = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    # Behavioral Analytics
    total_orders = Column(Integer, default=0)
    total_spent = Column(Float, default=0.0)
    avg_order_value = Column(Float, default=0.0)
    lifetime_value = Column(Float, default=0.0)
    churn_probability = Column(Float, default=0.0)
    satisfaction_score = Column(Float, default=5.0)
    
    # Segmentation
    customer_segment = Column(String(50), default="new_visitor")
    persona_vector = Column(Text)  # JSON string of behavioral embeddings
    preferences = Column(JSON)
    
    # Privacy & Consent
    data_consent = Column(Boolean, default=True)
    marketing_consent = Column(Boolean, default=False)
    
    # Relationships
    interactions = relationship("CustomerInteraction", back_populates="customer")
    orders = relationship("Order", back_populates="customer")
    recommendations = relationship("RecommendationHistory", back_populates="customer")
    
    __table_args__ = (
        Index('idx_customer_segment', 'customer_segment'),
        Index('idx_customer_location', 'location'),
        Index('idx_customer_last_active', 'last_active'),
    )

class Product(Base):
    """Comprehensive product model with multi-modal features"""
    __tablename__ = "products"
    
    product_id = Column(String(50), primary_key=True)
    name = Column(String(500), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    subcategory = Column(String(100))
    brand = Column(String(100))
    
    # Pricing
    price = Column(Float, nullable=False)
    original_price = Column(Float)
    discount_percentage = Column(Float, default=0.0)
    dynamic_price = Column(Float)  # AI-generated dynamic pricing
    
    # Inventory
    stock_quantity = Column(Integer, default=0)
    availability_status = Column(String(20), default="in_stock")
    
    # Product Features
    color = Column(String(50))
    size = Column(String(50))
    weight = Column(Float)
    dimensions = Column(String(100))
    material = Column(String(100))
    
    # Analytics
    popularity_score = Column(Float, default=0.0)
    rating = Column(Float, default=0.0)
    num_reviews = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    purchase_count = Column(Integer, default=0)
    
    # AI Features
    embedding_vector = Column(Text)  # JSON string of product embeddings
    image_embeddings = Column(Text)  # Multi-modal image features
    text_embeddings = Column(Text)   # Text-based features
    
    # Metadata
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    interactions = relationship("CustomerInteraction", back_populates="product")
    order_items = relationship("OrderItem", back_populates="product")
    recommendations = relationship("RecommendationHistory", back_populates="product")
    
    __table_args__ = (
        Index('idx_product_category', 'category'),
        Index('idx_product_brand', 'brand'),
        Index('idx_product_price', 'price'),
        Index('idx_product_popularity', 'popularity_score'),
        CheckConstraint('price >= 0', name='check_positive_price'),
        CheckConstraint('rating >= 0 AND rating <= 5', name='check_rating_range'),
    )

class CustomerInteraction(Base):
    """Detailed customer behavior tracking"""
    __tablename__ = "customer_interactions"
    
    interaction_id = Column(String(50), primary_key=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id"))
    product_id = Column(String(50), ForeignKey("products.product_id"))
    
    # Interaction Details
    interaction_type = Column(String(50))  # view, click, add_to_cart, purchase, review
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String(100))
    page_url = Column(String(500))
    referrer = Column(String(500))
    
    # Behavioral Data
    duration_seconds = Column(Integer)
    scroll_depth = Column(Float)  # Percentage of page scrolled
    click_coordinates = Column(String(100))
    device_type = Column(String(50))
    browser = Column(String(100))
    
    # Context
    search_query = Column(String(500))
    filter_applied = Column(JSON)
    recommendation_source = Column(String(100))
    
    # Sentiment & Engagement
    sentiment_score = Column(Float)
    engagement_score = Column(Float)
    
    # Relationships
    customer = relationship("Customer", back_populates="interactions")
    product = relationship("Product", back_populates="interactions")
    
    __table_args__ = (
        Index('idx_interaction_customer', 'customer_id'),
        Index('idx_interaction_product', 'product_id'),
        Index('idx_interaction_timestamp', 'timestamp'),
        Index('idx_interaction_type', 'interaction_type'),
    )

class Order(Base):
    """Order management with advanced analytics"""
    __tablename__ = "orders"
    
    order_id = Column(String(50), primary_key=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id"))
    
    # Order Details
    order_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="pending")
    total_amount = Column(Float, nullable=False)
    discount_amount = Column(Float, default=0.0)
    tax_amount = Column(Float, default=0.0)
    shipping_cost = Column(Float, default=0.0)
    
    # Delivery
    shipping_address = Column(JSON)
    estimated_delivery = Column(DateTime)
    actual_delivery = Column(DateTime)
    
    # AI Insights
    predicted_satisfaction = Column(Float)
    churn_risk_flag = Column(Boolean, default=False)
    recommendation_influence = Column(Float)  # How much recommendations influenced this order
    
    # Relationships
    customer = relationship("Customer", back_populates="orders")
    order_items = relationship("OrderItem", back_populates="order")
    
    __table_args__ = (
        Index('idx_order_customer', 'customer_id'),
        Index('idx_order_date', 'order_date'),
        Index('idx_order_status', 'status'),
    )

class OrderItem(Base):
    """Individual items within orders"""
    __tablename__ = "order_items"
    
    item_id = Column(String(50), primary_key=True)
    order_id = Column(String(50), ForeignKey("orders.order_id"))
    product_id = Column(String(50), ForeignKey("products.product_id"))
    
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    total_price = Column(Float, nullable=False)
    
    # Relationships
    order = relationship("Order", back_populates="order_items")
    product = relationship("Product", back_populates="order_items")

class RecommendationHistory(Base):
    """Track recommendation performance and explanations"""
    __tablename__ = "recommendation_history"
    
    recommendation_id = Column(String(50), primary_key=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id"))
    product_id = Column(String(50), ForeignKey("products.product_id"))
    
    # Recommendation Context
    timestamp = Column(DateTime, default=datetime.utcnow)
    algorithm_used = Column(String(100))
    confidence_score = Column(Float)
    ranking_position = Column(Integer)
    context = Column(String(200))  # homepage, product_page, cart, etc.
    
    # Performance Tracking
    was_clicked = Column(Boolean, default=False)
    was_purchased = Column(Boolean, default=False)
    click_timestamp = Column(DateTime)
    purchase_timestamp = Column(DateTime)
    
    # Explainable AI
    explanation = Column(JSON)  # Why this product was recommended
    feature_importance = Column(JSON)
    similar_customers = Column(JSON)
    
    # A/B Testing
    experiment_group = Column(String(50))
    variant = Column(String(50))
    
    # Relationships
    customer = relationship("Customer", back_populates="recommendations")
    product = relationship("Product", back_populates="recommendations")
    
    __table_args__ = (
        Index('idx_rec_customer', 'customer_id'),
        Index('idx_rec_product', 'product_id'),
        Index('idx_rec_timestamp', 'timestamp'),
        Index('idx_rec_algorithm', 'algorithm_used'),
    )

class AgentMemory(Base):
    """Long-term memory for multi-agent system"""
    __tablename__ = "agent_memory"
    
    memory_id = Column(String(50), primary_key=True)
    agent_type = Column(String(50), nullable=False)
    agent_instance_id = Column(String(50))
    
    # Memory Content
    memory_type = Column(String(50))  # customer_preference, product_insight, market_trend
    content = Column(JSON)
    embedding_vector = Column(Text)
    
    # Context
    customer_id = Column(String(50))
    product_id = Column(String(50))
    session_id = Column(String(100))
    
    # Metadata
    created_date = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    importance_score = Column(Float, default=0.5)
    
    # Expiry and Cleanup
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    __table_args__ = (
        Index('idx_memory_agent_type', 'agent_type'),
        Index('idx_memory_customer', 'customer_id'),
        Index('idx_memory_importance', 'importance_score'),
        Index('idx_memory_created', 'created_date'),
    )

class RealTimeMetrics(Base):
    """Real-time system metrics and KPIs"""
    __tablename__ = "real_time_metrics"
    
    metric_id = Column(String(50), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # System Metrics
    active_users = Column(Integer, default=0)
    concurrent_sessions = Column(Integer, default=0)
    recommendations_served = Column(Integer, default=0)
    api_response_time = Column(Float, default=0.0)
    
    # Business Metrics
    conversion_rate = Column(Float, default=0.0)
    average_order_value = Column(Float, default=0.0)
    revenue_per_visitor = Column(Float, default=0.0)
    cart_abandonment_rate = Column(Float, default=0.0)
    
    # AI Performance
    recommendation_accuracy = Column(Float, default=0.0)
    model_confidence = Column(Float, default=0.0)
    prediction_latency = Column(Float, default=0.0)
    
    # Agent Coordination
    agent_response_time = Column(JSON)
    agent_agreement_score = Column(Float, default=0.0)
    coordination_efficiency = Column(Float, default=0.0)
    
    __table_args__ = (
        Index('idx_metrics_timestamp', 'timestamp'),
    )

# Database utility functions
class DatabaseManager:
    """Advanced database operations manager"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
        
    def drop_tables(self):
        """Drop all tables (use with caution)"""
        Base.metadata.drop_all(bind=self.engine)
        
    def backup_database(self, backup_path: str):
        """Create database backup"""
        # Implementation for database backup
        pass
        
    def optimize_database(self):
        """Optimize database performance"""
        with self.engine.connect() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")

# Initialize database manager
def get_database_manager(database_url: str = "sqlite:///./smart_shopping.db"):
    return DatabaseManager(database_url)