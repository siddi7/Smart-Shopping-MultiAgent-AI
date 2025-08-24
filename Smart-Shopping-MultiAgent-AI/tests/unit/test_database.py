"""
Unit Tests for Database Models and Manager
Tests database operations, models, and data persistence
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
import sqlite3
from datetime import datetime, timedelta

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.models import (
    DatabaseManager, Customer, Product, CustomerInteraction, 
    RecommendationHistory, AgentMemory
)
from src.utils.config import Settings

class TestDatabaseModels:
    """Test cases for database models"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            tmp_db_path = tmp.name
        
        yield tmp_db_path
        
        # Cleanup
        if os.path.exists(tmp_db_path):
            os.unlink(tmp_db_path)
    
    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create database manager with temporary database"""
        return DatabaseManager(f"sqlite:///{temp_db_path}")
    
    def test_database_initialization(self, db_manager):
        """Test database initialization and table creation"""
        db_manager.initialize_database()
        
        # Check that tables were created
        with sqlite3.connect(db_manager.database_url.replace("sqlite:///", "")) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['customers', 'products', 'customer_interactions', 
                             'recommendation_history', 'agent_memory']
            for table in expected_tables:
                assert table in tables
    
    def test_customer_model_creation(self, db_manager):
        """Test customer model creation and validation"""
        db_manager.initialize_database()
        
        customer_data = {
            "customer_id": "cust_001",
            "name": "John Doe",
            "email": "john@example.com",
            "segment": "tech_enthusiast",
            "preferences": {"category": "electronics", "price_range": "mid"},
            "behavior_data": {"avg_session_time": 300, "pages_per_session": 5}
        }
        
        customer = Customer(**customer_data)
        
        assert customer.customer_id == "cust_001"
        assert customer.name == "John Doe"
        assert customer.email == "john@example.com"
        assert customer.segment == "tech_enthusiast"
        assert isinstance(customer.preferences, dict)
        assert isinstance(customer.behavior_data, dict)
    
    def test_product_model_creation(self, db_manager):
        """Test product model creation and validation"""
        db_manager.initialize_database()
        
        product_data = {
            "product_id": "prod_001",
            "name": "Smart Laptop",
            "category": "electronics",
            "price": 999.99,
            "description": "High-performance laptop",
            "features": ["wireless", "portable", "high-res"],
            "specifications": {"CPU": "Intel i7", "RAM": "16GB"},
            "content_data": {"images": ["img1.jpg"], "videos": ["demo.mp4"]}
        }
        
        product = Product(**product_data)
        
        assert product.product_id == "prod_001"
        assert product.name == "Smart Laptop"
        assert product.category == "electronics"
        assert product.price == 999.99
        assert isinstance(product.features, list)
        assert isinstance(product.specifications, dict)
    
    def test_customer_interaction_model(self, db_manager):
        """Test customer interaction model"""
        db_manager.initialize_database()
        
        interaction_data = {
            "interaction_id": "int_001",
            "customer_id": "cust_001",
            "product_id": "prod_001",
            "interaction_type": "view",
            "session_data": {"duration": 120, "page_views": 3},
            "context": {"page_type": "product_page", "referrer": "search"}
        }
        
        interaction = CustomerInteraction(**interaction_data)
        
        assert interaction.interaction_id == "int_001"
        assert interaction.customer_id == "cust_001"
        assert interaction.product_id == "prod_001"
        assert interaction.interaction_type == "view"
        assert isinstance(interaction.session_data, dict)
    
    def test_recommendation_history_model(self, db_manager):
        """Test recommendation history model"""
        db_manager.initialize_database()
        
        rec_data = {
            "recommendation_id": "rec_001",
            "customer_id": "cust_001",
            "product_id": "prod_001",
            "algorithm_variant": "hybrid",
            "confidence_score": 0.85,
            "explanation": {
                "primary_reason": "Similar products viewed",
                "factors": ["user_preference", "high_rating"]
            },
            "feedback": {"clicked": True, "purchased": False}
        }
        
        recommendation = RecommendationHistory(**rec_data)
        
        assert recommendation.recommendation_id == "rec_001"
        assert recommendation.customer_id == "cust_001"
        assert recommendation.algorithm_variant == "hybrid"
        assert recommendation.confidence_score == 0.85
        assert isinstance(recommendation.explanation, dict)
    
    def test_agent_memory_model(self, db_manager):
        """Test agent memory model"""
        db_manager.initialize_database()
        
        memory_data = {
            "memory_id": "mem_001",
            "agent_id": "customer_agent_001",
            "agent_type": "customer_agent",
            "memory_type": "long_term",
            "memory_data": {"customer_preferences": {"electronics": 0.9}},
            "context": {"session_id": "sess_001", "interaction_count": 5}
        }
        
        memory = AgentMemory(**memory_data)
        
        assert memory.memory_id == "mem_001"
        assert memory.agent_id == "customer_agent_001"
        assert memory.agent_type == "customer_agent"
        assert memory.memory_type == "long_term"
        assert isinstance(memory.memory_data, dict)

class TestDatabaseManager:
    """Test cases for database manager operations"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            tmp_db_path = tmp.name
        
        yield tmp_db_path
        
        # Cleanup
        if os.path.exists(tmp_db_path):
            os.unlink(tmp_db_path)
    
    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create database manager with temporary database"""
        manager = DatabaseManager(f"sqlite:///{temp_db_path}")
        manager.initialize_database()
        return manager
    
    def test_database_connection(self, db_manager):
        """Test database connection"""
        session = db_manager.get_session()
        assert session is not None
        session.close()
    
    def test_create_customer(self, db_manager):
        """Test creating a customer record"""
        customer_data = {
            "customer_id": "cust_001",
            "name": "Jane Smith",
            "email": "jane@example.com",
            "segment": "price_sensitive",
            "preferences": {"budget": "low"},
            "behavior_data": {"last_login": "2024-01-01"}
        }
        
        result = db_manager.create_customer(customer_data)
        assert result is True
        
        # Verify customer was created
        session = db_manager.get_session()
        customer = session.query(Customer).filter(
            Customer.customer_id == "cust_001"
        ).first()
        assert customer is not None
        assert customer.name == "Jane Smith"
        session.close()
    
    def test_create_product(self, db_manager):
        """Test creating a product record"""
        product_data = {
            "product_id": "prod_001",
            "name": "Wireless Mouse",
            "category": "electronics",
            "price": 29.99,
            "description": "Ergonomic wireless mouse",
            "features": ["wireless", "ergonomic"],
            "specifications": {"connectivity": "Bluetooth"},
            "content_data": {"images": ["mouse.jpg"]}
        }
        
        result = db_manager.create_product(product_data)
        assert result is True
        
        # Verify product was created
        session = db_manager.get_session()
        product = session.query(Product).filter(
            Product.product_id == "prod_001"
        ).first()
        assert product is not None
        assert product.name == "Wireless Mouse"
        session.close()
    
    def test_record_interaction(self, db_manager):
        """Test recording customer interaction"""
        # First create customer and product
        customer_data = {
            "customer_id": "cust_001",
            "name": "Test User",
            "email": "test@example.com"
        }
        product_data = {
            "product_id": "prod_001",
            "name": "Test Product",
            "category": "test",
            "price": 99.99
        }
        
        db_manager.create_customer(customer_data)
        db_manager.create_product(product_data)
        
        # Record interaction
        interaction_data = {
            "interaction_id": "int_001",
            "customer_id": "cust_001",
            "product_id": "prod_001",
            "interaction_type": "view",
            "session_data": {"duration": 60},
            "context": {"source": "homepage"}
        }
        
        result = db_manager.record_interaction(interaction_data)
        assert result is True
        
        # Verify interaction was recorded
        session = db_manager.get_session()
        interaction = session.query(CustomerInteraction).filter(
            CustomerInteraction.interaction_id == "int_001"
        ).first()
        assert interaction is not None
        assert interaction.interaction_type == "view"
        session.close()
    
    def test_save_recommendation(self, db_manager):
        """Test saving recommendation history"""
        # Create customer first
        customer_data = {
            "customer_id": "cust_001",
            "name": "Test User",
            "email": "test@example.com"
        }
        db_manager.create_customer(customer_data)
        
        rec_data = {
            "recommendation_id": "rec_001",
            "customer_id": "cust_001",
            "product_id": "prod_001",
            "algorithm_variant": "collaborative",
            "confidence_score": 0.78,
            "explanation": {"reason": "Similar users liked"},
            "feedback": {}
        }
        
        result = db_manager.save_recommendation(rec_data)
        assert result is True
        
        # Verify recommendation was saved
        session = db_manager.get_session()
        recommendation = session.query(RecommendationHistory).filter(
            RecommendationHistory.recommendation_id == "rec_001"
        ).first()
        assert recommendation is not None
        assert recommendation.algorithm_variant == "collaborative"
        session.close()
    
    def test_store_agent_memory(self, db_manager):
        """Test storing agent memory"""
        memory_data = {
            "memory_id": "mem_001",
            "agent_id": "test_agent",
            "agent_type": "customer_agent",
            "memory_type": "short_term",
            "memory_data": {"test_key": "test_value"},
            "context": {"session": "test_session"}
        }
        
        result = db_manager.store_agent_memory(memory_data)
        assert result is True
        
        # Verify memory was stored
        session = db_manager.get_session()
        memory = session.query(AgentMemory).filter(
            AgentMemory.memory_id == "mem_001"
        ).first()
        assert memory is not None
        assert memory.agent_id == "test_agent"
        session.close()
    
    def test_get_customer_data(self, db_manager):
        """Test retrieving customer data"""
        # Create customer first
        customer_data = {
            "customer_id": "cust_001",
            "name": "Test User",
            "email": "test@example.com",
            "segment": "tech_enthusiast",
            "preferences": {"category": "electronics"}
        }
        db_manager.create_customer(customer_data)
        
        # Retrieve customer data
        result = db_manager.get_customer_data("cust_001")
        assert result is not None
        assert result["customer_id"] == "cust_001"
        assert result["name"] == "Test User"
        assert result["segment"] == "tech_enthusiast"
    
    def test_get_product_data(self, db_manager):
        """Test retrieving product data"""
        # Create product first
        product_data = {
            "product_id": "prod_001",
            "name": "Test Product",
            "category": "electronics",
            "price": 199.99,
            "features": ["feature1", "feature2"]
        }
        db_manager.create_product(product_data)
        
        # Retrieve product data
        result = db_manager.get_product_data("prod_001")
        assert result is not None
        assert result["product_id"] == "prod_001"
        assert result["name"] == "Test Product"
        assert result["price"] == 199.99
    
    def test_get_customer_interactions(self, db_manager):
        """Test retrieving customer interactions"""
        # Create customer and record interaction
        customer_data = {"customer_id": "cust_001", "name": "Test User", "email": "test@example.com"}
        db_manager.create_customer(customer_data)
        
        interaction_data = {
            "interaction_id": "int_001",
            "customer_id": "cust_001",
            "product_id": "prod_001",
            "interaction_type": "view",
            "session_data": {},
            "context": {}
        }
        db_manager.record_interaction(interaction_data)
        
        # Retrieve interactions
        interactions = db_manager.get_customer_interactions("cust_001")
        assert len(interactions) == 1
        assert interactions[0]["interaction_id"] == "int_001"
        assert interactions[0]["interaction_type"] == "view"
    
    def test_get_recommendation_history(self, db_manager):
        """Test retrieving recommendation history"""
        # Create customer and save recommendation
        customer_data = {"customer_id": "cust_001", "name": "Test User", "email": "test@example.com"}
        db_manager.create_customer(customer_data)
        
        rec_data = {
            "recommendation_id": "rec_001",
            "customer_id": "cust_001",
            "product_id": "prod_001",
            "algorithm_variant": "hybrid",
            "confidence_score": 0.9,
            "explanation": {},
            "feedback": {}
        }
        db_manager.save_recommendation(rec_data)
        
        # Retrieve recommendation history
        history = db_manager.get_recommendation_history("cust_001")
        assert len(history) == 1
        assert history[0]["recommendation_id"] == "rec_001"
        assert history[0]["algorithm_variant"] == "hybrid"
    
    def test_execute_query(self, db_manager):
        """Test executing custom queries"""
        # Create some test data
        customer_data = {"customer_id": "cust_001", "name": "Test User", "email": "test@example.com"}
        db_manager.create_customer(customer_data)
        
        # Execute custom query
        query = "SELECT COUNT(*) as customer_count FROM customers"
        result = db_manager.execute_query(query)
        
        assert len(result) == 1
        assert result[0]["customer_count"] == 1
    
    def test_update_customer_segment(self, db_manager):
        """Test updating customer segment"""
        # Create customer
        customer_data = {
            "customer_id": "cust_001",
            "name": "Test User",
            "email": "test@example.com",
            "segment": "new_user"
        }
        db_manager.create_customer(customer_data)
        
        # Update segment
        result = db_manager.update_customer_segment("cust_001", "loyal_customer")
        assert result is True
        
        # Verify update
        customer_data = db_manager.get_customer_data("cust_001")
        assert customer_data["segment"] == "loyal_customer"
    
    def test_error_handling(self, db_manager):
        """Test error handling in database operations"""
        # Test with invalid customer ID
        result = db_manager.get_customer_data("nonexistent_customer")
        assert result is None
        
        # Test with invalid product ID
        result = db_manager.get_product_data("nonexistent_product")
        assert result is None
        
        # Test duplicate customer creation
        customer_data = {"customer_id": "cust_001", "name": "Test User", "email": "test@example.com"}
        result1 = db_manager.create_customer(customer_data)
        result2 = db_manager.create_customer(customer_data)  # Duplicate
        
        assert result1 is True
        assert result2 is False  # Should handle duplicate gracefully
    
    def test_transaction_handling(self, db_manager):
        """Test transaction handling and rollback"""
        # This test would check if transactions are properly handled
        # For now, we'll test basic session management
        session = db_manager.get_session()
        assert session is not None
        
        # Test that sessions can be closed without error
        session.close()
        
        # Test getting new session after close
        new_session = db_manager.get_session()
        assert new_session is not None
        new_session.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])