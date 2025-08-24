"""
Pytest Configuration and Shared Fixtures
Provides common test configuration and fixtures for the entire test suite
"""

import pytest
import asyncio
import tempfile
import os
import sqlite3
from unittest.mock import Mock, patch
from typing import Dict, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.models import DatabaseManager
from src.agents.customer_agent import CustomerAgent
from src.agents.product_agent import ProductAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.coordinator import AgentCoordinator
from src.utils.config import Settings

# Configure pytest for async testing
pytest_plugins = ("pytest_asyncio",)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_database():
    """Create a temporary database for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
        tmp_db_path = tmp.name
    
    yield tmp_db_path
    
    # Cleanup
    if os.path.exists(tmp_db_path):
        os.unlink(tmp_db_path)

@pytest.fixture
def test_settings():
    """Test configuration settings"""
    return Settings(
        DATABASE_URL="sqlite:///test.db",
        ENABLE_PERFORMANCE_MONITORING=True,
        ENABLE_EXPLAINABLE_AI=True,
        ENABLE_FEDERATED_LEARNING=False,  # Disable for testing
        ENABLE_CACHING=True,
        CACHE_TTL=300,
        API_RATE_LIMIT=1000,
        LOG_LEVEL="DEBUG"
    )

@pytest.fixture
def mock_database_manager():
    """Mock database manager for testing"""
    mock_db = Mock(spec=DatabaseManager)
    mock_session = Mock()
    mock_db.get_session.return_value = mock_session
    
    # Mock common database responses
    mock_db.get_customer_data.return_value = {
        "customer_id": "test_customer_001",
        "name": "Test Customer",
        "email": "test@example.com",
        "segment": "tech_enthusiast",
        "preferences": {"category": "electronics", "price_range": "mid"},
        "behavior_data": {"avg_session_time": 300, "pages_per_session": 5}
    }
    
    mock_db.get_product_data.return_value = {
        "product_id": "test_product_001",
        "name": "Smart Laptop",
        "category": "electronics",
        "price": 999.99,
        "description": "High-performance laptop for professionals",
        "features": ["wireless", "portable", "high-performance"],
        "specifications": {"CPU": "Intel i7", "RAM": "16GB", "Storage": "1TB SSD"}
    }
    
    mock_db.get_customer_interactions.return_value = [
        {
            "interaction_id": "int_001",
            "customer_id": "test_customer_001",
            "product_id": "test_product_001",
            "interaction_type": "view",
            "timestamp": "2024-01-01T10:00:00Z",
            "session_data": {"duration": 120, "page_views": 3},
            "context": {"page_type": "product_page", "referrer": "search"}
        }
    ]
    
    mock_db.get_recommendation_history.return_value = [
        {
            "recommendation_id": "rec_001",
            "customer_id": "test_customer_001",
            "product_id": "test_product_001",
            "algorithm_variant": "hybrid",
            "confidence_score": 0.85,
            "timestamp": "2024-01-01T10:00:00Z",
            "explanation": {
                "primary_reason": "Based on your browsing history",
                "factors": ["similar_products_viewed", "high_rating", "price_match"]
            },
            "feedback": {"clicked": True, "purchased": False}
        }
    ]
    
    # Mock successful operations
    mock_db.create_customer.return_value = True
    mock_db.create_product.return_value = True
    mock_db.record_interaction.return_value = True
    mock_db.save_recommendation.return_value = True
    mock_db.store_agent_memory.return_value = True
    mock_db.update_customer_segment.return_value = True
    
    return mock_db

@pytest.fixture
def real_database_manager(temp_database):
    """Real database manager with temporary database for integration tests"""
    db_url = f"sqlite:///{temp_database}"
    db_manager = DatabaseManager(db_url)
    db_manager.initialize_database()
    return db_manager

@pytest.fixture
def customer_agent(mock_database_manager, test_settings):
    """Customer agent instance for testing"""
    return CustomerAgent(
        agent_id="test_customer_agent",
        db_manager=mock_database_manager,
        config={"settings": test_settings}
    )

@pytest.fixture
def product_agent(mock_database_manager, test_settings):
    """Product agent instance for testing"""
    return ProductAgent(
        agent_id="test_product_agent",
        db_manager=mock_database_manager,
        config={"settings": test_settings}
    )

@pytest.fixture
def recommendation_agent(mock_database_manager, test_settings):
    """Recommendation agent instance for testing"""
    return RecommendationAgent(
        agent_id="test_recommendation_agent",
        db_manager=mock_database_manager,
        config={"settings": test_settings}
    )

@pytest.fixture
def agent_coordinator(mock_database_manager, customer_agent, product_agent, recommendation_agent):
    """Agent coordinator with all agents for testing"""
    coordinator = AgentCoordinator(mock_database_manager)
    coordinator.agents = {
        "customer_agent": customer_agent,
        "product_agent": product_agent,
        "recommendation_agent": recommendation_agent
    }
    return coordinator

@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing"""
    return {
        "customer_id": "test_customer_001",
        "name": "John Doe",
        "email": "john.doe@example.com",
        "segment": "tech_enthusiast",
        "preferences": {
            "categories": ["electronics", "books"],
            "price_range": "mid",
            "brands": ["Apple", "Samsung", "Sony"]
        },
        "behavior_data": {
            "avg_session_time": 450,
            "pages_per_session": 8,
            "last_login": "2024-01-01T10:00:00Z",
            "total_orders": 15,
            "total_spent": 2500.00
        }
    }

@pytest.fixture
def sample_product_data():
    """Sample product data for testing"""
    return {
        "product_id": "test_product_001",
        "name": "Smart Laptop Pro",
        "category": "electronics",
        "subcategory": "laptops",
        "price": 1299.99,
        "brand": "TechBrand",
        "description": "Professional-grade laptop with advanced AI capabilities",
        "features": ["AI-powered", "long-battery", "lightweight", "high-performance"],
        "specifications": {
            "CPU": "Intel i7-12700H",
            "RAM": "32GB DDR5",
            "Storage": "1TB NVMe SSD",
            "Display": "15.6\" 4K OLED",
            "Weight": "1.8kg",
            "Battery": "12 hours"
        },
        "content_data": {
            "images": ["laptop_front.jpg", "laptop_side.jpg", "laptop_open.jpg"],
            "videos": ["product_demo.mp4", "unboxing.mp4"],
            "reviews_summary": {
                "average_rating": 4.7,
                "total_reviews": 1250,
                "positive_keywords": ["fast", "reliable", "excellent display"]
            }
        }
    }

@pytest.fixture
def sample_interaction_data():
    """Sample interaction data for testing"""
    return {
        "interaction_id": "test_interaction_001",
        "customer_id": "test_customer_001",
        "product_id": "test_product_001",
        "interaction_type": "view",
        "timestamp": "2024-01-01T10:00:00Z",
        "session_data": {
            "session_id": "sess_12345",
            "duration": 180,
            "page_views": 5,
            "scroll_depth": 0.8,
            "time_on_product_page": 120
        },
        "context": {
            "page_type": "product_page",
            "referrer": "search_results",
            "search_query": "high performance laptop",
            "device_type": "desktop",
            "user_agent": "Mozilla/5.0..."
        }
    }

@pytest.fixture
def sample_recommendation_request():
    """Sample recommendation request for testing"""
    return {
        "customer_id": "test_customer_001",
        "context": {
            "page_type": "homepage",
            "time_of_day": "afternoon",
            "device_type": "mobile",
            "location": "US"
        },
        "num_recommendations": 10,
        "include_explanations": True,
        "algorithm_preference": "hybrid",
        "filters": {
            "max_price": 1500.00,
            "categories": ["electronics"],
            "exclude_viewed": True
        }
    }

@pytest.fixture
def performance_test_data():
    """Performance test data generator"""
    def generate_test_data(count=100):
        customers = [
            {
                "customer_id": f"perf_customer_{i}",
                "name": f"Test Customer {i}",
                "email": f"customer{i}@test.com",
                "segment": ["tech_enthusiast", "price_sensitive", "luxury_shopper"][i % 3]
            }
            for i in range(count)
        ]
        
        products = [
            {
                "product_id": f"perf_product_{i}",
                "name": f"Test Product {i}",
                "category": ["electronics", "books", "clothing"][i % 3],
                "price": 99.99 + (i * 10)
            }
            for i in range(count)
        ]
        
        interactions = [
            {
                "interaction_id": f"perf_interaction_{i}",
                "customer_id": f"perf_customer_{i % count}",
                "product_id": f"perf_product_{i % count}",
                "interaction_type": ["view", "click", "purchase"][i % 3]
            }
            for i in range(count * 2)
        ]
        
        return {
            "customers": customers,
            "products": products,
            "interactions": interactions
        }
    
    return generate_test_data

# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "database: Database tests")
    config.addinivalue_line("markers", "agent: Agent-specific tests")

# Test collection and reporting hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Add markers based on test name patterns
        if "api" in item.name.lower():
            item.add_marker(pytest.mark.api)
        if "database" in item.name.lower() or "db" in item.name.lower():
            item.add_marker(pytest.mark.database)
        if any(agent in item.name.lower() for agent in ["customer", "product", "recommendation", "agent"]):
            item.add_marker(pytest.mark.agent)

# Utility functions for tests
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def assert_response_structure(response_data, expected_keys):
        """Assert that response has expected structure"""
        assert isinstance(response_data, dict), "Response should be a dictionary"
        for key in expected_keys:
            assert key in response_data, f"Expected key '{key}' not found in response"
    
    @staticmethod
    def assert_agent_metrics(agent, min_requests=0):
        """Assert agent performance metrics are valid"""
        metrics = agent.get_performance_metrics()
        assert "total_requests" in metrics
        assert "success_rate" in metrics
        assert "average_response_time" in metrics
        assert metrics["total_requests"] >= min_requests
        assert 0 <= metrics["success_rate"] <= 1
        assert metrics["average_response_time"] >= 0
    
    @staticmethod
    def assert_recommendation_quality(recommendations, min_confidence=0.0):
        """Assert recommendation quality"""
        assert isinstance(recommendations, list), "Recommendations should be a list"
        for rec in recommendations:
            assert "product_id" in rec
            assert "confidence" in rec
            assert rec["confidence"] >= min_confidence
            assert 0 <= rec["confidence"] <= 1

@pytest.fixture
def test_utils():
    """Test utilities fixture"""
    return TestUtils

# Async test utilities
@pytest.fixture
def async_test_utils():
    """Async test utilities"""
    class AsyncTestUtils:
        @staticmethod
        async def measure_async_performance(async_func, *args, **kwargs):
            """Measure performance of async function"""
            import time
            start_time = time.time()
            result = await async_func(*args, **kwargs)
            end_time = time.time()
            return result, (end_time - start_time) * 1000  # Return result and time in ms
        
        @staticmethod
        async def run_concurrent_requests(async_func, num_requests, *args, **kwargs):
            """Run multiple concurrent requests"""
            tasks = [async_func(*args, **kwargs) for _ in range(num_requests)]
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    return AsyncTestUtils