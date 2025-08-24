"""
Unit Tests for Customer Agent
Tests customer behavior analysis, segmentation, and personalization functionality
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agents.customer_agent import CustomerAgent
from src.database.models import Customer, CustomerInteraction, DatabaseManager

class TestCustomerAgent:
    """Test suite for Customer Agent functionality"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing"""
        mock_db = Mock(spec=DatabaseManager)
        mock_session = Mock()
        mock_db.get_session.return_value = mock_session
        return mock_db
    
    @pytest.fixture
    def customer_agent(self, mock_db_manager):
        """Create customer agent instance for testing"""
        return CustomerAgent("test_customer_agent", mock_db_manager)
    
    @pytest.fixture
    def sample_customer(self):
        """Sample customer data for testing"""
        return Customer(
            customer_id="test_customer_001",
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            age=30,
            gender="male",
            location="New York, NY",
            total_orders=5,
            total_spent=500.0,
            avg_order_value=100.0,
            lifetime_value=750.0,
            customer_segment="frequent_buyer",
            preferences={"preferred_categories": ["electronics", "books"]}
        )
    
    @pytest.fixture
    def sample_interactions(self):
        """Sample customer interactions for testing"""
        return [
            CustomerInteraction(
                interaction_id="int_001",
                customer_id="test_customer_001",
                product_id="prod_001",
                interaction_type="view",
                timestamp=datetime.utcnow() - timedelta(days=1),
                duration_seconds=120,
                scroll_depth=0.8,
                device_type="desktop"
            ),
            CustomerInteraction(
                interaction_id="int_002",
                customer_id="test_customer_001",
                product_id="prod_002",
                interaction_type="purchase",
                timestamp=datetime.utcnow() - timedelta(days=2),
                duration_seconds=300,
                scroll_depth=1.0,
                device_type="mobile"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, customer_agent):
        """Test that customer agent initializes properly"""
        assert customer_agent.agent_id == "test_customer_agent"
        assert customer_agent.agent_type.value == "customer_agent"
        assert hasattr(customer_agent, 'segmentation_model')
        assert hasattr(customer_agent, 'behavior_patterns')
        assert hasattr(customer_agent, 'personalization_strategies')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, customer_agent):
        """Test that agent returns correct capabilities"""
        capabilities = customer_agent.get_capabilities()
        
        assert len(capabilities) == 4
        capability_names = [cap.name for cap in capabilities]
        assert "customer_analysis" in capability_names
        assert "customer_segmentation" in capability_names
        assert "personalization" in capability_names
        assert "churn_prediction" in capability_names
        
        # Test confidence levels
        for capability in capabilities:
            assert 0.7 <= capability.confidence_level <= 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_customer_success(self, customer_agent, sample_customer, sample_interactions):
        """Test successful customer analysis"""
        # Mock database session
        mock_session = customer_agent.db_manager.get_session.return_value
        mock_session.query.return_value.filter.return_value.first.return_value = sample_customer
        mock_session.query.return_value.filter.return_value.all.return_value = sample_interactions
        
        # Mock store_memory method
        customer_agent.store_memory = AsyncMock(return_value="memory_001")
        
        # Test customer analysis
        request = {
            "type": "analyze_customer",
            "customer_id": "test_customer_001",
            "interaction_data": {},
            "analysis_type": "comprehensive"
        }
        
        result = await customer_agent.process_request(request)
        
        assert "error" not in result
        assert "customer_id" in result
        assert "segment" in result
        assert "behavior_analysis" in result
        assert result["customer_id"] == "test_customer_001"
    
    @pytest.mark.asyncio
    async def test_analyze_customer_not_found(self, customer_agent):
        """Test customer analysis when customer not found"""
        # Mock database session
        mock_session = customer_agent.db_manager.get_session.return_value
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        request = {
            "type": "analyze_customer",
            "customer_id": "nonexistent_customer",
            "interaction_data": {}
        }
        
        result = await customer_agent.process_request(request)
        
        assert "error" in result
        assert result["error"] == "Customer not found"
    
    @pytest.mark.asyncio
    async def test_behavior_pattern_analysis(self, customer_agent, sample_customer, sample_interactions):
        """Test behavior pattern analysis functionality"""
        behavior_analysis = await customer_agent._analyze_behavior_patterns(sample_customer, sample_interactions)
        
        assert "session_frequency" in behavior_analysis
        assert "avg_session_duration" in behavior_analysis
        assert "preferred_device" in behavior_analysis
        assert "interaction_types" in behavior_analysis
        assert "engagement_score" in behavior_analysis
        assert "browsing_velocity" in behavior_analysis
        
        # Test calculated values
        assert behavior_analysis["session_frequency"] >= 0
        assert behavior_analysis["avg_session_duration"] >= 0
        assert behavior_analysis["preferred_device"] in ["desktop", "mobile", "tablet", "unknown"]
    
    @pytest.mark.asyncio
    async def test_customer_segmentation(self, customer_agent):
        """Test customer segmentation functionality"""
        # Mock customer data
        customer_data = [
            {"customer_id": "cust_001", "age": 25, "total_orders": 2, "total_spent": 100, "avg_order_value": 50, "lifetime_value": 150, "last_active_days": 5},
            {"customer_id": "cust_002", "age": 45, "total_orders": 15, "total_spent": 2000, "avg_order_value": 133, "lifetime_value": 3000, "last_active_days": 2},
            {"customer_id": "cust_003", "age": 35, "total_orders": 1, "total_spent": 25, "avg_order_value": 25, "lifetime_value": 30, "last_active_days": 30}
        ]
        
        # Mock database session
        mock_session = customer_agent.db_manager.get_session.return_value
        mock_session.query.return_value.all.return_value = []
        
        request = {
            "type": "segment_customers",
            "customer_data": customer_data
        }
        
        result = await customer_agent.process_request(request)
        
        assert "error" not in result
        assert "segments" in result
        assert "segment_mapping" in result
        assert "total_customers" in result
        assert result["total_customers"] == len(customer_data)
    
    @pytest.mark.asyncio
    async def test_personalize_experience(self, customer_agent, sample_customer):
        """Test personalization functionality"""
        # Mock customer analysis
        customer_agent._analyze_customer = AsyncMock(return_value={
            "customer_id": "test_customer_001",
            "segment": "tech_enthusiast",
            "preferences": {"preferred_categories": ["electronics"]},
            "behavior_analysis": {"engagement_score": 0.8}
        })
        
        # Mock store_memory
        customer_agent.store_memory = AsyncMock(return_value="memory_001")
        
        request = {
            "type": "personalize_experience",
            "customer_id": "test_customer_001",
            "context": {"page_type": "homepage"}
        }
        
        result = await customer_agent.process_request(request)
        
        assert "error" not in result
        assert "customer_id" in result
        assert "strategy" in result
        assert "personalization" in result
        assert "confidence" in result
        assert result["customer_id"] == "test_customer_001"
    
    @pytest.mark.asyncio
    async def test_churn_prediction(self, customer_agent):
        """Test churn prediction functionality"""
        # Mock customer analysis
        customer_agent._analyze_customer = AsyncMock(return_value={
            "customer_id": "test_customer_001",
            "behavior_analysis": {
                "engagement_score": 0.2,
                "session_frequency": 3,
                "sentiment_trend": -0.5,
                "browsing_velocity": 0.05
            }
        })
        
        # Mock store_memory
        customer_agent.store_memory = AsyncMock(return_value="memory_001")
        
        request = {
            "type": "predict_churn",
            "customer_id": "test_customer_001"
        }
        
        result = await customer_agent.process_request(request)
        
        assert "error" not in result
        assert "customer_id" in result
        assert "churn_probability" in result
        assert "risk_level" in result
        assert "risk_factors" in result
        assert "retention_strategies" in result
        assert 0 <= result["churn_probability"] <= 1
        assert result["risk_level"] in ["low", "medium", "high"]
    
    @pytest.mark.asyncio
    async def test_personalization_strategies(self, customer_agent):
        """Test different personalization strategies"""
        customer_profile = {
            "segment": "tech_enthusiast",
            "preferences": {"preferred_categories": ["electronics"]},
            "behavior_analysis": {"engagement_score": 0.8}
        }
        context = {"page_type": "homepage"}
        
        # Test tech enthusiast strategy
        result = await customer_agent._tech_enthusiast_strategy(customer_profile, context)
        assert "welcome_message" in result
        assert "recommended_categories" in result
        assert "tech_gadgets" in result["recommended_categories"]
        
        # Test frequent buyer strategy
        result = await customer_agent._frequent_buyer_strategy(customer_profile, context)
        assert "welcome_message" in result
        assert "ui_elements" in result
        assert result["ui_elements"]["quick_reorder"] is True
        
        # Test price sensitive strategy
        result = await customer_agent._price_sensitive_strategy(customer_profile, context)
        assert "on_sale" in result["recommended_categories"]
        assert result["ui_elements"]["price_comparison"] is True
    
    @pytest.mark.asyncio
    async def test_real_time_behavior_tracking(self, customer_agent):
        """Test real-time behavior tracking"""
        session_data = {
            "session_id": "session_001",
            "page_views": [{"url": "/products/laptop"}],
            "interactions": [{"type": "click"}],
            "duration_minutes": 5
        }
        
        request = {
            "type": "track_real_time_behavior",
            "customer_id": "test_customer_001",
            "session_data": session_data
        }
        
        result = await customer_agent.process_request(request)
        
        assert "error" not in result
        assert "customer_id" in result
        assert "session_id" in result
        assert "current_intent" in result
        assert "engagement_level" in result
        assert "next_best_action" in result
    
    @pytest.mark.asyncio
    async def test_error_handling(self, customer_agent):
        """Test error handling for invalid requests"""
        # Test unknown request type
        request = {"type": "unknown_type", "customer_id": "test_001"}
        result = await customer_agent.process_request(request)
        
        assert "error" in result
        assert "Unknown request type" in result["error"]
    
    @pytest.mark.asyncio
    async def test_performance_metrics_update(self, customer_agent):
        """Test that performance metrics are updated during operations"""
        initial_metrics = customer_agent.get_performance_metrics()
        initial_requests = initial_metrics["total_requests"]
        
        # Mock successful request
        customer_agent._analyze_customer = AsyncMock(return_value={"customer_id": "test_001"})
        
        request = {"type": "analyze_customer", "customer_id": "test_001"}
        await customer_agent.process_request(request)
        
        updated_metrics = customer_agent.get_performance_metrics()
        assert updated_metrics["total_requests"] > initial_requests
    
    @pytest.mark.asyncio
    async def test_memory_operations(self, customer_agent):
        """Test memory storage and retrieval operations"""
        # Mock embedding creation
        customer_agent._create_embedding = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
        
        # Mock database operations
        mock_session = customer_agent.db_manager.get_session.return_value
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.close = Mock()
        
        # Test memory storage
        memory_id = await customer_agent.store_memory(
            memory_type="customer_insight",
            content={"insight": "test insight"},
            customer_id="test_001",
            importance_score=0.8
        )
        
        assert memory_id is not None
        assert memory_id.startswith("test_customer_agent_")
        
        # Verify database operations were called
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    def test_interaction_weight_calculation(self, customer_agent):
        """Test interaction weight calculation for different types"""
        assert customer_agent._get_interaction_weights("frequent_buyer") != customer_agent._get_interaction_weights("price_sensitive")
        
        # Test weight distributions
        frequent_weights = customer_agent._get_interaction_weights("frequent_buyer")
        assert len(frequent_weights) == 5  # view, click, add_to_cart, purchase, review
        assert all(w >= 0 for w in frequent_weights)
        
        # Frequent buyers should have higher purchase weights
        price_sensitive_weights = customer_agent._get_interaction_weights("price_sensitive")
        assert frequent_weights[3] > price_sensitive_weights[3]  # Purchase weight
    
    @pytest.mark.asyncio
    async def test_explanation_generation(self, customer_agent):
        """Test explanation generation for agent decisions"""
        decision = {"recommended_action": "show_promotion", "confidence": 0.85}
        context = {"customer_segment": "price_sensitive", "current_page": "homepage"}
        
        explanation = await customer_agent._generate_explanation(decision, context)
        
        assert "reasoning" in explanation
        assert "factors" in explanation
        assert "data_sources" in explanation
        assert isinstance(explanation["reasoning"], list)
        assert len(explanation["reasoning"]) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])