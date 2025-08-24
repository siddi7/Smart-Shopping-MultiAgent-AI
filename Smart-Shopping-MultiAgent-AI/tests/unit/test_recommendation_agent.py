"""
Unit Tests for Recommendation Agent
Tests recommendation generation, explainable AI, and real-time personalization
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

from src.agents.recommendation_agent import RecommendationAgent
from src.database.models import Customer, Product, CustomerInteraction, RecommendationHistory, DatabaseManager

class TestRecommendationAgent:
    """Test suite for Recommendation Agent functionality"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing"""
        mock_db = Mock(spec=DatabaseManager)
        mock_session = Mock()
        mock_db.get_session.return_value = mock_session
        return mock_db
    
    @pytest.fixture
    def recommendation_agent(self, mock_db_manager):
        """Create recommendation agent instance for testing"""
        return RecommendationAgent("test_rec_agent", mock_db_manager)
    
    @pytest.fixture
    def sample_customer(self):
        """Sample customer for testing"""
        return Customer(
            customer_id="test_customer_001",
            email="test@example.com",
            customer_segment="tech_enthusiast",
            preferences={"preferred_categories": ["electronics", "gadgets"]}
        )
    
    @pytest.fixture
    def sample_products(self):
        """Sample products for testing"""
        return [
            Product(
                product_id="prod_001",
                name="Smart Laptop",
                category="electronics",
                brand="TechBrand",
                price=999.99,
                rating=4.5,
                popularity_score=0.8,
                is_active=True,
                stock_quantity=10
            ),
            Product(
                product_id="prod_002",
                name="Wireless Headphones",
                category="electronics",
                brand="AudioBrand",
                price=199.99,
                rating=4.2,
                popularity_score=0.7,
                is_active=True,
                stock_quantity=25
            ),
            Product(
                product_id="prod_003",
                name="Gaming Mouse",
                category="electronics",
                brand="GameBrand",
                price=79.99,
                rating=4.8,
                popularity_score=0.6,
                is_active=True,
                stock_quantity=15
            )
        ]
    
    @pytest.fixture
    def sample_interactions(self):
        """Sample customer interactions for testing"""
        return [
            CustomerInteraction(
                interaction_id="int_001",
                customer_id="test_customer_001",
                product_id="prod_001",
                interaction_type="view",
                timestamp=datetime.utcnow() - timedelta(days=1)
            ),
            CustomerInteraction(
                interaction_id="int_002",
                customer_id="test_customer_001",
                product_id="prod_002",
                interaction_type="purchase",
                timestamp=datetime.utcnow() - timedelta(days=2)
            )
        ]
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, recommendation_agent):
        """Test that recommendation agent initializes properly"""
        assert recommendation_agent.agent_id == "test_rec_agent"
        assert recommendation_agent.agent_type.value == "recommendation_agent"
        assert hasattr(recommendation_agent, 'recommendation_cache')
        assert hasattr(recommendation_agent, 'ab_test_variants')
        assert len(recommendation_agent.ab_test_variants) == 3
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, recommendation_agent):
        """Test that agent returns correct capabilities"""
        capabilities = recommendation_agent.get_capabilities()
        
        assert len(capabilities) == 3
        capability_names = [cap.name for cap in capabilities]
        assert "generate_recommendations" in capability_names
        assert "real_time_personalization" in capability_names
        assert "explain_recommendations" in capability_names
        
        # Test confidence levels
        for capability in capabilities:
            assert 0.8 <= capability.confidence_level <= 1.0
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_success(self, recommendation_agent, sample_customer, sample_products, sample_interactions):
        """Test successful recommendation generation"""
        # Mock database operations
        mock_session = recommendation_agent.db_manager.get_session.return_value
        mock_session.query.return_value.filter.return_value.first.return_value = sample_customer
        mock_session.query.return_value.filter.return_value.all.return_value = sample_interactions
        
        # Mock product query
        mock_session.query.return_value.filter.return_value.limit.return_value.all.return_value = sample_products
        
        # Mock helper methods
        recommendation_agent._extract_user_preferences = AsyncMock(return_value={
            "preferred_categories": ["electronics"],
            "preferred_brands": ["TechBrand"],
            "price_sensitivity": "medium"
        })
        recommendation_agent._store_recommendations = AsyncMock()
        
        request = {
            "type": "generate_recommendations",
            "customer_id": "test_customer_001",
            "context": {"page_type": "homepage"},
            "num_recommendations": 5
        }
        
        result = await recommendation_agent.process_request(request)
        
        assert "error" not in result
        assert "customer_id" in result
        assert "recommendations" in result
        assert "algorithm_variant" in result
        assert len(result["recommendations"]) <= 5
        assert result["customer_id"] == "test_customer_001"
    
    @pytest.mark.asyncio
    async def test_customer_not_found(self, recommendation_agent):
        """Test recommendation generation when customer not found"""
        mock_session = recommendation_agent.db_manager.get_session.return_value
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        request = {
            "type": "generate_recommendations",
            "customer_id": "nonexistent_customer"
        }
        
        result = await recommendation_agent.process_request(request)
        
        assert "error" in result
        assert result["error"] == "Customer not found"
    
    @pytest.mark.asyncio
    async def test_collaborative_filtering(self, recommendation_agent, sample_products):
        """Test collaborative filtering recommendations"""
        # Mock database operations for collaborative filtering
        mock_session = recommendation_agent.db_manager.get_session.return_value
        
        # Mock customer interactions
        mock_session.query.return_value.filter.return_value.all.return_value = [
            Mock(product_id="prod_001"),
            Mock(product_id="prod_002")
        ]
        
        # Mock similar customers
        mock_session.query.return_value.filter.return_value.distinct.return_value.limit.return_value.all.return_value = [
            ("similar_customer_1",),
            ("similar_customer_2",)
        ]
        
        # Mock recommended products
        mock_session.query.return_value.filter.return_value.distinct.return_value.limit.return_value.all.return_value = [
            ("prod_003",)
        ]
        
        # Mock product details
        mock_session.query.return_value.filter.return_value.first.return_value = sample_products[2]
        
        recommendations = await recommendation_agent._collaborative_recommendations(
            "test_customer_001", sample_products, 5
        )
        
        assert isinstance(recommendations, list)
        if recommendations:  # Only check if recommendations were generated
            assert all("product_id" in rec for rec in recommendations)
            assert all("confidence" in rec for rec in recommendations)
            assert all("algorithm" in rec for rec in recommendations)
            assert all(rec["algorithm"] == "collaborative" for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_content_based_filtering(self, recommendation_agent, sample_products):
        """Test content-based filtering recommendations"""
        user_preferences = {
            "preferred_categories": ["electronics"],
            "preferred_brands": ["TechBrand"],
            "price_sensitivity": "medium"
        }
        
        recommendations = await recommendation_agent._content_based_recommendations(
            user_preferences, sample_products, 5
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        
        for rec in recommendations:
            assert "product_id" in rec
            assert "confidence" in rec
            assert "algorithm" in rec
            assert rec["algorithm"] == "content_based"
            assert 0 <= rec["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_hybrid_recommendations(self, recommendation_agent, sample_products):
        """Test hybrid recommendation approach"""
        # Mock collaborative and content-based methods
        recommendation_agent._collaborative_recommendations = AsyncMock(return_value=[
            {"product_id": "prod_001", "confidence": 0.8, "algorithm": "collaborative"}
        ])
        recommendation_agent._content_based_recommendations = AsyncMock(return_value=[
            {"product_id": "prod_002", "confidence": 0.7, "algorithm": "content_based"}
        ])
        
        user_preferences = {"preferred_categories": ["electronics"]}
        
        recommendations = await recommendation_agent._hybrid_recommendations(
            "test_customer_001", user_preferences, sample_products, 5
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        
        for rec in recommendations:
            assert "product_id" in rec
            assert "confidence" in rec
            assert "algorithm" in rec
            assert rec["algorithm"] == "hybrid"
    
    @pytest.mark.asyncio
    async def test_real_time_personalization(self, recommendation_agent):
        """Test real-time personalization functionality"""
        # Mock generate_recommendations method
        recommendation_agent._generate_recommendations = AsyncMock(return_value={
            "recommendations": [
                {"product_id": "prod_001", "name": "Smart Laptop", "confidence": 0.9}
            ]
        })
        
        # Mock related products method
        recommendation_agent._get_related_products = AsyncMock(return_value=[
            {"product_id": "prod_002", "name": "Wireless Mouse", "confidence": 0.8}
        ])
        
        request = {
            "type": "real_time_personalization",
            "customer_id": "test_customer_001",
            "session_context": {
                "page_type": "product_page",
                "product_id": "prod_001"
            }
        }
        
        result = await recommendation_agent.process_request(request)
        
        assert "error" not in result
        assert "customer_id" in result
        assert "real_time_recommendations" in result
        assert "next_best_actions" in result
        assert result["customer_id"] == "test_customer_001"
    
    @pytest.mark.asyncio
    async def test_explain_recommendation(self, recommendation_agent):
        """Test recommendation explanation functionality"""
        # Mock database operations
        mock_session = recommendation_agent.db_manager.get_session.return_value
        mock_rec_history = Mock()
        mock_rec_history.algorithm_used = "hybrid"
        mock_rec_history.confidence_score = 0.85
        mock_session.query.return_value.filter.return_value.first.return_value = mock_rec_history
        
        request = {
            "type": "explain_recommendation",
            "recommendation_id": "rec_001",
            "customer_id": "test_customer_001"
        }
        
        result = await recommendation_agent.process_request(request)
        
        assert "error" not in result
        assert "recommendation_id" in result
        assert "algorithm_used" in result
        assert "confidence_score" in result
        assert "detailed_explanation" in result
        assert result["algorithm_used"] == "hybrid"
        assert result["confidence_score"] == 0.85
    
    @pytest.mark.asyncio
    async def test_user_preference_extraction(self, recommendation_agent, sample_interactions):
        """Test user preference extraction from interaction history"""
        # Mock database operations
        mock_session = recommendation_agent.db_manager.get_session.return_value
        mock_session.query.return_value.filter.return_value.all.return_value = sample_interactions
        
        # Mock products
        mock_products = [
            Mock(category="electronics", brand="TechBrand", price=999.99),
            Mock(category="electronics", brand="AudioBrand", price=199.99)
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = mock_products
        
        preferences = await recommendation_agent._extract_user_preferences("test_customer_001")
        
        assert "preferred_categories" in preferences
        assert "preferred_brands" in preferences
        assert "price_sensitivity" in preferences
        assert "avg_price" in preferences
        assert preferences["price_sensitivity"] in ["high", "medium", "low"]
    
    @pytest.mark.asyncio
    async def test_algorithm_variant_selection(self, recommendation_agent):
        """Test A/B testing algorithm variant selection"""
        # Test that same customer always gets same variant (consistent hashing)
        variant1 = recommendation_agent._select_algorithm_variant("customer_001")
        variant2 = recommendation_agent._select_algorithm_variant("customer_001")
        assert variant1 == variant2
        
        # Test that variant is from available options
        assert variant1 in recommendation_agent.ab_test_variants
    
    @pytest.mark.asyncio
    async def test_related_products(self, recommendation_agent, sample_products):
        """Test related products functionality"""
        # Mock database operations
        mock_session = recommendation_agent.db_manager.get_session.return_value
        mock_session.query.return_value.filter.return_value.first.return_value = sample_products[0]
        mock_session.query.return_value.filter.return_value.limit.return_value.all.return_value = sample_products[1:]
        
        related = await recommendation_agent._get_related_products("prod_001")
        
        assert isinstance(related, list)
        for product in related:
            assert "product_id" in product
            assert "confidence" in product
            assert "algorithm" in product
            assert product["algorithm"] == "similarity"
    
    @pytest.mark.asyncio
    async def test_explanation_generation_for_recommendation(self, recommendation_agent):
        """Test explanation generation for individual recommendations"""
        recommendation = {
            "product_id": "prod_001",
            "algorithm": "collaborative",
            "confidence": 0.85
        }
        user_preferences = {
            "preferred_categories": ["electronics"],
            "preferred_brands": ["TechBrand"]
        }
        
        explanation = recommendation_agent._generate_explanation_for_rec(recommendation, user_preferences)
        
        assert "primary_reason" in explanation
        assert "factors" in explanation
        assert isinstance(explanation["factors"], list)
        assert len(explanation["factors"]) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, recommendation_agent):
        """Test error handling for various scenarios"""
        # Test unknown request type
        request = {"type": "unknown_type", "customer_id": "test_001"}
        result = await recommendation_agent.process_request(request)
        
        assert "error" in result
        assert "Unknown request type" in result["error"]
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, recommendation_agent):
        """Test that performance metrics are properly tracked"""
        initial_metrics = recommendation_agent.get_performance_metrics()
        initial_requests = initial_metrics["total_requests"]
        
        # Mock successful request processing
        recommendation_agent._generate_recommendations = AsyncMock(return_value={
            "recommendations": []
        })
        
        request = {
            "type": "generate_recommendations",
            "customer_id": "test_001"
        }
        
        await recommendation_agent.process_request(request)
        
        updated_metrics = recommendation_agent.get_performance_metrics()
        assert updated_metrics["total_requests"] > initial_requests
        assert "average_response_time" in updated_metrics
    
    @pytest.mark.asyncio
    async def test_recommendation_storage(self, recommendation_agent):
        """Test recommendation storage for tracking"""
        recommendations = [
            {"product_id": "prod_001", "confidence": 0.9, "algorithm": "hybrid"},
            {"product_id": "prod_002", "confidence": 0.8, "algorithm": "hybrid"}
        ]
        
        # Mock database operations
        mock_session = recommendation_agent.db_manager.get_session.return_value
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.close = Mock()
        
        await recommendation_agent._store_recommendations("test_customer_001", recommendations)
        
        # Verify database operations
        assert mock_session.add.call_count == len(recommendations)
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    def test_explanation_types(self, recommendation_agent):
        """Test different explanation types for different algorithms"""
        # Test collaborative explanation
        rec_collab = {"algorithm": "collaborative", "confidence": 0.8}
        explanation_collab = recommendation_agent._generate_explanation_for_rec(rec_collab, {})
        assert "similar to you" in explanation_collab["primary_reason"].lower()
        
        # Test content-based explanation
        rec_content = {"algorithm": "content_based", "confidence": 0.8}
        explanation_content = recommendation_agent._generate_explanation_for_rec(rec_content, {})
        assert "matches your" in explanation_content["primary_reason"].lower()
        
        # Test hybrid explanation
        rec_hybrid = {"algorithm": "hybrid", "confidence": 0.8}
        explanation_hybrid = recommendation_agent._generate_explanation_for_rec(rec_hybrid, {})
        assert "based on your preferences" in explanation_hybrid["primary_reason"].lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])