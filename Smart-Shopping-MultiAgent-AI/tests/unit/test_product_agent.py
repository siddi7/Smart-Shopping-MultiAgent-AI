"""
Unit Tests for Product Agent
Tests product analysis, catalog optimization, and multi-modal processing
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agents.product_agent import ProductAgent
from src.database.models import DatabaseManager
from src.utils.config import Settings

class TestProductAgent:
    """Test cases for Product Agent"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager"""
        mock_db = Mock(spec=DatabaseManager)
        mock_session = Mock()
        mock_db.get_session.return_value = mock_session
        return mock_db
    
    @pytest.fixture
    def settings(self):
        """Test settings"""
        return Settings(
            DATABASE_URL="sqlite:///test.db",
            ENABLE_PERFORMANCE_MONITORING=True,
            ENABLE_EXPLAINABLE_AI=True
        )
    
    @pytest.fixture
    def product_agent(self, mock_db_manager, settings):
        """Create product agent instance"""
        return ProductAgent(
            agent_id="test_product_agent",
            db_manager=mock_db_manager,
            config={"settings": settings}
        )
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, product_agent):
        """Test product agent initialization"""
        assert product_agent.agent_id == "test_product_agent"
        assert product_agent.agent_type.value == "product_agent"
        assert hasattr(product_agent, 'performance_metrics')
        assert hasattr(product_agent, 'short_term_memory')
    
    @pytest.mark.asyncio
    async def test_product_analysis_request(self, product_agent):
        """Test product analysis request processing"""
        request_data = {
            "request_type": "analyze_product",
            "product_id": "prod_001",
            "analysis_type": "comprehensive"
        }
        
        with patch.object(product_agent, '_analyze_product') as mock_analyze:
            mock_analyze.return_value = {
                "product_insights": {
                    "product_id": "prod_001",
                    "performance": {"views": 1000, "conversions": 50}
                }
            }
            
            result = await product_agent.process_request(request_data)
            
            assert "product_insights" in result
            assert result["product_insights"]["product_id"] == "prod_001"
            mock_analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_similar_products_request(self, product_agent):
        """Test similar products request processing"""
        request_data = {
            "request_type": "find_similar_products",
            "product_id": "prod_001",
            "similarity_threshold": 0.7,
            "max_results": 5
        }
        
        with patch.object(product_agent, '_find_similar_products') as mock_similar:
            mock_similar.return_value = {
                "target_product_id": "prod_001",
                "similar_products": [
                    {"product_id": "prod_002", "similarity_score": 0.85}
                ]
            }
            
            result = await product_agent.process_request(request_data)
            
            assert "target_product_id" in result
            assert "similar_products" in result
            mock_similar.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_catalog_optimization_request(self, product_agent):
        """Test catalog optimization request processing"""
        request_data = {
            "request_type": "optimize_catalog",
            "category": "electronics",
            "optimization_goals": ["improve_visibility", "increase_conversions"]
        }
        
        with patch.object(product_agent, '_optimize_catalog') as mock_optimize:
            mock_optimize.return_value = {
                "category": "electronics",
                "optimization_actions": [
                    {"product_id": "prod_001", "action": "improve_seo"}
                ]
            }
            
            result = await product_agent.process_request(request_data)
            
            assert "category" in result
            assert "optimization_actions" in result
            mock_optimize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_product_analysis_comprehensive(self, product_agent):
        """Test comprehensive product analysis"""
        product_id = "prod_001"
        analysis_type = "comprehensive"
        
        # Mock database queries
        mock_session = product_agent.db_manager.get_session.return_value
        mock_product = Mock()
        mock_product.id = product_id
        mock_product.name = "Test Product"
        mock_product.category = "electronics"
        mock_product.price = 299.99
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_product
        mock_session.query.return_value.filter.return_value.count.return_value = 100  # views
        
        with patch.object(product_agent, '_analyze_content_quality') as mock_content:
            with patch.object(product_agent, '_calculate_performance_metrics') as mock_performance:
                mock_content.return_value = {"overall_score": 0.85}
                mock_performance.return_value = {
                    "total_views": 1000,
                    "conversion_rate": 0.05,
                    "revenue_generated": 15000
                }
                
                result = await product_agent._analyze_product(product_id, analysis_type)
                
                assert "product_insights" in result
                assert "optimization_suggestions" in result
                assert result["product_insights"]["product_id"] == product_id
    
    @pytest.mark.asyncio
    async def test_similar_products_calculation(self, product_agent):
        """Test similar products calculation"""
        product_id = "prod_001"
        similarity_threshold = 0.7
        max_results = 5
        
        # Mock database queries
        mock_session = product_agent.db_manager.get_session.return_value
        mock_products = [
            Mock(id="prod_002", name="Similar Product 1", category="electronics", price=250.0),
            Mock(id="prod_003", name="Similar Product 2", category="electronics", price=350.0)
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = mock_products
        
        with patch.object(product_agent, '_calculate_similarity_score') as mock_similarity:
            mock_similarity.side_effect = [0.85, 0.75]  # Similarity scores
            
            result = await product_agent._find_similar_products(
                product_id, similarity_threshold, max_results
            )
            
            assert "target_product_id" in result
            assert "similar_products" in result
            assert len(result["similar_products"]) == 2
            assert all(p["similarity_score"] >= similarity_threshold 
                      for p in result["similar_products"])
    
    @pytest.mark.asyncio
    async def test_catalog_optimization_analysis(self, product_agent):
        """Test catalog optimization analysis"""
        category = "electronics"
        optimization_goals = ["improve_visibility", "increase_conversions"]
        
        # Mock database queries
        mock_session = product_agent.db_manager.get_session.return_value
        mock_products = [
            Mock(id="prod_001", name="Product 1", category=category),
            Mock(id="prod_002", name="Product 2", category=category)
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = mock_products
        
        with patch.object(product_agent, '_analyze_optimization_opportunities') as mock_opportunities:
            mock_opportunities.return_value = [
                {"product_id": "prod_001", "action": "improve_seo", "priority": "high"},
                {"product_id": "prod_002", "action": "update_images", "priority": "medium"}
            ]
            
            result = await product_agent._optimize_catalog(category, optimization_goals)
            
            assert "category" in result
            assert "total_products_analyzed" in result
            assert "optimization_actions" in result
            assert result["category"] == category
            assert len(result["optimization_actions"]) == 2
    
    @pytest.mark.asyncio
    async def test_content_quality_analysis(self, product_agent):
        """Test content quality analysis"""
        product_data = {
            "name": "Smart Laptop Pro",
            "description": "High-performance laptop with advanced features",
            "images": ["image1.jpg", "image2.jpg"],
            "specifications": {"CPU": "Intel i7", "RAM": "16GB"}
        }
        
        result = await product_agent._analyze_content_quality(product_data)
        
        assert "overall_score" in result
        assert "details" in result
        assert 0 <= result["overall_score"] <= 1
        assert "description_quality" in result["details"]
        assert "image_quality" in result["details"]
        assert "specification_completeness" in result["details"]
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, product_agent):
        """Test performance metrics calculation"""
        product_id = "prod_001"
        
        # Mock database queries for metrics
        mock_session = product_agent.db_manager.get_session.return_value
        mock_session.query.return_value.filter.return_value.count.return_value = 1000  # views
        
        with patch.object(product_agent.db_manager, 'execute_query') as mock_execute:
            mock_execute.return_value = [
                {"conversions": 50, "revenue": 15000, "avg_rating": 4.5}
            ]
            
            result = await product_agent._calculate_performance_metrics(product_id)
            
            assert "total_views" in result
            assert "conversion_rate" in result
            assert "revenue_generated" in result
            assert "average_rating" in result
            assert result["total_views"] == 1000
            assert result["conversion_rate"] == 0.05  # 50/1000
    
    @pytest.mark.asyncio
    async def test_similarity_score_calculation(self, product_agent):
        """Test similarity score calculation between products"""
        product1_features = {
            "category": "electronics",
            "price": 299.99,
            "brand": "TechBrand",
            "features": ["wireless", "portable", "high-quality"]
        }
        
        product2_features = {
            "category": "electronics",
            "price": 259.99,
            "brand": "TechBrand",
            "features": ["wireless", "portable", "durable"]
        }
        
        similarity_score = await product_agent._calculate_similarity_score(
            product1_features, product2_features
        )
        
        assert 0 <= similarity_score <= 1
        assert similarity_score > 0.7  # Should be high similarity
    
    @pytest.mark.asyncio
    async def test_optimization_opportunities_analysis(self, product_agent):
        """Test optimization opportunities analysis"""
        product_data = {
            "id": "prod_001",
            "name": "Test Product",
            "views": 1000,
            "conversions": 20,  # Low conversion rate
            "description_length": 50,  # Short description
            "image_count": 1  # Few images
        }
        
        opportunities = await product_agent._analyze_optimization_opportunities(product_data)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        # Should identify low conversion rate
        conversion_actions = [op for op in opportunities if "conversion" in op.get("action", "").lower()]
        assert len(conversion_actions) > 0
    
    @pytest.mark.asyncio
    async def test_multi_modal_content_processing(self, product_agent):
        """Test multi-modal content processing capabilities"""
        content_data = {
            "text": "High-quality wireless headphones with noise cancellation",
            "images": ["headphone1.jpg", "headphone2.jpg"],
            "videos": ["demo_video.mp4"],
            "specifications": {"frequency_response": "20Hz-20kHz", "battery_life": "30 hours"}
        }
        
        with patch.object(product_agent, '_process_text_content') as mock_text:
            with patch.object(product_agent, '_process_image_content') as mock_image:
                with patch.object(product_agent, '_process_video_content') as mock_video:
                    mock_text.return_value = {"sentiment": "positive", "keywords": ["quality", "wireless"]}
                    mock_image.return_value = {"quality_score": 0.9, "features_detected": ["headphones"]}
                    mock_video.return_value = {"engagement_score": 0.8, "duration": 120}
                    
                    result = await product_agent._process_multimodal_content(content_data)
                    
                    assert "text_analysis" in result
                    assert "image_analysis" in result
                    assert "video_analysis" in result
                    assert "overall_content_score" in result
    
    @pytest.mark.asyncio
    async def test_agent_learning_and_adaptation(self, product_agent):
        """Test agent learning and adaptation capabilities"""
        # Simulate multiple requests to test learning
        for i in range(5):
            request_data = {
                "request_type": "analyze_product",
                "product_id": f"prod_00{i}",
                "analysis_type": "basic"
            }
            
            with patch.object(product_agent, '_analyze_product') as mock_analyze:
                mock_analyze.return_value = {"product_insights": {}}
                await product_agent.process_request(request_data)
        
        # Check that performance metrics are being tracked
        metrics = product_agent.get_performance_metrics()
        assert metrics["total_requests"] == 5
        assert "average_response_time" in metrics
        assert "success_rate" in metrics
    
    @pytest.mark.asyncio
    async def test_error_handling(self, product_agent):
        """Test error handling in product agent"""
        # Test with invalid request type
        invalid_request = {
            "request_type": "invalid_type",
            "product_id": "prod_001"
        }
        
        result = await product_agent.process_request(invalid_request)
        assert "error" in result
        
        # Test with missing required fields
        incomplete_request = {
            "request_type": "analyze_product"
            # Missing product_id
        }
        
        result = await product_agent.process_request(incomplete_request)
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_agent_memory_management(self, product_agent):
        """Test agent memory management"""
        # Add data to short-term memory
        product_agent.short_term_memory["test_key"] = "test_value"
        
        # Test memory persistence through requests
        request_data = {
            "request_type": "analyze_product",
            "product_id": "prod_001",
            "analysis_type": "basic"
        }
        
        with patch.object(product_agent, '_analyze_product') as mock_analyze:
            mock_analyze.return_value = {"product_insights": {}}
            await product_agent.process_request(request_data)
        
        # Memory should persist
        assert "test_key" in product_agent.short_term_memory
        assert product_agent.short_term_memory["test_key"] == "test_value"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])