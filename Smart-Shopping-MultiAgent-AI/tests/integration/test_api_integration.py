"""
Integration Tests for Smart Shopping Multi-Agent AI API
Tests the full API endpoints and agent coordination
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the FastAPI app
from main import app
from src.database.models import DatabaseManager

class TestAPIIntegration:
    """Integration tests for the API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_database(self):
        """Mock database for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            tmp_db_path = tmp.name
        
        yield tmp_db_path
        
        # Cleanup
        if os.path.exists(tmp_db_path):
            os.unlink(tmp_db_path)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] == "healthy"
    
    def test_system_status(self, client):
        """Test system status endpoint"""
        response = client.get("/system/status")
        
        # Should return 503 if system not initialized or 200 if initialized
        assert response.status_code in [200, 503]
    
    @patch('main.agents')
    def test_customer_analysis_endpoint(self, mock_agents, client):
        """Test customer analysis API endpoint"""
        # Mock customer agent
        mock_customer_agent = Mock()
        mock_customer_agent.process_request = AsyncMock(return_value={
            "customer_id": "test_customer_001",
            "segment": "tech_enthusiast",
            "behavior_analysis": {
                "engagement_score": 0.8,
                "session_frequency": 5
            }
        })
        mock_agents.get.return_value = mock_customer_agent
        
        request_data = {
            "customer_id": "test_customer_001",
            "interaction_data": {},
            "analysis_type": "comprehensive"
        }
        
        response = client.post("/api/customers/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "customer_id" in data
        assert data["customer_id"] == "test_customer_001"
    
    @patch('main.agents')
    def test_customer_analysis_agent_unavailable(self, mock_agents, client):
        """Test customer analysis when agent is unavailable"""
        mock_agents.get.return_value = None
        
        request_data = {
            "customer_id": "test_customer_001",
            "interaction_data": {}
        }
        
        response = client.post("/api/customers/analyze", json=request_data)
        assert response.status_code == 503
        
        data = response.json()
        assert "detail" in data
        assert "Customer agent not available" in data["detail"]
    
    @patch('main.agents')
    def test_recommendation_generation_endpoint(self, mock_agents, client):
        """Test recommendation generation API endpoint"""
        # Mock recommendation agent
        mock_rec_agent = Mock()
        mock_rec_agent.process_request = AsyncMock(return_value={
            "customer_id": "test_customer_001",
            "recommendations": [
                {
                    "product_id": "prod_001",
                    "name": "Smart Laptop",
                    "confidence": 0.9,
                    "explanation": {
                        "primary_reason": "Based on your preferences",
                        "factors": ["Similar products viewed", "High rating"]
                    }
                }
            ],
            "algorithm_variant": "hybrid"
        })
        mock_agents.get.return_value = mock_rec_agent
        
        request_data = {
            "customer_id": "test_customer_001",
            "context": {"page_type": "homepage"},
            "num_recommendations": 5,
            "include_explanations": True
        }
        
        response = client.post("/api/recommendations/generate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "customer_id" in data
        assert "recommendations" in data
        assert "algorithm_variant" in data
        assert len(data["recommendations"]) >= 1
        assert data["recommendations"][0]["product_id"] == "prod_001"
    
    @patch('main.agents')
    def test_product_analysis_endpoint(self, mock_agents, client):
        """Test product analysis API endpoint"""
        # Mock product agent
        mock_product_agent = Mock()
        mock_product_agent.process_request = AsyncMock(return_value={
            "product_insights": {
                "product_id": "prod_001",
                "name": "Smart Laptop",
                "performance": {
                    "total_views": 1000,
                    "conversion_rate": 0.05
                },
                "content_quality": {
                    "overall_score": 0.85
                }
            },
            "optimization_suggestions": [
                "Improve product description",
                "Add more product images"
            ]
        })
        mock_agents.get.return_value = mock_product_agent
        
        request_data = {
            "product_id": "prod_001",
            "analysis_type": "comprehensive"
        }
        
        response = client.post("/api/products/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "product_insights" in data
        assert "optimization_suggestions" in data
        assert data["product_insights"]["product_id"] == "prod_001"
    
    @patch('main.agents')
    def test_similar_products_endpoint(self, mock_agents, client):
        """Test similar products API endpoint"""
        # Mock product agent
        mock_product_agent = Mock()
        mock_product_agent.process_request = AsyncMock(return_value={
            "target_product_id": "prod_001",
            "similar_products": [
                {
                    "product_id": "prod_002",
                    "name": "Gaming Laptop",
                    "similarity_score": 0.85,
                    "price": 1199.99
                },
                {
                    "product_id": "prod_003",
                    "name": "Business Laptop",
                    "similarity_score": 0.78,
                    "price": 899.99
                }
            ]
        })
        mock_agents.get.return_value = mock_product_agent
        
        response = client.post("/api/products/prod_001/similar?similarity_threshold=0.7&max_results=5")
        assert response.status_code == 200
        
        data = response.json()
        assert "target_product_id" in data
        assert "similar_products" in data
        assert data["target_product_id"] == "prod_001"
        assert len(data["similar_products"]) == 2
    
    @patch('main.agents')
    def test_real_time_personalization_endpoint(self, mock_agents, client):
        """Test real-time personalization API endpoint"""
        # Mock recommendation agent
        mock_rec_agent = Mock()
        mock_rec_agent.process_request = AsyncMock(return_value={
            "customer_id": "test_customer_001",
            "personalized_content": {},
            "real_time_recommendations": [
                {
                    "product_id": "prod_001",
                    "confidence": 0.9,
                    "reasoning": "Based on current browsing"
                }
            ],
            "next_best_actions": ["view_reviews", "add_to_cart"]
        })
        mock_agents.get.return_value = mock_rec_agent
        
        request_data = {
            "customer_id": "test_customer_001",
            "session_context": {
                "page_type": "product_page",
                "product_id": "prod_001"
            },
            "page_context": {"time_on_page": 120}
        }
        
        response = client.post("/api/recommendations/real-time", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "customer_id" in data
        assert "real_time_recommendations" in data
        assert "next_best_actions" in data
    
    @patch('main.agent_coordinator')
    def test_agent_coordination_endpoint(self, mock_coordinator, client):
        """Test agent coordination API endpoint"""
        # Mock coordinator
        mock_coordinator.coordinate_task = AsyncMock(return_value={
            "task_id": "task_001",
            "results": {
                "customer_agent": {"status": "completed", "data": {}},
                "product_agent": {"status": "completed", "data": {}},
                "recommendation_agent": {"status": "completed", "data": {}}
            },
            "status": "completed"
        })
        
        request_data = {
            "task": {
                "type": "personalized_shopping_experience",
                "customer_id": "test_customer_001"
            },
            "required_agents": ["customer_agent", "product_agent", "recommendation_agent"]
        }
        
        response = client.post("/api/agents/coordinate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "task_id" in data
        assert "results" in data
        assert "status" in data
        assert data["status"] == "completed"
    
    @patch('main.agents')
    def test_agent_performance_endpoint(self, mock_agents, client):
        """Test agent performance metrics endpoint"""
        # Mock agent with performance metrics
        mock_agent = Mock()
        mock_agent.get_performance_metrics.return_value = {
            "agent_id": "test_agent",
            "total_requests": 100,
            "success_rate": 0.95,
            "average_response_time": 45.5,
            "average_confidence": 0.87
        }
        mock_agents.get.return_value = mock_agent
        
        response = client.get("/api/agents/test_agent/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert "agent_id" in data
        assert "total_requests" in data
        assert "success_rate" in data
        assert data["agent_id"] == "test_agent"
    
    def test_agent_performance_not_found(self, client):
        """Test agent performance endpoint when agent not found"""
        response = client.get("/api/agents/nonexistent_agent/performance")
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
        assert "Agent not found" in data["detail"]
    
    @patch('main.get_db_manager')
    def test_analytics_overview_endpoint(self, mock_get_db, client):
        """Test analytics overview endpoint"""
        # Mock database manager and session
        mock_db_manager = Mock()
        mock_session = Mock()
        mock_db_manager.get_session.return_value = mock_session
        mock_get_db.return_value = mock_db_manager
        
        # Mock query results
        mock_session.query.return_value.count.return_value = 100  # For counts
        
        response = client.get("/api/analytics/overview")
        assert response.status_code == 200
        
        data = response.json()
        assert "system_metrics" in data
        assert "performance_metrics" in data
        assert "agent_status" in data
        assert "timestamp" in data
    
    def test_feedback_endpoint(self, client):
        """Test recommendation feedback endpoint"""
        request_data = {
            "recommendation_id": "rec_001",
            "customer_id": "test_customer_001",
            "feedback_type": "clicked",
            "feedback_data": {"timestamp": "2024-01-01T10:00:00Z"}
        }
        
        response = client.post("/api/recommendations/feedback", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "recommendation_id" in data
        assert data["status"] == "feedback_received"
        assert data["recommendation_id"] == "rec_001"
    
    def test_demo_data_initialization(self, client):
        """Test demo data initialization endpoint"""
        response = client.post("/api/demo/initialize-sample-data")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert data["status"] == "sample_data_initialized"
    
    @patch('main.agents')
    def test_error_handling_in_endpoints(self, mock_agents, client):
        """Test error handling in API endpoints"""
        # Mock agent that raises an exception
        mock_agent = Mock()
        mock_agent.process_request = AsyncMock(side_effect=Exception("Test error"))
        mock_agents.get.return_value = mock_agent
        
        request_data = {
            "customer_id": "test_customer_001",
            "interaction_data": {}
        }
        
        response = client.post("/api/customers/analyze", json=request_data)
        assert response.status_code == 500
        
        data = response.json()
        assert "detail" in data
    
    def test_request_validation(self, client):
        """Test request validation for API endpoints"""
        # Test missing required fields
        invalid_request = {
            "interaction_data": {}
            # Missing customer_id
        }
        
        response = client.post("/api/customers/analyze", json=invalid_request)
        assert response.status_code == 422  # Validation error
        
        data = response.json()
        assert "detail" in data
    
    @patch('main.agents')
    def test_churn_prediction_endpoint(self, mock_agents, client):
        """Test churn prediction endpoint"""
        # Mock customer agent
        mock_customer_agent = Mock()
        mock_customer_agent.process_request = AsyncMock(return_value={
            "customer_id": "test_customer_001",
            "churn_probability": 0.25,
            "risk_level": "low",
            "risk_factors": ["low_engagement"],
            "retention_strategies": ["personalized_offers"]
        })
        mock_agents.get.return_value = mock_customer_agent
        
        response = client.post("/api/customers/test_customer_001/churn-prediction")
        assert response.status_code == 200
        
        data = response.json()
        assert "customer_id" in data
        assert "churn_probability" in data
        assert "risk_level" in data
        assert data["customer_id"] == "test_customer_001"
    
    @patch('main.agents')
    def test_catalog_optimization_endpoint(self, mock_agents, client):
        """Test catalog optimization endpoint"""
        # Mock product agent
        mock_product_agent = Mock()
        mock_product_agent.process_request = AsyncMock(return_value={
            "category": "electronics",
            "total_products_analyzed": 150,
            "optimization_actions": [
                {
                    "product_id": "prod_001",
                    "action": "improve_seo",
                    "priority": "high"
                }
            ],
            "priority_items": [
                {
                    "product_id": "prod_002",
                    "priority_score": 3,
                    "reasons": ["low_conversion"]
                }
            ]
        })
        mock_agents.get.return_value = mock_product_agent
        
        response = client.post("/api/products/catalog/optimize?category=electronics&optimization_goals=improve_visibility")
        assert response.status_code == 200
        
        data = response.json()
        assert "category" in data
        assert "total_products_analyzed" in data
        assert "optimization_actions" in data

class TestEndToEndScenarios:
    """End-to-end integration test scenarios"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @patch('main.agents')
    @patch('main.agent_coordinator')
    def test_complete_recommendation_flow(self, mock_coordinator, mock_agents, client):
        """Test complete recommendation flow from customer analysis to recommendation"""
        # Mock all agents
        mock_customer_agent = Mock()
        mock_product_agent = Mock()
        mock_rec_agent = Mock()
        
        mock_customer_agent.process_request = AsyncMock(return_value={
            "customer_id": "test_customer_001",
            "segment": "tech_enthusiast"
        })
        
        mock_rec_agent.process_request = AsyncMock(return_value={
            "customer_id": "test_customer_001",
            "recommendations": [{"product_id": "prod_001", "confidence": 0.9}]
        })
        
        def mock_agent_get(agent_type):
            if agent_type == "customer_agent":
                return mock_customer_agent
            elif agent_type == "recommendation_agent":
                return mock_rec_agent
            elif agent_type == "product_agent":
                return mock_product_agent
            return None
        
        mock_agents.get.side_effect = mock_agent_get
        
        # Step 1: Analyze customer
        response1 = client.post("/api/customers/analyze", json={
            "customer_id": "test_customer_001",
            "interaction_data": {}
        })
        assert response1.status_code == 200
        
        # Step 2: Generate recommendations
        response2 = client.post("/api/recommendations/generate", json={
            "customer_id": "test_customer_001",
            "context": {"page_type": "homepage"},
            "num_recommendations": 5
        })
        assert response2.status_code == 200
        
        # Verify data flow
        customer_data = response1.json()
        rec_data = response2.json()
        
        assert customer_data["customer_id"] == rec_data["customer_id"]
        assert "recommendations" in rec_data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])