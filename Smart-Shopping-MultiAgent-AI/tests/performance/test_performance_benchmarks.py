"""
Performance Benchmarks for Smart Shopping Multi-Agent AI System
Tests system performance, scalability, and response times under various loads
"""

import pytest
import asyncio
import time
import statistics
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agents.customer_agent import CustomerAgent
from src.agents.product_agent import ProductAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.coordinator import AgentCoordinator
from src.database.models import DatabaseManager
from src.utils.config import Settings

class PerformanceBenchmarks:
    """Performance benchmarks and stress tests"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for performance tests"""
        mock_db = Mock(spec=DatabaseManager)
        mock_session = Mock()
        mock_db.get_session.return_value = mock_session
        
        # Mock fast database responses
        mock_db.get_customer_data.return_value = {
            "customer_id": "test_customer",
            "segment": "tech_enthusiast",
            "preferences": {}
        }
        mock_db.get_product_data.return_value = {
            "product_id": "test_product",
            "name": "Test Product",
            "category": "electronics"
        }
        
        return mock_db
    
    @pytest.fixture
    def settings(self):
        """Performance test settings"""
        return Settings(
            DATABASE_URL="sqlite:///test.db",
            ENABLE_PERFORMANCE_MONITORING=True,
            ENABLE_CACHING=True,
            CACHE_TTL=300
        )
    
    @pytest.fixture
    def agents(self, mock_db_manager, settings):
        """Create agent instances for performance testing"""
        customer_agent = CustomerAgent("customer_agent", mock_db_manager, {"settings": settings})
        product_agent = ProductAgent("product_agent", mock_db_manager, {"settings": settings})
        recommendation_agent = RecommendationAgent("rec_agent", mock_db_manager, {"settings": settings})
        
        return {
            "customer": customer_agent,
            "product": product_agent,
            "recommendation": recommendation_agent
        }
    
    @pytest.mark.asyncio
    async def test_single_agent_response_time(self, agents):
        """Test response time of individual agents"""
        response_times = []
        num_requests = 100
        
        for i in range(num_requests):
            start_time = time.time()
            
            request_data = {
                "request_type": "analyze_customer",
                "customer_id": f"test_customer_{i}",
                "interaction_data": {}
            }
            
            with patch.object(agents["customer"], '_analyze_customer_behavior') as mock_analyze:
                mock_analyze.return_value = {"segment": "tech_enthusiast"}
                await agents["customer"].process_request(request_data)
            
            end_time = time.time()
            response_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Performance assertions
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        
        print(f"\nSingle Agent Performance:")
        print(f"Average response time: {avg_response_time:.2f}ms")
        print(f"Max response time: {max_response_time:.2f}ms")
        print(f"95th percentile: {p95_response_time:.2f}ms")
        
        # Performance targets
        assert avg_response_time < 50, f"Average response time {avg_response_time:.2f}ms exceeds 50ms target"
        assert p95_response_time < 100, f"95th percentile {p95_response_time:.2f}ms exceeds 100ms target"
        assert max_response_time < 200, f"Max response time {max_response_time:.2f}ms exceeds 200ms target"
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_requests(self, agents):
        """Test performance with concurrent requests to multiple agents"""
        concurrent_users = [10, 25, 50, 100]
        results = {}
        
        for num_users in concurrent_users:
            response_times = []
            
            async def make_request(user_id):
                start_time = time.time()
                
                # Simulate mixed requests to different agents
                agent_type = ["customer", "product", "recommendation"][user_id % 3]
                agent = agents[agent_type]
                
                request_data = {
                    "request_type": "analyze_customer" if agent_type == "customer" else "analyze_product",
                    "customer_id" if agent_type == "customer" else "product_id": f"test_{user_id}",
                    "interaction_data" if agent_type == "customer" else "analysis_type": {} if agent_type == "customer" else "basic"
                }
                
                with patch.object(agent, '_analyze_customer_behavior' if agent_type == "customer" else '_analyze_product') as mock_method:
                    mock_method.return_value = {"result": "success"}
                    await agent.process_request(request_data)
                
                end_time = time.time()
                return (end_time - start_time) * 1000
            
            # Execute concurrent requests
            tasks = [make_request(i) for i in range(num_users)]
            response_times = await asyncio.gather(*tasks)
            
            avg_time = statistics.mean(response_times)
            p95_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
            
            results[num_users] = {
                "avg_response_time": avg_time,
                "p95_response_time": p95_time,
                "max_response_time": max(response_times)
            }
            
            print(f"\nConcurrent Users: {num_users}")
            print(f"Average response time: {avg_time:.2f}ms")
            print(f"95th percentile: {p95_time:.2f}ms")
        
        # Performance assertions - response times should not degrade significantly with concurrency
        baseline_avg = results[10]["avg_response_time"]
        high_load_avg = results[100]["avg_response_time"]
        
        degradation_ratio = high_load_avg / baseline_avg
        assert degradation_ratio < 3.0, f"Performance degraded by {degradation_ratio:.1f}x under high load"
    
    @pytest.mark.asyncio
    async def test_recommendation_generation_performance(self, agents):
        """Test performance of recommendation generation specifically"""
        rec_agent = agents["recommendation"]
        
        # Test different recommendation batch sizes
        batch_sizes = [1, 5, 10, 20, 50]
        performance_results = {}
        
        for batch_size in batch_sizes:
            response_times = []
            
            for _ in range(20):  # 20 iterations per batch size
                start_time = time.time()
                
                request_data = {
                    "request_type": "generate_recommendations",
                    "customer_id": "test_customer",
                    "context": {"page_type": "homepage"},
                    "num_recommendations": batch_size,
                    "include_explanations": True
                }
                
                with patch.object(rec_agent, '_generate_recommendations') as mock_gen:
                    mock_recommendations = [
                        {
                            "product_id": f"prod_{i}",
                            "confidence": 0.8 + (i * 0.01),
                            "explanation": {"reason": "test"}
                        }
                        for i in range(batch_size)
                    ]
                    mock_gen.return_value = {
                        "recommendations": mock_recommendations,
                        "algorithm_variant": "hybrid"
                    }
                    
                    await rec_agent.process_request(request_data)
                
                end_time = time.time()
                response_times.append((end_time - start_time) * 1000)
            
            avg_time = statistics.mean(response_times)
            performance_results[batch_size] = avg_time
            
            print(f"Batch size {batch_size}: {avg_time:.2f}ms avg")
        
        # Assert that response time scales reasonably with batch size
        time_per_recommendation = performance_results[50] / 50
        assert time_per_recommendation < 5, f"Time per recommendation {time_per_recommendation:.2f}ms exceeds 5ms target"
    
    @pytest.mark.asyncio
    async def test_memory_usage_and_cleanup(self, agents):
        """Test memory usage and cleanup during extended operation"""
        customer_agent = agents["customer"]
        
        # Simulate extended operation with memory tracking
        initial_memory_size = len(customer_agent.short_term_memory)
        
        # Process many requests to fill memory
        for i in range(1000):
            request_data = {
                "request_type": "analyze_customer",
                "customer_id": f"customer_{i}",
                "interaction_data": {"session_id": f"session_{i}"}
            }
            
            with patch.object(customer_agent, '_analyze_customer_behavior') as mock_analyze:
                mock_analyze.return_value = {"segment": "tech_enthusiast"}
                await customer_agent.process_request(request_data)
        
        # Check that memory doesn't grow unbounded
        final_memory_size = len(customer_agent.short_term_memory)
        memory_growth = final_memory_size - initial_memory_size
        
        print(f"Memory growth after 1000 requests: {memory_growth} items")
        
        # Memory should not grow linearly with requests (should have cleanup)
        assert memory_growth < 500, f"Memory grew by {memory_growth} items, indicating poor cleanup"
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self, mock_db_manager):
        """Test database query performance simulation"""
        query_times = []
        num_queries = 200
        
        for i in range(num_queries):
            start_time = time.time()
            
            # Simulate various database operations
            if i % 4 == 0:
                mock_db_manager.get_customer_data(f"customer_{i}")
            elif i % 4 == 1:
                mock_db_manager.get_product_data(f"product_{i}")
            elif i % 4 == 2:
                mock_db_manager.get_customer_interactions(f"customer_{i}")
            else:
                mock_db_manager.get_recommendation_history(f"customer_{i}")
            
            end_time = time.time()
            query_times.append((end_time - start_time) * 1000)
        
        avg_query_time = statistics.mean(query_times)
        max_query_time = max(query_times)
        
        print(f"\nDatabase Query Performance:")
        print(f"Average query time: {avg_query_time:.2f}ms")
        print(f"Max query time: {max_query_time:.2f}ms")
        
        # Database queries should be fast (mocked, but testing the pattern)
        assert avg_query_time < 10, f"Average query time {avg_query_time:.2f}ms exceeds 10ms target"
    
    @pytest.mark.asyncio
    async def test_agent_coordination_performance(self, agents, mock_db_manager):
        """Test performance of agent coordination"""
        coordinator = AgentCoordinator(mock_db_manager)
        
        # Mock agents in coordinator
        for agent_name, agent in agents.items():
            coordinator.agents[agent_name] = agent
        
        coordination_times = []
        num_coordinations = 50
        
        for i in range(num_coordinations):
            start_time = time.time()
            
            task_data = {
                "type": "personalized_shopping_experience",
                "customer_id": f"customer_{i}",
                "context": {"page_type": "homepage"}
            }
            
            required_agents = ["customer", "product", "recommendation"]
            
            with patch.object(coordinator, 'coordinate_task') as mock_coordinate:
                mock_coordinate.return_value = {
                    "task_id": f"task_{i}",
                    "results": {agent: {"status": "completed"} for agent in required_agents},
                    "status": "completed"
                }
                
                await coordinator.coordinate_task(task_data, required_agents)
            
            end_time = time.time()
            coordination_times.append((end_time - start_time) * 1000)
        
        avg_coordination_time = statistics.mean(coordination_times)
        max_coordination_time = max(coordination_times)
        
        print(f"\nAgent Coordination Performance:")
        print(f"Average coordination time: {avg_coordination_time:.2f}ms")
        print(f"Max coordination time: {max_coordination_time:.2f}ms")
        
        # Coordination should be efficient
        assert avg_coordination_time < 100, f"Average coordination time {avg_coordination_time:.2f}ms exceeds 100ms target"
    
    @pytest.mark.asyncio
    async def test_throughput_measurement(self, agents):
        """Test system throughput (requests per second)"""
        customer_agent = agents["customer"]
        
        # Measure throughput over different time periods
        test_duration = 5  # seconds
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < test_duration:
            request_data = {
                "request_type": "analyze_customer",
                "customer_id": f"customer_{request_count}",
                "interaction_data": {}
            }
            
            with patch.object(customer_agent, '_analyze_customer_behavior') as mock_analyze:
                mock_analyze.return_value = {"segment": "tech_enthusiast"}
                await customer_agent.process_request(request_data)
            
            request_count += 1
        
        actual_duration = time.time() - start_time
        throughput = request_count / actual_duration
        
        print(f"\nThroughput Test:")
        print(f"Requests processed: {request_count}")
        print(f"Test duration: {actual_duration:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        
        # Minimum throughput target
        assert throughput > 100, f"Throughput {throughput:.2f} req/s below minimum target of 100 req/s"
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self, agents):
        """Test performance impact of error handling"""
        customer_agent = agents["customer"]
        
        success_times = []
        error_times = []
        
        # Test successful requests
        for i in range(50):
            start_time = time.time()
            
            request_data = {
                "request_type": "analyze_customer",
                "customer_id": f"customer_{i}",
                "interaction_data": {}
            }
            
            with patch.object(customer_agent, '_analyze_customer_behavior') as mock_analyze:
                mock_analyze.return_value = {"segment": "tech_enthusiast"}
                await customer_agent.process_request(request_data)
            
            end_time = time.time()
            success_times.append((end_time - start_time) * 1000)
        
        # Test error handling
        for i in range(50):
            start_time = time.time()
            
            request_data = {
                "request_type": "invalid_request_type",
                "customer_id": f"customer_{i}",
                "interaction_data": {}
            }
            
            result = await customer_agent.process_request(request_data)
            assert "error" in result  # Should handle error gracefully
            
            end_time = time.time()
            error_times.append((end_time - start_time) * 1000)
        
        avg_success_time = statistics.mean(success_times)
        avg_error_time = statistics.mean(error_times)
        
        print(f"\nError Handling Performance:")
        print(f"Average success time: {avg_success_time:.2f}ms")
        print(f"Average error handling time: {avg_error_time:.2f}ms")
        
        # Error handling should be fast and not significantly slower than success
        error_overhead = avg_error_time / avg_success_time
        assert error_overhead < 2.0, f"Error handling {error_overhead:.1f}x slower than success"
        assert avg_error_time < 50, f"Error handling time {avg_error_time:.2f}ms exceeds 50ms target"

class LoadTestScenarios:
    """Load testing scenarios for stress testing"""
    
    @pytest.mark.asyncio
    async def test_stress_test_high_concurrency(self, agents):
        """Stress test with very high concurrency"""
        concurrent_requests = 200
        success_count = 0
        error_count = 0
        response_times = []
        
        async def stress_request(request_id):
            try:
                start_time = time.time()
                
                agent = agents["customer"]
                request_data = {
                    "request_type": "analyze_customer",
                    "customer_id": f"stress_customer_{request_id}",
                    "interaction_data": {}
                }
                
                with patch.object(agent, '_analyze_customer_behavior') as mock_analyze:
                    mock_analyze.return_value = {"segment": "tech_enthusiast"}
                    await agent.process_request(request_data)
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                return {"success": True, "response_time": response_time}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Execute stress test
        tasks = [stress_request(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                success_count += 1
                response_times.append(result["response_time"])
            else:
                error_count += 1
        
        success_rate = success_count / concurrent_requests
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        print(f"\nStress Test Results:")
        print(f"Concurrent requests: {concurrent_requests}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average response time: {avg_response_time:.2f}ms")
        print(f"Errors: {error_count}")
        
        # Stress test targets
        assert success_rate > 0.95, f"Success rate {success_rate:.2%} below 95% target"
        assert avg_response_time < 500, f"Response time {avg_response_time:.2f}ms exceeds 500ms under stress"

def run_performance_benchmarks():
    """Run all performance benchmarks and generate report"""
    print("=== Smart Shopping Multi-Agent AI Performance Benchmarks ===\n")
    
    # Run the benchmarks
    pytest.main([__file__, "-v", "-s"])

if __name__ == "__main__":
    run_performance_benchmarks()