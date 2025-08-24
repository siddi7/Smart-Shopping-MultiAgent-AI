"""
Smart Shopping Multi-Agent AI System - Main Application
Advanced FastAPI application with comprehensive API endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uvicorn
import asyncio
import logging
from datetime import datetime
import json

from config.settings import settings
from src.database.models import DatabaseManager, get_database_manager
from src.agents.base_agent import AgentCoordinator
from src.agents.customer_agent import CustomerAgent
from src.agents.product_agent import ProductAgent
from src.agents.recommendation_agent import RecommendationAgent

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Advanced Smart Shopping Multi-Agent AI System for E-commerce Personalization",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
db_manager: DatabaseManager = None
agent_coordinator: AgentCoordinator = None
agents: Dict[str, Any] = {}

# Pydantic models for API requests/responses
class CustomerAnalysisRequest(BaseModel):
    customer_id: str
    interaction_data: Optional[Dict[str, Any]] = {}
    analysis_type: str = "comprehensive"

class RecommendationRequest(BaseModel):
    customer_id: str
    context: Optional[Dict[str, Any]] = {}
    num_recommendations: int = Field(default=10, ge=1, le=50)
    include_explanations: bool = True

class ProductAnalysisRequest(BaseModel):
    product_id: str
    analysis_type: str = "comprehensive"

class PersonalizationRequest(BaseModel):
    customer_id: str
    session_context: Dict[str, Any]
    page_context: Optional[Dict[str, Any]] = {}

class FeedbackRequest(BaseModel):
    recommendation_id: str
    customer_id: str
    feedback_type: str  # clicked, purchased, ignored
    feedback_data: Optional[Dict[str, Any]] = {}

# Dependency to get database manager
async def get_db_manager():
    global db_manager
    if db_manager is None:
        db_manager = get_database_manager(settings.DATABASE_URL)
        db_manager.create_tables()
    return db_manager

# Startup event
@app.on_event("startup")
async def startup_event():
    global db_manager, agent_coordinator, agents
    
    logger.info("Starting Smart Shopping Multi-Agent AI System")
    
    # Initialize database
    db_manager = await get_db_manager()
    
    # Initialize agent coordinator
    agent_coordinator = AgentCoordinator(db_manager)
    
    # Initialize agents
    agents = {
        "customer_agent": CustomerAgent("customer_001", db_manager),
        "product_agent": ProductAgent("product_001", db_manager),
        "recommendation_agent": RecommendationAgent("recommendation_001", db_manager)
    }
    
    # Register agents with coordinator
    for agent in agents.values():
        agent_coordinator.register_agent(agent)
    
    # Initialize agent models
    for agent in agents.values():
        await agent.initialize_models()
    
    logger.info("System initialized successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    global agents
    
    logger.info("Shutting down Smart Shopping Multi-Agent AI System")
    
    # Gracefully shutdown agents
    for agent in agents.values():
        await agent.shutdown()
    
    logger.info("System shutdown complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION,
        "agents_status": {
            agent_id: agent.get_performance_metrics()
            for agent_id, agent in agents.items()
        }
    }

# System status endpoint
@app.get("/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    global agent_coordinator
    
    if agent_coordinator:
        return agent_coordinator.get_system_status()
    else:
        raise HTTPException(status_code=503, detail="System not initialized")

# Customer Analysis Endpoints
@app.post("/api/customers/analyze")
async def analyze_customer(request: CustomerAnalysisRequest):
    """Analyze customer behavior and preferences"""
    try:
        customer_agent = agents.get("customer_agent")
        if not customer_agent:
            raise HTTPException(status_code=503, detail="Customer agent not available")
        
        result = await customer_agent.process_request({
            "type": "analyze_customer",
            "customer_id": request.customer_id,
            "interaction_data": request.interaction_data,
            "analysis_type": request.analysis_type
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in customer analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/customers/segment")
async def segment_customers():
    """Segment customers based on behavior"""
    try:
        customer_agent = agents.get("customer_agent")
        if not customer_agent:
            raise HTTPException(status_code=503, detail="Customer agent not available")
        
        result = await customer_agent.process_request({
            "type": "segment_customers"
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in customer segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/customers/{customer_id}/personalize")
async def personalize_experience(customer_id: str, request: PersonalizationRequest):
    """Generate personalized experience for customer"""
    try:
        customer_agent = agents.get("customer_agent")
        if not customer_agent:
            raise HTTPException(status_code=503, detail="Customer agent not available")
        
        result = await customer_agent.process_request({
            "type": "personalize_experience",
            "customer_id": customer_id,
            "context": request.session_context
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in personalization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/customers/{customer_id}/churn-prediction")
async def predict_churn(customer_id: str):
    """Predict customer churn probability"""
    try:
        customer_agent = agents.get("customer_agent")
        if not customer_agent:
            raise HTTPException(status_code=503, detail="Customer agent not available")
        
        result = await customer_agent.process_request({
            "type": "predict_churn",
            "customer_id": customer_id
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in churn prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Product Analysis Endpoints
@app.post("/api/products/analyze")
async def analyze_product(request: ProductAnalysisRequest):
    """Analyze product performance and content"""
    try:
        product_agent = agents.get("product_agent")
        if not product_agent:
            raise HTTPException(status_code=503, detail="Product agent not available")
        
        result = await product_agent.process_request({
            "type": "analyze_product",
            "product_id": request.product_id,
            "analysis_type": request.analysis_type
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in product analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/products/{product_id}/similar")
async def find_similar_products(product_id: str, similarity_threshold: float = 0.7, max_results: int = 10):
    """Find similar products"""
    try:
        product_agent = agents.get("product_agent")
        if not product_agent:
            raise HTTPException(status_code=503, detail="Product agent not available")
        
        result = await product_agent.process_request({
            "type": "find_similar_products",
            "product_id": product_id,
            "similarity_threshold": similarity_threshold,
            "max_results": max_results
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error finding similar products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/products/catalog/optimize")
async def optimize_catalog(category: Optional[str] = None, optimization_goals: List[str] = ["improve_visibility"]):
    """Optimize product catalog"""
    try:
        product_agent = agents.get("product_agent")
        if not product_agent:
            raise HTTPException(status_code=503, detail="Product agent not available")
        
        result = await product_agent.process_request({
            "type": "optimize_catalog",
            "category": category,
            "optimization_goals": optimization_goals
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error optimizing catalog: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Recommendation Endpoints
@app.post("/api/recommendations/generate")
async def generate_recommendations(request: RecommendationRequest):
    """Generate personalized recommendations"""
    try:
        recommendation_agent = agents.get("recommendation_agent")
        if not recommendation_agent:
            raise HTTPException(status_code=503, detail="Recommendation agent not available")
        
        result = await recommendation_agent.process_request({
            "type": "generate_recommendations",
            "customer_id": request.customer_id,
            "context": request.context,
            "num_recommendations": request.num_recommendations,
            "include_explanations": request.include_explanations
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommendations/real-time")
async def real_time_personalization(request: PersonalizationRequest):
    """Get real-time personalized content"""
    try:
        recommendation_agent = agents.get("recommendation_agent")
        if not recommendation_agent:
            raise HTTPException(status_code=503, detail="Recommendation agent not available")
        
        result = await recommendation_agent.process_request({
            "type": "real_time_personalization",
            "customer_id": request.customer_id,
            "session_context": request.session_context,
            "page_context": request.page_context
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in real-time personalization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommendations/{recommendation_id}/explain")
async def explain_recommendation(recommendation_id: str, customer_id: str):
    """Explain a specific recommendation"""
    try:
        recommendation_agent = agents.get("recommendation_agent")
        if not recommendation_agent:
            raise HTTPException(status_code=503, detail="Recommendation agent not available")
        
        result = await recommendation_agent.process_request({
            "type": "explain_recommendation",
            "recommendation_id": recommendation_id,
            "customer_id": customer_id
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error explaining recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommendations/feedback")
async def submit_recommendation_feedback(request: FeedbackRequest):
    """Submit feedback on recommendation performance"""
    try:
        # Process feedback and update recommendation models
        # This would typically involve updating the recommendation agent's learning
        
        feedback_data = {
            "recommendation_id": request.recommendation_id,
            "customer_id": request.customer_id,
            "feedback_type": request.feedback_type,
            "feedback_data": request.feedback_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store feedback in database and update agent learning
        # Implementation would go here
        
        return {
            "status": "feedback_received",
            "recommendation_id": request.recommendation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Agent Coordination Endpoints
@app.post("/api/agents/coordinate")
async def coordinate_agents(task: Dict[str, Any], required_agents: List[str]):
    """Coordinate a task across multiple agents"""
    try:
        global agent_coordinator
        
        if not agent_coordinator:
            raise HTTPException(status_code=503, detail="Agent coordinator not available")
        
        result = await agent_coordinator.coordinate_task(task, required_agents)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in agent coordination: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/{agent_id}/performance")
async def get_agent_performance(agent_id: str):
    """Get performance metrics for a specific agent"""
    try:
        agent = agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return agent.get_performance_metrics()
        
    except Exception as e:
        logger.error(f"Error getting agent performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics and Insights Endpoints
@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get system analytics overview"""
    try:
        overview = {
            "system_metrics": {
                "total_customers": 0,
                "total_products": 0,
                "total_interactions": 0,
                "total_recommendations": 0
            },
            "performance_metrics": {
                "average_response_time": 0.0,
                "success_rate": 0.0,
                "recommendation_accuracy": 0.0
            },
            "agent_status": {
                agent_id: agent.get_performance_metrics()
                for agent_id, agent in agents.items()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Get actual metrics from database
        db = await get_db_manager()
        session = db.get_session()
        
        try:
            from src.database.models import Customer, Product, CustomerInteraction, RecommendationHistory
            
            overview["system_metrics"]["total_customers"] = session.query(Customer).count()
            overview["system_metrics"]["total_products"] = session.query(Product).count()
            overview["system_metrics"]["total_interactions"] = session.query(CustomerInteraction).count()
            overview["system_metrics"]["total_recommendations"] = session.query(RecommendationHistory).count()
            
        finally:
            session.close()
        
        return overview
        
    except Exception as e:
        logger.error(f"Error getting analytics overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Demo and Testing Endpoints
@app.post("/api/demo/initialize-sample-data")
async def initialize_sample_data():
    """Initialize sample data for demonstration"""
    try:
        # This would create sample customers, products, and interactions
        # Implementation would go here
        
        return {
            "status": "sample_data_initialized",
            "message": "Sample data has been created for demonstration",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error initializing sample data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )