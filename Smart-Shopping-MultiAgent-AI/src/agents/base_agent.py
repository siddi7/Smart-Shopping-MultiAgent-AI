"""
Advanced Base Agent Framework for Smart Shopping Multi-Agent AI System
Implements core agent functionality with memory, learning, and coordination capabilities
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sqlalchemy.orm import Session

from ..database.models import AgentMemory, DatabaseManager
from ..utils.config import settings

class AgentType(Enum):
    """Enumeration of different agent types in the system"""
    CUSTOMER = "customer_agent"
    PRODUCT = "product_agent"
    RECOMMENDATION = "recommendation_agent"
    PRICING = "pricing_agent"
    ANALYTICS = "analytics_agent"
    COORDINATOR = "coordinator_agent"

class MessageType(Enum):
    """Types of messages agents can send/receive"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    COORDINATION = "coordination"
    LEARNING_UPDATE = "learning_update"

@dataclass
class AgentMessage:
    """Structured message format for agent communication"""
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    priority: int = 1  # 1=low, 5=critical

@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    confidence_level: float

class BaseAgent(ABC):
    """
    Advanced base agent class with memory, learning, and coordination capabilities
    
    Key Features:
    - Long-term memory storage and retrieval
    - Continuous learning from interactions
    - Explainable decision making
    - Real-time performance monitoring
    - Federated learning support
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        db_manager: DatabaseManager,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.db_manager = db_manager
        self.config = config or {}
        
        # Initialize logging
        self.logger = logging.getLogger(f"{agent_type.value}_{agent_id}")
        self.logger.setLevel(getattr(logging, settings.LOG_LEVEL))
        
        # Memory and learning
        self.short_term_memory: Dict[str, Any] = {}
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.memory_capacity = self.config.get("memory_capacity", 1000)
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_responses": 0,
            "average_response_time": 0.0,
            "confidence_scores": [],
            "last_update": datetime.utcnow()
        }
        
        # Coordination
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.active_conversations: Dict[str, List[AgentMessage]] = {}
        
        # AI Model components
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        
        # Initialize agent-specific components
        self._initialize_agent()
    
    @abstractmethod
    def _initialize_agent(self):
        """Initialize agent-specific components"""
        pass
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request and return response"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of agent capabilities"""
        pass
    
    async def initialize_models(self):
        """Initialize AI models for the agent"""
        try:
            if not self.embedding_model:
                model_name = settings.EMBEDDING_MODEL
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.embedding_model = AutoModel.from_pretrained(model_name)
                self.logger.info(f"Initialized embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
    
    async def store_memory(
        self,
        memory_type: str,
        content: Dict[str, Any],
        customer_id: Optional[str] = None,
        product_id: Optional[str] = None,
        importance_score: float = 0.5,
        expires_at: Optional[datetime] = None
    ) -> str:
        """Store information in long-term memory"""
        memory_id = f"{self.agent_id}_{datetime.utcnow().timestamp()}"
        
        # Create embedding for content
        embedding_vector = await self._create_embedding(str(content))
        
        memory = AgentMemory(
            memory_id=memory_id,
            agent_type=self.agent_type.value,
            agent_instance_id=self.agent_id,
            memory_type=memory_type,
            content=content,
            embedding_vector=json.dumps(embedding_vector.tolist()) if embedding_vector is not None else None,
            customer_id=customer_id,
            product_id=product_id,
            importance_score=importance_score,
            expires_at=expires_at
        )
        
        session = self.db_manager.get_session()
        try:
            session.add(memory)
            session.commit()
            self.logger.info(f"Stored memory: {memory_id}")
            return memory_id
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to store memory: {str(e)}")
            raise
        finally:
            session.close()
    
    async def retrieve_memory(
        self,
        query: str,
        memory_type: Optional[str] = None,
        customer_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on semantic similarity"""
        query_embedding = await self._create_embedding(query)
        
        session = self.db_manager.get_session()
        try:
            query_obj = session.query(AgentMemory).filter(
                AgentMemory.agent_type == self.agent_type.value,
                AgentMemory.is_active == True
            )
            
            if memory_type:
                query_obj = query_obj.filter(AgentMemory.memory_type == memory_type)
            if customer_id:
                query_obj = query_obj.filter(AgentMemory.customer_id == customer_id)
            
            memories = query_obj.order_by(AgentMemory.importance_score.desc()).limit(limit).all()
            
            # Calculate similarity scores
            relevant_memories = []
            for memory in memories:
                if memory.embedding_vector:
                    memory_embedding = np.array(json.loads(memory.embedding_vector))
                    similarity = self._calculate_similarity(query_embedding, memory_embedding)
                    
                    if similarity > 0.5:  # Threshold for relevance
                        relevant_memories.append({
                            "memory_id": memory.memory_id,
                            "content": memory.content,
                            "similarity": similarity,
                            "importance_score": memory.importance_score,
                            "created_date": memory.created_date
                        })
            
            # Sort by combined relevance score
            relevant_memories.sort(
                key=lambda x: x["similarity"] * x["importance_score"], 
                reverse=True
            )
            
            return relevant_memories[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory: {str(e)}")
            return []
        finally:
            session.close()
    
    async def _create_embedding(self, text: str) -> Optional[np.ndarray]:
        """Create embedding for text using the agent's model"""
        try:
            if not self.embedding_model:
                await self.initialize_models()
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to create embedding: {str(e)}")
            return None
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0
    
    async def send_message(self, message: AgentMessage):
        """Send message to another agent"""
        try:
            # Store in conversation history
            if message.correlation_id not in self.active_conversations:
                self.active_conversations[message.correlation_id] = []
            self.active_conversations[message.correlation_id].append(message)
            
            # Log the message
            self.logger.info(f"Sent message to {message.receiver_id}: {message.message_type.value}")
            
            # In a real system, this would use a message broker (Redis, RabbitMQ, etc.)
            # For now, we'll simulate by adding to receiver's queue
            await self._deliver_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {str(e)}")
    
    async def _deliver_message(self, message: AgentMessage):
        """Simulate message delivery (in real system, use message broker)"""
        # This would be handled by a message broker in production
        await self.message_queue.put(message)
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """Receive message from queue"""
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            self.logger.info(f"Received message from {message.sender_id}: {message.message_type.value}")
            return message
        except asyncio.TimeoutError:
            return None
    
    async def learn_from_interaction(
        self,
        interaction_data: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None
    ):
        """Learn and adapt from user interactions"""
        try:
            # Update performance metrics
            self.performance_metrics["total_requests"] += 1
            
            if feedback and feedback.get("success", False):
                self.performance_metrics["successful_responses"] += 1
            
            # Store interaction as memory for future reference
            await self.store_memory(
                memory_type="interaction",
                content={
                    "interaction": interaction_data,
                    "feedback": feedback,
                    "timestamp": datetime.utcnow().isoformat()
                },
                importance_score=0.7 if feedback and feedback.get("success") else 0.3
            )
            
            # Update short-term memory
            self.short_term_memory[f"interaction_{datetime.utcnow().timestamp()}"] = {
                "data": interaction_data,
                "feedback": feedback
            }
            
            # Cleanup old short-term memories
            await self._cleanup_short_term_memory()
            
            self.logger.info("Learned from interaction")
            
        except Exception as e:
            self.logger.error(f"Failed to learn from interaction: {str(e)}")
    
    async def _cleanup_short_term_memory(self):
        """Remove old entries from short-term memory"""
        if len(self.short_term_memory) > self.memory_capacity:
            # Remove oldest entries
            sorted_keys = sorted(
                self.short_term_memory.keys(),
                key=lambda k: k.split("_")[-1]
            )
            for key in sorted_keys[:-self.memory_capacity]:
                del self.short_term_memory[key]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        success_rate = 0.0
        if self.performance_metrics["total_requests"] > 0:
            success_rate = (
                self.performance_metrics["successful_responses"] / 
                self.performance_metrics["total_requests"]
            )
        
        avg_confidence = 0.0
        if self.performance_metrics["confidence_scores"]:
            avg_confidence = np.mean(self.performance_metrics["confidence_scores"])
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "total_requests": self.performance_metrics["total_requests"],
            "success_rate": success_rate,
            "average_response_time": self.performance_metrics["average_response_time"],
            "average_confidence": avg_confidence,
            "last_update": self.performance_metrics["last_update"],
            "active_conversations": len(self.active_conversations),
            "memory_usage": len(self.short_term_memory)
        }
    
    async def explain_decision(
        self,
        decision: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide explanation for agent's decision (Explainable AI)"""
        try:
            explanation = {
                "agent_id": self.agent_id,
                "decision": decision,
                "reasoning": [],
                "confidence": decision.get("confidence", 0.5),
                "factors": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add agent-specific explanations
            explanation.update(await self._generate_explanation(decision, context))
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Failed to generate explanation: {str(e)}")
            return {
                "error": "Failed to generate explanation",
                "decision": decision
            }
    
    @abstractmethod
    async def _generate_explanation(
        self,
        decision: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate agent-specific explanation"""
        pass
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        try:
            # Save any important state
            await self._save_state()
            
            # Clear queues
            while not self.message_queue.empty():
                await self.message_queue.get()
            
            self.logger.info(f"Agent {self.agent_id} shut down gracefully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    async def _save_state(self):
        """Save agent state before shutdown"""
        state = {
            "agent_id": self.agent_id,
            "performance_metrics": self.performance_metrics,
            "config": self.config,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.store_memory(
            memory_type="agent_state",
            content=state,
            importance_score=1.0  # High importance for state
        )

class AgentCoordinator:
    """
    Coordinates communication and collaboration between agents
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.agents: Dict[str, BaseAgent] = {}
        self.message_router: Dict[str, asyncio.Queue] = {}
        self.logger = logging.getLogger("AgentCoordinator")
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator"""
        self.agents[agent.agent_id] = agent
        self.message_router[agent.agent_id] = agent.message_queue
        self.logger.info(f"Registered agent: {agent.agent_id}")
    
    async def route_message(self, message: AgentMessage):
        """Route message to appropriate agent"""
        if message.receiver_id in self.message_router:
            await self.message_router[message.receiver_id].put(message)
        else:
            self.logger.warning(f"Unknown receiver: {message.receiver_id}")
    
    async def coordinate_task(
        self,
        task: Dict[str, Any],
        required_agents: List[str]
    ) -> Dict[str, Any]:
        """Coordinate a task across multiple agents"""
        task_id = f"task_{datetime.utcnow().timestamp()}"
        results = {}
        
        try:
            # Send task to required agents
            tasks = []
            for agent_id in required_agents:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    task_coroutine = agent.process_request(task)
                    tasks.append((agent_id, task_coroutine))
            
            # Wait for all agents to complete
            completed_tasks = await asyncio.gather(
                *[task_coro for _, task_coro in tasks],
                return_exceptions=True
            )
            
            # Collect results
            for i, (agent_id, _) in enumerate(tasks):
                result = completed_tasks[i]
                if isinstance(result, Exception):
                    results[agent_id] = {"error": str(result)}
                else:
                    results[agent_id] = result
            
            self.logger.info(f"Completed coordinated task: {task_id}")
            return {
                "task_id": task_id,
                "results": results,
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate task: {str(e)}")
            return {
                "task_id": task_id,
                "error": str(e),
                "status": "failed"
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent.get_performance_metrics()
        
        return {
            "total_agents": len(self.agents),
            "agent_statuses": agent_statuses,
            "timestamp": datetime.utcnow().isoformat()
        }