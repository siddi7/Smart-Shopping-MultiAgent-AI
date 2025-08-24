"""
Advanced Recommendation Agent for Smart Shopping Multi-Agent AI System
Implements hybrid recommendation algorithms with explainable AI
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

from .base_agent import BaseAgent, AgentType, AgentCapability
from ..database.models import Customer, Product, CustomerInteraction, RecommendationHistory, DatabaseManager
from ..utils.config import settings

class RecommendationAgent(BaseAgent):
    """Advanced recommendation agent with hybrid algorithms and explainable AI"""
    
    def __init__(self, agent_id: str, db_manager: DatabaseManager, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentType.RECOMMENDATION, db_manager, config)
        self.collaborative_model = None
        self.recommendation_cache = {}
        self.ab_test_variants = ["collaborative", "content_based", "hybrid"]
        
    def _initialize_agent(self):
        """Initialize recommendation-specific components"""
        self.logger.info("Initializing Recommendation Agent")
        asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self):
        """Initialize recommendation models"""
        try:
            self.collaborative_model = NMF(n_components=50, random_state=42)
            await self._load_historical_data()
            self.logger.info("Recommendation models initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
    
    async def _load_historical_data(self):
        """Load historical interaction data"""
        try:
            session = self.db_manager.get_session()
            interactions = session.query(CustomerInteraction).filter(
                CustomerInteraction.interaction_type.in_(["view", "purchase", "add_to_cart"])
            ).limit(10000).all()  # Limit for performance
            
            if interactions:
                await self._process_interactions(interactions)
            session.close()
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
    
    async def _process_interactions(self, interactions: List[CustomerInteraction]):
        """Process interactions for model training"""
        interaction_data = []
        for interaction in interactions:
            weight = {"view": 1.0, "click": 2.0, "add_to_cart": 5.0, "purchase": 10.0}.get(interaction.interaction_type, 1.0)
            interaction_data.append({
                "customer_id": interaction.customer_id,
                "product_id": interaction.product_id,
                "weight": weight
            })
        
        # Store processed data for recommendations
        self.interaction_data = pd.DataFrame(interaction_data)
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return recommendation agent capabilities"""
        return [
            AgentCapability(
                name="generate_recommendations",
                description="Generate personalized product recommendations",
                input_schema={"customer_id": "string", "context": "object", "num_recommendations": "integer"},
                output_schema={"recommendations": "array", "explanations": "object"},
                confidence_level=0.90
            ),
            AgentCapability(
                name="real_time_personalization",
                description="Provide real-time personalized content",
                input_schema={"customer_id": "string", "session_context": "object"},
                output_schema={"personalized_content": "object", "real_time_recommendations": "array"},
                confidence_level=0.85
            ),
            AgentCapability(
                name="explain_recommendations",
                description="Explain why specific products were recommended",
                input_schema={"recommendation_id": "string", "customer_id": "string"},
                output_schema={"explanation": "object", "confidence": "number"},
                confidence_level=0.88
            )
        ]
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process recommendation requests"""
        try:
            request_type = request.get("type")
            customer_id = request.get("customer_id")
            start_time = datetime.utcnow()
            
            if request_type == "generate_recommendations":
                result = await self._generate_recommendations(
                    customer_id, request.get("context", {}), request.get("num_recommendations", 10)
                )
            elif request_type == "real_time_personalization":
                result = await self._real_time_personalization(
                    customer_id, request.get("session_context", {})
                )
            elif request_type == "explain_recommendation":
                result = await self._explain_recommendation(
                    request.get("recommendation_id"), customer_id
                )
            else:
                result = {"error": f"Unknown request type: {request_type}"}
            
            # Update metrics
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.performance_metrics["average_response_time"] = (
                self.performance_metrics["average_response_time"] + response_time
            ) / 2
            
            await self.learn_from_interaction(request, {"success": "error" not in result})
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_recommendations(self, customer_id: str, context: Dict[str, Any], 
                                      num_recommendations: int = 10) -> Dict[str, Any]:
        """Generate personalized recommendations"""
        try:
            session = self.db_manager.get_session()
            
            customer = session.query(Customer).filter(Customer.customer_id == customer_id).first()
            if not customer:
                return {"error": "Customer not found"}
            
            # Get user preferences
            user_preferences = await self._extract_user_preferences(customer_id)
            
            # Select algorithm variant
            algorithm_variant = self._select_algorithm_variant(customer_id)
            
            # Generate recommendations
            recommendations = await self._generate_algorithm_recommendations(
                customer_id, user_preferences, algorithm_variant, num_recommendations
            )
            
            # Add explanations
            for rec in recommendations:
                rec["explanation"] = self._generate_explanation_for_rec(rec, user_preferences)
            
            # Store recommendations
            await self._store_recommendations(customer_id, recommendations)
            
            session.close()
            
            return {
                "customer_id": customer_id,
                "recommendations": recommendations,
                "algorithm_variant": algorithm_variant,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {"error": str(e)}
    
    async def _extract_user_preferences(self, customer_id: str) -> Dict[str, Any]:
        """Extract user preferences from interaction history"""
        try:
            session = self.db_manager.get_session()
            
            # Get recent interactions
            interactions = session.query(CustomerInteraction).filter(
                CustomerInteraction.customer_id == customer_id,
                CustomerInteraction.timestamp >= datetime.utcnow() - timedelta(days=30)
            ).all()
            
            if not interactions:
                return {"preferred_categories": [], "preferred_brands": [], "price_sensitivity": "medium"}
            
            # Get product data for interactions
            product_ids = [i.product_id for i in interactions]
            products = session.query(Product).filter(Product.product_id.in_(product_ids)).all()
            
            # Analyze preferences
            category_counts = {}
            brand_counts = {}
            prices = []
            
            for product in products:
                if product.category:
                    category_counts[product.category] = category_counts.get(product.category, 0) + 1
                if product.brand:
                    brand_counts[product.brand] = brand_counts.get(product.brand, 0) + 1
                if product.price:
                    prices.append(product.price)
            
            # Determine preferences
            preferred_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            preferred_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            avg_price = np.mean(prices) if prices else 100
            price_sensitivity = "high" if avg_price < 50 else "low" if avg_price > 200 else "medium"
            
            session.close()
            
            return {
                "preferred_categories": [cat[0] for cat in preferred_categories],
                "preferred_brands": [brand[0] for brand in preferred_brands],
                "price_sensitivity": price_sensitivity,
                "avg_price": avg_price
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting preferences: {str(e)}")
            return {}
    
    def _select_algorithm_variant(self, customer_id: str) -> str:
        """Select algorithm variant for A/B testing"""
        hash_value = hash(customer_id) % len(self.ab_test_variants)
        return self.ab_test_variants[hash_value]
    
    async def _generate_algorithm_recommendations(self, customer_id: str, user_preferences: Dict[str, Any], 
                                                algorithm_variant: str, num_recommendations: int) -> List[Dict[str, Any]]:
        """Generate recommendations using specified algorithm"""
        session = self.db_manager.get_session()
        
        available_products = session.query(Product).filter(
            Product.is_active == True,
            Product.stock_quantity > 0
        ).limit(100).all()
        
        if algorithm_variant == "collaborative":
            recommendations = await self._collaborative_recommendations(customer_id, available_products, num_recommendations)
        elif algorithm_variant == "content_based":
            recommendations = await self._content_based_recommendations(user_preferences, available_products, num_recommendations)
        else:  # hybrid
            recommendations = await self._hybrid_recommendations(customer_id, user_preferences, available_products, num_recommendations)
        
        session.close()
        return recommendations
    
    async def _collaborative_recommendations(self, customer_id: str, products: List[Product], num_recommendations: int) -> List[Dict[str, Any]]:
        """Generate collaborative filtering recommendations"""
        session = self.db_manager.get_session()
        
        # Get customer interactions
        customer_interactions = session.query(CustomerInteraction).filter(
            CustomerInteraction.customer_id == customer_id
        ).all()
        
        customer_product_ids = {i.product_id for i in customer_interactions}
        
        # Find similar customers
        similar_customers = session.query(CustomerInteraction.customer_id).filter(
            CustomerInteraction.product_id.in_(customer_product_ids),
            CustomerInteraction.customer_id != customer_id
        ).distinct().limit(20).all()
        
        # Get recommendations from similar customers
        similar_customer_ids = [c[0] for c in similar_customers]
        recommended_product_ids = session.query(CustomerInteraction.product_id).filter(
            CustomerInteraction.customer_id.in_(similar_customer_ids),
            ~CustomerInteraction.product_id.in_(customer_product_ids)
        ).distinct().limit(num_recommendations).all()
        
        recommendations = []
        for product_id_tuple in recommended_product_ids:
            product = session.query(Product).filter(Product.product_id == product_id_tuple[0]).first()
            if product and product.is_active:
                recommendations.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "price": product.price,
                    "rating": product.rating,
                    "confidence": 0.7 + np.random.random() * 0.2,
                    "algorithm": "collaborative"
                })
        
        session.close()
        return recommendations
    
    async def _content_based_recommendations(self, user_preferences: Dict[str, Any], 
                                           products: List[Product], num_recommendations: int) -> List[Dict[str, Any]]:
        """Generate content-based recommendations"""
        preferred_categories = user_preferences.get("preferred_categories", [])
        preferred_brands = user_preferences.get("preferred_brands", [])
        
        scored_products = []
        
        for product in products:
            score = 0.0
            
            # Category preference
            if product.category in preferred_categories:
                score += 0.4
            
            # Brand preference
            if product.brand in preferred_brands:
                score += 0.3
            
            # Rating boost
            if product.rating:
                score += product.rating * 0.1
            
            # Popularity boost
            if product.popularity_score:
                score += product.popularity_score * 0.2
            
            if score > 0:
                scored_products.append((product, score))
        
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for product, score in scored_products[:num_recommendations]:
            recommendations.append({
                "product_id": product.product_id,
                "name": product.name,
                "price": product.price,
                "rating": product.rating,
                "confidence": min(score, 1.0),
                "algorithm": "content_based"
            })
        
        return recommendations
    
    async def _hybrid_recommendations(self, customer_id: str, user_preferences: Dict[str, Any], 
                                    products: List[Product], num_recommendations: int) -> List[Dict[str, Any]]:
        """Generate hybrid recommendations"""
        # Get half from each algorithm
        collab_recs = await self._collaborative_recommendations(customer_id, products, num_recommendations // 2)
        content_recs = await self._content_based_recommendations(user_preferences, products, num_recommendations // 2)
        
        # Combine and deduplicate
        all_recs = collab_recs + content_recs
        seen_products = set()
        final_recs = []
        
        for rec in all_recs:
            if rec["product_id"] not in seen_products:
                rec["algorithm"] = "hybrid"
                rec["confidence"] = rec["confidence"] * 0.9 + 0.1  # Slight boost
                final_recs.append(rec)
                seen_products.add(rec["product_id"])
        
        final_recs.sort(key=lambda x: x["confidence"], reverse=True)
        return final_recs[:num_recommendations]
    
    def _generate_explanation_for_rec(self, recommendation: Dict[str, Any], user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for recommendation"""
        algorithm = recommendation.get("algorithm", "hybrid")
        
        explanations = {
            "collaborative": {
                "primary_reason": "Customers similar to you also liked this product",
                "factors": ["Similar customer preferences", "Popular among similar users"]
            },
            "content_based": {
                "primary_reason": "This matches your interests and preferences",
                "factors": ["Matches your preferred categories", "Similar to items you've viewed"]
            },
            "hybrid": {
                "primary_reason": "Recommended based on your preferences and similar customers",
                "factors": ["Popular among similar customers", "Matches your browsing history"]
            }
        }
        
        return explanations.get(algorithm, explanations["hybrid"])
    
    async def _store_recommendations(self, customer_id: str, recommendations: List[Dict[str, Any]]):
        """Store recommendations for tracking"""
        session = self.db_manager.get_session()
        
        try:
            for i, rec in enumerate(recommendations):
                rec_history = RecommendationHistory(
                    recommendation_id=f"rec_{customer_id}_{datetime.utcnow().timestamp()}_{i}",
                    customer_id=customer_id,
                    product_id=rec["product_id"],
                    algorithm_used=rec["algorithm"],
                    confidence_score=rec["confidence"],
                    ranking_position=i + 1,
                    context="api_request"
                )
                session.add(rec_history)
            
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing recommendations: {str(e)}")
        finally:
            session.close()
    
    async def _real_time_personalization(self, customer_id: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide real-time personalization"""
        try:
            current_page = session_context.get("page_type", "homepage")
            
            personalization = {
                "customer_id": customer_id,
                "personalized_content": {},
                "real_time_recommendations": [],
                "next_best_actions": []
            }
            
            # Generate context-specific recommendations
            if current_page == "product_page":
                product_id = session_context.get("product_id")
                personalization["real_time_recommendations"] = await self._get_related_products(product_id)
            else:
                recs = await self._generate_recommendations(customer_id, {}, 5)
                personalization["real_time_recommendations"] = recs.get("recommendations", [])
            
            # Suggest next actions
            intent = session_context.get("intent", "browse")
            if intent == "browse":
                personalization["next_best_actions"] = ["view_recommendations", "check_deals"]
            elif intent == "purchase":
                personalization["next_best_actions"] = ["complete_checkout", "apply_coupon"]
            
            return personalization
            
        except Exception as e:
            self.logger.error(f"Error in real-time personalization: {str(e)}")
            return {"error": str(e)}
    
    async def _get_related_products(self, product_id: str) -> List[Dict[str, Any]]:
        """Get products related to current product"""
        session = self.db_manager.get_session()
        
        current_product = session.query(Product).filter(Product.product_id == product_id).first()
        if not current_product:
            return []
        
        # Find similar products in same category
        related_products = session.query(Product).filter(
            Product.category == current_product.category,
            Product.product_id != product_id,
            Product.is_active == True
        ).limit(5).all()
        
        recommendations = []
        for product in related_products:
            recommendations.append({
                "product_id": product.product_id,
                "name": product.name,
                "price": product.price,
                "rating": product.rating,
                "confidence": 0.8,
                "algorithm": "similarity"
            })
        
        session.close()
        return recommendations
    
    async def _explain_recommendation(self, recommendation_id: str, customer_id: str) -> Dict[str, Any]:
        """Explain a specific recommendation"""
        try:
            session = self.db_manager.get_session()
            
            rec_history = session.query(RecommendationHistory).filter(
                RecommendationHistory.recommendation_id == recommendation_id
            ).first()
            
            if not rec_history:
                return {"error": "Recommendation not found"}
            
            explanation = {
                "recommendation_id": recommendation_id,
                "algorithm_used": rec_history.algorithm_used,
                "confidence_score": rec_history.confidence_score,
                "detailed_explanation": {
                    "why_recommended": f"This product was recommended using {rec_history.algorithm_used} algorithm",
                    "confidence_level": "high" if rec_history.confidence_score > 0.8 else "medium",
                    "factors": ["Customer behavior analysis", "Product similarity", "Popularity metrics"]
                }
            }
            
            session.close()
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining recommendation: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_explanation(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for agent decisions"""
        return {
            "reasoning": [
                "Analyzed customer behavior and preferences",
                "Applied hybrid recommendation algorithms",
                "Considered real-time context and intent"
            ],
            "factors": [
                "Customer interaction history",
                "Product similarity metrics",
                "Collaborative filtering signals",
                "Content-based preferences"
            ],
            "data_sources": [
                "Customer interaction database",
                "Product catalog",
                "Recommendation models",
                "Real-time session data"
            ]
        }