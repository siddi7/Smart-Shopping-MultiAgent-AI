"""
Advanced Customer Agent for Smart Shopping Multi-Agent AI System
Handles customer behavior analysis, personalization, and preference learning
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

from .base_agent import BaseAgent, AgentType, AgentCapability, AgentMessage
from ..database.models import Customer, CustomerInteraction, DatabaseManager
from ..utils.config import settings

class CustomerAgent(BaseAgent):
    """
    Specialized agent for customer behavior analysis and personalization
    
    Key Capabilities:
    - Real-time behavior tracking and analysis
    - Customer segmentation using advanced ML
    - Preference learning and prediction
    - Personalization strategy optimization
    - Churn prediction and prevention
    """
    
    def __init__(self, agent_id: str, db_manager: DatabaseManager, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentType.CUSTOMER, db_manager, config)
        
        # Customer analytics models
        self.segmentation_model = None
        self.preference_model = None
        self.churn_model = None
        
        # Customer behavior patterns
        self.behavior_patterns = {}
        self.real_time_sessions = {}
        
        # Personalization engines
        self.personalization_strategies = {
            "new_visitor": self._new_visitor_strategy,
            "frequent_buyer": self._frequent_buyer_strategy,
            "price_sensitive": self._price_sensitive_strategy,
            "luxury_shopper": self._luxury_shopper_strategy,
            "tech_enthusiast": self._tech_enthusiast_strategy
        }
    
    def _initialize_agent(self):
        """Initialize customer-specific components"""
        self.logger.info("Initializing Customer Agent")
        # Initialize ML models for customer analysis
        asyncio.create_task(self._initialize_ml_models())
    
    async def _initialize_ml_models(self):
        """Initialize machine learning models for customer analysis"""
        try:
            # Initialize segmentation model
            self.segmentation_model = KMeans(n_clusters=len(settings.CUSTOMER_SEGMENTS), random_state=42)
            
            # Initialize other models as needed
            self.logger.info("Customer ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {str(e)}")
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return customer agent capabilities"""
        return [
            AgentCapability(
                name="customer_analysis",
                description="Analyze customer behavior and preferences",
                input_schema={
                    "customer_id": "string",
                    "interaction_data": "object",
                    "analysis_type": "string"
                },
                output_schema={
                    "customer_profile": "object",
                    "behavior_insights": "object",
                    "recommendations": "array"
                },
                confidence_level=0.85
            ),
            AgentCapability(
                name="customer_segmentation",
                description="Segment customers based on behavior and preferences",
                input_schema={
                    "customer_data": "array",
                    "segmentation_criteria": "object"
                },
                output_schema={
                    "segments": "array",
                    "segment_characteristics": "object"
                },
                confidence_level=0.90
            ),
            AgentCapability(
                name="personalization",
                description="Generate personalized experiences for customers",
                input_schema={
                    "customer_id": "string",
                    "context": "object"
                },
                output_schema={
                    "personalized_content": "object",
                    "strategy": "string",
                    "confidence": "number"
                },
                confidence_level=0.80
            ),
            AgentCapability(
                name="churn_prediction",
                description="Predict customer churn risk and suggest retention strategies",
                input_schema={
                    "customer_id": "string",
                    "behavioral_data": "object"
                },
                output_schema={
                    "churn_probability": "number",
                    "risk_factors": "array",
                    "retention_strategies": "array"
                },
                confidence_level=0.75
            )
        ]
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process customer-related requests"""
        try:
            request_type = request.get("type")
            customer_id = request.get("customer_id")
            
            # Update performance metrics
            start_time = datetime.utcnow()
            
            if request_type == "analyze_customer":
                result = await self._analyze_customer(customer_id, request.get("interaction_data", {}))
            elif request_type == "segment_customers":
                result = await self._segment_customers(request.get("customer_data", []))
            elif request_type == "personalize_experience":
                result = await self._personalize_experience(customer_id, request.get("context", {}))
            elif request_type == "predict_churn":
                result = await self._predict_churn(customer_id)
            elif request_type == "track_real_time_behavior":
                result = await self._track_real_time_behavior(customer_id, request.get("session_data", {}))
            else:
                result = {"error": f"Unknown request type: {request_type}"}
            
            # Update metrics
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.performance_metrics["average_response_time"] = (
                self.performance_metrics["average_response_time"] + response_time
            ) / 2
            
            # Learn from this interaction
            await self.learn_from_interaction(request, {"success": "error" not in result})
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_customer(self, customer_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive customer analysis"""
        try:
            session = self.db_manager.get_session()
            
            # Get customer data
            customer = session.query(Customer).filter(Customer.customer_id == customer_id).first()
            if not customer:
                return {"error": "Customer not found"}
            
            # Get recent interactions
            recent_interactions = session.query(CustomerInteraction).filter(
                CustomerInteraction.customer_id == customer_id,
                CustomerInteraction.timestamp >= datetime.utcnow() - timedelta(days=30)
            ).all()
            
            # Analyze behavior patterns
            behavior_analysis = await self._analyze_behavior_patterns(customer, recent_interactions)
            
            # Update customer profile with new insights
            customer_profile = {
                "customer_id": customer_id,
                "segment": customer.customer_segment,
                "lifetime_value": customer.lifetime_value,
                "churn_probability": customer.churn_probability,
                "satisfaction_score": customer.satisfaction_score,
                "behavior_analysis": behavior_analysis,
                "preferences": customer.preferences or {},
                "last_analysis": datetime.utcnow().isoformat()
            }
            
            # Store insights in memory
            await self.store_memory(
                memory_type="customer_analysis",
                content=customer_profile,
                customer_id=customer_id,
                importance_score=0.8
            )
            
            session.close()
            return customer_profile
            
        except Exception as e:
            self.logger.error(f"Error analyzing customer {customer_id}: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_behavior_patterns(self, customer: Customer, interactions: List[CustomerInteraction]) -> Dict[str, Any]:
        """Analyze customer behavior patterns"""
        if not interactions:
            return {"message": "No recent interactions found"}
        
        # Convert interactions to DataFrame for analysis
        interaction_data = []
        for interaction in interactions:
            interaction_data.append({
                "interaction_type": interaction.interaction_type,
                "timestamp": interaction.timestamp,
                "duration_seconds": interaction.duration_seconds or 0,
                "scroll_depth": interaction.scroll_depth or 0,
                "device_type": interaction.device_type,
                "sentiment_score": interaction.sentiment_score or 0
            })
        
        df = pd.DataFrame(interaction_data)
        
        # Calculate behavior metrics
        behavior_patterns = {
            "session_frequency": len(df.groupby(df['timestamp'].dt.date)),
            "avg_session_duration": df['duration_seconds'].mean(),
            "preferred_device": df['device_type'].mode().iloc[0] if not df.empty else "unknown",
            "interaction_types": df['interaction_type'].value_counts().to_dict(),
            "engagement_score": df['scroll_depth'].mean() if 'scroll_depth' in df.columns else 0,
            "sentiment_trend": df['sentiment_score'].mean() if 'sentiment_score' in df.columns else 0,
            "peak_activity_hours": self._find_peak_activity_hours(df),
            "browsing_velocity": len(interactions) / 30 if interactions else 0  # interactions per day
        }
        
        return behavior_patterns
    
    def _find_peak_activity_hours(self, df: pd.DataFrame) -> List[int]:
        """Find peak activity hours for the customer"""
        if df.empty:
            return []
        
        df['hour'] = df['timestamp'].dt.hour
        hourly_activity = df['hour'].value_counts()
        peak_hours = hourly_activity.nlargest(3).index.tolist()
        return peak_hours
    
    async def _segment_customers(self, customer_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Advanced customer segmentation using ML"""
        try:
            if not customer_data:
                # Fetch customer data from database
                session = self.db_manager.get_session()
                customers = session.query(Customer).all()
                customer_data = []
                
                for customer in customers:
                    customer_data.append({
                        "customer_id": customer.customer_id,
                        "age": customer.age or 30,
                        "total_orders": customer.total_orders,
                        "total_spent": customer.total_spent,
                        "avg_order_value": customer.avg_order_value,
                        "lifetime_value": customer.lifetime_value,
                        "last_active_days": (datetime.utcnow() - customer.last_active).days if customer.last_active else 999
                    })
                session.close()
            
            if not customer_data:
                return {"error": "No customer data available for segmentation"}
            
            # Prepare features for clustering
            features = []
            customer_ids = []
            
            for customer in customer_data:
                features.append([
                    customer.get("age", 30),
                    customer.get("total_orders", 0),
                    customer.get("total_spent", 0),
                    customer.get("avg_order_value", 0),
                    customer.get("lifetime_value", 0),
                    customer.get("last_active_days", 999)
                ])
                customer_ids.append(customer["customer_id"])
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform clustering
            if self.segmentation_model is None:
                await self._initialize_ml_models()
            
            clusters = self.segmentation_model.fit_predict(features_scaled)
            
            # Analyze segments
            segments = {}
            for i, cluster in enumerate(clusters):
                customer_id = customer_ids[i]
                if cluster not in segments:
                    segments[cluster] = {
                        "customers": [],
                        "characteristics": {},
                        "size": 0
                    }
                segments[cluster]["customers"].append(customer_id)
                segments[cluster]["size"] += 1
            
            # Calculate segment characteristics
            for cluster_id, segment in segments.items():
                cluster_customers = [customer_data[i] for i, c in enumerate(clusters) if c == cluster_id]
                
                segment["characteristics"] = {
                    "avg_age": np.mean([c.get("age", 30) for c in cluster_customers]),
                    "avg_orders": np.mean([c.get("total_orders", 0) for c in cluster_customers]),
                    "avg_spent": np.mean([c.get("total_spent", 0) for c in cluster_customers]),
                    "avg_lifetime_value": np.mean([c.get("lifetime_value", 0) for c in cluster_customers])
                }
            
            # Map clusters to business segments
            segment_mapping = self._map_clusters_to_business_segments(segments)
            
            return {
                "segments": segments,
                "segment_mapping": segment_mapping,
                "total_customers": len(customer_data),
                "segmentation_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in customer segmentation: {str(e)}")
            return {"error": str(e)}
    
    def _map_clusters_to_business_segments(self, segments: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
        """Map ML clusters to meaningful business segments"""
        mapping = {}
        
        for cluster_id, segment in segments.items():
            characteristics = segment["characteristics"]
            
            # Rule-based mapping based on characteristics
            if characteristics["avg_lifetime_value"] > 1000:
                mapping[cluster_id] = "luxury_shopper"
            elif characteristics["avg_orders"] > 10:
                mapping[cluster_id] = "frequent_buyer"
            elif characteristics["avg_spent"] < 100:
                mapping[cluster_id] = "price_sensitive"
            elif characteristics["avg_age"] < 30:
                mapping[cluster_id] = "tech_enthusiast"
            else:
                mapping[cluster_id] = "regular_customer"
        
        return mapping
    
    async def _personalize_experience(self, customer_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized experience for customer"""
        try:
            # Get customer profile
            customer_analysis = await self._analyze_customer(customer_id, {})
            if "error" in customer_analysis:
                return customer_analysis
            
            segment = customer_analysis.get("segment", "new_visitor")
            preferences = customer_analysis.get("preferences", {})
            behavior_analysis = customer_analysis.get("behavior_analysis", {})
            
            # Apply personalization strategy based on segment
            if segment in self.personalization_strategies:
                strategy_func = self.personalization_strategies[segment]
                personalization = await strategy_func(customer_analysis, context)
            else:
                personalization = await self._default_personalization_strategy(customer_analysis, context)
            
            # Add context-specific personalization
            if context.get("page_type") == "homepage":
                personalization.update(await self._homepage_personalization(customer_analysis))
            elif context.get("page_type") == "product_page":
                personalization.update(await self._product_page_personalization(customer_analysis, context))
            
            # Store personalization decision
            await self.store_memory(
                memory_type="personalization",
                content={
                    "customer_id": customer_id,
                    "strategy": segment,
                    "personalization": personalization,
                    "context": context
                },
                customer_id=customer_id,
                importance_score=0.6
            )
            
            return {
                "customer_id": customer_id,
                "strategy": segment,
                "personalization": personalization,
                "confidence": 0.8,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error personalizing experience for {customer_id}: {str(e)}")
            return {"error": str(e)}
    
    async def _new_visitor_strategy(self, customer_profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Personalization strategy for new visitors"""
        return {
            "welcome_message": "Welcome to our store! Discover our most popular products.",
            "recommended_categories": ["bestsellers", "new_arrivals", "trending"],
            "special_offers": ["first_time_discount", "free_shipping"],
            "content_focus": "discovery",
            "ui_elements": {
                "show_tutorial": True,
                "highlight_navigation": True,
                "show_social_proof": True
            }
        }
    
    async def _frequent_buyer_strategy(self, customer_profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Personalization strategy for frequent buyers"""
        return {
            "welcome_message": "Welcome back! Here are some new items you might like.",
            "recommended_categories": ["based_on_history", "new_in_favorite_brands"],
            "special_offers": ["loyalty_discount", "early_access"],
            "content_focus": "personalized_recommendations",
            "ui_elements": {
                "quick_reorder": True,
                "saved_lists": True,
                "vip_treatment": True
            }
        }
    
    async def _price_sensitive_strategy(self, customer_profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Personalization strategy for price-sensitive customers"""
        return {
            "welcome_message": "Great deals await! Check out our latest discounts.",
            "recommended_categories": ["on_sale", "clearance", "value_packs"],
            "special_offers": ["price_match", "bulk_discount"],
            "content_focus": "value_proposition",
            "ui_elements": {
                "price_comparison": True,
                "savings_calculator": True,
                "deal_alerts": True
            }
        }
    
    async def _luxury_shopper_strategy(self, customer_profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Personalization strategy for luxury shoppers"""
        return {
            "welcome_message": "Exclusive collections curated just for you.",
            "recommended_categories": ["premium", "exclusive", "limited_edition"],
            "special_offers": ["white_glove_service", "personal_shopper"],
            "content_focus": "premium_experience",
            "ui_elements": {
                "concierge_chat": True,
                "exclusive_access": True,
                "premium_support": True
            }
        }
    
    async def _tech_enthusiast_strategy(self, customer_profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Personalization strategy for tech enthusiasts"""
        return {
            "welcome_message": "Discover the latest in technology innovation.",
            "recommended_categories": ["tech_gadgets", "innovation", "pre_orders"],
            "special_offers": ["beta_access", "tech_reviews"],
            "content_focus": "innovation_showcase",
            "ui_elements": {
                "spec_comparison": True,
                "tech_details": True,
                "innovation_timeline": True
            }
        }
    
    async def _default_personalization_strategy(self, customer_profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Default personalization strategy"""
        return {
            "welcome_message": "Welcome! Explore our curated selection.",
            "recommended_categories": ["popular", "trending"],
            "special_offers": ["standard_promotions"],
            "content_focus": "general_browse",
            "ui_elements": {
                "standard_layout": True
            }
        }
    
    async def _homepage_personalization(self, customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Homepage-specific personalization"""
        return {
            "hero_banner": "personalized_based_on_segment",
            "featured_products": "based_on_browsing_history",
            "content_blocks": ["recommendations", "trending", "deals"]
        }
    
    async def _product_page_personalization(self, customer_profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Product page-specific personalization"""
        return {
            "related_products": "based_on_customer_segment",
            "review_highlights": "relevant_to_customer",
            "pricing_display": "optimized_for_segment"
        }
    
    async def _predict_churn(self, customer_id: str) -> Dict[str, Any]:
        """Predict customer churn probability"""
        try:
            customer_analysis = await self._analyze_customer(customer_id, {})
            if "error" in customer_analysis:
                return customer_analysis
            
            behavior_analysis = customer_analysis.get("behavior_analysis", {})
            
            # Simple churn prediction based on behavior patterns
            churn_factors = {
                "low_engagement": behavior_analysis.get("engagement_score", 1) < 0.3,
                "decreased_frequency": behavior_analysis.get("session_frequency", 1) < 5,
                "negative_sentiment": behavior_analysis.get("sentiment_trend", 0) < 0,
                "long_inactivity": behavior_analysis.get("browsing_velocity", 1) < 0.1
            }
            
            # Calculate churn probability
            risk_factors = sum(churn_factors.values())
            churn_probability = min(risk_factors * 0.25, 1.0)
            
            # Generate retention strategies
            retention_strategies = []
            if churn_factors["low_engagement"]:
                retention_strategies.append("personalized_content_boost")
            if churn_factors["decreased_frequency"]:
                retention_strategies.append("re_engagement_campaign")
            if churn_factors["negative_sentiment"]:
                retention_strategies.append("customer_service_outreach")
            if churn_factors["long_inactivity"]:
                retention_strategies.append("win_back_offer")
            
            result = {
                "customer_id": customer_id,
                "churn_probability": churn_probability,
                "risk_level": "high" if churn_probability > 0.7 else "medium" if churn_probability > 0.4 else "low",
                "risk_factors": [factor for factor, present in churn_factors.items() if present],
                "retention_strategies": retention_strategies,
                "prediction_date": datetime.utcnow().isoformat()
            }
            
            # Store prediction
            await self.store_memory(
                memory_type="churn_prediction",
                content=result,
                customer_id=customer_id,
                importance_score=0.9
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting churn for {customer_id}: {str(e)}")
            return {"error": str(e)}
    
    async def _track_real_time_behavior(self, customer_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track and analyze real-time customer behavior"""
        try:
            session_id = session_data.get("session_id", f"session_{datetime.utcnow().timestamp()}")
            
            # Update real-time session tracking
            if customer_id not in self.real_time_sessions:
                self.real_time_sessions[customer_id] = {}
            
            self.real_time_sessions[customer_id][session_id] = {
                "start_time": session_data.get("start_time", datetime.utcnow()),
                "last_activity": datetime.utcnow(),
                "page_views": session_data.get("page_views", []),
                "interactions": session_data.get("interactions", []),
                "current_intent": self._analyze_current_intent(session_data)
            }
            
            # Real-time behavior insights
            insights = {
                "customer_id": customer_id,
                "session_id": session_id,
                "current_intent": self.real_time_sessions[customer_id][session_id]["current_intent"],
                "engagement_level": self._calculate_engagement_level(session_data),
                "next_best_action": await self._suggest_next_best_action(customer_id, session_data),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error tracking real-time behavior for {customer_id}: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_current_intent(self, session_data: Dict[str, Any]) -> str:
        """Analyze customer's current intent based on session data"""
        page_views = session_data.get("page_views", [])
        interactions = session_data.get("interactions", [])
        
        # Simple intent analysis
        if any("cart" in page.get("url", "") for page in page_views):
            return "purchase_intent"
        elif any("search" in interaction.get("type", "") for interaction in interactions):
            return "discovery_intent"
        elif any("product" in page.get("url", "") for page in page_views):
            return "browse_intent"
        else:
            return "general_browse"
    
    def _calculate_engagement_level(self, session_data: Dict[str, Any]) -> float:
        """Calculate current engagement level"""
        interactions = session_data.get("interactions", [])
        page_views = session_data.get("page_views", [])
        
        engagement_score = 0.0
        
        # Factor in number of interactions
        engagement_score += min(len(interactions) * 0.1, 0.5)
        
        # Factor in page views
        engagement_score += min(len(page_views) * 0.05, 0.3)
        
        # Factor in time spent
        session_duration = session_data.get("duration_minutes", 0)
        engagement_score += min(session_duration * 0.02, 0.2)
        
        return min(engagement_score, 1.0)
    
    async def _suggest_next_best_action(self, customer_id: str, session_data: Dict[str, Any]) -> str:
        """Suggest the next best action for the customer"""
        current_intent = self._analyze_current_intent(session_data)
        engagement_level = self._calculate_engagement_level(session_data)
        
        if current_intent == "purchase_intent" and engagement_level > 0.7:
            return "offer_assistance"
        elif current_intent == "discovery_intent":
            return "show_personalized_recommendations"
        elif engagement_level < 0.3:
            return "engage_with_promotion"
        else:
            return "continue_browsing"
    
    async def _generate_explanation(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for customer agent decisions"""
        return {
            "reasoning": [
                "Analyzed customer behavior patterns and preferences",
                "Applied segment-specific personalization strategy",
                "Considered real-time context and intent"
            ],
            "factors": [
                "Customer segment classification",
                "Historical behavior analysis",
                "Real-time session data",
                "Personalization rules"
            ],
            "data_sources": [
                "Customer profile",
                "Interaction history",
                "Behavioral analytics",
                "Segmentation model"
            ]
        }