"""
Advanced Product Agent for Smart Shopping Multi-Agent AI System
Handles product analysis, catalog optimization, and multi-modal content processing
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base_agent import BaseAgent, AgentType, AgentCapability, AgentMessage
from ..database.models import Product, CustomerInteraction, DatabaseManager
from ..utils.config import settings

class ProductAgent(BaseAgent):
    """
    Specialized agent for product management and optimization
    """
    
    def __init__(self, agent_id: str, db_manager: DatabaseManager, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentType.PRODUCT, db_manager, config)
        self.text_vectorizer = None
        self.product_embeddings = {}
        
    def _initialize_agent(self):
        """Initialize product-specific components"""
        self.logger.info("Initializing Product Agent")
        asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self):
        """Initialize ML models for product analysis"""
        try:
            self.text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
            self.logger.info("Product models initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return product agent capabilities"""
        return [
            AgentCapability(
                name="product_analysis",
                description="Comprehensive product analysis including content and performance",
                input_schema={"product_id": "string", "analysis_type": "string"},
                output_schema={"product_insights": "object", "optimization_suggestions": "array"},
                confidence_level=0.88
            ),
            AgentCapability(
                name="product_similarity",
                description="Find similar products using multi-modal analysis",
                input_schema={"product_id": "string", "similarity_threshold": "number"},
                output_schema={"similar_products": "array", "similarity_scores": "array"},
                confidence_level=0.92
            ),
            AgentCapability(
                name="catalog_optimization",
                description="Optimize product catalog for better performance",
                input_schema={"category": "string", "optimization_goals": "array"},
                output_schema={"optimized_catalog": "object", "action_items": "array"},
                confidence_level=0.85
            )
        ]
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process product-related requests"""
        try:
            request_type = request.get("type")
            start_time = datetime.utcnow()
            
            if request_type == "analyze_product":
                result = await self._analyze_product(request.get("product_id"), request.get("analysis_type", "comprehensive"))
            elif request_type == "find_similar_products":
                result = await self._find_similar_products(request.get("product_id"), request.get("similarity_threshold", 0.7))
            elif request_type == "optimize_catalog":
                result = await self._optimize_catalog(request.get("category"), request.get("optimization_goals", []))
            elif request_type == "enhance_content":
                result = await self._enhance_content(request.get("product_id"))
            else:
                result = {"error": f"Unknown request type: {request_type}"}
            
            # Update metrics
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.performance_metrics["average_response_time"] = (self.performance_metrics["average_response_time"] + response_time) / 2
            
            await self.learn_from_interaction(request, {"success": "error" not in result})
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_product(self, product_id: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Comprehensive product analysis"""
        try:
            session = self.db_manager.get_session()
            product = session.query(Product).filter(Product.product_id == product_id).first()
            
            if not product:
                return {"error": "Product not found"}
            
            # Get interactions for performance analysis
            interactions = session.query(CustomerInteraction).filter(
                CustomerInteraction.product_id == product_id,
                CustomerInteraction.timestamp >= datetime.utcnow() - timedelta(days=30)
            ).all()
            
            # Basic insights
            product_insights = {
                "product_id": product_id,
                "name": product.name,
                "category": product.category,
                "price": product.price,
                "popularity_score": product.popularity_score,
                "rating": product.rating,
                "performance": await self._analyze_performance(product, interactions),
                "content_quality": await self._analyze_content_quality(product),
                "seo_analysis": await self._analyze_seo_factors(product)
            }
            
            optimization_suggestions = await self._generate_optimization_suggestions(product, product_insights)
            
            await self.store_memory(
                memory_type="product_analysis",
                content=product_insights,
                product_id=product_id,
                importance_score=0.8
            )
            
            session.close()
            
            return {
                "product_insights": product_insights,
                "optimization_suggestions": optimization_suggestions,
                "analysis_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing product {product_id}: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_performance(self, product: Product, interactions: List[CustomerInteraction]) -> Dict[str, Any]:
        """Analyze product performance metrics"""
        if not interactions:
            return {"message": "No recent interactions found"}
        
        df = pd.DataFrame([{
            "interaction_type": i.interaction_type,
            "customer_id": i.customer_id
        } for i in interactions])
        
        performance = {
            "total_views": len(df[df["interaction_type"] == "view"]),
            "total_clicks": len(df[df["interaction_type"] == "click"]),
            "total_purchases": len(df[df["interaction_type"] == "purchase"]),
            "unique_visitors": df["customer_id"].nunique(),
            "conversion_rate": 0.0
        }
        
        if performance["total_views"] > 0:
            performance["conversion_rate"] = performance["total_purchases"] / performance["total_views"]
        
        return performance
    
    async def _analyze_content_quality(self, product: Product) -> Dict[str, Any]:
        """Analyze product content quality"""
        content_quality = {
            "title_quality": 0.0,
            "description_quality": 0.0,
            "completeness": 0.0,
            "issues": [],
            "suggestions": []
        }
        
        # Title analysis
        if product.name:
            title_length = len(product.name)
            if 10 <= title_length <= 100:
                content_quality["title_quality"] = 0.9
            else:
                content_quality["title_quality"] = 0.5
                content_quality["issues"].append("Title length not optimal")
        
        # Description analysis
        if product.description:
            desc_length = len(product.description)
            if 50 <= desc_length <= 2000:
                content_quality["description_quality"] = 0.9
            else:
                content_quality["description_quality"] = 0.5
                content_quality["issues"].append("Description length not optimal")
        
        # Completeness check
        fields = [product.name, product.description, product.category, product.brand]
        content_quality["completeness"] = sum(1 for f in fields if f) / len(fields)
        
        # Generate suggestions
        if content_quality["title_quality"] < 0.7:
            content_quality["suggestions"].append("Optimize product title")
        if content_quality["description_quality"] < 0.7:
            content_quality["suggestions"].append("Enhance product description")
        
        return content_quality
    
    async def _analyze_seo_factors(self, product: Product) -> Dict[str, Any]:
        """Analyze SEO factors for the product"""
        seo_analysis = {
            "title_optimization": 0.0,
            "description_optimization": 0.0,
            "suggestions": []
        }
        
        # Title SEO
        if product.name:
            brand_included = product.brand and product.brand.lower() in product.name.lower()
            category_included = product.category and product.category.lower() in product.name.lower()
            seo_analysis["title_optimization"] = (0.5 * brand_included + 0.5 * category_included)
        
        # Description SEO
        if product.description:
            word_count = len(product.description.split())
            seo_analysis["description_optimization"] = min(word_count / 150, 1.0)
        
        if seo_analysis["title_optimization"] < 0.7:
            seo_analysis["suggestions"].append("Include brand and category in title")
        if seo_analysis["description_optimization"] < 0.7:
            seo_analysis["suggestions"].append("Expand product description")
        
        return seo_analysis
    
    async def _generate_optimization_suggestions(self, product: Product, insights: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        performance = insights.get("performance", {})
        if performance.get("conversion_rate", 0) < 0.02:
            suggestions.append("Improve product description and images")
        
        content_quality = insights.get("content_quality", {})
        suggestions.extend(content_quality.get("suggestions", []))
        
        seo_analysis = insights.get("seo_analysis", {})
        suggestions.extend(seo_analysis.get("suggestions", []))
        
        if product.stock_quantity < 10:
            suggestions.append("Replenish inventory")
        
        return suggestions
    
    async def _find_similar_products(self, product_id: str, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Find similar products"""
        try:
            session = self.db_manager.get_session()
            
            target_product = session.query(Product).filter(Product.product_id == product_id).first()
            if not target_product:
                return {"error": "Product not found"}
            
            candidate_products = session.query(Product).filter(
                Product.category == target_product.category,
                Product.product_id != product_id,
                Product.is_active == True
            ).limit(50).all()
            
            similarities = []
            for candidate in candidate_products:
                similarity_score = await self._calculate_product_similarity(target_product, candidate)
                
                if similarity_score >= similarity_threshold:
                    similarities.append({
                        "product_id": candidate.product_id,
                        "name": candidate.name,
                        "similarity_score": similarity_score,
                        "price": candidate.price,
                        "rating": candidate.rating
                    })
            
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            session.close()
            
            return {
                "target_product_id": product_id,
                "similar_products": similarities[:10],
                "analysis_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error finding similar products: {str(e)}")
            return {"error": str(e)}
    
    async def _calculate_product_similarity(self, product1: Product, product2: Product) -> float:
        """Calculate similarity between two products"""
        try:
            # Text similarity
            text1 = f"{product1.name} {product1.description or ''}"
            text2 = f"{product2.name} {product2.description or ''}"
            
            emb1 = await self._create_embedding(text1)
            emb2 = await self._create_embedding(text2)
            
            text_similarity = 0.0
            if emb1 is not None and emb2 is not None:
                text_similarity = self._calculate_similarity(emb1, emb2)
            
            # Brand similarity
            brand_similarity = 1.0 if product1.brand == product2.brand else 0.0
            
            # Price similarity
            price_similarity = 0.0
            if product1.price and product2.price:
                price_diff = abs(product1.price - product2.price)
                max_price = max(product1.price, product2.price)
                price_similarity = 1.0 - (price_diff / max_price)
            
            # Weighted combination
            return (0.6 * text_similarity + 0.2 * brand_similarity + 0.2 * price_similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    async def _optimize_catalog(self, category: Optional[str], optimization_goals: List[str]) -> Dict[str, Any]:
        """Optimize product catalog"""
        try:
            session = self.db_manager.get_session()
            
            query = session.query(Product).filter(Product.is_active == True)
            if category:
                query = query.filter(Product.category == category)
            
            products = query.all()
            
            optimization_results = {
                "category": category or "all",
                "total_products_analyzed": len(products),
                "optimization_actions": [],
                "priority_items": []
            }
            
            # Analyze optimization goals
            for goal in optimization_goals:
                if goal == "improve_visibility":
                    actions = self._optimize_for_visibility(products)
                    optimization_results["optimization_actions"].extend(actions)
                elif goal == "increase_conversions":
                    actions = self._optimize_for_conversions(products)
                    optimization_results["optimization_actions"].extend(actions)
            
            # Identify priority products
            optimization_results["priority_items"] = self._identify_priority_products(products)
            
            session.close()
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing catalog: {str(e)}")
            return {"error": str(e)}
    
    def _optimize_for_visibility(self, products: List[Product]) -> List[Dict[str, Any]]:
        """Optimize products for visibility"""
        actions = []
        for product in products:
            if product.view_count < 100:
                actions.append({
                    "product_id": product.product_id,
                    "action": "improve_seo",
                    "description": "Optimize for better search visibility"
                })
        return actions
    
    def _optimize_for_conversions(self, products: List[Product]) -> List[Dict[str, Any]]:
        """Optimize products for conversions"""
        actions = []
        for product in products:
            conversion_rate = product.purchase_count / max(product.view_count, 1)
            if conversion_rate < 0.02:
                actions.append({
                    "product_id": product.product_id,
                    "action": "enhance_product_page",
                    "description": "Improve product details and images"
                })
        return actions
    
    def _identify_priority_products(self, products: List[Product]) -> List[Dict[str, Any]]:
        """Identify priority products"""
        priority_products = []
        
        for product in products:
            priority_score = 0
            reasons = []
            
            # High traffic but low conversion
            if product.view_count > 1000 and product.purchase_count < 20:
                priority_score += 3
                reasons.append("high_traffic_low_conversion")
            
            # Low stock
            if product.stock_quantity < 10:
                priority_score += 2
                reasons.append("low_stock")
            
            if priority_score >= 2:
                priority_products.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "priority_score": priority_score,
                    "reasons": reasons
                })
        
        priority_products.sort(key=lambda x: x["priority_score"], reverse=True)
        return priority_products[:20]
    
    async def _enhance_content(self, product_id: str) -> Dict[str, Any]:
        """Enhance product content"""
        try:
            session = self.db_manager.get_session()
            product = session.query(Product).filter(Product.product_id == product_id).first()
            
            if not product:
                return {"error": "Product not found"}
            
            enhanced_content = {
                "product_id": product_id,
                "original_description": product.description or "",
                "enhanced_description": await self._generate_enhanced_description(product),
                "improvements": ["Added feature highlights", "Improved readability", "Enhanced SEO"]
            }
            
            session.close()
            return enhanced_content
            
        except Exception as e:
            self.logger.error(f"Error enhancing content: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_enhanced_description(self, product: Product) -> str:
        """Generate enhanced description"""
        base_desc = product.description or ""
        
        enhancements = [
            f"Premium {product.category} from {product.brand or 'top brand'}",
            "Key Features:",
            "- High quality materials and construction",
            "- Perfect for everyday use",
            "- Available in multiple options"
        ]
        
        if base_desc:
            return base_desc + "\n\n" + "\n".join(enhancements)
        else:
            return "\n".join(enhancements)
    
    async def _generate_explanation(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for product agent decisions"""
        return {
            "reasoning": [
                "Analyzed product content and performance",
                "Evaluated optimization opportunities",
                "Applied best practices for catalog management"
            ],
            "factors": [
                "Content quality assessment",
                "Performance metrics",
                "SEO optimization factors"
            ],
            "data_sources": [
                "Product catalog data",
                "Customer interaction history",
                "Performance analytics"
            ]
        }