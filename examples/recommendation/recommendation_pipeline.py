"""
E-commerce Recommendation Pipeline
Based on Microsoft Recommenders
Repository: https://github.com/microsoft/recommenders
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class RecommendationPipeline:
    """E-commerce recommendation pipeline"""
    
    def __init__(self):
        self.model = None
        self.user_item_matrix = None
        self.feature_store = {
            'user_behaviors': defaultdict(list),
            'item_features': {},
            'user_features': {}
        }
        self.config = {
            'top_k': 10,
            'min_interactions': 5
        }
    
    def track_user_behavior(self, user_id: str, item_id: str, 
                           behavior_type: str, metadata: Dict[str, Any] = None):
        """User Behavior Tracking"""
        behavior = {
            'user_id': user_id,
            'item_id': item_id,
            'behavior_type': behavior_type,  # view, click, purchase, rating
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.feature_store['user_behaviors'][user_id].append(behavior)
        
        # Update item features
        if item_id not in self.feature_store['item_features']:
            self.feature_store['item_features'][item_id] = {
                'view_count': 0,
                'click_count': 0,
                'purchase_count': 0,
                'total_rating': 0,
                'rating_count': 0
            }
        
        item_features = self.feature_store['item_features'][item_id]
        if behavior_type == 'view':
            item_features['view_count'] += 1
        elif behavior_type == 'click':
            item_features['click_count'] += 1
        elif behavior_type == 'purchase':
            item_features['purchase_count'] += 1
        elif behavior_type == 'rating':
            item_features['total_rating'] += metadata.get('rating', 0)
            item_features['rating_count'] += 1
        
        logger.debug(f"✅ Behavior tracked: {user_id} -> {item_id} ({behavior_type})")
        return behavior
    
    def update_feature_store(self):
        """Feature Store Updates"""
        # Build user-item interaction matrix
        interactions = []
        for user_id, behaviors in self.feature_store['user_behaviors'].items():
            for behavior in behaviors:
                weight = {
                    'view': 1,
                    'click': 2,
                    'purchase': 5,
                    'rating': behavior['metadata'].get('rating', 3)
                }.get(behavior['behavior_type'], 1)
                
                interactions.append({
                    'user_id': user_id,
                    'item_id': behavior['item_id'],
                    'weight': weight
                })
        
        if interactions:
            df = pd.DataFrame(interactions)
            self.user_item_matrix = df.pivot_table(
                index='user_id', 
                columns='item_id', 
                values='weight', 
                aggfunc='sum', 
                fill_value=0
            )
            logger.info(f"✅ Feature store updated: {len(self.user_item_matrix)} users, {len(self.user_item_matrix.columns)} items")
        else:
            logger.warning("No interactions to build matrix")
    
    def train_model(self, algorithm: str = 'collaborative_filtering'):
        """Model Training Pipeline"""
        if self.user_item_matrix is None or len(self.user_item_matrix) == 0:
            logger.error("No user-item matrix available. Track some behaviors first.")
            return None
        
        try:
            if algorithm == 'collaborative_filtering':
                # Simple collaborative filtering using cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                
                # User-based collaborative filtering
                user_similarity = cosine_similarity(self.user_item_matrix)
                self.model = {
                    'type': 'user_based_cf',
                    'user_similarity': user_similarity,
                    'user_item_matrix': self.user_item_matrix
                }
                
            elif algorithm == 'matrix_factorization':
                # Matrix factorization (simplified)
                from sklearn.decomposition import NMF
                
                nmf = NMF(n_components=10, random_state=42)
                W = nmf.fit_transform(self.user_item_matrix)
                H = nmf.components_
                
                self.model = {
                    'type': 'matrix_factorization',
                    'W': W,
                    'H': H,
                    'nmf': nmf
                }
            
            logger.info(f"✅ Recommendation model trained: {algorithm}")
            return self.model
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return None
    
    def recommend(self, user_id: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Generate recommendations for a user"""
        if self.model is None:
            return []
        
        top_k = top_k or self.config['top_k']
        
        try:
            if self.model['type'] == 'user_based_cf':
                if user_id not in self.user_item_matrix.index:
                    return []
                
                user_idx = self.user_item_matrix.index.get_loc(user_id)
                user_similarity = self.model['user_similarity']
                user_item_matrix = self.model['user_item_matrix']
                
                # Get similar users
                similar_users = user_similarity[user_idx]
                similar_user_indices = np.argsort(similar_users)[::-1][1:top_k+1]
                
                # Aggregate recommendations from similar users
                recommendations = {}
                for similar_idx in similar_user_indices:
                    similar_user_id = user_item_matrix.index[similar_idx]
                    similar_user_items = user_item_matrix.loc[similar_user_id]
                    
                    for item_id, score in similar_user_items.items():
                        if item_id not in recommendations:
                            recommendations[item_id] = 0
                        recommendations[item_id] += score * similar_users[similar_idx]
                
                # Sort and return top K
                sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
                return [
                    {'item_id': item_id, 'score': float(score)}
                    for item_id, score in sorted_recs[:top_k]
                ]
            
            elif self.model['type'] == 'matrix_factorization':
                # Predict using matrix factorization
                if user_id not in self.user_item_matrix.index:
                    return []
                
                user_idx = self.user_item_matrix.index.get_loc(user_id)
                W = self.model['W']
                H = self.model['H']
                
                user_vector = W[user_idx]
                predicted_scores = np.dot(user_vector, H)
                
                item_indices = np.argsort(predicted_scores)[::-1][:top_k]
                recommendations = []
                for idx in item_indices:
                    item_id = self.user_item_matrix.columns[idx]
                    score = predicted_scores[idx]
                    recommendations.append({
                        'item_id': item_id,
                        'score': float(score)
                    })
                
                return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return []
    
    def ab_test(self, user_id: str, variant: str = 'A') -> List[Dict[str, Any]]:
        """A/B Testing Framework"""
        # Different recommendation strategies for A/B testing
        if variant == 'A':
            # Standard recommendations
            return self.recommend(user_id)
        elif variant == 'B':
            # Popular items + personalized
            popular_items = self.get_popular_items()
            personalized = self.recommend(user_id, top_k=5)
            return personalized + popular_items[:5]
        else:
            return self.recommend(user_id)
    
    def get_popular_items(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get popular items"""
        if not self.feature_store['item_features']:
            return []
        
        items = []
        for item_id, features in self.feature_store['item_features'].items():
            popularity_score = (
                features['view_count'] * 0.1 +
                features['click_count'] * 0.3 +
                features['purchase_count'] * 1.0
            )
            if features['rating_count'] > 0:
                avg_rating = features['total_rating'] / features['rating_count']
                popularity_score += avg_rating * 0.5
            
            items.append({
                'item_id': item_id,
                'score': popularity_score,
                'features': features
            })
        
        sorted_items = sorted(items, key=lambda x: x['score'], reverse=True)
        return sorted_items[:top_k]
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Performance Analytics"""
        total_users = len(self.feature_store['user_behaviors'])
        total_items = len(self.feature_store['item_features'])
        total_interactions = sum(
            len(behaviors) 
            for behaviors in self.feature_store['user_behaviors'].values()
        )
        
        return {
            'total_users': total_users,
            'total_items': total_items,
            'total_interactions': total_interactions,
            'model_loaded': self.model is not None,
            'matrix_shape': self.user_item_matrix.shape if self.user_item_matrix is not None else None
        }

