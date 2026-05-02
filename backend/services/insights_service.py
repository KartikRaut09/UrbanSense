from typing import Dict, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class InsightGenerationService:
    """Service for AI-powered insight generation"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
    
    def generate_region_insights(self, analysis_data: Dict) -> Dict:
        """Generate comprehensive insights for a region"""
        insights = {
            "critical_issues": self._identify_critical_issues(analysis_data),
            "opportunities": self._identify_opportunities(analysis_data),
            "recommendations": self._generate_recommendations(analysis_data),
            "priority_actions": self._rank_actions(analysis_data)
        }
        return insights
    
    def _identify_critical_issues(self, data: Dict) -> List[str]:
        """Identify critical urban issues"""
        issues = []
        
        if data.get("fire_risk", 0) > 0.7:
            issues.append("Critical fire risk in high-density areas")
        
        if data.get("flood_risk", 0) > 0.6:
            issues.append("Significant flood vulnerability detected")
        
        if data.get("accessibility_risk", 0) > 0.65:
            issues.append("Poor emergency response accessibility")
        
        if data.get("slum_growth_rate", 0) > 0.1:
            issues.append("Rapid informal settlement expansion")
        
        return issues
    
    def _identify_opportunities(self, data: Dict) -> List[str]:
        """Identify development opportunities"""
        opportunities = []
        
        if data.get("high_connectivity_zones"):
            opportunities.append("High connectivity zones for formal housing")
        
        if data.get("improved_infrastructure_potential"):
            opportunities.append("Infrastructure improvement zones")
        
        if data.get("livelihood_development_spots"):
            opportunities.append("Economic development opportunity areas")
        
        return opportunities
    
    def _generate_recommendations(self, data: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Fire risk recommendations
        if data.get("fire_risk", 0) > 0.6:
            recommendations.append({
                "category": "Fire Safety",
                "action": "Install fire hydrant network",
                "priority": "Critical",
                "estimated_cost": "$500K-$750K",
                "impact": 0.85
            })
        
        # Accessibility recommendations
        if data.get("accessibility_risk", 0) > 0.6:
            recommendations.append({
                "category": "Infrastructure",
                "action": "Develop emergency access roads",
                "priority": "High",
                "estimated_cost": "$750K-$1.2M",
                "impact": 0.80
            })
        
        # Drainage recommendations
        if data.get("flood_risk", 0) > 0.5:
            recommendations.append({
                "category": "Flood Management",
                "action": "Upgrade drainage systems",
                "priority": "High",
                "estimated_cost": "$300K-$500K",
                "impact": 0.70
            })
        
        return recommendations
    
    def _rank_actions(self, data: Dict) -> List[Dict]:
        """Rank actions by priority"""
        actions = [
            {"action": "Conduct baseline survey", "priority": 1, "timeline": "Immediate"},
            {"action": "Establish community engagement", "priority": 2, "timeline": "1-2 weeks"},
            {"action": "Infrastructure assessment", "priority": 3, "timeline": "2-4 weeks"},
            {"action": "Risk mitigation planning", "priority": 4, "timeline": "1 month"},
            {"action": "Implementation planning", "priority": 5, "timeline": "1-2 months"}
        ]
        return actions

class DataProcessingService:
    """Service for data processing and transformation"""
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = "minmax") -> np.ndarray:
        """Normalize data to 0-1 range"""
        if method == "minmax":
            return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        elif method == "zscore":
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        return data
    
    @staticmethod
    def aggregate_regions(region_data: List[Dict]) -> Dict:
        """Aggregate data from multiple regions"""
        aggregated = {
            "total_area_km2": sum(r.get("area_km2", 0) for r in region_data),
            "avg_fire_risk": np.mean([r.get("fire_risk", 0) for r in region_data]),
            "avg_flood_risk": np.mean([r.get("flood_risk", 0) for r in region_data]),
            "total_population": sum(r.get("population", 0) for r in region_data),
            "region_count": len(region_data)
        }
        return aggregated
