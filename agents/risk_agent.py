"""
RiskAgent
Responsible for converting fraud probabilities into risk levels and making decisions
"""

from typing import Dict, Any, List
from enum import Enum


class RiskLevel(Enum):
    """Risk level categories for transactions."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Action(Enum):
    """Actions to take based on risk assessment."""
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    FLAG = "FLAG"
    BLOCK = "BLOCK"


class RiskAgent:
    """
    Agent that assesses risk levels and determines appropriate actions.
    Converts fraud probabilities into actionable risk categories.
    """
    
    def __init__(
        self,
        low_threshold: float = 0.3,
        medium_threshold: float = 0.6,
        high_threshold: float = 0.85,
        verbose: bool = True
    ):
        """
        Initialize the Risk Agent with configurable thresholds.
        
        The agent categorizes fraud probabilities into risk levels:
        - LOW: probability < low_threshold
        - MEDIUM: low_threshold <= probability < medium_threshold
        - HIGH: probability >= medium_threshold
        
        And maps them to decisions:
        - LOW -> "allow"
        - MEDIUM -> "review"
        - HIGH -> "flag"
        
        Args:
            low_threshold: Probability threshold for LOW -> MEDIUM risk (default: 0.3)
            medium_threshold: Probability threshold for MEDIUM -> HIGH risk (default: 0.6)
            high_threshold: Probability threshold for HIGH -> CRITICAL risk (default: 0.85, for extended functionality)
            verbose: Whether to print risk assessment information
        """
        self.verbose = verbose
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.assessments_made = 0
        self.risk_counts = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 0,
            RiskLevel.HIGH: 0,
            RiskLevel.CRITICAL: 0
        }
        
        if self.verbose:
            print(f"[RiskAgent] Initialized with thresholds:")
            print(f"  LOW: < {low_threshold}")
            print(f"  MEDIUM: {low_threshold} - {medium_threshold}")
            print(f"  HIGH: >= {medium_threshold}")
    
    def assess_risk(self, probability: float) -> tuple:
        """
        Assess risk level and determine decision for a given fraud probability.
        
        This is the main method that converts a fraud probability into:
        - risk_level: "LOW", "MEDIUM", or "HIGH"
        - decision: "allow", "review", or "flag"
        
        Args:
            probability: Fraud probability as float (0.0 to 1.0)
            
        Returns:
            Tuple of (risk_level, decision) where:
                - risk_level is str: "LOW", "MEDIUM", or "HIGH"
                - decision is str: "allow", "review", or "flag"
                
        Example:
            >>> agent = RiskAgent(low_threshold=0.3, medium_threshold=0.6)
            >>> risk_level, decision = agent.assess_risk(0.45)
            >>> print(f"Risk: {risk_level}, Decision: {decision}")
            Risk: MEDIUM, Decision: review
        """
        # Determine risk level based on thresholds
        if probability < self.low_threshold:
            risk_level = "LOW"
            decision = "allow"
        elif probability < self.medium_threshold:
            risk_level = "MEDIUM"
            decision = "review"
        else:
            risk_level = "HIGH"
            decision = "flag"
        
        self.assessments_made += 1
        
        return (risk_level, decision)
    
    def assess_risk_level(self, fraud_probability: float) -> RiskLevel:
        """
        Assess the risk level based on fraud probability.
        
        Args:
            fraud_probability: Probability of fraud (0.0 to 1.0)
            
        Returns:
            RiskLevel enum value
        """
        if fraud_probability < self.low_threshold:
            return RiskLevel.LOW
        elif fraud_probability < self.medium_threshold:
            return RiskLevel.MEDIUM
        elif fraud_probability < self.high_threshold:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def determine_action(self, risk_level: RiskLevel) -> Action:
        """
        Determine the action to take based on risk level.
        
        Args:
            risk_level: RiskLevel enum value
            
        Returns:
            Action enum value
        """
        action_map = {
            RiskLevel.LOW: Action.ALLOW,
            RiskLevel.MEDIUM: Action.REVIEW,
            RiskLevel.HIGH: Action.FLAG,
            RiskLevel.CRITICAL: Action.BLOCK
        }
        
        return action_map[risk_level]
    
    def assess_transaction(
        self,
        fraud_probability: float,
        transaction_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Perform a complete risk assessment for a transaction.
        
        Args:
            fraud_probability: Probability of fraud (0.0 to 1.0)
            transaction_data: Optional transaction data for additional context
            
        Returns:
            Dictionary with complete risk assessment
        """
        # Assess risk level
        risk_level = self.assess_risk_level(fraud_probability)
        
        # Determine action
        action = self.determine_action(risk_level)
        
        # Update statistics
        self.assessments_made += 1
        self.risk_counts[risk_level] += 1
        
        # Create assessment result
        assessment = {
            'fraud_probability': fraud_probability,
            'risk_level': risk_level.value,
            'action': action.value,
            'is_suspicious': risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            'requires_review': action in [Action.REVIEW, Action.FLAG],
            'should_block': action == Action.BLOCK,
            'assessment_id': self.assessments_made
        }
        
        # Add risk description
        assessment['risk_description'] = self._get_risk_description(risk_level, fraud_probability)
        
        # Add action recommendation
        assessment['action_recommendation'] = self._get_action_recommendation(action, risk_level)
        
        # Add transaction context if provided
        if transaction_data:
            assessment['transaction_context'] = self._analyze_transaction_context(
                transaction_data, 
                risk_level
            )
        
        return assessment
    
    def assess_batch(
        self,
        fraud_probabilities: List[float],
        transaction_data_list: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform risk assessments for multiple transactions.
        
        Args:
            fraud_probabilities: List of fraud probabilities
            transaction_data_list: Optional list of transaction data
            
        Returns:
            List of risk assessment dictionaries
        """
        assessments = []
        
        for i, fraud_prob in enumerate(fraud_probabilities):
            trans_data = None
            if transaction_data_list and i < len(transaction_data_list):
                trans_data = transaction_data_list[i]
            
            assessment = self.assess_transaction(fraud_prob, trans_data)
            assessment['batch_index'] = i
            assessments.append(assessment)
        
        return assessments
    
    def _get_risk_description(self, risk_level: RiskLevel, probability: float) -> str:
        """Generate a human-readable risk description."""
        descriptions = {
            RiskLevel.LOW: f"Low risk transaction with {probability*100:.1f}% fraud probability. Transaction appears legitimate.",
            RiskLevel.MEDIUM: f"Medium risk transaction with {probability*100:.1f}% fraud probability. Warrants monitoring.",
            RiskLevel.HIGH: f"High risk transaction with {probability*100:.1f}% fraud probability. Manual review recommended.",
            RiskLevel.CRITICAL: f"Critical risk transaction with {probability*100:.1f}% fraud probability. Immediate action required."
        }
        
        return descriptions[risk_level]
    
    def _get_action_recommendation(self, action: Action, risk_level: RiskLevel) -> str:
        """Generate action recommendation text."""
        recommendations = {
            Action.ALLOW: "Process transaction normally. No additional action required.",
            Action.REVIEW: "Add to review queue for monitoring. Allow transaction but track for patterns.",
            Action.FLAG: "Flag for immediate review. Consider delaying transaction until review is complete.",
            Action.BLOCK: "Block transaction immediately. Contact customer for verification before processing."
        }
        
        return recommendations[action]
    
    def _analyze_transaction_context(
        self,
        transaction_data: Dict[str, Any],
        risk_level: RiskLevel
    ) -> Dict[str, Any]:
        """Analyze transaction context for additional insights."""
        context = {}
        
        # Analyze amount if present
        if 'Amount' in transaction_data:
            amount = transaction_data['Amount']
            context['amount'] = float(amount)
            
            # Flag high amounts with high risk
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] and amount > 1000:
                context['high_value_alert'] = True
                context['alert_reason'] = "High-value transaction with elevated fraud risk"
        
        # Analyze time if present
        if 'Time' in transaction_data:
            time = transaction_data['Time']
            context['time'] = float(time)
            
            # Check for unusual hours (converted to hours from seconds)
            hours = (time / 3600) % 24
            if (hours < 6 or hours > 22) and risk_level != RiskLevel.LOW:
                context['unusual_time_alert'] = True
                context['transaction_hour'] = int(hours)
        
        return context
    
    def filter_suspicious_transactions(
        self,
        assessments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter assessments to return only suspicious transactions.
        
        Args:
            assessments: List of risk assessment dictionaries
            
        Returns:
            List of suspicious transaction assessments
        """
        return [a for a in assessments if a['is_suspicious']]
    
    def filter_by_action(
        self,
        assessments: List[Dict[str, Any]],
        action: Action
    ) -> List[Dict[str, Any]]:
        """
        Filter assessments by specific action.
        
        Args:
            assessments: List of risk assessment dictionaries
            action: Action to filter by
            
        Returns:
            Filtered list of assessments
        """
        return [a for a in assessments if a['action'] == action.value]
    
    def get_risk_summary(self, assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of risk assessments.
        
        Args:
            assessments: List of risk assessment dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        if not assessments:
            return {
                'total_transactions': 0,
                'risk_distribution': {},
                'action_distribution': {},
                'suspicious_count': 0,
                'suspicious_rate': 0.0
            }
        
        # Count risk levels
        risk_dist = {}
        for level in RiskLevel:
            count = sum(1 for a in assessments if a['risk_level'] == level.value)
            risk_dist[level.value] = count
        
        # Count actions
        action_dist = {}
        for action in Action:
            count = sum(1 for a in assessments if a['action'] == action.value)
            action_dist[action.value] = count
        
        # Count suspicious
        suspicious_count = sum(1 for a in assessments if a['is_suspicious'])
        
        return {
            'total_transactions': len(assessments),
            'risk_distribution': risk_dist,
            'action_distribution': action_dist,
            'suspicious_count': suspicious_count,
            'suspicious_rate': (suspicious_count / len(assessments)) * 100,
            'average_fraud_probability': sum(a['fraud_probability'] for a in assessments) / len(assessments)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's assessment activity.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'assessments_made': self.assessments_made,
            'risk_counts': {level.value: count for level, count in self.risk_counts.items()},
            'thresholds': {
                'low': self.low_threshold,
                'medium': self.medium_threshold,
                'high': self.high_threshold
            }
        }
    
if __name__ == "__main__":
    # Test the RiskAgent
    print("Testing RiskAgent...")
    
    # Initialize agent with default thresholds
    agent = RiskAgent(low_threshold=0.3, medium_threshold=0.6, verbose=True)
    
    # Test main assess_risk method
    print("\n1. Testing assess_risk() method (main method)...")
    test_probabilities = [0.1, 0.25, 0.45, 0.65, 0.85]
    
    for prob in test_probabilities:
        risk_level, decision = agent.assess_risk(prob)
        print(f"   Probability {prob:.2f} -> Risk: {risk_level}, Decision: {decision}")
    
    # Test edge cases
    print("\n2. Testing edge cases...")
    edge_cases = [0.0, 0.3, 0.6, 1.0]
    for prob in edge_cases:
        risk_level, decision = agent.assess_risk(prob)
        print(f"   Probability {prob:.2f} -> Risk: {risk_level}, Decision: {decision}")
    
    # Test with custom thresholds
    print("\n3. Testing with custom thresholds...")
    custom_agent = RiskAgent(low_threshold=0.4, medium_threshold=0.7, verbose=False)
    risk_level, decision = custom_agent.assess_risk(0.5)
    print(f"   Probability 0.50 with custom thresholds (0.4, 0.7)")
    print(f"   Risk: {risk_level}, Decision: {decision}")
    
    # Test batch assessment (enhanced feature)
    print("\n4. Testing batch assessment (bonus feature)...")
    probabilities = [0.1, 0.4, 0.7, 0.95, 0.2]
    batch_assessments = agent.assess_batch(probabilities)
    print(f"   Processed {len(batch_assessments)} transactions")
    
    for i, a in enumerate(batch_assessments):
        print(f"   Transaction {i+1}: {a['risk_level']} - {a['action']}")
    
    # Test statistics
    print("\n5. Agent statistics...")
    stats = agent.get_statistics()
    print(f"   Total assessments made: {stats['assessments_made']}")
    print(f"   Thresholds: LOW={stats['thresholds']['low']}, MEDIUM={stats['thresholds']['medium']}")
    
    print("\n✓ RiskAgent test complete!")
    print("\n1. Assessing single transaction...")
    assessment = agent.assess_transaction(
        fraud_probability=0.75,
        transaction_data={'Amount': 1500.0, 'Time': 3600.0}
    )
    print(f"   Risk Level: {assessment['risk_level']}")
    print(f"   Action: {assessment['action']}")
    print(f"   Description: {assessment['risk_description']}")
    
    # Test batch assessment
    print("\n2. Assessing batch of transactions...")
    probabilities = [0.1, 0.4, 0.7, 0.95, 0.2]
    batch_assessments = agent.assess_batch(probabilities)
    print(f"   Processed {len(batch_assessments)} transactions")
    
    for i, a in enumerate(batch_assessments):
        print(f"   Transaction {i+1}: {a['risk_level']} - {a['action']}")
    
    # Test filtering
    print("\n3. Filtering suspicious transactions...")
    suspicious = agent.filter_suspicious_transactions(batch_assessments)
    print(f"   Found {len(suspicious)} suspicious transactions")
    
    # Test summary
    print("\n4. Generating risk summary...")
    summary = agent.get_risk_summary(batch_assessments)
    print(f"   Total: {summary['total_transactions']}")
    print(f"   Suspicious rate: {summary['suspicious_rate']:.1f}%")
    print(f"   Risk distribution: {summary['risk_distribution']}")
    
    # Test statistics
    print("\n5. Agent statistics...")
    stats = agent.get_statistics()
    print(f"   Assessments made: {stats['assessments_made']}")
    print(f"   Risk counts: {stats['risk_counts']}")
    
    print("\n✓ RiskAgent test complete!")
