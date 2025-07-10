"""
Intent Classification System for Customer Support Queries
Supports both transformer-based and LLM-based classification
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import yaml
from pydantic import BaseModel

from .llm_wrapper import LLMWrapper, LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentClass(str, Enum):
    TECHNICAL_SUPPORT = "technical_support"
    BILLING_ACCOUNT = "billing_account"
    FEATURE_REQUEST = "feature_request"


class IntentPrediction(BaseModel):
    intent: str
    confidence: float
    method: str  # transformer, llm, zero_shot


class IntentClassifier:
    def __init__(self, config_path: str = "app/config.yaml"):
        """Initialize intent classifier with configuration"""
        self.config = self._load_config(config_path)
        self.model = None
        self.classifier = None
        self.embedding_model = None
        self.llm_wrapper = None
        self.classes = self.config['intent']['classes']
        self.confidence_threshold = self.config['intent']['confidence_threshold']
        
        # Initialize based on model type
        model_type = self.config['intent']['model_type']
        if model_type == "transformer":
            self._setup_transformer_model()
        elif model_type == "llm":
            self._setup_llm_model()
        else:  # zero_shot
            self._setup_zero_shot_model()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'intent': {
                'model_type': 'transformer',
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'confidence_threshold': 0.7,
                'classes': ['technical_support', 'billing_account', 'feature_request']
            }
        }
    
    def _setup_transformer_model(self):
        """Setup transformer-based intent classification"""
        try:
            model_name = self.config['intent']['model_name']
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded transformer model: {model_name}")
            
            # Try to load pre-trained classifier
            self._load_classifier()
            
        except Exception as e:
            logger.error(f"Failed to setup transformer model: {e}")
    
    def _setup_llm_model(self):
        """Setup LLM-based intent classification"""
        try:
            self.llm_wrapper = LLMWrapper()
            logger.info("Setup LLM-based intent classification")
        except Exception as e:
            logger.error(f"Failed to setup LLM model: {e}")
    
    def _setup_zero_shot_model(self):
        """Setup zero-shot classification using LLM"""
        try:
            self.llm_wrapper = LLMWrapper()
            logger.info("Setup zero-shot intent classification")
        except Exception as e:
            logger.error(f"Failed to setup zero-shot model: {e}")
    
    def _load_classifier(self):
        """Load pre-trained classifier if available"""
        classifier_path = "embeddings/intent_classifier.pkl"
        if os.path.exists(classifier_path):
            try:
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                logger.info("Loaded pre-trained intent classifier")
            except Exception as e:
                logger.warning(f"Failed to load classifier: {e}")
    
    def _save_classifier(self):
        """Save trained classifier"""
        try:
            os.makedirs("embeddings", exist_ok=True)
            classifier_path = "embeddings/intent_classifier.pkl"
            with open(classifier_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            logger.info("Saved intent classifier")
        except Exception as e:
            logger.error(f"Failed to save classifier: {e}")
    
    def train_transformer_classifier(self, training_data: List[Tuple[str, str]]):
        """
        Train transformer-based classifier
        
        Args:
            training_data: List of (query, intent) tuples
        """
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return
        
        try:
            # Prepare data
            queries = [item[0] for item in training_data]
            labels = [item[1] for item in training_data]
            
            # Generate embeddings
            logger.info("Generating embeddings for training data...")
            embeddings = self.embedding_model.encode(queries)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Train classifier
            logger.info("Training logistic regression classifier...")
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
            self.classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Training completed. Accuracy: {accuracy:.3f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            # Save classifier
            self._save_classifier()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    async def predict_with_transformer(self, query: str) -> IntentPrediction:
        """Predict intent using transformer model"""
        if not self.embedding_model or not self.classifier:
            raise ValueError("Transformer model not properly initialized or trained")
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([query])
            
            # Predict
            prediction = self.classifier.predict(embedding)[0]
            probabilities = self.classifier.predict_proba(embedding)[0]
            
            # Get confidence for predicted class
            class_index = list(self.classifier.classes_).index(prediction)
            confidence = probabilities[class_index]
            
            return IntentPrediction(
                intent=prediction,
                confidence=float(confidence),
                method="transformer"
            )
            
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            # Fallback to zero-shot
            return await self.predict_with_zero_shot(query)
    
    async def predict_with_llm(self, query: str) -> IntentPrediction:
        """Predict intent using LLM with structured prompt"""
        if not self.llm_wrapper:
            raise ValueError("LLM wrapper not initialized")
        
        try:
            prompt = self._create_llm_classification_prompt(query)
            
            response = await self.llm_wrapper.generate(
                prompt, 
                provider=LLMProvider.AUTO,
                stream=False,  # Explicitly set stream=False
                temperature=0.1  # Low temperature for consistent classification
            )
            
            # Ensure we have an LLMResponse, not a generator
            if hasattr(response, 'content'):
                # Parse response
                intent, confidence = self._parse_llm_response(response.content)
                
                return IntentPrediction(
                    intent=intent,
                    confidence=confidence,
                    method="llm"
                )
            else:
                raise ValueError("Unexpected response type from LLM")
            
        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            # Fallback to default
            return IntentPrediction(
                intent=IntentClass.TECHNICAL_SUPPORT,
                confidence=0.5,
                method="fallback"
            )
    
    async def predict_with_zero_shot(self, query: str) -> IntentPrediction:
        """Predict intent using zero-shot classification"""
        if not self.llm_wrapper:
            raise ValueError("LLM wrapper not initialized")
        
        try:
            prompt = self._create_zero_shot_prompt(query)
            
            response = await self.llm_wrapper.generate(
                prompt,
                provider=LLMProvider.AUTO,
                stream=False,  # Explicitly set stream=False
                temperature=0.1
            )
            
            # Ensure we have an LLMResponse, not a generator
            if hasattr(response, 'content'):
                # Parse response
                intent, confidence = self._parse_llm_response(response.content)
                
                return IntentPrediction(
                    intent=intent,
                    confidence=confidence,
                    method="zero_shot"
                )
            else:
                raise ValueError("Unexpected response type from LLM")
            
        except Exception as e:
            logger.error(f"Zero-shot prediction failed: {e}")
            return IntentPrediction(
                intent=IntentClass.TECHNICAL_SUPPORT,
                confidence=0.5,
                method="fallback"
            )
    
    def _create_llm_classification_prompt(self, query: str) -> str:
        """Create structured prompt for LLM classification"""
        return f"""
You are an expert customer support intent classifier. Classify the following customer query into one of these categories:

1. technical_support: Questions about technical issues, API problems, bugs, integration help
2. billing_account: Questions about billing, payments, account management, subscriptions
3. feature_request: Requests for new features, improvements, or product suggestions

Customer Query: "{query}"

Respond ONLY with the classification in this exact format:
INTENT: [category]
CONFIDENCE: [0.0-1.0]

Classification:"""
    
    def _create_zero_shot_prompt(self, query: str) -> str:
        """Create zero-shot classification prompt"""
        return f"""
Classify this customer support query into one of these categories:
- technical_support: Technical issues, bugs, API problems
- billing_account: Billing, payments, account questions  
- feature_request: New feature requests or suggestions

Query: "{query}"

Reply with only the category name and confidence (0.0-1.0):
INTENT: [category]
CONFIDENCE: [score]"""
    
    def _parse_llm_response(self, response: str) -> Tuple[str, float]:
        """Parse LLM response to extract intent and confidence"""
        try:
            lines = response.strip().split('\n')
            intent = None
            confidence = 0.5
            
            for line in lines:
                line = line.strip()
                if line.startswith('INTENT:'):
                    intent = line.replace('INTENT:', '').strip().lower()
                elif line.startswith('CONFIDENCE:'):
                    confidence_str = line.replace('CONFIDENCE:', '').strip()
                    try:
                        confidence = float(confidence_str)
                    except ValueError:
                        confidence = 0.5
            
            # Validate intent and ensure it's not None
            if intent is None or intent not in self.classes:
                # Find closest match or use default
                intent = self._find_closest_intent(intent or "")
            
            return intent, confidence
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return IntentClass.TECHNICAL_SUPPORT, 0.5
    
    def _find_closest_intent(self, predicted_intent: str) -> str:
        """Find closest valid intent class"""
        # Simple keyword matching
        if 'tech' in predicted_intent or 'api' in predicted_intent or 'bug' in predicted_intent:
            return IntentClass.TECHNICAL_SUPPORT
        elif 'bill' in predicted_intent or 'pay' in predicted_intent or 'account' in predicted_intent:
            return IntentClass.BILLING_ACCOUNT
        elif 'feature' in predicted_intent or 'request' in predicted_intent:
            return IntentClass.FEATURE_REQUEST
        else:
            return IntentClass.TECHNICAL_SUPPORT  # Default
    
    async def predict(self, query: str) -> IntentPrediction:
        """
        Main prediction method that routes to appropriate classifier
        
        Args:
            query: Customer query text
            
        Returns:
            IntentPrediction with intent, confidence, and method
        """
        model_type = self.config['intent']['model_type']
        
        try:
            if model_type == "transformer":
                return await self.predict_with_transformer(query)
            elif model_type == "llm":
                return await self.predict_with_llm(query)
            else:  # zero_shot
                return await self.predict_with_zero_shot(query)
                
        except Exception as e:
            logger.error(f"Intent prediction failed: {e}")
            return IntentPrediction(
                intent=IntentClass.TECHNICAL_SUPPORT,
                confidence=0.5,
                method="fallback"
            )
    
    def is_confident(self, prediction: IntentPrediction) -> bool:
        """Check if prediction meets confidence threshold"""
        return prediction.confidence >= self.confidence_threshold


# Utility function for easy access
async def classify_intent(query: str, config_path: str = "app/config.yaml") -> IntentPrediction:
    """Convenience function for intent classification"""
    classifier = IntentClassifier(config_path)
    return await classifier.predict(query)


# Sample training data generator
def generate_sample_training_data() -> List[Tuple[str, str]]:
    """Generate sample training data for intent classification"""
    return [
        # Technical Support
        ("Why is the API returning a 403 error?", "technical_support"),
        ("How do I integrate your SDK with React?", "technical_support"),
        ("The webhook is not working", "technical_support"),
        ("I'm getting a connection timeout", "technical_support"),
        ("API documentation is unclear", "technical_support"),
        ("Bug in the dashboard", "technical_support"),
        ("Authentication not working", "technical_support"),
        ("Rate limiting issues", "technical_support"),
        ("Database connection error", "technical_support"),
        ("SSL certificate problem", "technical_support"),
        
        # Billing Account
        ("How much does the premium plan cost?", "billing_account"),
        ("I want to cancel my subscription", "billing_account"),
        ("Billing question about my account", "billing_account"),
        ("Payment failed", "billing_account"),
        ("Refund request", "billing_account"),
        ("Change payment method", "billing_account"),
        ("Invoice is incorrect", "billing_account"),
        ("Upgrade to enterprise plan", "billing_account"),
        ("Account billing history", "billing_account"),
        ("Credit card expired", "billing_account"),
        
        # Feature Request
        ("Can you add dark mode?", "feature_request"),
        ("Please support GraphQL", "feature_request"),
        ("New feature suggestion", "feature_request"),
        ("Add mobile app support", "feature_request"),
        ("Bulk export feature needed", "feature_request"),
        ("Integration with Slack", "feature_request"),
        ("Better search functionality", "feature_request"),
        ("Custom dashboard widgets", "feature_request"),
        ("Advanced filtering options", "feature_request"),
        ("Real-time notifications", "feature_request"),
    ]


if __name__ == "__main__":
    import asyncio
    
    async def test_intent_classifier():
        """Test the intent classifier"""
        classifier = IntentClassifier()
        
        # Test queries
        test_queries = [
            "Why is my API key not working?",
            "I need to update my billing information",
            "Can you add support for webhooks?",
            "The app crashes when I click submit",
            "How much does the pro plan cost?",
            "Please add a dark theme option"
        ]
        
        print("Testing Intent Classification:")
        print("-" * 50)
        
        for query in test_queries:
            prediction = await classifier.predict(query)
            print(f"Query: {query}")
            print(f"Intent: {prediction.intent}")
            print(f"Confidence: {prediction.confidence:.3f}")
            print(f"Method: {prediction.method}")
            print(f"Confident: {classifier.is_confident(prediction)}")
            print()
    
    asyncio.run(test_intent_classifier()) 