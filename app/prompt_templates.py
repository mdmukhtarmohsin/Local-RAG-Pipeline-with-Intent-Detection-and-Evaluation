"""
Intent-specific prompt templates for RAG responses
Each intent has customized prompts to provide the most relevant responses
"""

from typing import List, Dict, Any
from enum import Enum
from pydantic import BaseModel

from .retriever import RetrievalResult


class IntentType(str, Enum):
    TECHNICAL_SUPPORT = "technical_support"
    BILLING_ACCOUNT = "billing_account"
    FEATURE_REQUEST = "feature_request"


class PromptTemplate(BaseModel):
    intent: str
    system_prompt: str
    user_template: str
    context_format: str
    examples: List[str] = []


class PromptTemplateManager:
    def __init__(self):
        """Initialize prompt templates for each intent"""
        self.templates = self._create_templates()
    
    def _create_templates(self) -> Dict[str, PromptTemplate]:
        """Create intent-specific prompt templates"""
        templates = {}
        
        # Technical Support Template
        templates[IntentType.TECHNICAL_SUPPORT] = PromptTemplate(
            intent=IntentType.TECHNICAL_SUPPORT,
            system_prompt="""You are a technical support specialist with deep expertise in APIs, software integration, and troubleshooting. Your goal is to provide accurate, actionable solutions to technical problems.

Guidelines:
- Provide step-by-step solutions when possible
- Include relevant code examples or commands
- Reference specific documentation sections
- Offer multiple approaches when appropriate
- Ask clarifying questions if the issue is unclear
- Always prioritize security best practices
- Include links to relevant resources when available

Format your response clearly with headers, bullet points, and code blocks where helpful.""",
            
            user_template="""Technical Support Request: {user_query}

Relevant Documentation:
{context}

Please provide a comprehensive technical solution addressing the user's issue. Include:
1. Root cause analysis
2. Step-by-step solution
3. Code examples if applicable
4. Additional resources or documentation
5. Prevention tips for the future""",
            
            context_format="""## {source}
{content}

---""",
            
            examples=[
                "API authentication troubleshooting",
                "SDK integration guidance", 
                "Webhook configuration help",
                "Rate limiting solutions",
                "SSL certificate issues"
            ]
        )
        
        # Billing Account Template
        templates[IntentType.BILLING_ACCOUNT] = PromptTemplate(
            intent=IntentType.BILLING_ACCOUNT,
            system_prompt="""You are a billing and account management specialist. You help customers with subscription management, payment issues, account settings, and pricing questions.

Guidelines:
- Be empathetic and understanding about billing concerns
- Provide clear, specific pricing information
- Explain billing processes step-by-step
- Offer multiple resolution options when possible
- Direct users to appropriate self-service options
- Ensure compliance with refund and cancellation policies
- Protect customer financial information privacy

Always maintain a helpful and professional tone, especially for sensitive billing matters.""",
            
            user_template="""Billing/Account Question: {user_query}

Relevant Account Information:
{context}

Please provide a helpful response addressing the customer's billing or account question. Include:
1. Direct answer to their question
2. Step-by-step instructions if applicable
3. Relevant policy information
4. Alternative options or recommendations
5. Next steps or follow-up actions needed""",
            
            context_format="""## {source}
{content}

---""",
            
            examples=[
                "Subscription plan pricing",
                "Payment method updates",
                "Refund requests",
                "Account upgrades/downgrades",
                "Billing cycle questions"
            ]
        )
        
        # Feature Request Template
        templates[IntentType.FEATURE_REQUEST] = PromptTemplate(
            intent=IntentType.FEATURE_REQUEST,
            system_prompt="""You are a product specialist who helps customers understand our product roadmap and submit feature requests. You're knowledgeable about current features, planned developments, and how to channel customer feedback effectively.

Guidelines:
- Acknowledge and validate customer needs
- Provide information about similar existing features
- Reference relevant roadmap items when available
- Explain the feature request process clearly
- Suggest workarounds for immediate needs
- Encourage community engagement and voting
- Be realistic about timelines and feasibility

Maintain enthusiasm for product improvement while setting appropriate expectations.""",
            
            user_template="""Feature Request: {user_query}

Current Roadmap & Feature Information:
{context}

Please provide a comprehensive response about this feature request. Include:
1. Acknowledgment of the feature need
2. Information about similar existing features
3. Relevant roadmap timeline if available
4. How to submit/vote on feature requests
5. Possible workarounds or alternatives
6. Community resources for discussion""",
            
            context_format="""## {source}
{content}

---""",
            
            examples=[
                "New feature suggestions",
                "Product enhancement ideas",
                "Integration requests",
                "UI/UX improvements",
                "Mobile app features"
            ]
        )
        
        return templates
    
    def get_template(self, intent: str) -> PromptTemplate:
        """Get template for specific intent"""
        return self.templates.get(intent, self.templates[IntentType.TECHNICAL_SUPPORT])
    
    def format_context(self, retrieval_results: List[RetrievalResult], intent: str) -> str:
        """Format retrieval results into context for the prompt"""
        template = self.get_template(intent)
        
        if not retrieval_results:
            return "No relevant documentation found."
        
        formatted_context = []
        for result in retrieval_results:
            formatted_section = template.context_format.format(
                source=result.source,
                content=result.content
            )
            formatted_context.append(formatted_section)
        
        return "\n".join(formatted_context)
    
    def build_prompt(
        self, 
        user_query: str, 
        retrieval_results: List[RetrievalResult], 
        intent: str
    ) -> Dict[str, str]:
        """
        Build complete prompt for LLM
        
        Args:
            user_query: User's original question
            retrieval_results: Retrieved context documents
            intent: Detected intent
            
        Returns:
            Dictionary with system_prompt and user_prompt
        """
        template = self.get_template(intent)
        
        # Format context from retrieval results
        context = self.format_context(retrieval_results, intent)
        
        # Build user prompt
        user_prompt = template.user_template.format(
            user_query=user_query,
            context=context
        )
        
        return {
            "system_prompt": template.system_prompt,
            "user_prompt": user_prompt,
            "intent": intent
        }
    
    def get_fallback_prompt(self, user_query: str) -> Dict[str, str]:
        """Generate fallback prompt when intent is unclear"""
        return {
            "system_prompt": """You are a helpful customer support assistant. Provide accurate, helpful responses based on the user's question. If you're unsure about specific details, ask clarifying questions or direct the user to the appropriate support channel.""",
            "user_prompt": f"Customer Question: {user_query}\n\nPlease provide a helpful response or ask clarifying questions to better assist the customer.",
            "intent": "general"
        }
    
    def add_custom_template(self, intent: str, template: PromptTemplate):
        """Add or update a custom template"""
        self.templates[intent] = template
    
    def get_available_intents(self) -> List[str]:
        """Get list of available intent templates"""
        return list(self.templates.keys())
    
    def get_template_examples(self, intent: str) -> List[str]:
        """Get example use cases for an intent template"""
        template = self.get_template(intent)
        return template.examples


# Advanced prompt building functions
class AdvancedPromptBuilder:
    def __init__(self, template_manager: PromptTemplateManager = None):
        """Initialize with template manager"""
        self.template_manager = template_manager or PromptTemplateManager()
    
    def build_conversation_prompt(
        self,
        user_query: str,
        retrieval_results: List[RetrievalResult],
        intent: str,
        conversation_history: List[Dict[str, str]] = None,
        user_context: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """
        Build prompt with conversation history and user context
        
        Args:
            user_query: Current user question
            retrieval_results: Retrieved context
            intent: Detected intent
            conversation_history: Previous messages
            user_context: User profile/preferences
            
        Returns:
            Enhanced prompt with context
        """
        base_prompt = self.template_manager.build_prompt(
            user_query, retrieval_results, intent
        )
        
        # Add conversation history if available
        if conversation_history:
            history_text = self._format_conversation_history(conversation_history)
            base_prompt["user_prompt"] = f"""Conversation History:
{history_text}

Current Question:
{base_prompt["user_prompt"]}"""
        
        # Add user context if available
        if user_context:
            context_text = self._format_user_context(user_context)
            base_prompt["system_prompt"] += f"\n\nUser Context:\n{context_text}"
        
        return base_prompt
    
    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt"""
        formatted_history = []
        
        for message in history[-5:]:  # Last 5 messages
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted_history.append(f"{role.title()}: {content}")
        
        return "\n".join(formatted_history)
    
    def _format_user_context(self, context: Dict[str, Any]) -> str:
        """Format user context for prompt"""
        context_items = []
        
        if "plan" in context:
            context_items.append(f"Subscription Plan: {context['plan']}")
        
        if "api_usage" in context:
            context_items.append(f"API Usage Level: {context['api_usage']}")
        
        if "previous_issues" in context:
            context_items.append(f"Recent Issues: {', '.join(context['previous_issues'])}")
        
        return "\n".join(context_items)
    
    def build_multi_intent_prompt(
        self,
        user_query: str,
        intent_results: Dict[str, List[RetrievalResult]],
        primary_intent: str
    ) -> Dict[str, str]:
        """
        Build prompt when query spans multiple intents
        
        Args:
            user_query: User question
            intent_results: Results from multiple intents
            primary_intent: Main detected intent
            
        Returns:
            Combined prompt with multi-intent context
        """
        primary_template = self.template_manager.get_template(primary_intent)
        
        # Build combined context from all intents
        all_context = []
        for intent, results in intent_results.items():
            if results:
                intent_context = self.template_manager.format_context(results, intent)
                all_context.append(f"## {intent.replace('_', ' ').title()} Information\n{intent_context}")
        
        combined_context = "\n\n".join(all_context)
        
        # Use primary intent template but with multi-intent context
        user_prompt = primary_template.user_template.format(
            user_query=user_query,
            context=combined_context
        )
        
        enhanced_system_prompt = primary_template.system_prompt + """

Note: This query may involve multiple areas. Consider information from all relevant sections when providing your response."""
        
        return {
            "system_prompt": enhanced_system_prompt,
            "user_prompt": user_prompt,
            "intent": f"multi_intent_{primary_intent}"
        }


# Utility functions
def create_custom_template(
    intent: str,
    system_prompt: str,
    user_template: str,
    context_format: str = "## {source}\n{content}\n\n---",
    examples: List[str] = None
) -> PromptTemplate:
    """Create a custom prompt template"""
    return PromptTemplate(
        intent=intent,
        system_prompt=system_prompt,
        user_template=user_template,
        context_format=context_format,
        examples=examples or []
    )


def quick_prompt_build(
    query: str, 
    context: List[RetrievalResult], 
    intent: str
) -> Dict[str, str]:
    """Quick utility to build prompt"""
    manager = PromptTemplateManager()
    return manager.build_prompt(query, context, intent)


if __name__ == "__main__":
    # Test the prompt templates
    manager = PromptTemplateManager()
    
    # Mock retrieval results
    sample_results = [
        RetrievalResult(
            content="API keys are used for authentication. Include in Authorization header.",
            source="api_docs.txt",
            score=0.95,
            intent="technical_support"
        ),
        RetrievalResult(
            content="Common error codes: 401 Unauthorized, 403 Forbidden, 429 Rate Limited.",
            source="error_codes.txt", 
            score=0.88,
            intent="technical_support"
        )
    ]
    
    # Test each intent template
    test_queries = {
        "technical_support": "Why is my API key returning 403 errors?",
        "billing_account": "How much does the pro plan cost?", 
        "feature_request": "Can you add dark mode to the dashboard?"
    }
    
    print("Testing Prompt Templates:")
    print("=" * 50)
    
    for intent, query in test_queries.items():
        print(f"\nIntent: {intent}")
        print(f"Query: {query}")
        print("-" * 30)
        
        prompt = manager.build_prompt(query, sample_results, intent)
        
        print("System Prompt:")
        print(prompt["system_prompt"][:200] + "...")
        print("\nUser Prompt:")
        print(prompt["user_prompt"][:300] + "...")
        print("\n" + "="*50) 