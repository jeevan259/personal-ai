"""
Manages conversation flow and context
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path  # ADD THIS IMPORT

from src.llm.voice_optimized_llm import VoiceOptimizedLLM
from src.memory.audio_memory import AudioMemory
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


@dataclass
class ConversationState:
    """Represents the current state of a conversation"""
    user_id: str
    session_id: str
    messages: List[Dict[str, str]]
    context: Dict[str, Any]
    turn_count: int = 0
    last_interaction: Optional[datetime] = None
    active_skill: Optional[str] = None


class ConversationManager:
    """Manages conversation flow, context, and state"""
    
    def __init__(self, llm: VoiceOptimizedLLM, memory: AudioMemory, config: Dict[str, Any]):
        self.llm = llm
        self.memory = memory
        self.config = config
        
        # Conversation states by user/session
        self.conversations: Dict[str, ConversationState] = {}
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from files"""
        templates = {}
        template_dir = Path("src/core/prompt_templates")
        
        # Create directory if it doesn't exist
        template_dir.mkdir(parents=True, exist_ok=True)
        
        for template_file in template_dir.glob("*.jinja2"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    templates[template_file.stem] = f.read()
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")
        
        # If no templates loaded, create defaults
        if not templates:
            templates = {
                'voice_system': "You are a helpful voice assistant.",
                'short_response': "Provide a brief response.",
                'conversation': "Continue the conversation naturally."
            }
        
        return templates
    
    def get_or_create_conversation(self, user_id: str, session_id: str) -> ConversationState:
        """Get existing conversation or create new one"""
        key = f"{user_id}_{session_id}"
        
        if key not in self.conversations:
            self.conversations[key] = ConversationState(
                user_id=user_id,
                session_id=session_id,
                messages=[],
                context={
                    'user_preferences': {},
                    'conversation_history': [],
                    'current_topic': '',
                    'mood': 'neutral'
                },
                turn_count=0,
                last_interaction=datetime.now()
            )
            
            # Load user preferences from memory
            self._load_user_preferences(user_id, self.conversations[key])
            
            logger.info(f"Created new conversation: {key}")
        
        return self.conversations[key]
    
    def _load_user_preferences(self, user_id: str, conversation: ConversationState):
        """Load user preferences from memory"""
        # This would typically load from a database
        # For now, using default preferences
        conversation.context['user_preferences'] = {
            'preferred_name': 'User',
            'speech_rate': 'normal',
            'formality': 'casual',
            'topics_of_interest': []
        }
    
    async def generate_response(
        self,
        user_input: str,
        external_context: Dict[str, Any],
        voice_context: Any
    ) -> Dict[str, Any]:
        """Generate a response to user input"""
        
        # Get or create conversation
        conversation = self.get_or_create_conversation(
            voice_context.user_id,
            voice_context.session_id
        )
        
        # Update conversation state
        conversation.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        conversation.turn_count += 1
        conversation.last_interaction = datetime.now()
        
        # Prepare context for LLM
        llm_context = self._prepare_llm_context(
            user_input,
            conversation,
            external_context,
            voice_context
        )
        
        # Generate response using LLM
        try:
            response = await self.llm.generate_response(
                prompt=user_input,
                context=llm_context,
                conversation_history=conversation.messages[-10:],  # Last 10 messages
                persona=voice_context.persona
            )
            
            # Update conversation with response
            conversation.messages.append({
                'role': 'assistant',
                'content': response['text'],
                'metadata': response.get('metadata', {})
            })
            
            # Update conversation context
            self._update_conversation_context(conversation, response)
            
            # Check if response indicates a skill should be activated
            skill_response = await self._check_for_skill_activation(
                response['text'],
                user_input,
                conversation
            )
            
            if skill_response:
                # Override LLM response with skill response
                response = skill_response
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            
            # Fallback response
            return {
                'text': "I'm having trouble processing that right now. Could you please repeat?",
                'metadata': {
                    'error': str(e),
                    'is_fallback': True
                }
            }
    
    def _prepare_llm_context(
        self,
        user_input: str,
        conversation: ConversationState,
        external_context: Dict[str, Any],
        voice_context: Any
    ) -> Dict[str, Any]:
        """Prepare comprehensive context for the LLM"""
        
        # Extract entities and intent from user input
        entities = self._extract_entities(user_input)
        intent = self._classify_intent(user_input)
        
        # Build context dictionary
        context = {
            # User information
            'user_id': voice_context.user_id,
            'persona': voice_context.persona,
            'user_preferences': conversation.context['user_preferences'],
            
            # Conversation state
            'turn_count': conversation.turn_count,
            'current_topic': conversation.context['current_topic'],
            'conversation_mood': conversation.context['mood'],
            
            # Current input analysis
            'user_input': user_input,
            'detected_intent': intent,
            'extracted_entities': entities,
            
            # External context
            'memory_context': external_context,
            'timestamp': datetime.now().isoformat(),
            
            # System state
            'requires_short_response': self._should_use_short_response(user_input),
            'allow_proactive_suggestions': conversation.turn_count > 2,
        }
        
        # Add recent conversation history (last 3 exchanges)
        if len(conversation.messages) >= 6:
            context['recent_history'] = conversation.messages[-6:]
        else:
            context['recent_history'] = conversation.messages
        
        return context
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        # This is a simplified version
        # In production, use NER models or rule-based extraction
        
        entities = {
            'dates': [],
            'times': [],
            'locations': [],
            'people': [],
            'topics': []
        }
        
        # Simple keyword matching (production would use proper NER)
        date_keywords = ['today', 'tomorrow', 'yesterday', 'week', 'month', 'year']
        time_keywords = ['morning', 'afternoon', 'evening', 'night', 'hour', 'minute']
        
        words = text.lower().split()
        
        for word in words:
            if word in date_keywords:
                entities['dates'].append(word)
            elif word in time_keywords:
                entities['times'].append(word)
        
        return entities
    
    def _classify_intent(self, text: str) -> str:
        """Classify the intent of user input"""
        # Simplified intent classification
        # Production would use ML-based classification
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['weather', 'temperature', 'forecast']):
            return 'weather_inquiry'
        elif any(word in text_lower for word in ['time', 'clock', 'hour']):
            return 'time_inquiry'
        elif any(word in text_lower for word in ['remind', 'remember', 'alert']):
            return 'set_reminder'
        elif any(word in text_lower for word in ['tell me about', 'what is', 'who is']):
            return 'knowledge_query'
        elif any(word in text_lower for word in ['thank', 'thanks', 'appreciate']):
            return 'gratitude'
        elif any(word in text_lower for word in ['bye', 'goodbye', 'see you']):
            return 'goodbye'
        else:
            return 'general_conversation'
    
    def _should_use_short_response(self, text: str) -> bool:
        """Determine if a short response is appropriate"""
        # Short responses for brief questions or commands
        
        short_triggers = [
            'what time',
            'weather',
            'set timer',
            'remind me',
            'stop',
            'cancel'
        ]
        
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in short_triggers)
    
    def _update_conversation_context(self, conversation: ConversationState, response: Dict[str, Any]):
        """Update conversation context based on response"""
        
        # Update current topic if response has strong topic indication
        if 'metadata' in response and 'primary_topic' in response['metadata']:
            conversation.context['current_topic'] = response['metadata']['primary_topic']
        
        # Update mood based on response sentiment
        if 'metadata' in response and 'sentiment' in response['metadata']:
            conversation.context['mood'] = response['metadata']['sentiment']
        
        # Store conversation snippet in memory
        if len(conversation.messages) >= 2:
            last_exchange = conversation.messages[-2:]
            conversation.context['conversation_history'].append(last_exchange)
    
    async def _check_for_skill_activation(
        self,
        response_text: str,
        user_input: str,
        conversation: ConversationState
    ) -> Optional[Dict[str, Any]]:
        """Check if a skill should be activated based on the conversation"""
        
        # This would integrate with the skill registry
        # For now, return None (no skill activation)
        
        return None
    
    def end_conversation(self, user_id: str, session_id: str):
        """End a conversation and clean up resources"""
        key = f"{user_id}_{session_id}"
        
        if key in self.conversations:
            # Save conversation to long-term memory
            conversation = self.conversations.pop(key)
            
            # Here you would typically save to database
            logger.info(f"Ended conversation: {key} (turns: {conversation.turn_count})")
    
    def get_conversation_summary(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get summary of a conversation"""
        key = f"{user_id}_{session_id}"
        
        if key in self.conversations:
            conversation = self.conversations[key]
            
            return {
                'user_id': user_id,
                'session_id': session_id,
                'turn_count': conversation.turn_count,
                'start_time': conversation.messages[0]['timestamp'] if conversation.messages else None,
                'last_interaction': conversation.last_interaction.isoformat() if conversation.last_interaction else None,
                'current_topic': conversation.context['current_topic'],
                'active_skill': conversation.active_skill
            }
        
        return {}


# Simple version for testing
class SimpleConversationManager:
    """Simplified conversation manager for testing"""
    
    def __init__(self):
        self.conversations = {}
        
    async def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simple response generation"""
        # For testing, just echo back
        return {
            'text': f"You said: {user_input}",
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'is_test': True
            }
        }