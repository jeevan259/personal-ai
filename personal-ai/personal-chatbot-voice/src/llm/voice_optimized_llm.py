"""
Voice-optimized LLM with short responses and prosody hints
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


# Define ResponseShortener class here since it's referenced
class ResponseShortener:
    """Utility for shortening LLM responses for voice output"""
    
    def __init__(self, max_sentences: int = 2, max_words: int = 50):
        self.max_sentences = max_sentences
        self.max_words = max_words
    
    def shorten(self, text: str, max_sentences: Optional[int] = None, max_words: Optional[int] = None) -> str:
        """Shorten text for voice output"""
        if max_sentences is None:
            max_sentences = self.max_sentences
        if max_words is None:
            max_words = self.max_words
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        # Take only the first few sentences
        shortened = ' '.join(sentences[:max_sentences])
        
        # Limit words
        words = shortened.split()
        if len(words) > max_words:
            shortened = ' '.join(words[:max_words]) + '...'
        
        return shortened.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        import re
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Filter out empty strings and strip
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Add punctuation back
        sentences = [s + '.' for s in sentences]
        
        return sentences


@dataclass
class LLMResponse:
    """Structured LLM response for voice applications"""
    text: str
    metadata: Dict[str, Any]
    prosody_hints: Optional[Dict[str, Any]] = None
    should_interrupt: bool = False
    requires_confirmation: bool = False


class VoiceOptimizedLLM:
    """LLM optimized for voice interactions with short responses"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # LLM clients
        self.clients = {}
        self.active_client = config.get('model', 'gpt-4-turbo')
        
        # Response optimization
        self.max_tokens = config.get('max_tokens', 150)
        self.temperature = config.get('temperature', 0.7)
        self.response_shortener = ResponseShortener()  # Now this will work
        
        # Caching
        self.cache_enabled = config.get('caching', {}).get('enabled', True)
        self.response_cache = {}
        
        # Prosody hints generation
        self.prosody_enabled = config.get('prosody_hints', True)
        
    async def initialize(self):
        """Initialize LLM clients"""
        logger.info("Initializing VoiceOptimizedLLM...")
        
        try:
            # Initialize OpenAI client
            if await self._init_openai():
                logger.info("OpenAI client initialized")
            
            # Initialize Anthropic client (if configured)
            if self.config.get('anthropic_api_key'):
                if await self._init_anthropic():
                    logger.info("Anthropic client initialized")
            
            # Initialize local LLM (if configured)
            if self.config.get('local_llm_path'):
                if await self._init_local_llm():
                    logger.info("Local LLM initialized")
            
            if not self.clients:
                raise RuntimeError("No LLM clients could be initialized")
            
            logger.info(f"VoiceOptimizedLLM initialized with active client: {self.active_client}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize VoiceOptimizedLLM: {e}", exc_info=True)
            return False
    
    async def _init_openai(self) -> bool:
        """Initialize OpenAI client"""
        try:
            from openai import AsyncOpenAI
            
            api_key = self.config.get('openai_api_key')
            if not api_key:
                logger.warning("OpenAI API key not provided")
                return False
            
            client = AsyncOpenAI(api_key=api_key)
            
            self.clients['openai'] = {
                'client': client,
                'models': ['gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
                'supports_streaming': True,
                'max_tokens_limit': 4096
            }
            
            return True
            
        except ImportError:
            logger.warning("OpenAI not installed. Install with: pip install openai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            return False
    
    async def _init_anthropic(self) -> bool:
        """Initialize Anthropic client"""
        try:
            import anthropic
            
            api_key = self.config.get('anthropic_api_key')
            if not api_key:
                return False
            
            client = anthropic.AsyncAnthropic(api_key=api_key)
            
            self.clients['anthropic'] = {
                'client': client,
                'models': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                'supports_streaming': True,
                'max_tokens_limit': 4096
            }
            
            return True
            
        except ImportError:
            logger.warning("Anthropic not installed. Install with: pip install anthropic")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
            return False
    
    async def _init_local_llm(self) -> bool:
        """Initialize local LLM (Ollama, llama.cpp, etc.)"""
        try:
            local_llm_path = self.config.get('local_llm_path')
            if not local_llm_path:
                return False
            
            # This is a placeholder for local LLM integration
            # In production, you'd integrate with Ollama, llama.cpp, etc.
            
            self.clients['local'] = {
                'type': 'local',
                'model_path': local_llm_path,
                'models': ['local-llama', 'local-mistral'],
                'supports_streaming': False,
                'max_tokens_limit': 2048
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}")
            return False
    
    async def generate_response(
        self,
        prompt: str,
        context: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        persona: str = "default"
    ) -> LLMResponse:
        """Generate a voice-optimized response"""
        
        # Check cache
        cache_key = self._create_cache_key(prompt, context, persona)
        if self.cache_enabled and cache_key in self.response_cache:
            logger.debug("Cache hit for LLM response")
            return self.response_cache[cache_key]
        
        # Prepare messages for LLM
        messages = self._prepare_messages(prompt, context, conversation_history, persona)
        
        # Generate response
        try:
            llm_response = await self._call_llm(messages, context)
            
            # Post-process response for voice
            processed_response = self._process_response_for_voice(llm_response, context)
            
            # Generate prosody hints if enabled
            prosody_hints = None
            if self.prosody_enabled:
                prosody_hints = await self._generate_prosody_hints(processed_response.text, context)
            
            # Create response object
            response = LLMResponse(
                text=processed_response.text,
                metadata={
                    'model': self.active_client,
                    'tokens_used': processed_response.metadata.get('tokens_used', 0),
                    'processing_time': processed_response.metadata.get('processing_time', 0),
                    'persona': persona,
                    **processed_response.metadata
                },
                prosody_hints=prosody_hints,
                should_interrupt=processed_response.metadata.get('should_interrupt', False),
                requires_confirmation=processed_response.metadata.get('requires_confirmation', False)
            )
            
            # Cache response
            if self.cache_enabled:
                self.response_cache[cache_key] = response
                # Limit cache size
                if len(self.response_cache) > 1000:
                    # Remove oldest entry
                    oldest_key = next(iter(self.response_cache))
                    del self.response_cache[oldest_key]
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            return self._create_fallback_response()
    
    def _create_cache_key(self, prompt: str, context: Dict[str, Any], persona: str) -> str:
        """Create cache key for LLM response"""
        import hashlib
        
        cache_string = f"{prompt}_{persona}_{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _prepare_messages(
        self,
        prompt: str,
        context: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        persona: str
    ) -> List[Dict[str, str]]:
        """Prepare messages for LLM including system prompt"""
        
        # Load persona-specific system prompt
        system_prompt = self._get_system_prompt(persona, context)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for message in conversation_history[-6:]:  # Last 6 messages
            messages.append({
                "role": message['role'],
                "content": message['content']
            })
        
        # Add current prompt with context
        enhanced_prompt = self._enhance_prompt_with_context(prompt, context)
        messages.append({"role": "user", "content": enhanced_prompt})
        
        return messages
    
    def _get_system_prompt(self, persona: str, context: Dict[str, Any]) -> str:
        """Get system prompt for specific persona"""
        
        # Load persona template
        persona_path = Path(f"config/personas/{persona}.yaml")
        if persona_path.exists():
            import yaml
            with open(persona_path, 'r') as f:
                persona_config = yaml.safe_load(f)
            
            # Get persona instructions
            instructions = persona_config.get('instructions', '')
            
            # Add voice-specific instructions
            voice_instructions = """
            You are a voice assistant. Keep responses:
            1. Concise (1-2 sentences maximum)
            2. Natural for speech (avoid complex punctuation)
            3. Clear and easy to understand aloud
            4. Use contractions for natural speech
            5. Avoid markdown, URLs, or complex formatting
            
            For questions requiring longer answers, ask if the user wants more details.
            """
            
            return f"{instructions}\n\n{voice_instructions}"
        
        # Default system prompt
        return """
        You are a helpful, friendly voice assistant. Keep responses short and natural for speech.
        Use casual language and contractions. Be concise but helpful.
        
        Response guidelines:
        - Maximum 2 sentences
        - Use natural speech patterns
        - Avoid complex terminology
        - Be warm and engaging
        - Focus on clarity for audio delivery
        """
    
    def _enhance_prompt_with_context(self, prompt: str, context: Dict[str, Any]) -> str:
        """Enhance user prompt with context"""
        
        enhanced_parts = [prompt]
        
        # Add context about conversation
        if context.get('current_topic'):
            enhanced_parts.append(f"(Current topic: {context['current_topic']})")
        
        if context.get('requires_short_response'):
            enhanced_parts.append("(Please give a very brief response)")
        
        if context.get('conversation_mood'):
            enhanced_parts.append(f"(Current mood: {context['conversation_mood']})")
        
        # Add time context
        from datetime import datetime
        current_time = datetime.now().strftime("%I:%M %p")
        enhanced_parts.append(f"(Current time: {current_time})")
        
        return " ".join(enhanced_parts)
    
    async def _call_llm(
        self, 
        messages: List[Dict[str, str]], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call the LLM API"""
        
        client_type = self.active_client.split('-')[0] if '-' in self.active_client else 'openai'
        
        if client_type not in self.clients:
            # Fallback to first available client
            available_clients = list(self.clients.keys())
            if available_clients:
                client_type = available_clients[0]
            else:
                raise RuntimeError("No LLM clients available")
        
        client_info = self.clients[client_type]
        
        try:
            import time
            start_time = time.time()
            
            if client_type == 'openai':
                response = await self._call_openai(messages, client_info, context)
            elif client_type == 'anthropic':
                response = await self._call_anthropic(messages, client_info, context)
            elif client_type == 'local':
                response = await self._call_local_llm(messages, client_info, context)
            else:
                raise ValueError(f"Unsupported client type: {client_type}")
            
            processing_time = time.time() - start_time
            
            response['metadata']['processing_time'] = processing_time
            response['metadata']['client_type'] = client_type
            
            return response
            
        except Exception as e:
            logger.error(f"Error calling {client_type} LLM: {e}")
            raise
    
    async def _call_openai(
        self, 
        messages: List[Dict[str, str]], 
        client_info: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call OpenAI API"""
        
        client = client_info['client']
        model = self.active_client if self.active_client in client_info['models'] else client_info['models'][0]
        
        # Prepare parameters
        params = {
            'model': model,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'stream': False
        }
        
        # Add presence and frequency penalties if configured
        if 'presence_penalty' in self.config:
            params['presence_penalty'] = self.config['presence_penalty']
        if 'frequency_penalty' in self.config:
            params['frequency_penalty'] = self.config['frequency_penalty']
        
        response = await client.chat.completions.create(**params)
        
        return {
            'text': response.choices[0].message.content,
            'metadata': {
                'tokens_used': response.usage.total_tokens if response.usage else 0,
                'model': model
            }
        }
    
    async def _call_anthropic(
        self, 
        messages: List[Dict[str, str]], 
        client_info: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call Anthropic API"""
        
        client = client_info['client']
        model = self.active_client if self.active_client in client_info['models'] else client_info['models'][0]
        
        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                anthropic_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        response = await client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_message,
            messages=anthropic_messages
        )
        
        text = response.content[0].text
        
        return {
            'text': text,
            'metadata': {
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens,
                'model': model
            }
        }
    
    async def _call_local_llm(
        self, 
        messages: List[Dict[str, str]], 
        client_info: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call local LLM"""
        # Placeholder for local LLM integration
        # In production, this would call Ollama, llama.cpp, etc.
        
        return {
            'text': "This is a placeholder response from local LLM.",
            'metadata': {
                'tokens_used': 0,
                'model': 'local'
            }
        }
    
    def _process_response_for_voice(
        self, 
        llm_response: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process LLM response for voice delivery"""
        
        text = llm_response['text']
        
        # Shorten response if needed
        if context.get('requires_short_response', False):
            text = self.response_shortener.shorten(text, max_sentences=1)
        
        # Clean up text for speech
        text = self._clean_text_for_speech(text)
        
        # Add metadata
        metadata = llm_response['metadata'].copy()
        
        # Detect if response indicates interruption
        metadata['should_interrupt'] = self._should_interrupt_response(text)
        
        # Detect if response requires confirmation
        metadata['requires_confirmation'] = self._requires_confirmation(text)
        
        return {
            'text': text,
            'metadata': metadata
        }
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for natural speech delivery"""
        
        # Remove markdown
        import re
        text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)  # Remove bold/italic
        text = re.sub(r'#{1,6}\s*', '', text)  # Remove headings
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove links, keep text
        
        # Fix common issues for speech
        text = text.replace('...', '.')  # Replace ellipsis
        text = text.replace(' - ', ', ')  # Replace dashes with commas
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        
        # Ensure proper sentence endings
        if text and text[-1] not in ['.', '!', '?']:
            text += '.'
        
        return text.strip()
    
    def _should_interrupt_response(self, text: str) -> bool:
        """Detect if response should interrupt current speech"""
        
        interruption_keywords = [
            'stop', 'cancel', 'wait', 'hold on', 'actually',
            'never mind', 'ignore that', 'abort'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in interruption_keywords)
    
    def _requires_confirmation(self, text: str) -> bool:
        """Detect if response requires user confirmation"""
        
        confirmation_patterns = [
            'are you sure', 'confirm', 'proceed', 'continue',
            'is that correct', 'do you want', 'shall I'
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in confirmation_patterns)
    
    async def _generate_prosody_hints(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prosody hints for TTS"""
        
        # Simple prosody hints based on text content
        hints = {
            'speech_rate': 'normal',
            'pitch': 'medium',
            'emphasis': [],
            'pauses': []
        }
        
        # Adjust based on punctuation
        if '!' in text:
            hints['pitch'] = 'high'
            hints['speech_rate'] = 'fast'
        elif '?' in text:
            hints['pitch'] = 'rising'
        elif len(text.split()) > 20:
            hints['speech_rate'] = 'slow'
        
        # Add emphasis on important words
        important_words = ['important', 'urgent', 'critical', 'warning', 'note']
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in important_words:
                hints['emphasis'].append(i)
        
        # Add pauses after commas and periods
        if ',' in text or '.' in text:
            hints['pauses'] = [0.1]  # Short pause
        
        return hints
    
    def _create_fallback_response(self) -> LLMResponse:
        """Create a fallback response when LLM fails"""
        
        return LLMResponse(
            text="I'm having trouble processing that right now. Could you please repeat or rephrase your question?",
            metadata={
                'is_fallback': True,
                'error': 'LLM generation failed'
            },
            prosody_hints={
                'speech_rate': 'slow',
                'pitch': 'normal',
                'emphasis': []
            }
        )
    
    async def streaming_generate(
        self,
        prompt: str,
        context: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        persona: str = "default"
    ):
        """Stream generated response token by token"""
        
        # Prepare messages
        messages = self._prepare_messages(prompt, context, conversation_history, persona)
        
        # Determine client type
        client_type = self.active_client.split('-')[0] if '-' in self.active_client else 'openai'
        
        if client_type not in self.clients:
            yield self._create_fallback_response()
            return
        
        client_info = self.clients[client_type]
        
        if not client_info.get('supports_streaming', False):
            # Fallback to non-streaming
            response = await self.generate_response(prompt, context, conversation_history, persona)
            yield response
            return
        
        try:
            if client_type == 'openai':
                async for chunk in self._stream_openai(messages, client_info, context):
                    yield chunk
            elif client_type == 'anthropic':
                async for chunk in self._stream_anthropic(messages, client_info, context):
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield self._create_fallback_response()
    
    async def _stream_openai(self, messages, client_info, context):
        """Stream from OpenAI"""
        
        client = client_info['client']
        model = self.active_client if self.active_client in client_info['models'] else client_info['models'][0]
        
        params = {
            'model': model,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'stream': True
        }
        
        full_response = ""
        
        async for chunk in await client.chat.completions.create(**params):
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                
                # Yield partial response
                yield LLMResponse(
                    text=full_response,
                    metadata={
                        'is_partial': True,
                        'model': model
                    }
                )
        
        # Yield final processed response
        processed = self._process_response_for_voice(
            {'text': full_response, 'metadata': {'model': model}},
            context
        )
        
        prosody_hints = None
        if self.prosody_enabled:
            prosody_hints = await self._generate_prosody_hints(processed['text'], context)
        
        yield LLMResponse(
            text=processed['text'],
            metadata=processed['metadata'],
            prosody_hints=prosody_hints
        )
    
    async def _stream_anthropic(self, messages, client_info, context):
        """Stream from Anthropic"""
        # Similar implementation for Anthropic streaming
        # Placeholder for brevity
        
        yield self._create_fallback_response()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        models = []
        
        for client_info in self.clients.values():
            models.extend(client_info.get('models', []))
        
        return list(set(models))
    
    def set_active_model(self, model_name: str) -> bool:
        """Set the active model"""
        # Check if model is available in any client
        for client_type, client_info in self.clients.items():
            if model_name in client_info.get('models', []):
                self.active_client = model_name
                logger.info(f"Active model set to: {model_name}")
                return True
        
        logger.error(f"Model not available: {model_name}")
        return False
    
    async def stop(self):
        """Cleanup resources"""
        # Close client connections
        for client_info in self.clients.values():
            if 'client' in client_info and hasattr(client_info['client'], 'close'):
                await client_info['client'].close()
        
        self.clients.clear()
        logger.info("VoiceOptimizedLLM stopped")