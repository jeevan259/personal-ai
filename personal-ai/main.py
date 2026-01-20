#!/usr/bin/env python3
"""
Personal Voice Chatbot - Main entry point
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

class VoiceChatbot:
    """Main chatbot class"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.voice_engine = None
        self.voice_interface = None
        self.running = False
    
    async def _load_modules(self):
        """Dynamically load modules with fallbacks"""
        try:
            # Check if src directory exists
            src_dir = Path(__file__).parent / "src"
            
            if src_dir.exists():
                print(f"Found src directory at: {src_dir}")
                
                # Try to import from src
                try:
                    import importlib.util
                    
                    # Try to import voice_engine
                    engine_spec = importlib.util.spec_from_file_location(
                        "voice_engine", 
                        src_dir / "core" / "voice_engine.py"
                    )
                    if engine_spec:
                        voice_engine_module = importlib.util.module_from_spec(engine_spec)
                        engine_spec.loader.exec_module(voice_engine_module)
                        VoiceEngine = voice_engine_module.VoiceEngine
                        
                        # Try to import voice_interface
                        interface_spec = importlib.util.spec_from_file_location(
                            "voice_interface",
                            src_dir / "interface" / "voice_interface.py"
                        )
                        if interface_spec:
                            voice_interface_module = importlib.util.module_from_spec(interface_spec)
                            interface_spec.loader.exec_module(voice_interface_module)
                            VoiceInterface = voice_interface_module.VoiceInterface
                            
                            print("✅ Successfully imported from src package")
                            return VoiceEngine, VoiceInterface
                except Exception as e:
                    print(f"Could not import from src: {e}")
            
            # If we get here, create dummy classes
            print("⚠️ Creating dummy modules (real modules not found)")
            
            class DummyVoiceEngine:
                def __init__(self, config):
                    self.config = config
                    print("Created DummyVoiceEngine")
                
                async def initialize(self):
                    print("DummyVoiceEngine initialized")
                    return True
                
                async def generate_response(self, message):
                    return f"Dummy response to: '{message}'"
            
            class DummyVoiceInterface:
                def __init__(self, engine):
                    self.engine = engine
                    print("Created DummyVoiceInterface")
            
            return DummyVoiceEngine, DummyVoiceInterface
            
        except Exception as e:
            print(f"Error loading modules: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    async def initialize(self):
        """Initialize chatbot components"""
        print("\n" + "="*50)
        print("Initializing Voice Chatbot...")
        print("="*50)
        
        # Load modules dynamically
        VoiceEngine, VoiceInterface = await self._load_modules()
        
        if not VoiceEngine or not VoiceInterface:
            print("❌ Failed to load required modules")
            return False
        
        try:
            # Create configuration
            config = {
                "debug": True,
                "model": "gpt-3.5-turbo",
                "voice_enabled": False
            }
            
            # Initialize components
            print("\nCreating VoiceEngine...")
            self.voice_engine = VoiceEngine(config)
            
            print("Initializing VoiceEngine...")
            try:
                if hasattr(self.voice_engine, 'initialize'):
                    await self.voice_engine.initialize()
                    print("✅ VoiceEngine initialized")
                else:
                    print("⚠️ VoiceEngine has no initialize method")
            except Exception as e:
                print(f"⚠️ VoiceEngine initialization error: {e}")
            
            print("Creating VoiceInterface...")
            self.voice_interface = VoiceInterface(self.voice_engine)
            
            print("\n" + "="*50)
            print("✅ Initialization complete!")
            print("="*50)
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def process_text(self, message):
        """Process text input - FIXED METHOD NAME"""
        print(f"\n📝 Processing: '{message}'")
        
        if not self.voice_engine:
            print("❌ Voice engine not available")
            return
        
        try:
            # Try different response method names
            response_methods = ['generate_response', '_generate_response', 'process_message']
            
            for method_name in response_methods:
                if hasattr(self.voice_engine, method_name):
                    method = getattr(self.voice_engine, method_name)
                    
                    if callable(method):
                        response = await method(message)
                        print(f"\n🤖 Assistant: {response}")
                        return
            
            # Fallback
            print(f"\n🤖 Assistant: I received your message: '{message}'")
            
        except Exception as e:
            print(f"❌ Error processing text: {e}")
            print(f"\n🤖 Assistant: Sorry, I encountered an error processing your message.")
    
    async def process_audio(self):
        """Process audio input"""
        print("\n🎤 Audio Input Mode")
        print("-" * 30)
        
        if not self.voice_interface:
            print("❌ Voice interface not available")
            return
        
        try:
            # Check for audio methods
            if hasattr(self.voice_interface, 'start_listening'):
                print("Starting audio listening...")
                await self.voice_interface.start_listening()
            elif hasattr(self.voice_interface, 'record_audio'):
                print("Recording audio...")
                await self.voice_interface.record_audio()
            else:
                print("Audio features not implemented.")
                print("\nTo enable audio, you need:")
                print("1. Install: pip install pyaudio soundfile")
                print("2. Implement audio methods in voice_interface.py")
        
        except Exception as e:
            print(f"❌ Audio error: {e}")
    
    def show_status(self):
        """Show system status"""
        print("\n" + "="*30)
        print("SYSTEM STATUS")
        print("="*30)
        print(f"VoiceEngine: {'✅ Available' if self.voice_engine else '❌ Not available'}")
        print(f"VoiceInterface: {'✅ Available' if self.voice_interface else '❌ Not available'}")
        
        if self.voice_engine:
            engine_type = type(self.voice_engine).__name__
            print(f"Engine Type: {engine_type}")
            
            # Show available methods
            methods = [m for m in dir(self.voice_engine) 
                      if not m.startswith('_') and callable(getattr(self.voice_engine, m))]
            print(f"Available methods: {', '.join(methods[:5])}{'...' if len(methods) > 5 else ''}")
        
        print("="*30)
    
    def show_help(self):
        """Show help message"""
        print("\n" + "="*30)
        print("HELP")
        print("="*30)
        print("text <message>  - Send text to AI")
        print("audio           - Enter audio mode")
        print("status          - Show system status")
        print("help            - Show this message")
        print("exit            - Quit the program")
        print("="*30)
        print("\nExample: text Hello, how are you?")
        print("="*30)
    
    async def _fallback_mode(self):
        """Fallback text-only mode"""
        print("\n" + "="*50)
        print("🤖 TEXT-ONLY MODE")
        print("="*50)
        print("\nVoice modules not found. Running in text-only mode.")
        print("Commands: text <message>, exit, help")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye! 👋")
                    break
                
                elif user_input.lower().startswith('text '):
                    message = user_input[5:]
                    print(f"\n🤖 Assistant: '{message}' (Echo mode)")
                
                elif user_input.lower() == 'help':
                    print("\nText-only mode commands:")
                    print("text <message> - Echo your message")
                    print("exit           - Quit")
                    print("help           - Show this help")
                
                else:
                    print("Type 'help' for commands")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
    
    async def run(self):
        """Main chatbot loop - FIXED VERSION"""
        if not await self.initialize():
            print("\nRunning in text-only fallback mode...")
            return await self._fallback_mode()
        
        self.running = True
        
        print("\n" + "="*50)
        print("🤖 PERSONAL VOICE CHATBOT")
        print("="*50)
        print("\nAvailable Commands:")
        print("  text <message>  - Send text to AI")
        print("  audio           - Audio input mode")
        print("  status          - Show system status")
        print("  help            - Show this help")
        print("  exit            - Quit")
        print("="*50)
        
        while self.running:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit first
                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye! 👋")
                    break
                
                # Check for text command
                elif user_input.lower().startswith('text '):
                    message = user_input[5:].strip()
                    if message:
                        await self.process_text(message)
                    else:
                        print("Please provide a message after 'text'")
                        print("Example: text Hello, how are you?")
                
                # Check for other commands
                elif user_input.lower() == 'audio':
                    await self.process_audio()
                
                elif user_input.lower() == 'status':
                    self.show_status()
                
                elif user_input.lower() == 'help':
                    self.show_help()
                
                else:
                    print(f"Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")


async def main():
    """Main entry point"""
    print("\n" + "="*50)
    print("PERSONAL VOICE CHATBOT")
    print("="*50)
    
    chatbot = VoiceChatbot()
    await chatbot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProgram interrupted")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()