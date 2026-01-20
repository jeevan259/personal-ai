import asyncio
import aiohttp
import json
from typing import Dict, Any
import subprocess
import sys


class VoiceEngine:
    """Voice engine with Ollama (local, free, private)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("model", "llama2")  # Default to llama2
        self.base_url = config.get("base_url", "http://localhost:11434/api")
        self.session = None
        self.ollama_running = False
        
        print(f"🤖 Initializing Ollama VoiceEngine")
        print(f"   Model: {self.model}")
        print(f"   URL: {self.base_url}")
    
    async def initialize(self):
        """Initialize Ollama connection"""
        print(f"\n🔍 Checking Ollama...")
        
        # Check if Ollama is installed
        if not self._is_ollama_installed():
            print("❌ Ollama is not installed!")
            print("\n📥 Please install Ollama:")
            print("   1. Download from: https://ollama.com")
            print("   2. Install it")
            print("   3. Open new terminal and run: ollama pull llama2")
            return False
        
        # Check if Ollama service is running
        self.ollama_running = await self._check_ollama_running()
        
        if not self.ollama_running:
            print("⚠️ Ollama service not running. Starting it...")
            success = await self._start_ollama()
            if not success:
                print("❌ Could not start Ollama")
                print("\n💡 Try running manually:")
                print("   1. Open new terminal")
                print("   2. Run: ollama serve")
                print("   3. Then restart this program")
                return False
        
        # Check if model is available
        model_available = await self._check_model_available()
        
        if not model_available:
            print(f"❌ Model '{self.model}' not found!")
            print(f"\n📥 Download it with:")
            print(f"   ollama pull {self.model}")
            print("\n💡 Available models: llama2, mistral, phi, gemma:2b, etc.")
            return False
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Test with a simple prompt
        print("🧪 Testing Ollama connection...")
        test_result = await self._test_ollama()
        
        if test_result:
            print("✅ Ollama is ready!")
            return True
        else:
            print("⚠️ Ollama test failed but continuing...")
            return True  # Continue anyway
    
    def _is_ollama_installed(self) -> bool:
        """Check if Ollama is installed"""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    async def _check_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            async with aiohttp.ClientSession() as temp_session:
                async with temp_session.get(
                    f"{self.base_url}/tags",
                    timeout=2
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _start_ollama(self) -> bool:
        """Try to start Ollama service"""
        print("   Starting Ollama in background...")
        
        try:
            # Try to start Ollama as subprocess
            import os
            if os.name == 'nt':  # Windows
                subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:  # Mac/Linux
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            # Wait a bit for it to start
            for i in range(10):
                print(f"   Waiting... ({i+1}/10)")
                await asyncio.sleep(1)
                if await self._check_ollama_running():
                    return True
            
            return False
        except Exception as e:
            print(f"   Error starting Ollama: {e}")
            return False
    
    async def _check_model_available(self) -> bool:
        """Check if the model is downloaded"""
        try:
            async with aiohttp.ClientSession() as temp_session:
                async with temp_session.get(
                    f"{self.base_url}/tags",
                    timeout=5
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m.get("name", "") for m in data.get("models", [])]
                        return any(self.model in m for m in models)
                    return False
        except Exception:
            return False
    
    async def _test_ollama(self) -> bool:
        """Test Ollama with a simple prompt"""
        try:
            test_prompt = "Say 'OK'"
            response = await self._call_ollama(test_prompt, max_tokens=2)
            return bool(response and len(response) > 0)
        except Exception:
            return False
    
    async def generate_response(self, message: str) -> str:
        """Generate response using Ollama"""
        
        if not self.ollama_running:
            # Try to reconnect
            self.ollama_running = await self._check_ollama_running()
            if not self.ollama_running:
                return "❌ Ollama is not running. Start it with 'ollama serve'"
        
        print(f"\n📤 Sending to Ollama ({self.model}): '{message[:50]}...'")
        
        try:
            response = await self._call_ollama(message)
            return response
        except Exception as e:
            error_msg = str(e)
            
            if "Connection refused" in error_msg or "Cannot connect" in error_msg:
                return "❌ Cannot connect to Ollama. Is 'ollama serve' running?"
            elif "timeout" in error_msg:
                return "⏱️  Ollama is taking too long. Try a smaller model like 'phi'."
            else:
                return f"⚠️ Ollama error: {error_msg[:80]}"
    
    async def _call_ollama(self, message: str, max_tokens: int = 200) -> str:
        """Call Ollama API"""
        url = f"{self.base_url}/generate"
        
        data = {
            "model": self.model,
            "prompt": message,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": max_tokens
            }
        }
        
        async with self.session.post(
            url,
            json=data,
            timeout=30  # Ollama can be slow
        ) as response:
            
            if response.status == 200:
                result = await response.json()
                return result.get("response", "").strip()
            else:
                error_text = await response.text()
                raise Exception(f"Ollama error {response.status}: {error_text[:150]}")
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()


# Interactive setup helper
class OllamaSetupHelper:
    """Helps set up Ollama"""
    
    @staticmethod
    def show_guide():
        print("\n" + "="*60)
        print("🦙 OLLAMA SETUP GUIDE")
        print("="*60)
        print("\n1. INSTALL OLLAMA:")
        print("   Download from: https://ollama.com")
        print("   Run the installer")
        
        print("\n2. DOWNLOAD A MODEL (run in terminal):")
        print("   ollama pull llama2      # Recommended (7B)")
        print("   ollama pull phi         # Faster (2.7B)")
        print("   ollama pull mistral     # Better quality (7B)")
        print("   ollama pull gemma:2b    # Small & fast (2B)")
        
        print("\n3. START OLLAMA:")
        print("   Open terminal and run: ollama serve")
        print("   Keep it running in background")
        
        print("\n4. TEST INSTALLATION:")
        print("   Open another terminal and run:")
        print("   ollama run llama2")
        print("   Then type: 'Hello'")
        
        print("\n5. RUN YOUR CHATBOT:")
        print("   python main.py")
        print("="*60)
    
    @staticmethod
    def quick_install_check():
        """Quick check if Ollama is ready"""
        import subprocess
        try:
            # Check ollama command
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ Ollama is installed")
                print(f"   Version: {result.stdout.strip()}")
                
                # Check if any models are pulled
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True
                )
                if "NAME" in result.stdout:
                    print("✅ Models available:")
                    for line in result.stdout.strip().split('\n')[1:]:
                        if line.strip():
                            print(f"   - {line.split()[0]}")
                else:
                    print("❌ No models downloaded")
                    print("   Run: ollama pull llama2")
                    
                return True
            else:
                print("❌ Ollama not found")
                return False
        except FileNotFoundError:
            print("❌ Ollama is not installed")
            return False


# Test the engine
if __name__ == "__main__":
    async def test():
        print("🧪 Testing Ollama VoiceEngine...")
        
        # Show setup guide first
        OllamaSetupHelper.show_guide()
        
        print("\n" + "="*60)
        print("Running test...")
        print("="*60)
        
        config = {
            "model": "llama2",  # Change to your preferred model
            "base_url": "http://localhost:11434/api"
        }
        
        engine = VoiceEngine(config)
        
        if await engine.initialize():
            print("\n✅ Ollama connected!")
            
            # Test with a simple message
            test_messages = [
                "Hello, how are you?",
                "What is AI?",
                "Tell me a short joke"
            ]
            
            for msg in test_messages:
                print(f"\n" + "="*40)
                print(f"💬 You: {msg}")
                print("🤖 Thinking...")
                
                response = await engine.generate_response(msg)
                print(f"🦙 Ollama: {response}")
                
                await asyncio.sleep(1)  # Brief pause
            
            await engine.close()
            print("\n🎉 Test complete! Ollama is working!")
        else:
            print("\n❌ Ollama setup incomplete")
            OllamaSetupHelper.quick_install_check()
    
    asyncio.run(test())