# debug_api_fixed.py - Save in E:\personal-ai\personal-ai\
import asyncio
import aiohttp
import sys
import os
from pathlib import Path

async def test_api_key(api_key: str):
    """Test API key"""
    print(f"🔑 Testing key: {api_key[:15]}...")
    print(f"   Length: {len(api_key)} characters")
    
    if not api_key.startswith("sk-"):
        print("❌ ERROR: Key doesn't start with 'sk-'")
        return False
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Say just 'OK'"}],
        "max_tokens": 2
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            print("📡 Connecting to OpenAI...")
            async with session.post(url, headers=headers, json=data, timeout=10) as response:
                print(f"📊 Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    response_text = result["choices"][0]["message"]["content"]
                    print(f"✅ SUCCESS! Response: '{response_text}'")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ ERROR {response.status}:")
                    print(f"   {error_text[:200]}")
                    
                    if "rate limit" in error_text.lower() or "429" in str(response.status):
                        print("\n💡 You're being rate limited on a NEW key!")
                        print("   This usually means:")
                        print("   1. Your account needs billing setup")
                        print("   2. Or the key format is wrong")
                    
                    return False
                    
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

async def read_key_from_file():
    """Read API key from voice_engine.py"""
    print("\n📖 Reading voice_engine.py...")
    
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent / "src" / "core" / "voice_engine.py",  # Current dir
        Path.cwd() / "src" / "core" / "voice_engine.py",  # Working dir
        Path("src/core/voice_engine.py"),  # Relative
    ]
    
    for path in possible_paths:
        print(f"   Checking: {path}")
        if path.exists():
            print(f"✅ Found at: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for API key
            import re
            
            # Pattern 1: self.api_key = "key"
            pattern = r'self\.api_key\s*=\s*["\']([^"\']+)["\']'
            matches = re.findall(pattern, content)
            
            if matches:
                key = matches[0].strip()
                print(f"🔍 Found key in file: {key[:20]}...")
                return key
            
            # Pattern 2: api_key = "key" (without self.)
            pattern2 = r'api_key\s*=\s*["\']([^"\']+)["\']'
            matches2 = re.findall(pattern2, content)
            
            if matches2:
                key = matches2[0].strip()
                print(f"🔍 Found key in file: {key[:20]}...")
                return key
    
    print("❌ Could not find API key in voice_engine.py")
    return None

async def main():
    print("="*60)
    print("OpenAI API Debugger - Fixed Version")
    print("="*60)
    
    print(f"\n📍 Current directory: {Path.cwd()}")
    print(f"📁 Script location: {Path(__file__).parent}")
    
    # Get API key
    api_key = None
    
    # Try command line argument
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        print(f"\n🔑 Using key from command line: {api_key[:15]}...")
    
    # Try environment variable
    if not api_key:
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            api_key = env_key
            print(f"\n🔑 Using key from environment: {env_key[:15]}...")
    
    # Try reading from file
    if not api_key:
        api_key = await read_key_from_file()
    
    # Ask user if still no key
    if not api_key:
        print("\n❌ No API key found!")
        print("\nTo add your key, edit:")
        print("   personal-ai/src/core/voice_engine.py")
        print("\nFind the line with self.api_key = ...")
        print("Replace with your actual OpenAI API key")
        return
    
    # Test the key
    print("\n" + "="*60)
    print("Testing API Key...")
    print("="*60)
    
    success = await test_api_key(api_key)
    
    print("\n" + "="*60)
    if success:
        print("🎉 SUCCESS! Your API key is working!")
        print("\n💡 If chatbot still has issues:")
        print("   - Check main.py is importing correctly")
        print("   - Make sure voice_engine.py has: async def generate_response()")
    else:
        print("❌ API key test failed")
        print("\n🔧 Next steps:")
        print("   1. Check your OpenAI account at platform.openai.com")
        print("   2. Make sure you have credits/billing set up")
        print("   3. Generate a NEW key if needed")

if __name__ == "__main__":
    asyncio.run(main())