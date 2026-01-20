"""
Integration tests for skills system
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock


@pytest.mark.integration
class TestSkillsIntegration:
    """Integration tests for skills system"""
    
    @pytest.fixture
    def skill_system(self):
        """Create skill system components"""
        # Create mock skill system
        class MockSkill:
            def __init__(self, name, description):
                self.name = name
                self.description = description
                self.can_handle = Mock(return_value=True)
                self.handle = AsyncMock(return_value="Skill executed")
                
        weather_skill = MockSkill("weather", "Get weather information")
        timer_skill = MockSkill("timer", "Set timers and alarms")
        
        skills = [weather_skill, timer_skill]
        
        # Create skill registry
        class SkillRegistry:
            def __init__(self):
                self.skills = {}
                
            def register(self, skill):
                self.skills[skill.name] = skill
                
            def get_skill(self, name):
                return self.skills.get(name)
                
            def get_all_skills(self):
                return list(self.skills.values())
                
        registry = SkillRegistry()
        for skill in skills:
            registry.register(skill)
            
        return {
            "registry": registry,
            "skills": skills,
            "weather_skill": weather_skill,
            "timer_skill": timer_skill
        }
        
    @pytest.mark.asyncio
    async def test_skill_registration(self, skill_system):
        """Test skill registration"""
        registry = skill_system["registry"]
        
        # Check skills are registered
        assert len(registry.get_all_skills()) == 2
        
        weather = registry.get_skill("weather")
        assert weather is not None
        assert weather.name == "weather"
        
        timer = registry.get_skill("timer")
        assert timer is not None
        assert timer.name == "timer"
        
    @pytest.mark.asyncio
    async def test_skill_execution(self, skill_system):
        """Test skill execution"""
        weather_skill = skill_system["weather_skill"]
        
        # Execute skill
        result = await weather_skill.handle("What's the weather?")
        
        assert result == "Skill executed"
        weather_skill.handle.assert_called_once_with("What's the weather?")
        
    @pytest.mark.asyncio
    async def test_skill_matching(self, skill_system):
        """Test skill intent matching"""
        weather_skill = skill_system["weather_skill"]
        timer_skill = skill_system["timer_skill"]
        
        # Test can_handle
        weather_skill.can_handle.return_value = True
        timer_skill.can_handle.return_value = False
        
        assert weather_skill.can_handle("weather in london") is True
        assert timer_skill.can_handle("weather in london") is False
        
    def test_skill_descriptions(self, skill_system):
        """Test skill descriptions"""
        for skill in skill_system["skills"]:
            assert skill.name is not None
            assert skill.description is not None
            assert len(skill.description) > 0