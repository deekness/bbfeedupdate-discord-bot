import discord
from discord.ext import commands, tasks
import feedparser
import asyncio
import sqlite3
import hashlib
import re
import os
import sys
import signal
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional
import logging
from dataclasses import dataclass
import json
import time
import traceback
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict, Counter, deque
import anthropic

# Configuration constants
DEFAULT_CONFIG = {
    "bot_token": "",
    "update_channel_id": None,
    "rss_check_interval": 2,
    "max_retries": 3,
    "retry_delay": 5,
    "database_path": "bb_updates.db",
    "enable_heartbeat": True,
    "heartbeat_interval": 300,
    "max_update_age_hours": 168,
    "enable_auto_restart": True,
    "max_consecutive_errors": 10,
    "anthropic_api_key": "",
    "enable_llm_summaries": True,
    "llm_model": "claude-3-haiku-20240307",
    "batch_mode": "intelligent",
    "min_batch_size": 3,
    "max_batch_wait_minutes": 30,
    "urgent_batch_threshold": 2,
    "timeline_mode": "smart",
    "max_timeline_embeds": 3,
    "show_importance_timeline": True,
    # Rate limiting settings
    "llm_requests_per_minute": 10,
    "llm_requests_per_hour": 100,
    "max_processed_hashes": 10000
}

# Environment variable mappings
ENV_MAPPINGS = {
    'BOT_TOKEN': 'bot_token',
    'UPDATE_CHANNEL_ID': 'update_channel_id',
    'ANTHROPIC_API_KEY': 'anthropic_api_key',
    'RSS_CHECK_INTERVAL': 'rss_check_interval',
    'LLM_MODEL': 'llm_model',
    'BATCH_MODE': 'batch_mode',
    'OWNER_ID': 'owner_id'
}

# Game-specific constants
COMPETITION_KEYWORDS = [
    'hoh', 'head of household', 'power of veto', 'pov', 'nomination', 
    'eviction', 'ceremony', 'competition', 'challenge', 'immunity'
]

STRATEGY_KEYWORDS = [
    'alliance', 'backdoor', 'target', 'scheme', 'plan', 'strategy',
    'vote', 'voting', 'campaigning', 'deal', 'promise', 'betrayal'
]

DRAMA_KEYWORDS = [
    'argument', 'fight', 'confrontation', 'drama', 'tension',
    'called out', 'blowup', 'heated', 'angry', 'upset'
]

RELATIONSHIP_KEYWORDS = [
    'showmance', 'romance', 'flirting', 'cuddle', 'kiss',
    'relationship', 'attracted', 'feelings', 'friendship', 'bond'
]

ENTERTAINMENT_KEYWORDS = [
    'funny', 'joke', 'laugh', 'prank', 'hilarious', 'comedy',
    'entertaining', 'memorable', 'quirky', 'silly'
]

URGENT_KEYWORDS = [
    'evicted', 'eliminated', 'wins hoh', 'wins pov', 'backdoor', 
    'self-evict', 'expelled', 'quit', 'medical', 'pandora', 'coup',
    'diamond veto', 'secret power', 'battle back', 'return', 'winner',
    'final', 'jury vote', 'finale night'
]

EXCLUDE_WORDS = {
    'The', 'This', 'That', 'They', 'Some', 'Many', 'Other', 'First', 'Last',
    'Big', 'Brother', 'Julie', 'Host', 'Diary', 'Room', 'Have', 'Not'
}

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging for 24/7 operation"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # File handler for all logs
    file_handler = logging.FileHandler(log_dir / "bb_bot.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Error file handler
    error_handler = logging.FileHandler(log_dir / "bb_bot_errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()

class RateLimiter:
    """Rate limiter for API calls to prevent excessive costs"""
    
    def __init__(self, max_requests_per_minute: int = 10, max_requests_per_hour: int = 100):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.minute_requests = deque()
        self.hour_requests = deque()
        self.total_requests = 0
        
    async def wait_if_needed(self) -> bool:
        """Wait if rate limit would be exceeded. Returns True if request can proceed."""
        now = time.time()
        
        # Clean old requests
        self._clean_old_requests(now)
        
        # Check minute limit
        if len(self.minute_requests) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.minute_requests[0])
            if wait_time > 0:
                logger.warning(f"Rate limit: waiting {wait_time:.1f}s for minute window")
                await asyncio.sleep(wait_time)
                self._clean_old_requests(time.time())
        
        # Check hour limit
        if len(self.hour_requests) >= self.max_requests_per_hour:
            wait_time = 3600 - (now - self.hour_requests[0])
            if wait_time > 0:
                logger.warning(f"Rate limit: waiting {wait_time:.1f}s for hour window")
                await asyncio.sleep(wait_time)
                self._clean_old_requests(time.time())
        
        # Record the request
        current_time = time.time()
        self.minute_requests.append(current_time)
        self.hour_requests.append(current_time)
        self.total_requests += 1
        
        return True
    
    def _clean_old_requests(self, now: float):
        """Remove requests older than the window"""
        # Remove requests older than 1 minute
        while self.minute_requests and self.minute_requests[0] < now - 60:
            self.minute_requests.popleft()
        
        # Remove requests older than 1 hour
        while self.hour_requests and self.hour_requests[0] < now - 3600:
            self.hour_requests.popleft()
    
    def get_stats(self) -> Dict[str, int]:
        """Get current rate limit statistics"""
        now = time.time()
        self._clean_old_requests(now)
        
        return {
            'requests_this_minute': len(self.minute_requests),
            'requests_this_hour': len(self.hour_requests),
            'total_requests': self.total_requests,
            'minute_limit': self.max_requests_per_minute,
            'hour_limit': self.max_requests_per_hour
        }

class Config:
    """Enhanced configuration management with validation"""
    
    def __init__(self):
        self.config_file = Path("config.json")
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> dict:
        """Load configuration with better validation"""
        config = DEFAULT_CONFIG.copy()
        
        # Environment variables (priority 1)
        for env_var, config_key in ENV_MAPPINGS.items():
            env_value = os.getenv(env_var)
            if env_value:
                config[config_key] = self._convert_env_value(config_key, env_value)
        
        # Config file (priority 2)
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    # Only use file config for missing env vars
                    for key, value in file_config.items():
                        if key in config and not os.getenv(key.upper()):
                            config[key] = value
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        return config
    
    def _convert_env_value(self, config_key: str, env_value: str):
        """Convert environment variable to proper type"""
        if config_key == 'update_channel_id':
            try:
                return int(env_value) if env_value != '0' else None
            except ValueError:
                logger.warning(f"Invalid channel ID: {env_value}")
                return None
        elif config_key in ['rss_check_interval', 'max_retries', 'retry_delay', 'owner_id',
                           'llm_requests_per_minute', 'llm_requests_per_hour', 'max_processed_hashes']:
            try:
                return int(env_value)
            except ValueError:
                logger.warning(f"Invalid integer for {config_key}: {env_value}")
                return DEFAULT_CONFIG.get(config_key, 0)
        elif config_key in ['enable_heartbeat', 'enable_llm_summaries']:
            return env_value.lower() == 'true'
        else:
            return env_value
    
    def _validate_config(self):
        """Validate critical configuration values"""
        if not self.config.get('bot_token'):
            raise ValueError("BOT_TOKEN is required")
        
        # Validate rate limiting settings
        if self.config.get('llm_requests_per_minute', 0) <= 0:
            self.config['llm_requests_per_minute'] = DEFAULT_CONFIG['llm_requests_per_minute']
        
        if self.config.get('llm_requests_per_hour', 0) <= 0:
            self.config['llm_requests_per_hour'] = DEFAULT_CONFIG['llm_requests_per_hour']
        
        logger.info("Configuration validated successfully")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value"""
        self.config[key] = value

@dataclass
class BBUpdate:
    """Represents a Big Brother update"""
    title: str
    description: str
    link: str
    pub_date: datetime
    content_hash: str
    author: str = ""

class BBAnalyzer:
    """Analyzes Big Brother updates for strategic insights and social dynamics"""
    
    def categorize_update(self, update: BBUpdate) -> List[str]:
        """Categorize an update based on its content"""
        content = f"{update.title} {update.description}".lower()
        categories = []
        
        if any(keyword in content for keyword in COMPETITION_KEYWORDS):
            categories.append("🏆 Competition")
        
        if any(keyword in content for keyword in STRATEGY_KEYWORDS):
            categories.append("🎯 Strategy")
        
        if any(keyword in content for keyword in DRAMA_KEYWORDS):
            categories.append("💥 Drama")
        
        if any(keyword in content for keyword in RELATIONSHIP_KEYWORDS):
            categories.append("💕 Romance")
        
        if any(keyword in content for keyword in ENTERTAINMENT_KEYWORDS):
            categories.append("🎬 Entertainment")
        
        return categories if categories else ["📝 General"]
    
    def extract_houseguests(self, text: str) -> List[str]:
        """Extract houseguest names from text"""
        houseguest_pattern = r'\b[A-Z][a-z]+\b'
        potential_names = re.findall(houseguest_pattern, text)
        return [name for name in potential_names if name not in EXCLUDE_WORDS]
    
    def analyze_strategic_importance(self, update: BBUpdate) -> int:
        """Rate importance from 1-10 with balanced strategic/social weighting"""
        content = f"{update.title} {update.description}".lower()
        score = 1
        
        # Strategic moments (high importance)
        if any(word in content for word in ['eviction', 'nomination', 'backdoor']):
            score += 4
        if any(word in content for word in ['hoh', 'head of household', 'power of veto']):
            score += 3
        if any(word in content for word in ['alliance', 'target', 'strategy']):
            score += 2
        if any(word in content for word in ['vote', 'voting', 'campaign']):
            score += 2
        
        # Social moments (medium-high importance for superfans)
        if any(word in content for word in ['showmance', 'kiss', 'cuddle', 'romance']):
            score += 3
        if any(word in content for word in ['fight', 'argument', 'confrontation', 'blowup']):
            score += 3
        if any(word in content for word in ['friendship', 'bond', 'close', 'trust']):
            score += 2
        
        # Entertainment moments (medium importance)
        if any(word in content for word in ['funny', 'joke', 'laugh', 'prank']):
            score += 2
        if any(word in content for word in ['crying', 'emotional', 'breakdown']):
            score += 2
        
        # House culture (low-medium importance)
        if any(word in content for word in ['tradition', 'routine', 'habit', 'inside joke']):
            score += 1
        
        # Finale night special scoring
        if any(word in content for word in ['finale', 'winner', 'crowned', 'julie']):
            if any(word in content for word in ['america', 'favorite']):
                score += 4
        
        return min(score, 10)

class UpdateBatcher:
    """Groups and analyzes updates like a BB superfan using LLM intelligence"""
    
    def __init__(self, analyzer: BBAnalyzer, config: Config):
        self.analyzer = analyzer
        self.config = config
        self.update_queue = []
        self.last_batch_time = datetime.now()
        self.processed_hashes = set()
        self.max_processed_hashes = config.get('max_processed_hashes', 10000)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=config.get('llm_requests_per_minute', 10),
            max_requests_per_hour=config.get('llm_requests_per_hour', 100)
        )
        
        # Initialize Anthropic client
        self.llm_client = None
        self.llm_model = config.get('llm_model', 'claude-3-haiku-20240307')
        self._init_llm_client()
    
    def _init_llm_client(self):
        """Initialize LLM client with proper error handling"""
        api_key = self.config.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY', '')
        
        if not api_key.strip():
            logger.warning("No valid Anthropic API key provided")
            return
        
        try:
            self.llm_client = anthropic.Anthropic(api_key=api_key.strip())
            logger.info("LLM integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm_client = None
    
    def _manage_processed_hashes(self):
        """Manage processed hashes to prevent memory bloat"""
        if len(self.processed_hashes) >= self.max_processed_hashes:
            # Remove oldest 50% of hashes (simple approach)
            # In production, you might want to use a more sophisticated LRU cache
            hashes_to_remove = len(self.processed_hashes) // 2
            hashes_list = list(self.processed_hashes)
            for _ in range(hashes_to_remove):
                self.processed_hashes.remove(hashes_list.pop(0))
            logger.info(f"Cleaned {hashes_to_remove} processed hashes to prevent memory bloat")
    
    def should_send_batch(self) -> bool:
        """Determine if we should send a batch now"""
        if not self.update_queue:
            return False
            
        time_elapsed = (datetime.now() - self.last_batch_time).total_seconds() / 60
        
        # Check for urgent updates
        has_urgent = any(self._is_urgent(update) for update in self.update_queue)
        if has_urgent and len(self.update_queue) >= 2:
            return True
        
        # High activity mode: 10+ updates or 15 minutes
        if len(self.update_queue) >= 10 or time_elapsed >= 15:
            return True
            
        # Normal mode: 5+ updates or 30 minutes
        if len(self.update_queue) >= 5 or time_elapsed >= 30:
            return True
            
        return False
    
    def _is_urgent(self, update: BBUpdate) -> bool:
        """Check if update contains game-critical information"""
        content = f"{update.title} {update.description}".lower()
        return any(keyword in content for keyword in URGENT_KEYWORDS)
    
    def add_update(self, update: BBUpdate):
        """Add update to queue if not already processed"""
        if update.content_hash not in self.processed_hashes:
            self.update_queue.append(update)
            self.processed_hashes.add(update.content_hash)
            self._manage_processed_hashes()
    
    async def create_batch_summary(self) -> List[discord.Embed]:
        """Create intelligent summary embeds using LLM if available"""
        if not self.update_queue:
            return []
        
        embeds = []
        
        # Use LLM if available and rate limits allow
        if self.llm_client and await self._can_make_llm_request():
            try:
                embeds = await self._create_llm_summary()
            except Exception as e:
                logger.error(f"LLM summary failed: {e}")
                embeds = self._create_pattern_summary_with_explanation("LLM analysis failed")
        else:
            reason = "LLM unavailable" if not self.llm_client else "Rate limit reached"
            embeds = self._create_pattern_summary_with_explanation(reason)
        
        # Clear queue after processing
        self.update_queue.clear()
        self.last_batch_time = datetime.now()
        
        return embeds
    
    async def _can_make_llm_request(self) -> bool:
        """Check if we can make an LLM request without hitting rate limits"""
        stats = self.rate_limiter.get_stats()
        return (stats['requests_this_minute'] < stats['minute_limit'] and 
                stats['requests_this_hour'] < stats['hour_limit'])
    
    async def _create_llm_summary(self) -> List[discord.Embed]:
        """Use Claude to create intelligent summaries with rate limiting"""
        # Wait for rate limit if needed
        await self.rate_limiter.wait_if_needed()
        
        # Prepare update data for LLM
        updates_data = []
        for update in self.update_queue:
            time_str = update.pub_date.strftime('%I:%M %p')
            updates_data.append({
                'time': time_str,
                'title': update.title,
                'description': update.description[:200] if update.description != update.title else ""
            })
        
        # Detect finale night vs regular season
        is_finale_night = any(
            keyword in update.title.lower() 
            for update in self.update_queue 
            for keyword in ['winner', 'crowned', 'finale', 'americas favorite']
        )
        
        # Create appropriate prompt
        prompt = self._create_llm_prompt(updates_data, is_finale_night)
        
        try:
            # Make LLM request
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.llm_model,
                max_tokens=1200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            analysis = self._parse_llm_response(response.content[0].text)
            
            # Create main embed
            embeds = self._create_embeds_from_analysis(analysis, is_finale_night)
            
            # Add highlights embed for batches with 7+ updates if rate limits allow
            if len(self.update_queue) >= 7:
                logger.info(f"Creating highlights embed for {len(self.update_queue)} updates")
                try:
                    # Check if we can make another LLM call
                    if await self._can_make_llm_request():
                        highlights_embed = await self._create_llm_highlights_embed(analysis.get('game_phase', 'current'))
                        if highlights_embed:
                            embeds.append(highlights_embed)
                            logger.info("Added LLM-curated highlights embed")
                        else:
                            logger.warning("LLM highlights embed creation returned None")
                    else:
                        logger.warning("Rate limit reached - creating pattern-based highlights embed")
                        highlights_embed = self._create_pattern_highlights_embed()
                        if highlights_embed:
                            embeds.append(highlights_embed)
                            logger.info("Added pattern-based highlights embed")
                except Exception as e:
                    logger.error(f"Error creating highlights embed: {e}")
                    # Fallback to pattern-based highlights
                    highlights_embed = self._create_pattern_highlights_embed()
                    if highlights_embed:
                        embeds.append(highlights_embed)
                        logger.info("Added fallback highlights embed")
            
            return embeds
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def _create_llm_prompt(self, updates_data: List[Dict], is_finale_night: bool) -> str:
        """Create appropriate LLM prompt based on context"""
        updates_text = "\n".join([
            f"{u['time']} - {u['title']}" + (f" ({u['description']})" if u['description'] else "")
            for u in updates_data
        ])
        
        if is_finale_night:
            return f"""You are Taran Armstrong, the ultimate Big Brother superfan analyzing the FINALE NIGHT of Big Brother.

Analyze these {len(updates_data)} finale night updates:

{updates_text}

For FINALE NIGHT coverage, provide comprehensive analysis including:
- Strategic game outcome and jury decision
- Social moments and houseguest interactions
- Host segments and special announcements
- America's Favorite Houseguest results
- Post-game relationships and connections
- Jury questioning and final speeches

Provide your analysis in this JSON format:
{{
    "headline": "Finale night headline capturing the winner and key moments",
    "summary": "3-4 sentence summary covering winner, jury decision, and notable social moments",
    "strategic_analysis": "Why the winner won and what their victory means strategically",
    "social_highlights": "Notable social moments, relationships, and interactions from finale night",
    "key_players": ["winner", "runner-up", "afh", "other", "notable", "houseguests"],
    "game_phase": "finale_night",
    "strategic_importance": 10,
    "jury_analysis": "How the jury voted and what influenced their decision",
    "finale_moments": "Special finale segments, host interactions, and memorable moments"
}}

Focus on providing complete finale night coverage for superfans who want both strategic analysis AND social moments."""
        else:
            return f"""You are the ultimate Big Brother superfan - part Taran Armstrong's strategic genius, part live feed obsessive who loves ALL aspects of the BB experience.

Analyze these {len(updates_data)} Big Brother live feed updates:

{updates_text}

As a complete Big Brother superfan, provide analysis covering BOTH strategic gameplay AND social dynamics:

{{
    "headline": "Compelling headline that captures the most significant development (strategic OR social)",
    "summary": "3-4 sentence summary balancing strategic implications with social dynamics and entertainment value",
    "strategic_analysis": "Strategic implications - alliances, targets, power shifts, competition positioning",
    "social_dynamics": "Social relationships, showmances, conflicts, friendships, house dynamics",
    "entertainment_highlights": "Funny moments, drama, memorable quotes, personality clashes, or unique interactions",
    "key_players": ["houseguests", "involved", "in", "strategic", "and", "social", "moments"],
    "game_phase": "one of: early_game, jury_phase, final_weeks, finale_night",
    "strategic_importance": 7,
    "house_culture": "Inside jokes, daily routines, house traditions, or quirky moments that define this group",
    "relationship_updates": "Showmance developments, friendship changes, or alliance shifts"
}}

Remember: Big Brother superfans want strategic depth BUT also love the social experiment aspects."""
    
    def _parse_llm_response(self, response_text: str) -> dict:
        """Parse LLM response with fallback handling"""
        try:
            # Try to extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start:json_end]
                return json.loads(json_text)
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parsing failed: {e}, using text response")
            return self._parse_text_response(response_text)
    
    def _parse_text_response(self, response_text: str) -> dict:
        """Parse LLM response when JSON parsing fails"""
        # Create a basic analysis from the text response
        analysis = {
            "headline": "Big Brother Update",
            "summary": response_text[:300] + "..." if len(response_text) > 300 else response_text,
            "strategic_analysis": "Analysis available in summary above.",
            "key_players": [],
            "game_phase": "current",
            "strategic_importance": 5,
            "social_dynamics": "See summary for details.",
            "entertainment_highlights": "Various moments occurred."
        }
        
        # Try to extract key information
        try:
            names = re.findall(r'\b[A-Z][a-z]+\b', response_text)
            analysis['key_players'] = [name for name in names if name not in EXCLUDE_WORDS][:5]
        except Exception as e:
            logger.debug(f"Text parsing error: {e}")
        
        return analysis
    
    def _create_embeds_from_analysis(self, analysis: dict, is_finale_night: bool) -> List[discord.Embed]:
        """Create Discord embeds from LLM analysis"""
        game_phase_colors = {
            "early_game": 0x3498db,
            "jury_phase": 0xf39c12,
            "final_weeks": 0xe74c3c,
            "finale_night": 0xffd700
        }
        
        color = game_phase_colors.get(analysis.get('game_phase', 'early_game'), 0x3498db)
        
        main_embed = discord.Embed(
            title=f"🎭 {analysis['headline']}",
            description=f"**{len(self.update_queue)} feed updates** • {analysis.get('game_phase', 'Current Phase').replace('_', ' ').title()}\n\n{analysis['summary']}",
            color=color,
            timestamp=datetime.now()
        )
        
        # Add strategic analysis
        if analysis.get('strategic_analysis'):
            main_embed.add_field(
                name="🎯 Strategic Analysis",
                value=analysis['strategic_analysis'],
                inline=False
            )
        
        # Add content based on finale vs regular season
        if is_finale_night:
            self._add_finale_fields(main_embed, analysis)
        else:
            self._add_regular_fields(main_embed, analysis)
        
        # Add common fields
        self._add_common_fields(main_embed, analysis, is_finale_night)
        
        return [main_embed]
    
    def _add_finale_fields(self, embed: discord.Embed, analysis: dict):
        """Add finale-specific fields to embed"""
        if analysis.get('social_highlights'):
            embed.add_field(
                name="🎭 Finale Night Highlights",
                value=analysis['social_highlights'],
                inline=False
            )
        
        if analysis.get('jury_analysis'):
            embed.add_field(
                name="⚖️ Jury Decision Analysis",
                value=analysis['jury_analysis'],
                inline=False
            )
        
        if analysis.get('finale_moments'):
            embed.add_field(
                name="✨ Special Finale Moments",
                value=analysis['finale_moments'],
                inline=False
            )
    
    def _add_regular_fields(self, embed: discord.Embed, analysis: dict):
        """Add regular season fields to embed"""
        if analysis.get('social_dynamics'):
            embed.add_field(
                name="👥 Social Dynamics",
                value=analysis['social_dynamics'],
                inline=False
            )
        
        if analysis.get('entertainment_highlights'):
            embed.add_field(
                name="🎬 Entertainment Highlights",
                value=analysis['entertainment_highlights'],
                inline=False
            )
        
        if analysis.get('relationship_updates'):
            embed.add_field(
                name="💕 Relationship Updates",
                value=analysis['relationship_updates'],
                inline=False
            )
        
        if analysis.get('house_culture'):
            embed.add_field(
                name="🏠 House Culture",
                value=analysis['house_culture'],
                inline=False
            )
    
    def _add_common_fields(self, embed: discord.Embed, analysis: dict, is_finale_night: bool):
        """Add common fields to embed"""
        # Key players
        if analysis.get('key_players'):
            players = analysis['key_players'][:8]
            embed.add_field(
                name="⭐ Key Players",
                value=" • ".join(players),
                inline=False
            )
        
        # Strategic importance
        importance = analysis.get('strategic_importance', 5)
        importance_bar = "🔥" * min(importance, 10)
        embed.add_field(
            name="🎲 Overall Importance",
            value=f"{importance_bar} {importance}/10",
            inline=True
        )
        
        # Footer
        footer_text = ("Finale Night Analysis • BB Superfan AI" if is_finale_night 
                      else "Strategic Analysis + Social Dynamics • BB Superfan AI")
        embed.set_footer(text=footer_text)
    
    def _create_pattern_summary_with_explanation(self, reason: str) -> List[discord.Embed]:
        """Enhanced pattern-based summary with explanation"""
        grouped = self._group_updates_pattern()
        
        total_updates = sum(len(updates) for updates in grouped.values())
        headline = self._get_headline_pattern(grouped)
        
        embed = discord.Embed(
            title=f"🎭 {headline}",
            description=f"**{total_updates} updates** from the last batch period\n\n*{reason} - Using pattern analysis*",
            color=self._get_batch_color_pattern(grouped),
            timestamp=datetime.now()
        )
        
        # Add grouped summaries
        for category, updates in grouped.items():
            if updates:
                summary = self._create_category_summary(category, updates)
                embed.add_field(
                    name=f"{self._get_category_emoji(category)} {category.title()} ({len(updates)})",
                    value=summary,
                    inline=False
                )
        
        return [embed]
    
    def _group_updates_pattern(self) -> Dict[str, List[BBUpdate]]:
        """Pattern-based grouping when LLM unavailable"""
        groups = defaultdict(list)
        
        for update in self.update_queue:
            content = f"{update.title} {update.description}".lower()
            
            if 'vote' in content or 'voting' in content:
                groups['votes'].append(update)
            elif any(word in content for word in ['wins', 'won', 'hoh', 'pov', 'veto']):
                groups['competitions'].append(update)
            elif any(word in content for word in ['nominat', 'ceremony']):
                groups['nominations'].append(update)
            else:
                groups['general'].append(update)
        
        return groups
    
    def _get_headline_pattern(self, grouped: Dict[str, List[BBUpdate]]) -> str:
        """Generate headline without LLM"""
        if grouped.get('votes'):
            return "Jury Votes Revealed!"
        elif grouped.get('competitions'):
            return "Competition Results!"
        elif grouped.get('nominations'):
            return "Nomination Ceremony!"
        else:
            return "Big Brother Update"
    
    def _get_batch_color_pattern(self, grouped: Dict[str, List[BBUpdate]]) -> int:
        """Determine color without LLM"""
        if grouped.get('votes'):
            return 0xe74c3c  # Red
        elif grouped.get('competitions'):
            return 0xf39c12  # Orange
        else:
            return 0x3498db  # Blue
    
    def _create_category_summary(self, category: str, updates: List[BBUpdate]) -> str:
        """Create better category summaries"""
        if len(updates) <= 3:
            return "\n".join([
                f"• {u.title[:80]}..." if len(u.title) > 80 else f"• {u.title}"
                for u in updates
            ])
        else:
            return (f"• {updates[0].title[:80]}...\n"
                   f"• {updates[1].title[:80]}...\n"
                   f"• And {len(updates)-2} more updates")
    
    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for category"""
        emoji_map = {
            'votes': '🗳️',
            'competitions': '🏆',
            'nominations': '🎯',
            'general': '📝',
            'strategy': '🧠',
            'drama': '💥'
        }
        return emoji_map.get(category, '📝')
    
    def get_rate_limit_stats(self) -> Dict[str, int]:
        """Get current rate limiting statistics"""
        return self.rate_limiter.get_stats()

    def _create_pattern_highlights_embed(self) -> Optional[discord.Embed]:
        """Create highlights embed using pattern matching when LLM unavailable"""
        try:
            # Sort updates by importance score
            updates_with_importance = [
                (update, self.analyzer.analyze_strategic_importance(update))
                for update in self.update_queue
            ]
            updates_with_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Select top 5-8 most important updates
            selected_updates = updates_with_importance[:min(8, len(updates_with_importance))]
            
            if not selected_updates:
                return None
            
            embed = discord.Embed(
                title="🎯 Feed Highlights - What Mattered",
                description=f"Key moments from this period ({len(selected_updates)} of {len(self.update_queue)} updates)",
                color=0x95a5a6,
                timestamp=datetime.now()
            )
            
            # Add selected highlights
            for i, (update, importance) in enumerate(selected_updates, 1):
                time_str = update.pub_date.strftime("%I:%M %p")
                
                # Format the highlight
                title = update.title
                if len(title) > 100:
                    title = title[:97] + "..."
                
                # Add importance indicators
                importance_emoji = "🔥" if importance >= 7 else "⭐" if importance >= 5 else "📝"
                
                embed.add_field(
                    name=f"{importance_emoji} {time_str}",
                    value=title,
                    inline=False
                )
            
            embed.set_footer(text=f"Pattern-based Analysis • {len(selected_updates)} key moments selected")
            
            return embed
            
        except Exception as e:
            logger.error(f"Pattern highlights creation failed: {e}")
            return None

class BBDatabase:
    """Handles database operations with connection pooling and error recovery"""
    
    def __init__(self, db_path: str = "bb_updates.db"):
        self.db_path = db_path
        self.connection_timeout = 30
        self.init_database()
    
    def get_connection(self):
        """Get database connection with proper error handling"""
        try:
            conn = sqlite3.connect(
                self.db_path, 
                timeout=self.connection_timeout,
                check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def init_database(self):
        """Initialize the database schema with proper indexing"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT UNIQUE,
                    title TEXT,
                    description TEXT,
                    link TEXT,
                    pub_date TIMESTAMP,
                    author TEXT,
                    importance_score INTEGER DEFAULT 1,
                    categories TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON updates(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pub_date ON updates(pub_date)")
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
        finally:
            conn.close()
    
    def is_duplicate(self, content_hash: str) -> bool:
        """Check if update already exists"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT 1 FROM updates WHERE content_hash = ?", (content_hash,))
            result = cursor.fetchone()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            logger.error(f"Database duplicate check error: {e}")
            return False
    
    def store_update(self, update: BBUpdate, importance_score: int = 1, categories: List[str] = None):
        """Store a new update"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            categories_str = ",".join(categories) if categories else ""
            
            cursor.execute("""
                INSERT INTO updates (content_hash, title, description, link, pub_date, author, importance_score, categories)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (update.content_hash, update.title, update.description, 
                  update.link, update.pub_date, update.author, importance_score, categories_str))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database store error: {e}")
            raise
    
    def get_recent_updates(self, hours: int = 24) -> List[BBUpdate]:
        """Get updates from the last N hours"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cursor.execute("""
                SELECT title, description, link, pub_date, content_hash, author
                FROM updates 
                WHERE pub_date > ?
                ORDER BY pub_date DESC
            """, (cutoff_time,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [BBUpdate(*row) for row in results]
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []

class BBDiscordBot(commands.Bot):
    """Main Discord bot class with 24/7 reliability features"""
    
    def __init__(self):
        self.config = Config()
        
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        super().__init__(command_prefix='!bb', intents=intents)
        
        self.rss_url = "https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php"
        self.db = BBDatabase(self.config.get('database_path', 'bb_updates.db'))
        self.analyzer = BBAnalyzer()
        self.update_batcher = UpdateBatcher(self.analyzer, self.config)
        
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.remove_command('help')
        self.setup_commands()
    
    def setup_commands(self):
        """Setup all slash commands"""
        
        @self.tree.command(name="status", description="Show bot status and statistics")
        async def status_slash(interaction: discord.Interaction):
            """Show bot status"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                embed = discord.Embed(
                    title="Big Brother Bot Status",
                    color=0x2ecc71 if self.consecutive_errors == 0 else 0xe74c3c,
                    timestamp=datetime.now()
                )
                
                embed.add_field(name="RSS Feed", value=self.rss_url, inline=False)
                embed.add_field(name="Update Channel", 
                               value=f"<#{self.config.get('update_channel_id')}>" if self.config.get('update_channel_id') else "Not set", 
                               inline=True)
                embed.add_field(name="Updates Processed", value=str(self.total_updates_processed), inline=True)
                embed.add_field(name="Consecutive Errors", value=str(self.consecutive_errors), inline=True)
                
                time_since_check = datetime.now() - self.last_successful_check
                embed.add_field(name="Last RSS Check", value=f"{time_since_check.total_seconds():.0f} seconds ago", inline=True)
                
                queue_size = len(self.update_batcher.update_queue)
                embed.add_field(name="Updates in Queue", value=str(queue_size), inline=True)
                
                llm_status = "✅ Enabled" if self.update_batcher.llm_client else "❌ Disabled"
                embed.add_field(name="LLM Summaries", value=llm_status, inline=True)
                
                # Add rate limiting stats
                rate_stats = self.update_batcher.get_rate_limit_stats()
                embed.add_field(
                    name="Rate Limits",
                    value=f"Minute: {rate_stats['requests_this_minute']}/{rate_stats['minute_limit']}\n"
                          f"Hour: {rate_stats['requests_this_hour']}/{rate_stats['hour_limit']}\n"
                          f"Total: {rate_stats['total_requests']}",
                    inline=True
                )
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error generating status: {e}")
                await interaction.followup.send("Error generating status.", ephemeral=True)

        @self.tree.command(name="summary", description="Get a summary of recent Big Brother updates")
        async def summary_slash(interaction: discord.Interaction, hours: int = 24):
            """Generate a summary of updates"""
            try:
                if hours < 1 or hours > 168:
                    await interaction.response.send_message("Hours must be between 1 and 168", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                updates = self.db.get_recent_updates(hours)
                
                if not updates:
                    await interaction.followup.send(f"No updates found in the last {hours} hours.", ephemeral=True)
                    return
                
                categories = {}
                for update in updates:
                    update_categories = self.analyzer.categorize_update(update)
                    for category in update_categories:
                        if category not in categories:
                            categories[category] = []
                        categories[category].append(update)
                
                embed = discord.Embed(
                    title=f"Big Brother Updates Summary ({hours}h)",
                    description=f"**{len(updates)} total updates**",
                    color=0x3498db,
                    timestamp=datetime.now()
                )
                
                for category, cat_updates in categories.items():
                    top_updates = sorted(cat_updates, 
                                       key=lambda x: self.analyzer.analyze_strategic_importance(x), 
                                       reverse=True)[:3]
                    
                    summary_text = "\n".join([f"• {update.title[:100]}..." 
                                            if len(update.title) > 100 
                                            else f"• {update.title}" 
                                            for update in top_updates])
                    
                    embed.add_field(
                        name=f"{category} ({len(cat_updates)} updates)",
                        value=summary_text or "No updates",
                        inline=False
                    )
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                await interaction.followup.send("Error generating summary. Please try again.", ephemeral=True)

        @self.tree.command(name="setchannel", description="Set the channel for Big Brother updates")
        async def setchannel_slash(interaction: discord.Interaction, channel: discord.TextChannel):
            """Set the channel for RSS updates"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                    
                if not channel.permissions_for(interaction.guild.me).send_messages:
                    await interaction.response.send_message(
                        f"I don't have permission to send messages in {channel.mention}", 
                        ephemeral=True
                    )
                    return
                
                self.config.set('update_channel_id', channel.id)
                
                await interaction.response.send_message(
                    f"Update channel set to {channel.mention}", 
                    ephemeral=True
                )
                logger.info(f"Update channel set to {channel.id}")
                
            except Exception as e:
                logger.error(f"Error setting channel: {e}")
                await interaction.response.send_message("Error setting channel. Please try again.", ephemeral=True)

        @self.tree.command(name="commands", description="Show all available commands")
        async def commands_slash(interaction: discord.Interaction):
            """Show available commands"""
            embed = discord.Embed(
                title="Big Brother Bot Commands",
                description="Monitor Jokers Updates RSS feed with intelligent analysis",
                color=0x3498db
            )
            
            commands_list = [
                ("/summary", "Get a summary of recent updates (default: 24h)"),
                ("/status", "Show bot status and statistics"),
                ("/setchannel", "Set update channel (Admin only)"),
                ("/commands", "Show this help message"),
                ("/forcebatch", "Force send any queued updates (Admin only)"),
                ("/testllm", "Test LLM connection (Admin only)"),
                ("/sync", "Sync slash commands (Owner only)")
            ]
            
            for name, description in commands_list:
                embed.add_field(name=name, value=description, inline=False)
            
            embed.set_footer(text="All commands are ephemeral (only you can see the response)")
            
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @self.tree.command(name="forcebatch", description="Force send any queued updates")
        async def forcebatch_slash(interaction: discord.Interaction):
            """Force send batch update"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                queue_size = len(self.update_batcher.update_queue)
                if queue_size == 0:
                    await interaction.followup.send("No updates in queue to send.", ephemeral=True)
                    return
                
                await self.send_batch_update()
                
                await interaction.followup.send(f"Force sent batch of {queue_size} updates!", ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error forcing batch: {e}")
                await interaction.followup.send("Error sending batch.", ephemeral=True)

        @self.tree.command(name="testllm", description="Test LLM connection and functionality")
        async def test_llm_slash(interaction: discord.Interaction):
            """Test LLM integration"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                if not self.update_batcher.llm_client:
                    await interaction.followup.send("❌ LLM client not initialized - check API key", ephemeral=True)
                    return
                
                # Check rate limits
                if not await self.update_batcher._can_make_llm_request():
                    stats = self.update_batcher.get_rate_limit_stats()
                    await interaction.followup.send(
                        f"❌ Rate limit reached\n"
                        f"Minute: {stats['requests_this_minute']}/{stats['minute_limit']}\n"
                        f"Hour: {stats['requests_this_hour']}/{stats['hour_limit']}", 
                        ephemeral=True
                    )
                    return
                
                # Test API call
                await self.update_batcher.rate_limiter.wait_if_needed()
                
                test_response = await asyncio.to_thread(
                    self.update_batcher.llm_client.messages.create,
                    model=self.update_batcher.llm_model,
                    max_tokens=100,
                    messages=[{
                        "role": "user", 
                        "content": "You are a Big Brother superfan. Respond with 'LLM connection successful!' and briefly explain why you love both strategic gameplay and social dynamics in Big Brother."
                    }]
                )
                
                response_text = test_response.content[0].text
                
                embed = discord.Embed(
                    title="✅ LLM Connection Test",
                    description=f"**Model**: {self.update_batcher.llm_model}\n**Response**: {response_text}",
                    color=0x2ecc71,
                    timestamp=datetime.now()
                )
                
                # Add rate limit info
                stats = self.update_batcher.get_rate_limit_stats()
                embed.add_field(
                    name="Rate Limits After Test",
                    value=f"Minute: {stats['requests_this_minute']}/{stats['minute_limit']}\n"
                          f"Hour: {stats['requests_this_hour']}/{stats['hour_limit']}",
                    inline=False
                )
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error testing LLM: {e}")
                await interaction.followup.send(f"❌ LLM test failed: {str(e)}", ephemeral=True)

        @self.tree.command(name="sync", description="Sync slash commands (Owner only)")
        async def sync_slash(interaction: discord.Interaction):
            """Manually sync slash commands"""
            try:
                owner_id = self.config.get('owner_id')
                if not owner_id or interaction.user.id != owner_id:
                    await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                try:
                    synced = await self.tree.sync()
                    await interaction.followup.send(f"✅ Synced {len(synced)} slash commands!", ephemeral=True)
                    logger.info(f"Manually synced {len(synced)} commands")
                except Exception as e:
                    await interaction.followup.send(f"❌ Failed to sync commands: {e}", ephemeral=True)
                    logger.error(f"Manual sync failed: {e}")
                    
            except Exception as e:
                logger.error(f"Error in sync command: {e}")
                await interaction.followup.send("Error syncing commands.", ephemeral=True)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_shutting_down = True
        asyncio.create_task(self.close())
    
    async def on_ready(self):
        """Bot startup event"""
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')
        
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash command(s)")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")
        
        try:
            self.check_rss_feed.start()
            logger.info("RSS feed monitoring started")
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    def create_content_hash(self, title: str, description: str) -> str:
        """Create a unique hash for content deduplication"""
        content = f"{title}|{description}".lower()
        content = re.sub(r'\d{1,2}:\d{2}[ap]m', '', content)
        content = re.sub(r'\d{1,2}/\d{1,2}', '', content)
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_rss_entries(self, entries) -> List[BBUpdate]:
        """Process RSS entries into BBUpdate objects"""
        updates = []
        
        for entry in entries:
            try:
                title = entry.get('title', 'No title')
                description = entry.get('description', 'No description')
                link = entry.get('link', '')
                
                pub_date = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                
                content_hash = self.create_content_hash(title, description)
                author = entry.get('author', '')
                
                updates.append(BBUpdate(
                    title=title,
                    description=description,
                    link=link,
                    pub_date=pub_date,
                    content_hash=content_hash,
                    author=author
                ))
                
            except Exception as e:
                logger.error(f"Error processing RSS entry: {e}")
                continue
        
        return updates
    
    def filter_duplicates(self, updates: List[BBUpdate]) -> List[BBUpdate]:
        """Filter out duplicate updates"""
        new_updates = []
        seen_hashes = set()
        
        for update in updates:
            if not self.db.is_duplicate(update.content_hash):
                if update.content_hash not in seen_hashes:
                    new_updates.append(update)
                    seen_hashes.add(update.content_hash)
        
        return new_updates
    
    @tasks.loop(minutes=2)
    async def check_rss_feed(self):
        """Check RSS feed for new updates with intelligent batching"""
        if self.is_shutting_down:
            return
        
        try:
            feed = feedparser.parse(self.rss_url)
            
            if not feed.entries:
                logger.warning("No entries returned from RSS feed")
                return
            
            updates = self.process_rss_entries(feed.entries)
            new_updates = self.filter_duplicates(updates)
            
            for update in new_updates:
                try:
                    categories = self.analyzer.categorize_update(update)
                    importance = self.analyzer.analyze_strategic_importance(update)
                    
                    self.db.store_update(update, importance, categories)
                    self.update_batcher.add_update(update)
                    
                    self.total_updates_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing update: {e}")
                    self.consecutive_errors += 1
            
            if self.update_batcher.should_send_batch():
                await self.send_batch_update()
            
            self.last_successful_check = datetime.now()
            self.consecutive_errors = 0
            
            if new_updates:
                logger.info(f"Added {len(new_updates)} updates to batch queue")
                
        except Exception as e:
            logger.error(f"Error in RSS check: {e}")
            self.consecutive_errors += 1
    
    async def send_batch_update(self):
        """Send intelligent batch summary to Discord"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            logger.warning("Update channel not configured")
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found")
                return
            
            embeds = await self.update_batcher.create_batch_summary()
            
            for embed in embeds[:10]:  # Discord limit
                await channel.send(embed=embed)
            
            logger.info(f"Sent batch update with {len(embeds)} embeds")
            
        except Exception as e:
            logger.error(f"Error sending batch update: {e}")

# Create bot instance
bot = BBDiscordBot()

def main():
    """Main function to run the bot"""
    try:
        bot_token = bot.config.get('bot_token')
        if not bot_token:
            logger.error("No bot token found!")
            return
        
        logger.info("Starting Big Brother Discord Bot...")
        bot.run(bot_token, reconnect=True)
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    main()
