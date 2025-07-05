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
from collections import defaultdict, Counter
import anthropic

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

class Config:
    """Enhanced configuration management"""
    
    def __init__(self):
        self.config_file = Path("config.json")
        self.default_config = {
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
            "show_importance_timeline": True
        }
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration with better validation"""
        config = self.default_config.copy()
        
        # Environment variables (priority 1)
        env_mappings = {
            'BOT_TOKEN': 'bot_token',
            'UPDATE_CHANNEL_ID': 'update_channel_id',
            'ANTHROPIC_API_KEY': 'anthropic_api_key',
            'RSS_CHECK_INTERVAL': 'rss_check_interval',
            'LLM_MODEL': 'llm_model',
            'BATCH_MODE': 'batch_mode'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Type conversion
                if config_key == 'update_channel_id':
                    try:
                        config[config_key] = int(env_value) if env_value != '0' else None
                    except ValueError:
                        logger.warning(f"Invalid channel ID: {env_value}")
                elif config_key in ['rss_check_interval', 'max_retries', 'retry_delay']:
                    try:
                        config[config_key] = int(env_value)
                    except ValueError:
                        logger.warning(f"Invalid integer for {config_key}: {env_value}")
                elif config_key in ['enable_heartbeat', 'enable_llm_summaries']:
                    config[config_key] = env_value.lower() == 'true'
                else:
                    config[config_key] = env_value
        
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
    
    def __init__(self):
        self.competition_keywords = [
            'hoh', 'head of household', 'power of veto', 'pov', 'nomination', 
            'eviction', 'ceremony', 'competition', 'challenge', 'immunity'
        ]
        
        self.strategy_keywords = [
            'alliance', 'backdoor', 'target', 'scheme', 'plan', 'strategy',
            'vote', 'voting', 'campaigning', 'deal', 'promise', 'betrayal'
        ]
        
        self.drama_keywords = [
            'argument', 'fight', 'confrontation', 'drama', 'tension',
            'called out', 'blowup', 'heated', 'angry', 'upset'
        ]
        
        self.relationship_keywords = [
            'showmance', 'romance', 'flirting', 'cuddle', 'kiss',
            'relationship', 'attracted', 'feelings', 'friendship', 'bond'
        ]
        
        self.entertainment_keywords = [
            'funny', 'joke', 'laugh', 'prank', 'hilarious', 'comedy',
            'entertaining', 'memorable', 'quirky', 'silly'
        ]
    
    def categorize_update(self, update: BBUpdate) -> List[str]:
        """Categorize an update based on its content"""
        content = f"{update.title} {update.description}".lower()
        categories = []
        
        if any(keyword in content for keyword in self.competition_keywords):
            categories.append("🏆 Competition")
        
        if any(keyword in content for keyword in self.strategy_keywords):
            categories.append("🎯 Strategy")
        
        if any(keyword in content for keyword in self.drama_keywords):
            categories.append("💥 Drama")
        
        if any(keyword in content for keyword in self.relationship_keywords):
            categories.append("💕 Romance")
        
        if any(keyword in content for keyword in self.entertainment_keywords):
            categories.append("🎬 Entertainment")
        
        return categories if categories else ["📝 General"]
    
    def extract_houseguests(self, text: str) -> List[str]:
        """Extract houseguest names from text"""
        houseguest_pattern = r'\b[A-Z][a-z]+\b'
        potential_names = re.findall(houseguest_pattern, text)
        
        exclude_words = {'The', 'This', 'That', 'They', 'Some', 'Many', 'Other', 'First', 'Last',
                        'Big', 'Brother', 'Julie', 'Host', 'Diary', 'Room', 'Have', 'Not'}
        return [name for name in potential_names if name not in exclude_words]
    
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
            score += 3  # Showmances are big for superfans
        if any(word in content for word in ['fight', 'argument', 'confrontation', 'blowup']):
            score += 3  # Drama is entertainment value
        if any(word in content for word in ['friendship', 'bond', 'close', 'trust']):
            score += 2  # Relationships matter strategically
        
        # Entertainment moments (medium importance)
        if any(word in content for word in ['funny', 'joke', 'laugh', 'prank']):
            score += 2  # Superfans love personality moments
        if any(word in content for word in ['crying', 'emotional', 'breakdown']):
            score += 2  # Emotional moments are significant
        
        # House culture (low-medium importance)
        if any(word in content for word in ['tradition', 'routine', 'habit', 'inside joke']):
            score += 1  # House culture builds over time
        
        # Finale night special scoring
        if any(word in content for word in ['finale', 'winner', 'crowned', 'julie']):
            if any(word in content for word in ['america', 'favorite']):
                score += 4  # AFH is always important
        
        return min(score, 10)

class UpdateBatcher:
    """Groups and analyzes updates like a BB superfan using LLM intelligence"""
    
    def __init__(self, analyzer: BBAnalyzer, api_key: Optional[str] = None):
        self.analyzer = analyzer
        self.update_queue = []
        self.last_batch_time = datetime.now()
        self.processed_hashes = set()
        
        # Initialize Anthropic client with better error handling
        self.llm_client = None
        self.llm_model = "claude-3-haiku-20240307"
        
        if api_key and api_key.strip():
            try:
                self.llm_client = anthropic.Anthropic(api_key=api_key.strip())
                # Test the connection
                test_response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Test"}]
                )
                logger.info("LLM integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                self.llm_client = None
        else:
            logger.warning("No valid Anthropic API key provided")
        
        # Game-defining moments that need immediate attention
        self.urgent_keywords = [
            'evicted', 'eliminated', 'wins hoh', 'wins pov', 'backdoor', 
            'self-evict', 'expelled', 'quit', 'medical', 'pandora', 'coup',
            'diamond veto', 'secret power', 'battle back', 'return', 'winner',
            'final', 'jury vote', 'finale night'
        ]
    
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
        return any(keyword in content for keyword in self.urgent_keywords)
    
    def add_update(self, update: BBUpdate):
        """Add update to queue if not already processed"""
        if update.content_hash not in self.processed_hashes:
            self.update_queue.append(update)
            self.processed_hashes.add(update.content_hash)
    
    def create_batch_summary(self) -> List[discord.Embed]:
        """Create intelligent summary embeds using LLM if available"""
        if not self.update_queue:
            return []
        
        embeds = []
        
        # Use LLM if available, otherwise fall back to pattern matching
        if self.llm_client:
            embeds = self._create_llm_summary()
        else:
            embeds = self._create_pattern_summary_with_explanation("LLM unavailable")
        
        # Clear queue after processing
        self.update_queue.clear()
        self.last_batch_time = datetime.now()
        
        return embeds
    
    def _create_llm_summary(self) -> List[discord.Embed]:
        """Use Claude to create intelligent summaries with balanced superfan coverage"""
        try:
            # Prepare comprehensive update data for LLM
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
            
            if is_finale_night:
                # Use finale-specific prompt (strategic focus)
                prompt = f"""You are Taran Armstrong, the ultimate Big Brother superfan analyzing the FINALE NIGHT of Big Brother.

Analyze these {len(updates_data)} finale night updates:

{chr(10).join([f"{u['time']} - {u['title']}" + (f" ({u['description']})" if u['description'] else "") for u in updates_data])}

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
                # Use balanced superfan prompt for regular season
                prompt = f"""You are the ultimate Big Brother superfan - part Taran Armstrong's strategic genius, part live feed obsessive who loves ALL aspects of the BB experience.

Analyze these {len(updates_data)} Big Brother live feed updates:

{chr(10).join([f"{u['time']} - {u['title']}" + (f" ({u['description']})" if u['description'] else "") for u in updates_data])}

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

Remember: Big Brother superfans want strategic depth BUT also love the social experiment aspects. Include:
- Strategic gameplay (your specialty)
- Social relationships and dynamics
- Entertainment value and memorable moments
- House culture and personality interactions
- Relationship developments (romantic and platonic)

Don't dismiss moments as "surface-level" - social dynamics ARE strategic in Big Brother, and entertainment value matters to superfans."""

            # Get LLM response with proper error handling
            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=1200,  # Increased for comprehensive coverage
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response with fallback
            try:
                response_text = response.content[0].text
                logger.debug(f"LLM Raw Response: {response_text}")
                
                # Try to extract JSON if it's wrapped in other text
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_text = response_text[json_start:json_end]
                    analysis = json.loads(json_text)
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON parsing failed: {e}, using text response")
                # Fallback to structured text parsing
                analysis = self._parse_text_response(response_text)
            
            # Create main embed with enhanced design
            game_phase_colors = {
                "early_game": 0x3498db,    # Blue
                "jury_phase": 0xf39c12,    # Orange  
                "final_weeks": 0xe74c3c,   # Red
                "finale_night": 0xffd700   # Gold
            }
            
            color = game_phase_colors.get(analysis.get('game_phase', 'early_game'), 0x3498db)
            
            main_embed = discord.Embed(
                title=f"🎭 {analysis['headline']}",
                description=f"**{len(self.update_queue)} feed updates** • {analysis.get('game_phase', 'Current Phase').replace('_', ' ').title()}\n\n{analysis['summary']}",
                color=color,
                timestamp=datetime.now()
            )
            
            # Strategic analysis (always included)
            if analysis.get('strategic_analysis'):
                main_embed.add_field(
                    name="🎯 Strategic Analysis",
                    value=analysis['strategic_analysis'],
                    inline=False
                )
            
            # Content based on finale vs regular season
            if is_finale_night:
                # Finale-specific fields
                if analysis.get('social_highlights'):
                    main_embed.add_field(
                        name="🎭 Finale Night Highlights",
                        value=analysis['social_highlights'],
                        inline=False
                    )
                
                if analysis.get('jury_analysis'):
                    main_embed.add_field(
                        name="⚖️ Jury Decision Analysis",
                        value=analysis['jury_analysis'],
                        inline=False
                    )
                
                if analysis.get('finale_moments'):
                    main_embed.add_field(
                        name="✨ Special Finale Moments",
                        value=analysis['finale_moments'],
                        inline=False
                    )
            else:
                # Regular season fields
                if analysis.get('social_dynamics'):
                    main_embed.add_field(
                        name="👥 Social Dynamics",
                        value=analysis['social_dynamics'],
                        inline=False
                    )
                
                if analysis.get('entertainment_highlights'):
                    main_embed.add_field(
                        name="🎬 Entertainment Highlights",
                        value=analysis['entertainment_highlights'],
                        inline=False
                    )
                
                if analysis.get('relationship_updates'):
                    main_embed.add_field(
                        name="💕 Relationship Updates",
                        value=analysis['relationship_updates'],
                        inline=False
                    )
                
                if analysis.get('house_culture'):
                    main_embed.add_field(
                        name="🏠 House Culture",
                        value=analysis['house_culture'],
                        inline=False
                    )
            
            # Key players with better formatting
            if analysis.get('key_players'):
                players = analysis['key_players'][:8]  # Limit to prevent embed overflow
                main_embed.add_field(
                    name="⭐ Key Players",
                    value=" • ".join(players),
                    inline=False
                )
            
            # Strategic importance indicator
            importance = analysis.get('strategic_importance', 5)
            importance_bar = "🔥" * min(importance, 10)
            main_embed.add_field(
                name="🎲 Overall Importance",
                value=f"{importance_bar} {importance}/10",
                inline=True
            )
            
            # Add footer with superfan branding
            footer_text = "Strategic Analysis + Social Dynamics • BB Superfan AI" if not is_finale_night else "Finale Night Analysis • BB Superfan AI"
            main_embed.set_footer(text=footer_text)
            
            embeds = [main_embed]
            
            # Add enhanced timeline based on number of updates
            if len(self.update_queue) > 15:
                # For large batches, use smart pagination
                timeline_embeds = self._create_smart_timeline_embeds(analysis.get('game_phase', 'current'))
                embeds.extend(timeline_embeds)
            elif len(self.update_queue) > 7:
                # For medium batches, use multiple embeds
                timeline_embeds = self._create_timeline_embeds(analysis.get('game_phase', 'current'))
                embeds.extend(timeline_embeds)
            
            return embeds
            
        except Exception as e:
            logger.error(f"LLM summary failed: {e}")
            logger.error(traceback.format_exc())
            # Fall back to pattern matching with explanation
            return self._create_pattern_summary_with_explanation("LLM analysis unavailable")
    
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
        
        # Try to extract key information with regex
        try:
            # Look for houseguest names (capitalized words)
            names = re.findall(r'\b[A-Z][a-z]+\b', response_text)
            # Filter out common words
            exclude = {'The', 'This', 'That', 'Big', 'Brother', 'House', 'Game', 'Vote', 'Player'}
            analysis['key_players'] = [name for name in names if name not in exclude][:5]
            
        except Exception as e:
            logger.debug(f"Text parsing error: {e}")
        
        return analysis
    
    def _create_smart_timeline_embeds(self, game_phase: str) -> List[discord.Embed]:
        """Create timeline embeds with smart pagination based on importance"""
        phase_colors = {
            'early_game': 0x3498db,
            'jury_phase': 0xf39c12,
            'final_weeks': 0xe74c3c,
            'finale_night': 0xffd700
        }
        
        color = phase_colors.get(game_phase, 0x95a5a6)
        
        # Analyze and sort updates by importance and time
        updates_with_importance = []
        for update in self.update_queue:
            importance = self.analyzer.analyze_strategic_importance(update)
            updates_with_importance.append((update, importance))
        
        # Sort by importance first, then by time
        updates_with_importance.sort(key=lambda x: (x[1], x[0].pub_date), reverse=True)
        
        embeds = []
        
        # First embed: Top strategic moments
        high_importance = [u for u, i in updates_with_importance if i >= 7]
        if high_importance:
            embed = discord.Embed(
                title="🔥 Key Strategic Moments",
                description=f"Most important updates from this batch ({len(high_importance)} of {len(self.update_queue)})",
                color=0xe74c3c,  # Red for high importance
                timestamp=datetime.now()
            )
            
            for i, update in enumerate(high_importance[:12], 1):  # Limit to 12 for readability
                time_str = update.pub_date.strftime("%I:%M %p")
                importance = self.analyzer.analyze_strategic_importance(update)
                
                content = update.title
                if len(content) > 150:
                    content = content[:147] + "..."
                
                embed.add_field(
                    name=f"{'🔥' * min(importance, 3)} {time_str}",
                    value=content,
                    inline=False
                )
            
            embeds.append(embed)
        
        # Second embed: Complete chronological timeline
        embed = discord.Embed(
            title="📋 Complete Live Feed Timeline",
            description=f"All {len(self.update_queue)} updates in chronological order",
            color=color,
            timestamp=datetime.now()
        )
        
        # Sort chronologically for this embed
        sorted_updates = sorted(self.update_queue, key=lambda x: x.pub_date, reverse=True)
        
        # Use a more compact format to fit more updates
        timeline_sections = []
        current_section = ""
        
        for i, update in enumerate(sorted_updates, 1):
            time_str = update.pub_date.strftime("%I:%M %p")
            
            # Don't truncate - show full content up to Discord limits
            title = update.title
            if len(title) > 150:
                title = title[:147] + "..."
            
            line = f"**{i}.** {time_str} - {title}\n"
            
            # Check if adding this line would exceed field limit
            if len(current_section + line) > 1000:
                timeline_sections.append(current_section)
                current_section = line
            else:
                current_section += line
        
        # Add the last section
        if current_section:
            timeline_sections.append(current_section)
        
        # Add sections as fields
        for i, section in enumerate(timeline_sections):
            field_name = f"Timeline Part {i + 1}" if len(timeline_sections) > 1 else "Complete Timeline"
            embed.add_field(
                name=field_name,
                value=section,
                inline=False
            )
        
        embeds.append(embed)
        
        return embeds
    
    def _create_timeline_embeds(self, game_phase: str) -> List[discord.Embed]:
        """Create multiple timeline embeds to show all updates without truncation"""
        phase_colors = {
            'early_game': 0x3498db,
            'jury_phase': 0xf39c12,
            'final_weeks': 0xe74c3c,
            'finale_night': 0xffd700
        }
        
        color = phase_colors.get(game_phase, 0x95a5a6)
        embeds = []
        
        # Sort updates chronologically (newest first for finale/important events)
        sorted_updates = sorted(self.update_queue, key=lambda x: x.pub_date, reverse=True)
        
        # Split updates into chunks that fit Discord's limits
        chunk_size = 10
        chunks = [sorted_updates[i:i + chunk_size] for i in range(0, len(sorted_updates), chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            # Create embed for this chunk
            if chunk_idx == 0:
                title = f"📋 Live Feed Timeline - Part {chunk_idx + 1}"
                description = f"Complete breakdown of all {len(self.update_queue)} updates"
            else:
                title = f"📋 Live Feed Timeline - Part {chunk_idx + 1}"
                description = f"Continued timeline (Updates {chunk_idx * chunk_size + 1}-{min((chunk_idx + 1) * chunk_size, len(sorted_updates))})"
            
            embed = discord.Embed(
                title=title,
                description=description,
                color=color,
                timestamp=datetime.now()
            )
            
            # Add updates from this chunk
            for i, update in enumerate(chunk, 1):
                time_str = update.pub_date.strftime("%I:%M %p")
                
                # Don't truncate - show full content
                content = update.title
                if len(content) > 1000:  # Discord field value limit is 1024
                    content = content[:997] + "..."
                
                # Use enumeration for the full timeline
                update_number = chunk_idx * chunk_size + i
                
                embed.add_field(
                    name=f"{update_number}. {time_str}",
                    value=content,
                    inline=False
                )
            
            # Add footer with navigation info
            if len(chunks) > 1:
                embed.set_footer(text=f"Part {chunk_idx + 1} of {len(chunks)} • {len(self.update_queue)} total updates")
            else:
                embed.set_footer(text=f"All {len(self.update_queue)} updates shown")
            
            embeds.append(embed)
        
        return embeds
    
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
    
    def _create_category_summary(self, category: str, updates: List[BBUpdate]) -> str:
        """Create better category summaries"""
        if category == 'votes':
            return self._summarize_votes_pattern(updates)
        elif category == 'competitions':
            return self._summarize_competitions(updates)
        elif category == 'nominations':
            return self._summarize_nominations(updates)
        else:
            # General summary
            if len(updates) <= 3:
                return "\n".join([f"• {u.title[:80]}..." if len(u.title) > 80 else f"• {u.title}" for u in updates])
            else:
                return f"• {updates[0].title[:80]}...\n• {updates[1].title[:80]}...\n• And {len(updates)-2} more updates"
    
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
    
    def _summarize_votes_pattern(self, vote_updates: List[BBUpdate]) -> str:
        """Summarize votes without LLM"""
        vote_counts = defaultdict(list)
        
        for update in vote_updates:
            content = update.title
            # Simple pattern: "X votes for Y"
            parts = content.lower().split('votes for')
            if len(parts) == 2:
                voter = parts[0].strip().split()[-1].title()
                votee = parts[1].strip().split()[0].title()
                vote_counts[votee].append(voter)
        
        summaries = []
        for votee, voters in vote_counts.items():
            if voters:
                summaries.append(f"**{votee}** received votes from: {', '.join(voters)} ({len(voters)} votes)")
        
        return "\n".join(summaries) if summaries else "Votes were cast"
    
    def _summarize_competitions(self, comp_updates: List[BBUpdate]) -> str:
        """Summarize competition updates"""
        if not comp_updates:
            return "No competition updates"
        
        summaries = []
        for update in comp_updates[:3]:  # Show top 3
            summaries.append(f"• {update.title[:80]}...")
        
        if len(comp_updates) > 3:
            summaries.append(f"• And {len(comp_updates) - 3} more competition updates")
        
        return "\n".join(summaries)
    
    def _summarize_nominations(self, nom_updates: List[BBUpdate]) -> str:
        """Summarize nomination updates"""
        if not nom_updates:
            return "No nomination updates"
        
        summaries = []
        for update in nom_updates[:3]:  # Show top 3
            summaries.append(f"• {update.title[:80]}...")
        
        if len(nom_updates) > 3:
            summaries.append(f"• And {len(nom_updates) - 3} more nomination updates")
        
        return "\n".join(summaries)

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
        
        if not self.config.get('bot_token'):
            logger.error("Bot token not configured! Please set BOT_TOKEN environment variable")
            sys.exit(1)
        
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        super().__init__(command_prefix='!bb', intents=intents)
        
        self.rss_url = "https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php"
        self.db = BBDatabase(self.config.get('database_path', 'bb_updates.db'))
        self.analyzer = BBAnalyzer()
        
        # Initialize LLM-enhanced batcher
        anthropic_key = self.config.get('anthropic_api_key')
        if not anthropic_key:
            anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
        
        self.update_batcher = UpdateBatcher(self.analyzer, api_key=anthropic_key)
        
        if anthropic_key:
            logger.info("LLM summaries enabled with Claude")
        else:
            logger.warning("No Anthropic API key found - using pattern matching for summaries")
        
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Remove all slash commands and add them properly
        self.remove_command('help')
        
        # Setup commands after initialization
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
                
                # Add batch queue status
                queue_size = len(self.update_batcher.update_queue)
                embed.add_field(name="Updates in Queue", value=str(queue_size), inline=True)
                
                # Add LLM status
                llm_status = "✅ Enabled" if self.update_batcher.llm_client else "❌ Disabled (Pattern Mode)"
                embed.add_field(name="LLM Summaries", value=llm_status, inline=True)
                
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
                
                # Force send the batch
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
                
                # Test API key presence
                api_key = self.config.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY', '')
                if not api_key:
                    await interaction.followup.send("❌ No Anthropic API key found!", ephemeral=True)
                    return
                
                # Test connection
                if not self.update_batcher.llm_client:
                    await interaction.followup.send("❌ LLM client not initialized", ephemeral=True)
                    return
                
                # Test actual API call
                try:
                    test_response = self.update_batcher.llm_client.messages.create(
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
                    
                    await interaction.followup.send(embed=embed, ephemeral=True)
                    
                except Exception as e:
                    await interaction.followup.send(f"❌ LLM API call failed: {str(e)}", ephemeral=True)
                    
            except Exception as e:
                logger.error(f"Error testing LLM: {e}")
                await interaction.followup.send("Error testing LLM connection.", ephemeral=True)

        @self.tree.command(name="sync", description="Sync slash commands (Owner only)")
        async def sync_slash(interaction: discord.Interaction):
            """Manually sync slash commands"""
            try:
                # Add your Discord user ID here for owner check
                # Replace 123456789 with your actual Discord user ID
                if interaction.user.id != 123456789:  # Replace with your user ID
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
        
        # Sync slash commands properly
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
    
    def create_update_embed(self, update: BBUpdate) -> discord.Embed:
        """Create a Discord embed for an update"""
        categories = self.analyzer.categorize_update(update)
        importance = self.analyzer.analyze_strategic_importance(update)
        
        colors = {
            1: 0x95a5a6, 2: 0x3498db, 3: 0x2ecc71, 4: 0xf39c12, 5: 0xe74c3c
        }
        color = colors.get(min(importance // 2 + 1, 5), 0x95a5a6)
        
        title = update.title[:256] if len(update.title) <= 256 else update.title[:253] + "..."
        description = update.description[:2048] if len(update.description) <= 2048 else update.description[:2045] + "..."
        
        embed = discord.Embed(
            title=title,
            description=description,
            color=color,
            url=update.link,
            timestamp=update.pub_date
        )
        
        if categories:
            embed.add_field(name="Categories", value=" | ".join(categories), inline=True)
        
        importance_stars = "⭐" * importance
        embed.add_field(name="Strategic Importance", value=f"{importance_stars} ({importance}/10)", inline=True)
        
        houseguests = self.analyzer.extract_houseguests(f"{update.title} {update.description}")
        if houseguests:
            houseguests_text = ", ".join(houseguests[:5])
            if len(houseguests) > 5:
                houseguests_text += f" +{len(houseguests) - 5} more"
            embed.add_field(name="Houseguests Mentioned", value=houseguests_text, inline=False)
        
        if update.author:
            embed.set_footer(text=f"Reported by: {update.author}")
        
        return embed
    
    async def send_update_to_channel(self, update: BBUpdate):
        """Send an update to the configured channel"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            logger.warning("Update channel not configured")
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found")
                return
            
            embed = self.create_update_embed(update)
            await channel.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error sending update to channel: {e}")
    
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
            
            # Add new updates to the batcher instead of sending immediately
            for update in new_updates:
                try:
                    # Still need to analyze for categorization
                    categories = self.analyzer.categorize_update(update)
                    importance = self.analyzer.analyze_strategic_importance(update)
                    
                    # Store in database
                    self.db.store_update(update, importance, categories)
                    
                    # Add to batcher queue
                    self.update_batcher.add_update(update)
                    
                    self.total_updates_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing update: {e}")
                    self.consecutive_errors += 1
            
            # Check if we should send a batch
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
            
            # Get batch summary embeds
            embeds = self.update_batcher.create_batch_summary()
            
            # Send all embeds
            for embed in embeds[:10]:  # Discord limit is 10 embeds per message
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
