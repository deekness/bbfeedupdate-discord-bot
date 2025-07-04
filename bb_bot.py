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
from typing import List, Dict, Set, Tuple
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
import statistics

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging for 24/7 operation"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_dir / "bb_bot.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    error_handler = logging.FileHandler(log_dir / "bb_bot_errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()

class Config:
    """Configuration management for the bot"""
    
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
            "current_season": 27,
            "bb27_houseguests": [
                "Angela", "Tucker", "Makensy", "Cam", "Chelsie", 
                "Rubina", "Kimo", "Leah", "Quinn", "Joseph",
                "T'Kor", "Cedric", "Brooklyn", "Kenney", "Lisa"
            ]
        }
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration from environment variables or file"""
        config = {
            "bot_token": os.getenv('BOT_TOKEN', ''),
            "update_channel_id": int(os.getenv('UPDATE_CHANNEL_ID', '0')) or None,
            "rss_check_interval": int(os.getenv('RSS_CHECK_INTERVAL', '2')),
            "max_retries": int(os.getenv('MAX_RETRIES', '3')),
            "retry_delay": int(os.getenv('RETRY_DELAY', '5')),
            "database_path": os.getenv('DATABASE_PATH', 'bb_updates.db'),
            "enable_heartbeat": os.getenv('ENABLE_HEARTBEAT', 'true').lower() == 'true',
            "heartbeat_interval": int(os.getenv('HEARTBEAT_INTERVAL', '300')),
            "max_update_age_hours": int(os.getenv('MAX_UPDATE_AGE_HOURS', '168')),
            "enable_auto_restart": os.getenv('ENABLE_AUTO_RESTART', 'true').lower() == 'true',
            "max_consecutive_errors": int(os.getenv('MAX_CONSECUTIVE_ERRORS', '10')),
            "current_season": int(os.getenv('CURRENT_SEASON', '27')),
            "bb27_houseguests": [
                "Angela", "Tucker", "Makensy", "Cam", "Chelsie", 
                "Rubina", "Kimo", "Leah", "Quinn", "Joseph",
                "T'Kor", "Cedric", "Brooklyn", "Kenney", "Lisa"
            ]
        }
        
        if not config["bot_token"] and self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
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
    """Enhanced Big Brother analyzer with advanced features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.houseguests = config.get('bb27_houseguests', [])
        
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
            'relationship', 'attracted', 'feelings'
        ]
        
        # Advanced analytics
        self.sentiment_positive = [
            'happy', 'excited', 'love', 'amazing', 'great', 'wonderful', 
            'perfect', 'fantastic', 'awesome', 'good', 'fun', 'laugh'
        ]
        
        self.sentiment_negative = [
            'angry', 'hate', 'mad', 'upset', 'annoyed', 'frustrated',
            'terrible', 'awful', 'bad', 'worst', 'annoying', 'drama'
        ]
        
        self.alliance_tracker = {}
        self.relationship_matrix = defaultdict(lambda: defaultdict(int))
        self.houseguest_stats = {}
    
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
        
        return categories if categories else ["📝 General"]
    
    def extract_houseguests(self, text: str) -> List[str]:
        """Extract houseguest names from text"""
        mentioned_houseguests = []
        text_lower = text.lower()
        
        for houseguest in self.houseguests:
            if houseguest.lower() in text_lower:
                mentioned_houseguests.append(houseguest)
        
        return mentioned_houseguests
    
    def analyze_strategic_importance(self, update: BBUpdate) -> int:
        """Rate strategic importance from 1-10"""
        content = f"{update.title} {update.description}".lower()
        score = 1
        
        if any(word in content for word in ['eviction', 'nomination', 'backdoor']):
            score += 4
        if any(word in content for word in ['hoh', 'head of household', 'power of veto']):
            score += 3
        if any(word in content for word in ['alliance', 'target', 'strategy']):
            score += 2
        if any(word in content for word in ['vote', 'voting', 'campaign']):
            score += 2
        
        return min(score, 10)
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        text_lower = text.lower()
        
        positive_score = sum(1 for word in self.sentiment_positive if word in text_lower)
        negative_score = sum(1 for word in self.sentiment_negative if word in text_lower)
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        positive_ratio = positive_score / total_words
        negative_ratio = negative_score / total_words
        neutral_ratio = 1 - (positive_ratio + negative_ratio)
        
        return {
            "positive": positive_ratio,
            "negative": negative_ratio, 
            "neutral": max(0, neutral_ratio)
        }
    
    def track_alliances(self, update: BBUpdate) -> List[Dict]:
        """Track alliance formations"""
        content = f"{update.title} {update.description}".lower()
        
        alliance_keywords = ['alliance', 'group', 'team', 'together', 'meet', 'plan']
        if not any(keyword in content for keyword in alliance_keywords):
            return []
        
        mentioned_houseguests = self.extract_houseguests(content)
        
        if len(mentioned_houseguests) < 2:
            return []
        
        # Track relationships
        for i, hg1 in enumerate(mentioned_houseguests):
            for hg2 in mentioned_houseguests[i+1:]:
                self.relationship_matrix[hg1][hg2] += 1
                self.relationship_matrix[hg2][hg1] += 1
        
        # Detect new alliances
        new_alliances = []
        if len(mentioned_houseguests) >= 3:
            alliance_key = tuple(sorted(mentioned_houseguests))
            
            if alliance_key not in self.alliance_tracker:
                self.alliance_tracker[alliance_key] = {
                    'members': mentioned_houseguests,
                    'strength': 1.0,
                    'first_detected': update.pub_date,
                    'last_mentioned': update.pub_date,
                    'mentions_count': 1
                }
                new_alliances.append({
                    'members': mentioned_houseguests,
                    'type': 'new_alliance',
                    'strength': 1.0
                })
        
        return new_alliances
    
    def predict_eviction(self, recent_updates: List[BBUpdate]) -> Dict[str, float]:
        """Predict eviction likelihood"""
        eviction_indicators = {}
        
        for update in recent_updates:
            content = f"{update.title} {update.description}".lower()
            
            campaign_keywords = ['campaign', 'vote', 'evict', 'target', 'backdoor', 'pawn']
            if any(keyword in content for keyword in campaign_keywords):
                for houseguest in self.houseguests:
                    if houseguest.lower() in content:
                        if houseguest not in eviction_indicators:
                            eviction_indicators[houseguest] = 0.0
                        
                        if any(word in content for word in ['target', 'backdoor', 'evict']):
                            eviction_indicators[houseguest] += 2.0
                        elif 'campaign' in content:
                            eviction_indicators[houseguest] += 1.0
        
        if eviction_indicators:
            max_score = max(eviction_indicators.values())
            if max_score > 0:
                for houseguest in eviction_indicators:
                    eviction_indicators[houseguest] = min(1.0, eviction_indicators[houseguest] / max_score)
        
        return eviction_indicators
    
    def calculate_power_rankings(self, updates: List[BBUpdate]) -> List[Dict]:
        """Calculate power rankings"""
        rankings = {}
        
        for houseguest in self.houseguests:
            rankings[houseguest] = {
                'name': houseguest,
                'power_ranking': 0.0,
                'hoh_wins': 0,
                'pov_wins': 0,
                'strategy_mentions': 0,
                'social_connections': 0,
                'target_level': 0.0
            }
        
        for update in updates:
            content = f"{update.title} {update.description}".lower()
            
            for houseguest in self.houseguests:
                if houseguest.lower() in content:
                    stats = rankings[houseguest]
                    
                    if any(word in content for word in ['strategy', 'plan', 'alliance']):
                        stats['power_ranking'] += 2.0
                        stats['strategy_mentions'] += 1
                    
                    if 'wins' in content and ('hoh' in content or 'pov' in content):
                        stats['power_ranking'] += 3.0
                        if 'hoh' in content:
                            stats['hoh_wins'] += 1
                        if 'pov' in content:
                            stats['pov_wins'] += 1
                    
                    if any(word in content for word in ['target', 'backdoor', 'evict']):
                        stats['power_ranking'] -= 2.0
                        stats['target_level'] += 1.0
        
        return sorted(rankings.values(), key=lambda x: x['power_ranking'], reverse=True)
    
    def get_activity_heatmap(self, updates: List[BBUpdate]) -> Dict[str, int]:
        """Get activity heatmap"""
        activity = defaultdict(int)
        
        for update in updates:
            houseguests = self.extract_houseguests(f"{update.title} {update.description}")
            for houseguest in houseguests:
                activity[houseguest] += 1
        
        return dict(activity)

class BBDatabase:
    """Database with enhanced analytics"""
    
    def __init__(self, db_path: str = "bb_updates.db"):
        self.db_path = db_path
        self.connection_timeout = 30
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        try:
            conn = sqlite3.connect(
                self.db_path, 
                timeout=self.connection_timeout,
                check_same_thread=False
            )
            
            alliances = self.analyzer.alliance_tracker
            
            if not alliances:
                embed.add_field(name="No Alliances", value="No clear alliances detected yet", inline=False)
            else:
                for i, (alliance_key, alliance_data) in enumerate(alliances.items(), 1):
                    members_str = ", ".join(alliance_data['members'])
                    strength = alliance_data['strength']
                    strength_bar = "█" * int(strength) + "░" * (10 - int(strength))
                    
                    embed.add_field(
                        name=f"Alliance #{i}",
                        value=f"**Members:** {members_str}\n**Strength:** {strength_bar} ({strength:.1f}/10)",
                        inline=False
                    )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error showing alliances: {e}")
            await ctx.send("Error retrieving alliance data. Please try again.")
    
    @commands.command(name='evictionpredict')
    async def eviction_prediction(self, ctx):
        """Predict eviction likelihood"""
        try:
            recent_updates = self.db.get_recent_updates(72)
            predictions = self.analyzer.predict_eviction(recent_updates)
            
            embed = discord.Embed(
                title="🎯 Eviction Predictions",
                description="Likelihood of eviction based on recent activity",
                color=0xe74c3c,
                timestamp=datetime.now()
            )
            
            if not predictions:
                embed.add_field(name="No Predictions", value="Not enough data for eviction predictions", inline=False)
            else:
                sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                
                for houseguest, likelihood in sorted_predictions[:5]:
                    risk_level = "🔴 High" if likelihood > 0.7 else "🟡 Medium" if likelihood > 0.3 else "🟢 Low"
                    percentage = f"{likelihood * 100:.1f}%"
                    
                    embed.add_field(
                        name=f"{houseguest}",
                        value=f"**Risk:** {risk_level}\n**Likelihood:** {percentage}",
                        inline=True
                    )
            
            embed.set_footer(text="Based on targeting language and campaign activity")
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating eviction predictions: {e}")
            await ctx.send("Error generating predictions. Please try again.")
    
    @commands.command(name='sentiment')
    async def house_sentiment(self, ctx, hours: int = 24):
        """Show house sentiment analysis"""
        try:
            recent_updates = self.db.get_recent_updates(hours)
            
            if not recent_updates:
                await ctx.send(f"No updates found in the last {hours} hours.")
                return
            
            total_positive = 0
            total_negative = 0
            total_neutral = 0
            
            for update in recent_updates:
                sentiment = self.analyzer.analyze_sentiment(f"{update.title} {update.description}")
                total_positive += sentiment["positive"]
                total_negative += sentiment["negative"]
                total_neutral += sentiment["neutral"]
            
            count = len(recent_updates)
            avg_positive = total_positive / count
            avg_negative = total_negative / count
            avg_neutral = total_neutral / count
            
            if avg_positive > avg_negative and avg_positive > avg_neutral:
                mood = "😊 Positive"
                color = 0x2ecc71
            elif avg_negative > avg_positive and avg_negative > avg_neutral:
                mood = "😤 Tense"
                color = 0xe74c3c
            else:
                mood = "😐 Neutral"
                color = 0x95a5a6
            
            embed = discord.Embed(
                title="🏠 House Sentiment Analysis",
                description=f"Overall house mood: {mood}",
                color=color,
                timestamp=datetime.now()
            )
            
            embed.add_field(name="😊 Positive", value=f"{avg_positive * 100:.1f}%", inline=True)
            embed.add_field(name="😤 Negative", value=f"{avg_negative * 100:.1f}%", inline=True)
            embed.add_field(name="😐 Neutral", value=f"{avg_neutral * 100:.1f}%", inline=True)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating sentiment analysis: {e}")
            await ctx.send("Error generating sentiment analysis. Please try again.")
    
    @commands.command(name='heatmap')
    async def activity_heatmap(self, ctx, hours: int = 24):
        """Show activity heatmap"""
        try:
            recent_updates = self.db.get_recent_updates(hours)
            activity = self.analyzer.get_activity_heatmap(recent_updates)
            
            embed = discord.Embed(
                title="🔥 Activity Heatmap",
                description=f"Houseguest mentions in the last {hours} hours",
                color=0xf39c12,
                timestamp=datetime.now()
            )
            
            if not activity:
                embed.add_field(name="No Activity", value="No houseguests mentioned in recent updates", inline=False)
            else:
                sorted_activity = sorted(activity.items(), key=lambda x: x[1], reverse=True)
                
                for houseguest, mentions in sorted_activity[:10]:
                    heat_level = "🔥🔥🔥" if mentions > 10 else "🔥🔥" if mentions > 5 else "🔥" if mentions > 2 else "💤"
                    
                    embed.add_field(
                        name=f"{houseguest}",
                        value=f"{heat_level} {mentions} mentions",
                        inline=True
                    )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating activity heatmap: {e}")
            await ctx.send("Error generating heatmap. Please try again.")
    
    @commands.command(name='predict')
    async def prediction_game(self, ctx, prediction_type: str = None, target: str = None, *, value: str = None):
        """Make predictions"""
        if not prediction_type:
            embed = discord.Embed(
                title="🔮 Prediction Game",
                description="Make predictions about Big Brother events",
                color=0x9b59b6
            )
            
            embed.add_field(
                name="How to Use",
                value="!bbpredict [type] [target] [value]",
                inline=False
            )
            
            embed.add_field(
                name="Examples",
                value="• `!bbpredict eviction Michael`\n• `!bbpredict hoh Sarah`",
                inline=False
            )
            
            await ctx.send(embed=embed)
            return
        
        if not target:
            await ctx.send("Please provide a target. Use `!bbpredict` for help.")
            return
        
        try:
            self.db.store_prediction(str(ctx.author.id), prediction_type, target, value or target, 0.5)
            
            embed = discord.Embed(
                title="🔮 Prediction Recorded",
                description=f"Your prediction has been saved!",
                color=0x2ecc71,
                timestamp=datetime.now()
            )
            
            embed.add_field(name="Type", value=prediction_type.title(), inline=True)
            embed.add_field(name="Target", value=target, inline=True)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            await ctx.send("Error storing prediction. Please try again.")
    
    # ORIGINAL COMMANDS
    @commands.command(name='summary')
    async def daily_summary(self, ctx, hours: int = 24):
        """Generate enhanced summary"""
        try:
            if hours < 1 or hours > 168:
                await ctx.send("Hours must be between 1 and 168")
                return
                
            updates = self.db.get_recent_updates(hours)
            
            if not updates:
                await ctx.send(f"No updates found in the last {hours} hours.")
                return
            
            categories = {}
            for update in updates:
                update_categories = self.analyzer.categorize_update(update)
                for category in update_categories:
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(update)
            
            embed = discord.Embed(
                title=f"📊 Enhanced Big Brother Summary ({hours}h)",
                description=f"**{len(updates)} total updates**",
                color=0x3498db,
                timestamp=datetime.now()
            )
            
            for category, cat_updates in categories.items():
                top_updates = sorted(cat_updates, 
                                   key=lambda x: self.analyzer.analyze_strategic_importance(x), 
                                   reverse=True)[:3]
                
                summary_text = "\n".join([f"• {update.title[:80]}..." 
                                        if len(update.title) > 80 
                                        else f"• {update.title}" 
                                        for update in top_updates])
                
                embed.add_field(
                    name=f"{category} ({len(cat_updates)} updates)",
                    value=summary_text or "No updates",
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            await ctx.send("Error generating summary. Please try again.")
    
    @commands.command(name='setchannel')
    @commands.has_permissions(administrator=True)
    async def set_update_channel(self, ctx, channel_id: int):
        """Set update channel"""
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                await ctx.send(f"Channel with ID {channel_id} not found.")
                return
            
            if not channel.permissions_for(ctx.guild.me).send_messages:
                await ctx.send(f"I don't have permission to send messages in <#{channel_id}>")
                return
            
            self.config.set('update_channel_id', channel_id)
            await ctx.send(f"Update channel set to <#{channel_id}>")
            logger.info(f"Update channel set to {channel_id}")
            
        except Exception as e:
            logger.error(f"Error setting channel: {e}")
            await ctx.send("Error setting channel. Please try again.")
    
    @commands.command(name='status')
    async def bot_status(self, ctx):
        """Show bot status"""
        try:
            embed = discord.Embed(
                title="🤖 Enhanced Big Brother Bot Status",
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
            
            embed.add_field(name="🧠 AI Features", value="✅ Active", inline=True)
            embed.add_field(name="📊 Analytics", value="✅ Tracking", inline=True)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating status: {e}")
            await ctx.send("Error generating status.")
    
    # TEST COMMANDS
    @commands.command(name='testanalyzer')
    async def test_analyzer(self, ctx):
        """Test enhanced analyzer"""
        sample_updates = [
            {
                "title": "HOH Competition Results - Sarah Wins",
                "description": "Sarah wins Head of Household after intense endurance challenge. She's planning to backdoor Michael and is meeting with her alliance tonight.",
                "author": "BBUpdater1"
            },
            {
                "title": "Kitchen Drama Escalates",
                "description": "Huge argument between Lisa and Michael over dishes. Lisa called him out and the house is taking sides. Very tense atmosphere.",
                "author": "DramaAlert"
            }
        ]
        
        embed = discord.Embed(
            title="🧪 Enhanced AI Test Results",
            description="Testing advanced Big Brother analytics",
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        for i, sample in enumerate(sample_updates, 1):
            mock_update = BBUpdate(
                title=sample["title"],
                description=sample["description"],
                link=f"https://example.com/update{i}",
                pub_date=datetime.now(),
                content_hash=f"test_hash_{i}",
                author=sample["author"]
            )
            
            categories = self.analyzer.categorize_update(mock_update)
            importance = self.analyzer.analyze_strategic_importance(mock_update)
            houseguests = self.analyzer.extract_houseguests(f"{mock_update.title} {mock_update.description}")
            sentiment = self.analyzer.analyze_sentiment(f"{mock_update.title} {mock_update.description}")
            
            analysis = []
            analysis.append(f"**Categories:** {' | '.join(categories)}")
            analysis.append(f"**Importance:** {'⭐' * importance} ({importance}/10)")
            
            if sentiment["positive"] > 0.1:
                analysis.append(f"**Sentiment:** 😊 Positive")
            elif sentiment["negative"] > 0.1:
                analysis.append(f"**Sentiment:** 😤 Negative")
            else:
                analysis.append(f"**Sentiment:** 😐 Neutral")
            
            if houseguests:
                analysis.append(f"**Houseguests:** {', '.join(houseguests[:3])}")
            
            embed.add_field(
                name=f"#{i}: {sample['title'][:40]}...",
                value="\n".join(analysis),
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='testhelp')
    async def test_help(self, ctx):
        """Show test commands"""
        embed = discord.Embed(
            title="🧪 Enhanced Test Commands",
            description="Test all advanced features",
            color=0x9b59b6
        )
        
        embed.add_field(
            name="**Basic Tests**",
            value="• `!bbtestanalyzer` - Test enhanced AI\n• `!bbpowerrankings` - Test power rankings",
            inline=False
        )
        
        embed.add_field(
            name="**Analytics Tests**",
            value="• `!bballiances` - Test alliance detection\n• `!bbsentiment` - Test sentiment analysis\n• `!bbheatmap` - Test activity heatmap",
            inline=False
        )
        
        embed.add_field(
            name="**Prediction Tests**",
            value="• `!bbpredict eviction Angela` - Test predictions\n• `!bbevictionpredict` - Test eviction predictions",
            inline=False
        )
        
        await ctx.send(embed=embed)

def main():
    """Main function"""
    try:
        bot = BBDiscordBot()
        
        @bot.event
        async def on_command_error(ctx, error):
            if isinstance(error, commands.MissingPermissions):
                await ctx.send("You don't have permission to use this command.")
            elif isinstance(error, commands.CommandNotFound):
                await ctx.send("Command not found. Use `!bbcommands` for available commands.")
            else:
                logger.error(f"Command error: {error}")
                await ctx.send("An error occurred while processing the command.")
        
        @bot.command(name='commands')
        async def commands_help(ctx):
            """Show all commands"""
            embed = discord.Embed(
                title="🏠 Enhanced Big Brother Bot Commands",
                description="AI-powered Big Brother analysis with advanced features",
                color=0x3498db
            )
            
            embed.add_field(
                name="**📊 Main Commands**",
                value="• `!bbsummary [hours]` - Enhanced summary\n• `!bbstatus` - Bot status\n• `!bbsetchannel [ID]` - Set channel (Admin)",
                inline=False
            )
            
            embed.add_field(
                name="**🏆 Analytics Commands**",
                value="• `!bbpowerrankings` - Power rankings\n• `!bballiances` - Alliance detection\n• `!bbsentiment` - House mood\n• `!bbheatmap` - Activity heatmap",
                inline=False
            )
            
            embed.add_field(
                name="**🔮 Prediction Commands**",
                value="• `!bbpredict [type] [target]` - Make predictions\n• `!bbevictionpredict` - Eviction likelihood",
                inline=False
            )
            
            embed.add_field(
                name="**🧪 Test Commands**",
                value="• `!bbtesthelp` - Show all test commands\n• `!bbtestanalyzer` - Test AI features",
                inline=False
            )
            
            await ctx.send(embed=embed)
        
        bot_token = bot.config.get('bot_token')
        if not bot_token:
            logger.error("No bot token found!")
            return
        
        logger.info("Starting Enhanced Big Brother Discord Bot...")
        bot.run(bot_token, reconnect=True)
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    main()
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def init_database(self):
        """Initialize database"""
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
                    sentiment_positive REAL DEFAULT 0.0,
                    sentiment_negative REAL DEFAULT 0.0,
                    sentiment_neutral REAL DEFAULT 1.0,
                    mentioned_houseguests TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    prediction_type TEXT,
                    prediction_target TEXT,
                    prediction_value TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    was_correct BOOLEAN
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON updates(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pub_date ON updates(pub_date)")
            
            conn.commit()
            logger.info("Enhanced database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
        finally:
            conn.close()
    
    def is_duplicate(self, content_hash: str) -> bool:
        """Check if update exists"""
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
    
    def store_update(self, update: BBUpdate, importance_score: int = 1, categories: List[str] = None, 
                    sentiment: Dict[str, float] = None, mentioned_houseguests: List[str] = None):
        """Store update with analytics"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            categories_str = ",".join(categories) if categories else ""
            houseguests_str = ",".join(mentioned_houseguests) if mentioned_houseguests else ""
            
            sentiment = sentiment or {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
            cursor.execute("""
                INSERT INTO updates (content_hash, title, description, link, pub_date, author, 
                                   importance_score, categories, sentiment_positive, sentiment_negative, 
                                   sentiment_neutral, mentioned_houseguests)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (update.content_hash, update.title, update.description, update.link, 
                  update.pub_date, update.author, importance_score, categories_str,
                  sentiment["positive"], sentiment["negative"], sentiment["neutral"],
                  houseguests_str))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database store error: {e}")
            raise
    
    def get_recent_updates(self, hours: int = 24) -> List[BBUpdate]:
        """Get recent updates"""
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
    
    def store_prediction(self, user_id: str, prediction_type: str, target: str, value: str, confidence: float):
        """Store prediction"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (user_id, prediction_type, prediction_target, prediction_value, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, prediction_type, target, value, confidence))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")

class BBDiscordBot(commands.Bot):
    """Enhanced Discord bot"""
    
    def __init__(self):
        self.config = Config()
        
        if not self.config.get('bot_token'):
            logger.error("Bot token not configured! Please set BOT_TOKEN environment variable")
            sys.exit(1)
        
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!bb', intents=intents)
        
        self.rss_url = "https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php"
        self.db = BBDatabase(self.config.get('database_path', 'bb_updates.db'))
        self.analyzer = BBAnalyzer(self.config)
        
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_shutting_down = True
        asyncio.create_task(self.close())
    
    async def on_ready(self):
        """Bot startup event"""
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')
        
        try:
            self.check_rss_feed.start()
            logger.info("RSS feed monitoring started")
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    def create_content_hash(self, title: str, description: str) -> str:
        """Create content hash"""
        content = f"{title}|{description}".lower()
        content = re.sub(r'\d{1,2}:\d{2}[ap]m', '', content)
        content = re.sub(r'\d{1,2}/\d{1,2}', '', content)
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_rss_entries(self, entries) -> List[BBUpdate]:
        """Process RSS entries"""
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
        """Filter duplicates"""
        new_updates = []
        seen_hashes = set()
        
        for update in updates:
            if not self.db.is_duplicate(update.content_hash):
                if update.content_hash not in seen_hashes:
                    new_updates.append(update)
                    seen_hashes.add(update.content_hash)
        
        return new_updates
    
    def create_update_embed(self, update: BBUpdate) -> discord.Embed:
        """Create Discord embed"""
        categories = self.analyzer.categorize_update(update)
        importance = self.analyzer.analyze_strategic_importance(update)
        houseguests = self.analyzer.extract_houseguests(f"{update.title} {update.description}")
        sentiment = self.analyzer.analyze_sentiment(f"{update.title} {update.description}")
        
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
        
        if sentiment["positive"] > 0.1:
            embed.add_field(name="House Mood", value="😊 Positive", inline=True)
        elif sentiment["negative"] > 0.1:
            embed.add_field(name="House Mood", value="😤 Tense", inline=True)
        else:
            embed.add_field(name="House Mood", value="😐 Neutral", inline=True)
        
        if houseguests:
            houseguests_text = ", ".join(houseguests[:5])
            if len(houseguests) > 5:
                houseguests_text += f" +{len(houseguests) - 5} more"
            embed.add_field(name="Houseguests Mentioned", value=houseguests_text, inline=False)
        
        if update.author:
            embed.set_footer(text=f"Reported by: {update.author}")
        
        return embed
    
    async def send_update_to_channel(self, update: BBUpdate):
        """Send update to channel"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                return
            
            embed = self.create_update_embed(update)
            await channel.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error sending update to channel: {e}")
    
    @tasks.loop(minutes=2)
    async def check_rss_feed(self):
        """Check RSS feed"""
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
                    houseguests = self.analyzer.extract_houseguests(f"{update.title} {update.description}")
                    sentiment = self.analyzer.analyze_sentiment(f"{update.title} {update.description}")
                    
                    self.db.store_update(update, importance, categories, sentiment, houseguests)
                    self.analyzer.track_alliances(update)
                    
                    await self.send_update_to_channel(update)
                    
                    self.total_updates_processed += 1
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing update: {e}")
                    self.consecutive_errors += 1
            
            self.last_successful_check = datetime.now()
            self.consecutive_errors = 0
            
            if new_updates:
                logger.info(f"Processed {len(new_updates)} new updates")
                
        except Exception as e:
            logger.error(f"Error in RSS check: {e}")
            self.consecutive_errors += 1
    
    # ENHANCED COMMANDS
    @commands.command(name='powerrankings')
    async def power_rankings(self, ctx):
        """Show power rankings"""
        try:
            recent_updates = self.db.get_recent_updates(168)
            rankings = self.analyzer.calculate_power_rankings(recent_updates)
            
            embed = discord.Embed(
                title="🏆 Big Brother Power Rankings",
                description="Weekly power rankings based on strategic positioning",
                color=0xf39c12,
                timestamp=datetime.now()
            )
            
            for i, player in enumerate(rankings[:10], 1):
                rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
                
                embed.add_field(
                    name=f"{rank_emoji} {player['name']}",
                    value=f"**Score:** {player['power_ranking']:.1f}\n**Strategy:** {player['strategy_mentions']} mentions",
                    inline=True
                )
            
            embed.set_footer(text="Based on strategic mentions and game activity")
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating power rankings: {e}")
            await ctx.send("Error generating power rankings. Please try again.")
    
    @commands.command(name='alliances')
    async def show_alliances(self, ctx):
        """Show detected alliances"""
        try:
            embed = discord.Embed(
                title="🤝 Detected Alliances",
                description="Active alliances based on update analysis",
                color=0x9b59b6,
                timestamp=datetime.now()
