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
from typing import List, Dict, Set
import logging
from dataclasses import dataclass
import json
import time
import traceback
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
            "max_consecutive_errors": 10
        }
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration from environment variables or file"""
        # Try environment variables first (for cloud deployment)
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
            "max_consecutive_errors": int(os.getenv('MAX_CONSECUTIVE_ERRORS', '10'))
        }
        
        # If no bot token from environment, try config file
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
    """Analyzes Big Brother updates for strategic insights"""
    
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
            'relationship', 'attracted', 'feelings'
        ]
        
        # Eviction likelihood analysis keywords
        self.targeting_keywords = [
            'target', 'backdoor', 'going home', 'evict', 'vote out', 
            'get rid of', 'eliminate', 'wants out', 'sending home'
        ]
        
        self.safety_keywords = [
            'safe', 'protected', 'immune', 'alliance', 'tight with',
            'good with', 'working with', 'keeping', 'trust'
        ]
        
        self.campaign_keywords = [
            'campaigning', 'campaign', 'votes', 'staying', 'fighting',
            'talking to', 'convincing', 'pleading', 'begging'
        ]
        
        self.isolation_keywords = [
            'alone', 'isolated', 'no allies', 'everyone against',
            'outsider', 'bottom', 'expendable', 'floater'
        ]
        
        # Current BB27 houseguests for analysis
        self.houseguests = [
            'Angela', 'Tucker', 'Makensy', 'Cam', 'Chelsie', 
            'Rubina', 'Kimo', 'Leah', 'Quinn', 'Joseph',
            'T\'Kor', 'Cedric', 'Brooklyn', 'Kenney', 'Lisa'
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
        
        return categories if categories else ["📝 General"]
    
    def extract_houseguests(self, text: str) -> List[str]:
        """Extract houseguest names from text using known houseguest list"""
        mentioned_houseguests = []
        text_lower = text.lower()
        
        for houseguest in self.houseguests:
            if houseguest.lower() in text_lower:
                mentioned_houseguests.append(houseguest)
        
        return mentioned_houseguests
    
    def analyze_eviction_likelihood(self, updates: List[BBUpdate]) -> Dict[str, Dict]:
        """Analyze eviction likelihood for each houseguest based on recent updates"""
        
        # Initialize likelihood scores for each houseguest
        likelihood_scores = {}
        for houseguest in self.houseguests:
            likelihood_scores[houseguest] = {
                'target_score': 0,
                'safety_score': 0,
                'campaign_score': 0,
                'isolation_score': 0,
                'total_mentions': 0,
                'reasons': [],
                'likelihood_percentage': 0,
                'trend': 'stable'
            }
        
        # Analyze each update
        for update in updates:
            content = f"{update.title} {update.description}".lower()
            mentioned_houseguests = self.extract_houseguests(content)
            
            for houseguest in mentioned_houseguests:
                hg_data = likelihood_scores[houseguest]
                hg_data['total_mentions'] += 1
                
                # Check for targeting language
                if any(keyword in content for keyword in self.targeting_keywords):
                    hg_data['target_score'] += 3
                    hg_data['reasons'].append(f"Mentioned as target in: {update.title[:50]}...")
                
                # Check for safety indicators
                if any(keyword in content for keyword in self.safety_keywords):
                    hg_data['safety_score'] += 2
                    hg_data['reasons'].append(f"Mentioned as safe in: {update.title[:50]}...")
                
                # Check for campaign activity
                if any(keyword in content for keyword in self.campaign_keywords):
                    hg_data['campaign_score'] += 1
                    hg_data['reasons'].append(f"Campaign activity in: {update.title[:50]}...")
                
                # Check for isolation indicators
                if any(keyword in content for keyword in self.isolation_keywords):
                    hg_data['isolation_score'] += 2
                    hg_data['reasons'].append(f"Isolation mentioned in: {update.title[:50]}...")
        
        # Calculate final likelihood percentages
        for houseguest, data in likelihood_scores.items():
            if data['total_mentions'] == 0:
                data['likelihood_percentage'] = 0
                data['confidence'] = 'No Data'
                continue
            
            # Calculate weighted score
            danger_score = (data['target_score'] * 1.5) + (data['isolation_score'] * 1.2) + (data['campaign_score'] * 0.8)
            safety_score = data['safety_score']
            
            # Net risk score
            net_score = danger_score - safety_score
            
            # Convert to percentage (0-100)
            if net_score <= 0:
                likelihood = 0
            else:
                # Scale based on mention frequency and danger indicators
                likelihood = min(100, (net_score / data['total_mentions']) * 20)
            
            data['likelihood_percentage'] = likelihood
            
            # Determine confidence level
            if data['total_mentions'] >= 5:
                data['confidence'] = 'High'
            elif data['total_mentions'] >= 3:
                data['confidence'] = 'Medium'
            elif data['total_mentions'] >= 1:
                data['confidence'] = 'Low'
            else:
                data['confidence'] = 'No Data'
            
            # Limit reasons to top 3
            data['reasons'] = data['reasons'][:3]
        
        return likelihood_scores
    
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
            
            # Updates table
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
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    user_name TEXT,
                    prediction_type TEXT,
                    prediction_target TEXT,
                    prediction_value TEXT,
                    confidence INTEGER DEFAULT 5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    was_correct BOOLEAN,
                    season INTEGER DEFAULT 27
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON updates(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pub_date ON updates(pub_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type)")
            
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
    
    def store_prediction(self, user_id: str, user_name: str, prediction_type: str, 
                        prediction_target: str, prediction_value: str, confidence: int = 5):
        """Store a user prediction"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (user_id, user_name, prediction_type, prediction_target, 
                                       prediction_value, confidence, season)
                VALUES (?, ?, ?, ?, ?, ?, 27)
            """, (user_id, user_name, prediction_type, prediction_target, prediction_value, confidence))
            
            conn.commit()
            conn.close()
            logger.info(f"Stored prediction: {user_name} predicts {prediction_type} - {prediction_target}")
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            raise
    
    def get_user_predictions(self, user_id: str) -> List[Dict]:
        """Get all predictions for a user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT prediction_type, prediction_target, prediction_value, confidence, 
                       created_at, was_correct, resolved_at
                FROM predictions 
                WHERE user_id = ? AND season = 27
                ORDER BY created_at DESC
            """, (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            predictions = []
            for row in results:
                predictions.append({
                    'type': row[0],
                    'target': row[1],
                    'value': row[2],
                    'confidence': row[3],
                    'created_at': row[4],
                    'was_correct': row[5],
                    'resolved_at': row[6]
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting user predictions: {e}")
            return []
    
    def get_user_prediction_stats(self, user_id: str) -> Dict:
        """Get prediction statistics for a user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ? AND season = 27", (user_id,))
            total = cursor.fetchone()[0]
            
            # Correct predictions
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ? AND season = 27 AND was_correct = 1", (user_id,))
            correct = cursor.fetchone()[0]
            
            # Resolved predictions
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ? AND season = 27 AND resolved_at IS NOT NULL", (user_id,))
            resolved = cursor.fetchone()[0]
            
            conn.close()
            
            accuracy = (correct / resolved * 100) if resolved > 0 else 0
            
            return {
                'total_predictions': total,
                'correct_predictions': correct,
                'resolved_predictions': resolved,
                'accuracy_percentage': accuracy
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction stats: {e}")
            return {'total_predictions': 0, 'correct_predictions': 0, 'resolved_predictions': 0, 'accuracy_percentage': 0}

class BBDiscordBot(commands.Bot):
    """Main Discord bot class with 24/7 reliability features"""
    
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
        self.analyzer = BBAnalyzer()
        
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
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
        """Check RSS feed for new updates"""
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
    
    # EVICTION LIKELIHOOD COMMAND
    @commands.command(name='eviction')
    async def eviction_likelihood(self, ctx, hours: int = 72):
        """Show eviction likelihood analysis based on recent updates"""
        try:
            if hours < 1 or hours > 168:
                await ctx.send("Hours must be between 1 and 168 (1 week max)")
                return
            
            # Get recent updates
            updates = self.db.get_recent_updates(hours)
            
            if not updates:
                await ctx.send(f"No updates found in the last {hours} hours to analyze.")
                return
            
            # Analyze eviction likelihood
            likelihood_data = self.analyzer.analyze_eviction_likelihood(updates)
            
            # Filter out houseguests with no mentions and sort by likelihood
            analyzed_houseguests = {k: v for k, v in likelihood_data.items() if v['total_mentions'] > 0}
            
            if not analyzed_houseguests:
                await ctx.send("No houseguests mentioned in recent updates.")
                return
            
            # Sort by likelihood percentage (highest first)
            sorted_houseguests = sorted(analyzed_houseguests.items(), 
                                      key=lambda x: x[1]['likelihood_percentage'], 
                                      reverse=True)
            
            # Create main embed
            embed = discord.Embed(
                title="🎯 Eviction Likelihood Analysis",
                description=f"Based on {len(updates)} updates from the last {hours} hours",
                color=0xe74c3c,
                timestamp=datetime.now()
            )
            
            # Add top 8 houseguests (to fit in embed)
            for i, (houseguest, data) in enumerate(sorted_houseguests[:8]):
                likelihood = data['likelihood_percentage']
                confidence = data['confidence']
                
                # Risk level indicators
                if likelihood >= 70:
                    risk_emoji = "🔴"
                    risk_level = "EXTREME"
                elif likelihood >= 50:
                    risk_emoji = "🟠"
                    risk_level = "HIGH"
                elif likelihood >= 30:
                    risk_emoji = "🟡"
                    risk_level = "MEDIUM"
                elif likelihood >= 10:
                    risk_emoji = "🟢"
                    risk_level = "LOW"
                else:
                    risk_emoji = "⚪"
                    risk_level = "MINIMAL"
                
                # Create progress bar
                bar_length = 10
                filled_length = int(likelihood / 10)
                progress_bar = "█" * filled_length + "░" * (bar_length - filled_length)
                
                field_value = f"{risk_emoji} **{risk_level}** - {likelihood:.1f}%\n"
                field_value += f"{progress_bar}\n"
                field_value += f"**Confidence:** {confidence} ({data['total_mentions']} mentions)"
                
                embed.add_field(
                    name=f"#{i+1} {houseguest}",
                    value=field_value,
                    inline=True
                )
            
            # Add analysis summary
            total_analyzed = len(analyzed_houseguests)
            high_risk_count = sum(1 for _, data in analyzed_houseguests.items() if data['likelihood_percentage'] >= 50)
            
            embed.add_field(
                name="📊 Analysis Summary",
                value=f"• **{total_analyzed}** houseguests analyzed\n• **{high_risk_count}** in high danger\n• **{len(updates)}** updates processed",
                inline=False
            )
            
            embed.set_footer(text="Analysis based on targeting language, campaign activity, and social positioning")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating eviction likelihood: {e}")
            await ctx.send("Error analyzing eviction likelihood. Please try again.")
    
    @commands.command(name='danger')
    async def danger_details(self, ctx, houseguest: str = None):
        """Show detailed danger analysis for a specific houseguest"""
        try:
            if not houseguest:
                await ctx.send("Please specify a houseguest! Example: `!bbdanger Tucker`")
                return
            
            # Find matching houseguest (case insensitive)
            houseguest_match = None
            for hg in self.analyzer.houseguests:
                if hg.lower() == houseguest.lower():
                    houseguest_match = hg
                    break
            
            if not houseguest_match:
                await ctx.send(f"Houseguest '{houseguest}' not found. Available: {', '.join(self.analyzer.houseguests)}")
                return
            
            # Get recent updates and analyze
            updates = self.db.get_recent_updates(72)  # Last 3 days
            likelihood_data = self.analyzer.analyze_eviction_likelihood(updates)
            
            hg_data = likelihood_data.get(houseguest_match)
            if not hg_data or hg_data['total_mentions'] == 0:
                await ctx.send(f"{houseguest_match} hasn't been mentioned in recent updates.")
                return
            
            likelihood = hg_data['likelihood_percentage']
            
            # Determine color based on risk level
            if likelihood >= 70:
                color = 0xe74c3c  # Red
                risk_level = "🔴 EXTREME DANGER"
            elif likelihood >= 50:
                color = 0xf39c12  # Orange
                risk_level = "🟠 HIGH DANGER"
            elif likelihood >= 30:
                color = 0xf1c40f  # Yellow
                risk_level = "🟡 MEDIUM DANGER"
            elif likelihood >= 10:
                color = 0x2ecc71  # Green
                risk_level = "🟢 LOW DANGER"
            else:
                color = 0x95a5a6  # Gray
                risk_level = "⚪ MINIMAL DANGER"
            
            embed = discord.Embed(
                title=f"🎯 Danger Analysis: {houseguest_match}",
                description=f"**Risk Level:** {risk_level}",
                color=color,
                timestamp=datetime.now()
            )
            
            # Likelihood with progress bar
            bar_length = 20
            filled_length = int(likelihood / 5)
            progress_bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            embed.add_field(
                name="Eviction Likelihood",
                value=f"{progress_bar}\n**{likelihood:.1f}%** chance of eviction",
                inline=False
            )
            
            # Detailed scores
            embed.add_field(
                name="📊 Threat Indicators",
                value=f"**Target Score:** {hg_data['target_score']}\n**Safety Score:** {hg_data['safety_score']}\n**Campaign Activity:** {hg_data['campaign_score']}\n**Isolation Level:** {hg_data['isolation_score']}",
                inline=True
            )
            
            embed.add_field(
                name="📈 Analysis Data",
                value=f"**Total Mentions:** {hg_data['total_mentions']}\n**Confidence:** {hg_data['confidence']}\n**Analysis Period:** 72 hours",
                inline=True
            )
            
            # Recent reasons
            if hg_data['reasons']:
                reasons_text = "\n".join([f"• {reason}" for reason in hg_data['reasons']])
                embed.add_field(
                    name="🔍 Recent Activity",
                    value=reasons_text,
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating danger details: {e}")
            await ctx.send("Error analyzing danger details. Please try again.")
    @commands.command(name='predict')
    async def make_prediction(self, ctx, prediction_type: str = None, *, prediction_target: str = None):
        """Make a prediction about Big Brother events"""
        
        if not prediction_type:
            # Show help message
            embed = discord.Embed(
                title="🔮 Big Brother Prediction Game",
                description="Make predictions about Big Brother events and track your accuracy!",
                color=0x9b59b6
            )
            
            embed.add_field(
                name="How to Use",
                value="`!bbpredict [type] [target/details]`",
                inline=False
            )
            
            embed.add_field(
                name="Prediction Types",
                value="• `eviction` - Who will be evicted\n• `hoh` - Who will win HOH\n• `pov` - Who will win Power of Veto\n• `winner` - Who will win Big Brother\n• `jury` - Who will make jury",
                inline=False
            )
            
            embed.add_field(
                name="Examples",
                value="• `!bbpredict eviction Tucker`\n• `!bbpredict hoh Angela week 4`\n• `!bbpredict winner Chelsie`\n• `!bbpredict jury Makensy and Cam`",
                inline=False
            )
            
            embed.set_footer(text="Your predictions are tracked for accuracy scoring!")
            await ctx.send(embed=embed)
            return
        
        if not prediction_target:
            await ctx.send("Please specify what you're predicting! Use `!bbpredict` for examples.")
            return
        
        # Validate prediction type
        valid_types = ['eviction', 'hoh', 'pov', 'winner', 'jury', 'finale']
        if prediction_type.lower() not in valid_types:
            await ctx.send(f"Invalid prediction type. Valid types: {', '.join(valid_types)}")
            return
        
        try:
            # Store the prediction
            user_id = str(ctx.author.id)
            user_name = ctx.author.display_name
            
            self.db.store_prediction(
                user_id=user_id,
                user_name=user_name,
                prediction_type=prediction_type.lower(),
                prediction_target=prediction_target.lower(),
                prediction_value=prediction_target,
                confidence=5
            )
            
            # Create confirmation embed
            embed = discord.Embed(
                title="🔮 Prediction Recorded!",
                description=f"Your prediction has been saved and will be tracked for accuracy.",
                color=0x2ecc71,
                timestamp=datetime.now()
            )
            
            embed.add_field(name="Type", value=prediction_type.title(), inline=True)
            embed.add_field(name="Prediction", value=prediction_target, inline=True)
            embed.add_field(name="Predictor", value=ctx.author.display_name, inline=True)
            
            embed.set_footer(text="Use !bbmypredictions to see all your predictions")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            await ctx.send("Error saving your prediction. Please try again.")
    
    @commands.command(name='mypredictions')
    async def show_my_predictions(self, ctx):
        """Show all predictions made by the user"""
        try:
            user_id = str(ctx.author.id)
            predictions = self.db.get_user_predictions(user_id)
            
            if not predictions:
                embed = discord.Embed(
                    title="🔮 Your Predictions",
                    description="You haven't made any predictions yet! Use `!bbpredict` to get started.",
                    color=0x9b59b6
                )
                await ctx.send(embed=embed)
                return
            
            embed = discord.Embed(
                title="🔮 Your Big Brother Predictions",
                description=f"You've made {len(predictions)} predictions for BB27",
                color=0x9b59b6,
                timestamp=datetime.now()
            )
            
            # Show recent predictions (last 10)
            recent_predictions = predictions[:10]
            
            for prediction in recent_predictions:
                # Format date
                created_date = datetime.fromisoformat(prediction['created_at'])
                date_str = created_date.strftime("%m/%d")
                
                # Status
                if prediction['was_correct'] is None:
                    status = "⏳ Pending"
                elif prediction['was_correct']:
                    status = "✅ Correct"
                else:
                    status = "❌ Wrong"
                
                embed.add_field(
                    name=f"{prediction['type'].title()}: {prediction['value']}",
                    value=f"**Made:** {date_str}\n**Status:** {status}",
                    inline=True
                )
            
            if len(predictions) > 10:
                embed.set_footer(text=f"Showing 10 most recent predictions out of {len(predictions)} total")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error showing predictions: {e}")
            await ctx.send("Error retrieving your predictions. Please try again.")
    
    @commands.command(name='mystats')
    async def show_my_stats(self, ctx):
        """Show prediction accuracy statistics"""
        try:
            user_id = str(ctx.author.id)
            stats = self.db.get_user_prediction_stats(user_id)
            
            if stats['total_predictions'] == 0:
                embed = discord.Embed(
                    title="📊 Your Prediction Stats",
                    description="You haven't made any predictions yet! Use `!bbpredict` to get started.",
                    color=0x9b59b6
                )
                await ctx.send(embed=embed)
                return
            
            # Determine color based on accuracy
            accuracy = stats['accuracy_percentage']
            if accuracy >= 80:
                color = 0x2ecc71  # Green
            elif accuracy >= 60:
                color = 0xf39c12  # Orange
            elif accuracy >= 40:
                color = 0xe74c3c  # Red
            else:
                color = 0x95a5a6  # Gray
            
            embed = discord.Embed(
                title="📊 Your Prediction Stats",
                description=f"Season 27 prediction performance for {ctx.author.display_name}",
                color=color,
                timestamp=datetime.now()
            )
            
            embed.add_field(
                name="Total Predictions",
                value=str(stats['total_predictions']),
                inline=True
            )
            
            embed.add_field(
                name="Resolved",
                value=str(stats['resolved_predictions']),
                inline=True
            )
            
            embed.add_field(
                name="Correct",
                value=str(stats['correct_predictions']),
                inline=True
            )
            
            # Accuracy with visual representation
            embed.add_field(
                name="Accuracy",
                value=f"{accuracy_bar}\n{accuracy:.1f}%",
                inline=False
            )
            
            # Add some encouragement based on performance
            if accuracy >= 80:
                embed.add_field(name="🏆 Status", value="Big Brother Oracle! Amazing accuracy!", inline=False)
            elif accuracy >= 60:
                embed.add_field(name="🎯 Status", value="Strategic Mastermind! Great predictions!", inline=False)
            elif accuracy >= 40:
                embed.add_field(name="📈 Status", value="Getting better! Keep making predictions!", inline=False)
            elif stats['resolved_predictions'] == 0:
                embed.add_field(name="⏳ Status", value="Waiting for predictions to be resolved!", inline=False)
            else:
                embed.add_field(name="🎲 Status", value="Everyone starts somewhere! Keep predicting!", inline=False)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error showing prediction stats: {e}")
            await ctx.send("Error retrieving your stats. Please try again.")
    
    # ORIGINAL COMMANDS
    @commands.command(name='summary')
    async def daily_summary(self, ctx, hours: int = 24):
        """Generate a summary of updates"""
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
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            await ctx.send("Error generating summary. Please try again.")
    
    @commands.command(name='setchannel')
    @commands.has_permissions(administrator=True)
    async def set_update_channel(self, ctx, channel_id: int):
        """Set the channel for RSS updates"""
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
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating status: {e}")
            await ctx.send("Error generating status.")

def main():
    """Main function to run the bot"""
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
            """Show available commands"""
            embed = discord.Embed(
                title="Big Brother Bot Commands",
                description="Monitor Jokers Updates RSS feed with intelligent analysis",
                color=0x3498db
            )
            
            embed.add_field(name="📊 **Main Commands**", value="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", inline=False)
            embed.add_field(name="!bbsummary [hours]", value="Generate summary of updates", inline=False)
            embed.add_field(name="!bbstatus", value="Show bot status", inline=False)
            embed.add_field(name="!bbsetchannel [ID]", value="Set update channel (Admin only)", inline=False)
            
            embed.add_field(name="🎯 **Eviction Analysis**", value="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", inline=False)
            embed.add_field(name="!bbeviction [hours]", value="Show eviction likelihood rankings", inline=False)
            embed.add_field(name="!bbdanger [houseguest]", value="Detailed danger analysis for specific houseguest", inline=False)
            
            embed.add_field(name="🔮 **Prediction Game**", value="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", inline=False)
            embed.add_field(name="!bbpredict [type] [target]", value="Make predictions about BB events", inline=False)
            embed.add_field(name="!bbmypredictions", value="View your prediction history", inline=False)
            embed.add_field(name="!bbmystats", value="View your prediction accuracy", inline=False)
            
            embed.set_footer(text="Start making predictions now - they'll be tracked for accuracy!")
            
            await ctx.send(embed=embed)
        
        bot_token = bot.config.get('bot_token')
        if not bot_token:
            logger.error("No bot token found!")
            return
        
        logger.info("Starting Big Brother Discord Bot with Prediction Game...")
        bot.run(bot_token, reconnect=True)
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    main()
