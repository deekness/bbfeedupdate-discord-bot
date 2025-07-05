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
from collections import defaultdict, Counter

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
        houseguest_pattern = r'\b[A-Z][a-z]+\b'
        potential_names = re.findall(houseguest_pattern, text)
        
        exclude_words = {'The', 'This', 'That', 'They', 'Some', 'Many', 'Other', 'First', 'Last',
                        'Big', 'Brother', 'Julie', 'Host', 'Diary', 'Room', 'Have', 'Not'}
        return [name for name in potential_names if name not in exclude_words]
    
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

class UpdateBatcher:
    """Groups and analyzes updates like a BB superfan"""
    
    def __init__(self, analyzer: BBAnalyzer):
        self.analyzer = analyzer
        self.update_queue = []
        self.last_batch_time = datetime.now()
        
        # Superfan knowledge patterns
        self.vote_patterns = re.compile(r'(votes?|voted|voting)\s+for\s+(\w+)', re.IGNORECASE)
        self.comp_win_patterns = re.compile(r'(\w+)\s+(wins?|won)\s+(HOH|Head of Household|POV|Power of Veto|Veto)', re.IGNORECASE)
        self.nomination_patterns = re.compile(r'(nominates?|nominated|noms?)\s+(\w+)(?:\s+and\s+(\w+))?', re.IGNORECASE)
        self.alliance_patterns = re.compile(r'(alliance|deal|final \d|working with)', re.IGNORECASE)
        
        # Game-defining moments that need immediate attention
        self.urgent_keywords = [
            'evicted', 'eliminated', 'wins hoh', 'wins pov', 'backdoor', 
            'self-evict', 'expelled', 'quit', 'medical', 'pandora', 'coup',
            'diamond veto', 'secret power', 'battle back', 'return'
        ]
    
    def should_send_batch(self) -> bool:
        """Determine if we should send a batch now"""
        if not self.update_queue:
            return False
            
        time_elapsed = (datetime.now() - self.last_batch_time).total_seconds() / 60
        
        # Check for urgent updates
        has_urgent = any(self._is_urgent(update) for update in self.update_queue)
        if has_urgent and len(self.update_queue) >= 3:
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
        """Add update to queue"""
        self.update_queue.append(update)
    
    def create_batch_summary(self) -> List[discord.Embed]:
        """Create intelligent summary embeds like a superfan would"""
        if not self.update_queue:
            return []
        
        # Group updates by type
        grouped = self._group_updates()
        embeds = []
        
        # Create main summary embed
        main_embed = self._create_main_embed(grouped)
        embeds.append(main_embed)
        
        # Create detailed embeds for important groups
        for group_type, updates in grouped.items():
            if group_type in ['votes', 'competition_wins', 'nominations'] and updates:
                detail_embed = self._create_detail_embed(group_type, updates)
                if detail_embed:
                    embeds.append(detail_embed)
        
        # Clear queue after processing
        self.update_queue.clear()
        self.last_batch_time = datetime.now()
        
        return embeds
    
    def _group_updates(self) -> Dict[str, List[BBUpdate]]:
        """Group updates by game event type"""
        groups = defaultdict(list)
        
        for update in self.update_queue:
            content = f"{update.title} {update.description}"
            
            # Check for vote patterns
            vote_match = self.vote_patterns.search(content)
            if vote_match:
                groups['votes'].append(update)
                continue
            
            # Check for competition wins
            comp_match = self.comp_win_patterns.search(content)
            if comp_match:
                groups['competition_wins'].append(update)
                continue
            
            # Check for nominations
            nom_match = self.nomination_patterns.search(content)
            if nom_match:
                groups['nominations'].append(update)
                continue
            
            # Check for alliance/strategy
            if self.alliance_patterns.search(content):
                groups['strategy'].append(update)
                continue
            
            # Categorize by analyzer categories
            categories = self.analyzer.categorize_update(update)
            if "💥 Drama" in categories:
                groups['drama'].append(update)
            elif "💕 Romance" in categories:
                groups['showmance'].append(update)
            else:
                groups['general'].append(update)
        
        return groups
    
    def _create_main_embed(self, grouped: Dict[str, List[BBUpdate]]) -> discord.Embed:
        """Create the main summary embed with superfan analysis"""
        total_updates = sum(len(updates) for updates in grouped.values())
        
        # Determine the headline based on most important events
        headline = self._get_headline(grouped)
        
        embed = discord.Embed(
            title=f"🎭 {headline}",
            description=f"**{total_updates} updates** from the last batch period",
            color=self._get_batch_color(grouped),
            timestamp=datetime.now()
        )
        
        # Add quick summary for each category
        if grouped['competition_wins']:
            winners = self._extract_comp_winners(grouped['competition_wins'])
            embed.add_field(
                name="🏆 Competition Results",
                value=winners,
                inline=False
            )
        
        if grouped['votes']:
            vote_summary = self._summarize_votes(grouped['votes'])
            embed.add_field(
                name="🗳️ Voting Update",
                value=vote_summary,
                inline=False
            )
        
        if grouped['nominations']:
            nom_summary = self._summarize_nominations(grouped['nominations'])
            embed.add_field(
                name="🎯 Nomination Ceremony",
                value=nom_summary,
                inline=False
            )
        
        if grouped['drama']:
            embed.add_field(
                name="💥 House Drama",
                value=f"{len(grouped['drama'])} incidents - check details below",
                inline=True
            )
        
        if grouped['strategy']:
            embed.add_field(
                name="🧠 Strategic Moves",
                value=f"{len(grouped['strategy'])} game moves detected",
                inline=True
            )
        
        # Add superfan analysis
        analysis = self._get_superfan_analysis(grouped)
        if analysis:
            embed.add_field(
                name="🎯 Superfan Analysis",
                value=analysis,
                inline=False
            )
        
        return embed
    
    def _get_headline(self, grouped: Dict[str, List[BBUpdate]]) -> str:
        """Generate a smart headline based on the most important events"""
        if grouped['competition_wins']:
            return "New Competition Winner Changes the Game!"
        elif grouped['votes']:
            return "Jury Votes Revealed - Winner Emerging!"
        elif grouped['nominations']:
            return "Nomination Ceremony Shakes Up the House!"
        elif len(grouped['drama']) >= 3:
            return "House Explodes in Drama!"
        elif grouped['strategy']:
            return "Strategic Gameplay in Motion"
        else:
            return "Big Brother House Update"
    
    def _get_batch_color(self, grouped: Dict[str, List[BBUpdate]]) -> int:
        """Color code based on update importance"""
        if grouped['competition_wins'] or grouped['votes']:
            return 0xe74c3c  # Red for game-critical
        elif grouped['nominations'] or len(grouped['strategy']) >= 2:
            return 0xf39c12  # Orange for important
        elif grouped['drama']:
            return 0x9b59b6  # Purple for drama
        else:
            return 0x3498db  # Blue for standard
    
    def _extract_comp_winners(self, updates: List[BBUpdate]) -> str:
        """Extract and format competition winners"""
        winners = []
        for update in updates:
            content = f"{update.title} {update.description}"
            match = self.comp_win_patterns.search(content)
            if match:
                winner = match.group(1)
                comp = match.group(3)
                winners.append(f"**{winner}** won {comp}")
        
        return "\n".join(winners) if winners else "Competition results detected"
    
    def _summarize_votes(self, updates: List[BBUpdate]) -> str:
        """Summarize voting patterns like a superfan would notice"""
        vote_counts = defaultdict(list)
        
        for update in updates:
            content = f"{update.title} {update.description}"
            match = self.vote_patterns.search(content)
            if match:
                votee = match.group(2)
                # Extract voter name from the update
                voter_match = re.search(r'(\w+)\s+votes?', content, re.IGNORECASE)
                if voter_match:
                    voter = voter_match.group(1)
                    vote_counts[votee].append(voter)
        
        # Format the summary
        summary_parts = []
        for votee, voters in vote_counts.items():
            if len(voters) > 1:
                summary_parts.append(f"**{votee}** received votes from: {', '.join(voters)} ({len(voters)} votes)")
            else:
                summary_parts.append(f"**{votee}** received a vote from {voters[0]}")
        
        return "\n".join(summary_parts) if summary_parts else "Votes were cast"
    
    def _summarize_nominations(self, updates: List[BBUpdate]) -> str:
        """Summarize nomination ceremony results"""
        all_nominees = set()
        nominators = []
        
        for update in updates:
            content = f"{update.title} {update.description}"
            match = self.nomination_patterns.search(content)
            if match:
                nominator_match = re.search(r'(\w+)\s+nominates?', content, re.IGNORECASE)
                if nominator_match:
                    nominators.append(nominator_match.group(1))
                
                nominee1 = match.group(2)
                if nominee1:
                    all_nominees.add(nominee1)
                nominee2 = match.group(3)
                if nominee2:
                    all_nominees.add(nominee2)
        
        if all_nominees:
            return f"**Nominees**: {', '.join(all_nominees)}\nNominated by: {', '.join(set(nominators))}"
        return "Nomination ceremony completed"
    
    def _get_superfan_analysis(self, grouped: Dict[str, List[BBUpdate]]) -> str:
        """Provide strategic analysis like Taran would"""
        analysis_parts = []
        
        # Analyze competition wins impact
        if grouped['competition_wins']:
            analysis_parts.append("💡 **Comp Win Impact**: This changes the entire week's trajectory. Watch for deals being made in the next few hours.")
        
        # Analyze voting patterns
        if grouped['votes'] and len(grouped['votes']) >= 3:
            vote_count = len(grouped['votes'])
            if vote_count >= 5:
                analysis_parts.append("💡 **Jury Analysis**: This looks like a steamroll victory. The jury is clearly unified.")
            else:
                analysis_parts.append("💡 **Jury Analysis**: Split votes detected. We might see a closer finale than expected.")
        
        # Analyze strategic patterns
        if grouped['strategy'] and grouped['drama']:
            analysis_parts.append("💡 **House Dynamics**: Strategy mixed with drama usually means someone's game is blown up. Expect shifting alliances.")
        
        return "\n".join(analysis_parts) if analysis_parts else None
    
    def _create_detail_embed(self, group_type: str, updates: List[BBUpdate]) -> discord.Embed:
        """Create detailed embed for specific update groups"""
        titles = {
            'votes': "🗳️ Detailed Voting Breakdown",
            'competition_wins': "🏆 Competition Results Details",
            'nominations': "🎯 Nomination Ceremony Details",
            'drama': "💥 Drama Breakdown",
            'strategy': "🧠 Strategic Moves Breakdown"
        }
        
        colors = {
            'votes': 0xe74c3c,
            'competition_wins': 0xf1c40f,
            'nominations': 0xe67e22,
            'drama': 0x9b59b6,
            'strategy': 0x3498db
        }
        
        embed = discord.Embed(
            title=titles.get(group_type, "Update Details"),
            color=colors.get(group_type, 0x95a5a6),
            timestamp=datetime.now()
        )
        
        # Add individual updates with timestamps
        for i, update in enumerate(updates[:10]):  # Limit to 10 to avoid embed limits
            time_str = update.pub_date.strftime("%I:%M %p")
            
            # Extract key info instead of showing full title/description
            key_info = self._extract_key_info(update, group_type)
            
            embed.add_field(
                name=f"{time_str}",
                value=key_info,
                inline=False
            )
        
        if len(updates) > 10:
            embed.set_footer(text=f"Showing 10 of {len(updates)} updates")
        
        return embed
    
    def _extract_key_info(self, update: BBUpdate, group_type: str) -> str:
        """Extract the most important info from an update"""
        content = f"{update.title} {update.description}"
        
        # Remove duplicate title/description
        if update.title == update.description:
            content = update.title
        
        # Extract key information based on type
        if group_type == 'votes':
            match = self.vote_patterns.search(content)
            if match:
                voter_match = re.search(r'(\w+)\s+votes?', content, re.IGNORECASE)
                if voter_match:
                    return f"{voter_match.group(1)} → {match.group(2)}"
        
        # For other types, return a shortened version
        if len(content) > 100:
            return content[:97] + "..."
        return content

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
        self.update_batcher = UpdateBatcher(self.analyzer)
        
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Remove all slash commands and add them properly
        self.remove_command('help')
        
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

# Define slash commands using the bot.tree decorator
@bot.tree.command(name="status", description="Show bot status and statistics")
async def status_slash(interaction: discord.Interaction):
    """Show bot status"""
    try:
        await interaction.response.defer(ephemeral=True)
        
        embed = discord.Embed(
            title="Big Brother Bot Status",
            color=0x2ecc71 if bot.consecutive_errors == 0 else 0xe74c3c,
            timestamp=datetime.now()
        )
        
        embed.add_field(name="RSS Feed", value=bot.rss_url, inline=False)
        embed.add_field(name="Update Channel", 
                       value=f"<#{bot.config.get('update_channel_id')}>" if bot.config.get('update_channel_id') else "Not set", 
                       inline=True)
        embed.add_field(name="Updates Processed", value=str(bot.total_updates_processed), inline=True)
        embed.add_field(name="Consecutive Errors", value=str(bot.consecutive_errors), inline=True)
        
        time_since_check = datetime.now() - bot.last_successful_check
        embed.add_field(name="Last RSS Check", value=f"{time_since_check.total_seconds():.0f} seconds ago", inline=True)
        
        # Add batch queue status
        queue_size = len(bot.update_batcher.update_queue)
        embed.add_field(name="Updates in Queue", value=str(queue_size), inline=True)
        
        await interaction.followup.send(embed=embed, ephemeral=True)
        
    except Exception as e:
        logger.error(f"Error generating status: {e}")
        await interaction.followup.send("Error generating status.", ephemeral=True)

@bot.tree.command(name="summary", description="Get a summary of recent Big Brother updates")
async def summary_slash(interaction: discord.Interaction, hours: int = 24):
    """Generate a summary of updates"""
    try:
        if hours < 1 or hours > 168:
            await interaction.response.send_message("Hours must be between 1 and 168", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=True)
        
        updates = bot.db.get_recent_updates(hours)
        
        if not updates:
            await interaction.followup.send(f"No updates found in the last {hours} hours.", ephemeral=True)
            return
        
        categories = {}
        for update in updates:
            update_categories = bot.analyzer.categorize_update(update)
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
                               key=lambda x: bot.analyzer.analyze_strategic_importance(x), 
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

@bot.tree.command(name="setchannel", description="Set the channel for Big Brother updates")
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
        
        bot.config.set('update_channel_id', channel.id)
        
        await interaction.response.send_message(
            f"Update channel set to {channel.mention}", 
            ephemeral=True
        )
        logger.info(f"Update channel set to {channel.id}")
        
    except Exception as e:
        logger.error(f"Error setting channel: {e}")
        await interaction.response.send_message("Error setting channel. Please try again.", ephemeral=True)

@bot.tree.command(name="commands", description="Show all available commands")
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
        ("/forcebatch", "Force send any queued updates (Admin only)")
    ]
    
    for name, description in commands_list:
        embed.add_field(name=name, value=description, inline=False)
    
    embed.set_footer(text="All commands are ephemeral (only you can see the response)")
    
    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="forcebatch", description="Force send any queued updates")
async def forcebatch_slash(interaction: discord.Interaction):
    """Force send batch update"""
    try:
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=True)
        
        queue_size = len(bot.update_batcher.update_queue)
        if queue_size == 0:
            await interaction.followup.send("No updates in queue to send.", ephemeral=True)
            return
        
        # Force send the batch
        await bot.send_batch_update()
        
        await interaction.followup.send(f"Force sent batch of {queue_size} updates!", ephemeral=True)
        
    except Exception as e:
        logger.error(f"Error forcing batch: {e}")
        await interaction.followup.send("Error sending batch.", ephemeral=True)

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
