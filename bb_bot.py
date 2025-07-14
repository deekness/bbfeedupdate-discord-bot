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
import pytz
from typing import List, Dict, Set, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import time
import traceback
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict, Counter, deque, OrderedDict
import anthropic
from enum import Enum
import psycopg2
import psycopg2.extras
from urllib.parse import urlparse

# Configure SQLite to handle datetime properly
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter("timestamp", lambda b: datetime.fromisoformat(b.decode()))
sqlite3.register_converter("TIMESTAMP", lambda b: datetime.fromisoformat(b.decode()))

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

# Zing constants
PG_ZINGS = [
    # 60 PG zings (30%)
    "{target}, you're so forgettable, production had to label your microphone twice!",
    "{target}, your diary room sessions are so boring, even the cameras fall asleep!",
    "{target}, you float so much, you should come with a pool noodle!",
    "{target}, your game moves are like a sloth doing yoga - slow and pointless!",
    "{target}, you're playing Big Brother like it's Big Sleeper!",
    "{target}, your strategy is so invisible, it should come with a magnifying glass!",
    "{target}, you're about as threatening as a rubber duck in the HOH competition!",
    "{target}, your alliance building skills are like a house of cards in a tornado!",
    "{target}, you blend into the background so well, you should be wallpaper!",
    "{target}, your competition record is like a phone with no signal - zero bars!",
    "{target}, you're playing chess while everyone else is playing... well, chess. You're just bad at it!",
    "{target}, your social game is like Wi-Fi in the basement - barely connected!",
    "{target}, you're so out of the loop, you think POV stands for 'Point of Vanilla'!",
    "{target}, your jury management is like a GPS with no satellites - completely lost!",
    "{target}, you're coasting so hard, you should be wearing roller skates!",
    "{target}, your HOH reign was shorter than a commercial break!",
    "{target}, you're playing Big Brother like it's Little Sister!",
    "{target}, your competition wins are rarer than a unicorn sighting!",
    "{target}, you're so predictable, we could replace you with a magic 8-ball!",
    "{target}, your strategy talks are like bedtime stories - they put everyone to sleep!",
    "{target}, you're about as intimidating as a kitten in mittens!",
    "{target}, your game resume is shorter than a haiku!",
    "{target}, you're playing hide and seek while everyone else plays Big Brother!",
    "{target}, your big moves are smaller than a Tic Tac!",
    "{target}, you're so far behind in the game, you're still in sequester!",
    "{target}, your diary room confessions are like reading a blank book!",
    "{target}, you're floating so much, NASA wants to study you!",
    "{target}, your competition performance is like dial-up internet - outdated and slow!",
    "{target}, you're playing Big Brother like it's a library - too quiet!",
    "{target}, your strategic mind is like a maze with no exit!",
    "{target}, you're so passive, you make a sloth look hyperactive!",
    "{target}, your game awareness is like a bat without sonar!",
    "{target}, you're riding coattails so hard, you should be a fashion designer!",
    "{target}, your threat level is lower than the basement!",
    "{target}, you're playing 4D chess... badly... in a checkers game!",
    "{target}, your social connections are like a phone on airplane mode!",
    "{target}, you're so forgettable, your own alliance forgets you exist!",
    "{target}, your competition skills need more training than a newborn giraffe!",
    "{target}, you're playing Big Brother like it's Big Naptime!",
    "{target}, your strategic planning is like a GPS going in circles!",
    "{target}, you're so under the radar, submarines are jealous!",
    "{target}, your game moves are like a snail racing a cheetah!",
    "{target}, you're playing so safe, you should wear a helmet in the diary room!",
    "{target}, your big brother game is more like little cousin energy!",
    "{target}, you're coasting harder than a bike going downhill!",
    "{target}, your competition record is like a broken scoreboard - all zeros!",
    "{target}, you're so out of touch, you think the veto meeting is a coffee date!",
    "{target}, your strategy is like a book with all blank pages!",
    "{target}, you're playing Big Brother like it's Big Background Character!",
    "{target}, your game influence is like a whisper in a hurricane!",
    "{target}, you're so forgettable, the memory wall forgot to light up your picture!",
    "{target}, your HOH letters are probably just participation certificates!",
    "{target}, you're playing so quietly, closed captions can't even pick you up!",
    "{target}, your competition training must have been watching paint dry!",
    "{target}, you're floating so professionally, you should charge admission!",
    "{target}, your strategic conversations are like silent movies without subtitles!",
    "{target}, you're so far from winning, you need a telescope to see the prize!",
    "{target}, your game awareness is like a security camera that's unplugged!",
    "{target}, you're playing Big Brother like it's Big Spectator!",
    "{target}, your influence in the house is like a fan with no blades!"
]

PG13_ZINGS = [
    # 140 PG-13 zings (70%)
    "{target}, you're so boring in bed, your showmance partner counts sheep... while awake!",
    "{target}, your kissing technique is like CPR - technically correct but nobody's enjoying it!",
    "{target}, you're so thirsty, the Have-Not cold showers are jealous!",
    "{target}, your flirting is so bad, even the cameras switch to fish!",
    "{target}, you shower so rarely, Febreze is considering you for a sponsorship!",
    "{target}, your game is like your love life - all talk, no action!",
    "{target}, you're so desperate for attention, you'd showmance with a mannequin!",
    "{target}, your HOH room action is like a nature documentary - rare and disappointing!",
    "{target}, you're trying so hard to be America's Favorite, you'd kiss a cactus for votes!",
    "{target}, your strategy is like your hygiene - questionable at best!",
    "{target}, you're so clingy, your showmance needs a restraining order!",
    "{target}, your diary room sessions are like your dating profile - full of lies!",
    "{target}, you're so bad at comps, you'd lose a staring contest to a blind person!",
    "{target}, your social game is like your personal hygiene - it stinks!",
    "{target}, you're showmancing so hard, production needs a cold shower!",
    "{target}, your game is weaker than your pull-out game... from competitions!",
    "{target}, you're so fake, your tears come with a warranty!",
    "{target}, your loyalty flip-flops more than your shower shoes!",
    "{target}, you're riding [name] harder than a mechanical bull at a bar!",
    "{target}, your competition performance is like your dating history - brief and embarrassing!",
    "{target}, you're so delusional, you think your showmance actually likes you!",
    "{target}, your strategic mind is like your love life - completely fictional!",
    "{target}, you're trying to play puppet master but you can't even master basic hygiene!",
    "{target}, your game is so messy, it needs a hazmat team!",
    "{target}, you're so desperate for camera time, you'd streak through the backyard!",
    "{target}, your HOH reign was shorter than your last relationship!",
    "{target}, you kiss so much ass, you should carry mouthwash!",
    "{target}, your game is like a bad hookup - regrettable and forgettable!",
    "{target}, you're so paranoid, you check under the bed for hidden vetoes!",
    "{target}, your showmance strategy is like your personality - shallow and transparent!",
    "{target}, you're playing everyone like a fiddle... a broken fiddle... that nobody wants to hear!",
    "{target}, your DR sessions are faker than your showmance feelings!",
    "{target}, you're so obsessed with camera time, you'd make out with your own reflection!",
    "{target}, your competition beast mode is more like competition deceased mode!",
    "{target}, you're riding coattails so hard, you should be charged rent!",
    "{target}, your flirting technique makes everyone want to self-evict!",
    "{target}, you're so two-faced, you need two microphones!",
    "{target}, your game is like your shower schedule - inconsistent and concerning!",
    "{target}, you're playing so dirty, the backyard needs to be power-washed!",
    "{target}, your strategic talks are like foreplay - awkward and nobody finishes satisfied!",
    "{target}, you're so thirsty for attention, the pool is jealous!",
    "{target}, your social game is like a bad date - everyone's looking for an exit!",
    "{target}, you've been in more beds than a hotel maid!",
    "{target}, your competition record is like your dating record - a lot of participation, no victories!",
    "{target}, you're so desperate for allies, you'd align with the ants!",
    "{target}, your game moves are like your dance moves - embarrassing and ineffective!",
    "{target}, you're playing so sloppy, you need a bib!",
    "{target}, your showmance is so forced, it needs a safe word!",
    "{target}, you're backstabbing so much, you should be a chiropractor!",
    "{target}, your HOH room saw less action than the storage room!",
    "{target}, you're so fake, your diary room needs a fact-checker!",
    "{target}, your strategy is like your skincare routine - non-existent!",
    "{target}, you're trying to be a mastermind but you can't even master basic math!",
    "{target}, your game is so weak, it needs Viagra!",
    "{target}, you're so desperate for screen time, you'd shower with the door open!",
    "{target}, your loyalty changes faster than your underwear... which isn't saying much!",
    "{target}, you're playing Big Brother like it's The Bachelor - wrong show!",
    "{target}, your competition performance is like your pickup lines - a total flop!",
    "{target}, you're so clingy, your alliance needs therapy!",
    "{target}, your game is dirtier than the kitchen after taco night!",
    "{target}, you're trying to be a villain but you're more like a Disney Channel antagonist!",
    "{target}, your strategic mind is like your dating standards - set way too low!",
    "{target}, you're so thirsty, you make the Have-Nots look hydrated!",
    "{target}, your DR sessions are more scripted than a soap opera!",
    "{target}, you're playing everyone but the only thing you're playing is yourself!",
    "{target}, your game is like a bad Tinder date - all swipe, no substance!",
    "{target}, you're so desperate for attention, you'd cuddle with the cameras!",
    "{target}, your competition skills are like your flirting skills - non-existent!",
    "{target}, you're riding [name] so hard, you should pay for gas!",
    "{target}, your game is messier than the bathroom after slop week!",
    "{target}, you're so fake, your emotions come with a director's cut!",
    "{target}, your strategic planning is like your shower planning - it doesn't happen!",
    "{target}, you're playing so many sides, you're basically a geometry lesson!",
    "{target}, your showmance is so awkward, it makes the feeds switch to fish!",
    "{target}, you're so bad at comps, you'd lose strip poker fully clothed!",
    "{target}, your social game is like a bad rash - irritating and spreading!",
    "{target}, you're trying to be memorable but you're as forgettable as last night's slop!",
    "{target}, your game is like your hygiene habits - everyone's talking about how bad it is!",
    "{target}, you're so desperate for a showmance, you'd date a Have-Not restriction!",
    "{target}, your competition record is like your love life - a series of unfortunate events!",
    "{target}, you're playing puppet master but your strings are more tangled than earbuds!",
    "{target}, your strategy sessions are like bad foreplay - confusing and unsatisfying!",
    "{target}, you're so thirsty, you're dehydrating the other houseguests!",
    "{target}, your game moves are like your bowel movements - irregular and concerning!",
    "{target}, you're trying to be Brad Pitt but you're more like arm pit!",
    "{target}, your alliance loyalty is like your deodorant - it doesn't last long!",
    "{target}, you're so fake, your tears need a stunt double!",
    "{target}, your HOH reign was like bad sex - over before anyone noticed it started!",
    "{target}, you're playing so many people, you need a spreadsheet!",
    "{target}, your social game is like your shower game - avoided by everyone!",
    "{target}, you're so desperate for votes, you'd promise your firstborn!",
    "{target}, your competition performance is like your personality - a total letdown!",
    "{target}, you're clinging to [name] like a bad STD!",
    "{target}, your game is dirtier than the hot tub after a showmance session!",
    "{target}, you're so delusional, you think production likes you!",
    "{target}, your strategic mind is like your sex appeal - imaginary!",
    "{target}, you're backstabbing so much, you need a spinal surgeon!",
    "{target}, your DR sessions are faker than your orgasms... I mean, emotions!",
    "{target}, you're so bad at this game, you make first boots look like masterminds!",
    "{target}, your showmance chemistry is like oil and water - it doesn't mix!",
    "{target}, you're playing so sloppy, you need a mop and bucket!",
    "{target}, your game is like a bad hookup - everyone regrets it happened!",
    "{target}, you're so thirsty for attention, you'd streak through the diary room!",
    "{target}, your competition skills are like your kissing skills - all teeth!",
    "{target}, you're riding coattails harder than a rodeo champion!",
    "{target}, your strategy is like your underwear - it needs to be changed!",
    "{target}, you're so fake, your showmance thinks you're CGI!",
    "{target}, your game moves are like your sexual moves - predictable and disappointing!",
    "{target}, you're playing everyone but yourself - and badly!",
    "{target}, your social game is like body odor - offensive and hard to ignore!",
    "{target}, you're so desperate for allies, you'd align with production!",
    "{target}, your HOH room saw less action than a nun's bedroom!",
    "{target}, you're trying to be a mastermind but you can't even mind your own business!",
    "{target}, your game is weaker than your bladder during a lockdown!",
    "{target}, you're so clingy, your showmance filed for emotional support!",
    "{target}, your competition record is like your hygiene record - needs improvement!",
    "{target}, you're playing dirty but not in the fun way!",
    "{target}, your strategic planning is like your family planning - accidental at best!",
    "{target}, you're so thirsty, the pool lost water!",
    "{target}, your game is messier than the sheets after your HOH reign!",
    "{target}, you're so fake, your diary room needs subtitles for the truth!",
    "{target}, your loyalty flip-flops more than a fish out of water!",
    "{target}, you're riding [name] like it's your job - and you're working overtime!",
    "{target}, your social game is like a communicable disease - everyone's trying to avoid it!",
    "{target}, you're so bad at comps, you'd lose a sleeping contest to an insomniac!",
    "{target}, your game moves are like your romantic moves - desperate and cringeworthy!",
    "{target}, you're playing so many sides, you're basically a rubik's cube!",
    "{target}, your showmance is so forced, it needs a lubricant!",
    "{target}, you're so desperate for screen time, you'd do diary rooms naked!",
    "{target}, your strategy is like your sex life - all in your head!",
    "{target}, you're backstabbing so much, you should open a sushi restaurant!",
    "{target}, your game is dirtier than the thoughts you have in the HOH shower!",
    "{target}, you're so delusional, you think your edit is accurate!",
    "{target}, your competition performance is like erectile dysfunction - a failure to rise to the occasion!",
    "{target}, you're clinging to power like you cling to your showmance - desperately!",
    "{target}, your DR sessions have more fiction than the Bible!",
    "{target}, you're so thirsty, you're sponsored by Gatorade!",
    "{target}, your game is like a wet dream - seemed good at the time but embarrassing in daylight!",
    "{target}, you're playing everyone like a cheap kazoo - badly and annoying everyone!"
]

# Combine all zings
ALL_ZINGS = PG_ZINGS + PG13_ZINGS

# Game-specific constants
# Add this with your other constants (around line 50-100)
BB27_HOUSEGUESTS = [
    "Adrian", "Amy", "Ashley", "Ava", "Jimmy", "Katherine", "Keanu", 
    "Kelley", "Lauren", "Mickey", "Morgan", "Rachel", "Rylie", "Vince", "Will", "Zach", "Zae"
    # Replace these with your actual 20 houseguests
]

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
    'Big', 'Brother', 'Julie', 'Host', 'Diary', 'Room', 'Have', 'Not',
    'America', 'Favorite', 'Winner', 'Finale', 'Vote', 'Jury', 'Eviction',
    'Competition', 'Ceremony', 'Veto', 'Power', 'Head', 'Household'
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

# Alliance tracking enums
class AllianceStatus(Enum):
    ACTIVE = "active"
    BROKEN = "broken"
    SUSPECTED = "suspected"
    DISSOLVED = "dissolved"

class AllianceEventType(Enum):
    FORMED = "formed"
    EXPANDED = "expanded"
    BETRAYAL = "betrayal"
    DISSOLVED = "dissolved"
    SUSPECTED = "suspected"

# Prediction system enums
class PredictionType(Enum):
    SEASON_WINNER = "season_winner"
    FIRST_BOOT = "first_boot" 
    WEEKLY_HOH = "weekly_hoh"
    WEEKLY_VETO = "weekly_veto"
    WEEKLY_EVICTION = "weekly_eviction"

class PredictionStatus(Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"

@dataclass
class Prediction:
    prediction_id: int
    title: str
    description: str
    prediction_type: PredictionType
    options: List[str]
    created_by: int
    created_at: datetime
    closes_at: datetime
    status: PredictionStatus
    correct_option: Optional[str] = None
    week_number: Optional[int] = None

@dataclass
class UserPrediction:
    user_id: int
    prediction_id: int
    option: str
    created_at: datetime

class LRUCache:
    """Thread-safe LRU Cache implementation for tracking processed hashes"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self._lock = asyncio.Lock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    async def contains(self, key: str) -> bool:
        """Check if key exists in cache (async for thread safety)"""
        async with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats['hits'] += 1
                return True
            self.stats['misses'] += 1
            return False
    
    async def add(self, key: str, value: Optional[float] = None) -> None:
        """Add key to cache with optional timestamp value"""
        async with self._lock:
            if key in self.cache:
                # Update existing and move to end
                self.cache.move_to_end(key)
                self.cache[key] = value or time.time()
            else:
                # Add new entry
                self.cache[key] = value or time.time()
                
                # Check capacity and evict if necessary
                if len(self.cache) > self.capacity:
                    evicted_key, _ = self.cache.popitem(last=False)  # Remove oldest
                    self.stats['evictions'] += 1
                    logger.debug(f"Evicted hash from cache: {evicted_key[:8]}...")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'hit_rate': f"{hit_rate:.1f}%"
        }
    
    async def clear(self) -> None:
        """Clear the cache"""
        async with self._lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    async def get_oldest_timestamp(self) -> Optional[float]:
        """Get timestamp of oldest entry"""
        async with self._lock:
            if self.cache:
                first_key = next(iter(self.cache))
                return self.cache[first_key]
            return None

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
        """Set configuration value and save to file"""
        self.config[key] = value
        self.save_to_file()  # Add this line
        
    def save_to_file(self):
        """Save current config to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved to file")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

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

class HistoricalContextTracker:
    """Tracks historical context for Big Brother events"""
    
    # Event type constants
    EVENT_TYPES = {
        'HOH_WIN': 'hoh_win',
        'VETO_WIN': 'veto_win', 
        'NOMINATION': 'nomination',
        'EVICTION': 'eviction',
        'ALLIANCE_FORM': 'alliance_form',
        'ALLIANCE_BREAK': 'alliance_break',
        'SHOWMANCE_START': 'showmance_start',
        'SHOWMANCE_END': 'showmance_end',
        'FIGHT': 'fight',
        'BETRAYAL': 'betrayal'
    }
    
    # Competition detection patterns
    COMPETITION_PATTERNS = [
        (r'(\w+)\s+wins?\s+(?:the\s+)?hoh', 'HOH_WIN'),
        (r'(\w+)\s+wins?\s+(?:the\s+)?(?:power\s+of\s+)?veto', 'VETO_WIN'),
        (r'(\w+)\s+wins?\s+(?:the\s+)?pov', 'VETO_WIN'),
        (r'(\w+)\s+(?:is\s+)?nominated?', 'NOMINATION'),
        (r'(\w+)\s+(?:gets?\s+)?evicted', 'EVICTION'),
        (r'(\w+)\s+(?:was\s+)?eliminated', 'EVICTION')
    ]
    
    # Social event patterns  
    SOCIAL_PATTERNS = [
        (r'(\w+)\s+and\s+(\w+)\s+(?:kiss|kissed|make\s+out)', 'SHOWMANCE_START'),
        (r'(\w+)\s+and\s+(\w+)\s+(?:fight|argue|confrontation)', 'FIGHT'),
        (r'(\w+)\s+betrays?\s+(\w+)', 'BETRAYAL'),
        (r'(\w+)\s+(?:throws?\s+)?(\w+)\s+under\s+the\s+bus', 'BETRAYAL')
    ]
    
    def __init__(self, database_url: str = None, use_postgresql: bool = True):
        self.database_url = database_url
        self.use_postgresql = use_postgresql
        
    def get_connection(self):
        """Get database connection"""
        if self.use_postgresql and self.database_url:
            import psycopg2
            import psycopg2.extras
            return psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            # Fallback for SQLite - you'd need to implement this
            raise NotImplementedError("SQLite context tracking not implemented")
    
    async def analyze_update_for_events(self, update: BBUpdate) -> List[Dict]:
        """Analyze an update for trackable events"""
        detected_events = []
        content = f"{update.title} {update.description}".lower()
        
        # Skip finale/voting updates (like we do for alliances)
        skip_phrases = [
            'votes for', 'voted for', 'to be the winner', 'winner of big brother',
            'jury vote', 'crown the winner', 'wins bb', 'finale', 'final vote'
        ]
        
        if any(phrase in content for phrase in skip_phrases):
            return []
        
        # Detect competition events
        for pattern, event_type in self.COMPETITION_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                houseguest = match.group(1).strip().title()
                
                # Filter out common false positives
                if houseguest not in EXCLUDE_WORDS and len(houseguest) > 2:
                    detected_events.append({
                        'type': self.EVENT_TYPES[event_type],
                        'houseguest': houseguest,
                        'description': f"{houseguest} {event_type.lower().replace('_', ' ')}",
                        'update': update,
                        'confidence': self._calculate_event_confidence(event_type, content)
                    })
        
        # Detect social events (involving 2+ people)
        for pattern, event_type in self.SOCIAL_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    hg1 = match.group(1).strip().title()
                    hg2 = match.group(2).strip().title()
                    
                    if (hg1 not in EXCLUDE_WORDS and hg2 not in EXCLUDE_WORDS and 
                        len(hg1) > 2 and len(hg2) > 2):
                        detected_events.append({
                            'type': self.EVENT_TYPES[event_type],
                            'houseguests': [hg1, hg2],
                            'description': f"{hg1} and {hg2} {event_type.lower().replace('_', ' ')}",
                            'update': update,
                            'confidence': self._calculate_event_confidence(event_type, content)
                        })
        
        return detected_events
    
    def _calculate_event_confidence(self, event_type: str, content: str) -> int:
        """Calculate confidence level for detected event"""
        confidence_map = {
            'HOH_WIN': 90,
            'VETO_WIN': 90, 
            'NOMINATION': 85,
            'EVICTION': 95,
            'ALLIANCE_FORM': 70,
            'ALLIANCE_BREAK': 75,
            'SHOWMANCE_START': 80,
            'FIGHT': 85,
            'BETRAYAL': 80
        }
        
        base_confidence = confidence_map.get(event_type, 60)
        
        # Boost confidence for certain keywords
        boost_words = {
            'HOH_WIN': ['officially', 'crowned', 'winner'],
            'VETO_WIN': ['wins', 'winner', 'golden'],
            'NOMINATION': ['ceremony', 'officially', 'nominated'],
            'EVICTION': ['evicted', 'eliminated', 'voted out']
        }
        
        if event_type in boost_words:
            for word in boost_words[event_type]:
                if word in content:
                    base_confidence += 5
        
        return min(base_confidence, 100)
    
    async def record_event(self, event_data: Dict) -> bool:
        """Record an event in the database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Calculate current week and day
            current_week = self._get_current_week()
            current_day = self._get_current_day()
            
            # Record in houseguest_events table
            if 'houseguest' in event_data:
                # Single houseguest event
                cursor.execute("""
                    INSERT INTO houseguest_events 
                    (houseguest_name, event_type, description, week_number, season_day, update_hash, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING event_id
                """, (
                    event_data['houseguest'],
                    event_data['type'],
                    event_data['description'],
                    current_week,
                    current_day,
                    event_data['update'].content_hash,
                    json.dumps({
                        'confidence': event_data['confidence'],
                        'update_title': event_data['update'].title
                    })
                ))
                
                event_id = cursor.fetchone()[0]
                
                # Update statistics
                await self._update_houseguest_stats(cursor, event_data['houseguest'], event_data['type'])
                
            elif 'houseguests' in event_data:
                # Multi-houseguest event
                for hg in event_data['houseguests']:
                    cursor.execute("""
                        INSERT INTO houseguest_events 
                        (houseguest_name, event_type, description, week_number, season_day, update_hash, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        hg,
                        event_data['type'],
                        event_data['description'],
                        current_week,
                        current_day,
                        event_data['update'].content_hash,
                        json.dumps({
                            'confidence': event_data['confidence'],
                            'other_houseguests': [h for h in event_data['houseguests'] if h != hg],
                            'update_title': event_data['update'].title
                        })
                    ))
                
                # Update relationship tracking for social events
                if len(event_data['houseguests']) == 2:
                    await self._update_relationship(cursor, event_data['houseguests'], event_data['type'])
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded event: {event_data['type']} - {event_data['description']}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording event: {e}")
            return False
    
    async def _update_houseguest_stats(self, cursor, houseguest: str, event_type: str):
        """Update running statistics for a houseguest"""
        stat_type = event_type.lower()
        
        cursor.execute("""
            INSERT INTO houseguest_stats (houseguest_name, stat_type, stat_value, season_total)
            VALUES (%s, %s, 1, 1)
            ON CONFLICT (houseguest_name, stat_type)
            DO UPDATE SET 
                stat_value = houseguest_stats.stat_value + 1,
                season_total = houseguest_stats.season_total + 1,
                last_updated = CURRENT_TIMESTAMP
        """, (houseguest, stat_type))
    
    async def _update_relationship(self, cursor, houseguests: List[str], event_type: str):
        """Update relationship tracking between houseguests"""
        if len(houseguests) != 2:
            return
        
        # Ensure consistent ordering
        hg1, hg2 = sorted(houseguests)
        
        # Determine relationship type and strength change
        relationship_map = {
            'showmance_start': ('showmance', +20),
            'fight': ('conflict', -15),
            'betrayal': ('betrayal', -25),
            'alliance_form': ('alliance', +15)
        }
        
        rel_type, strength_change = relationship_map.get(event_type, ('unknown', 0))
        
        cursor.execute("""
            INSERT INTO houseguest_relationships 
            (houseguest_1, houseguest_2, relationship_type, strength_score, duration_days)
            VALUES (%s, %s, %s, %s, 0)
            ON CONFLICT (houseguest_1, houseguest_2)
            DO UPDATE SET
                strength_score = GREATEST(0, LEAST(100, houseguest_relationships.strength_score + %s)),
                last_updated = CURRENT_TIMESTAMP,
                duration_days = EXTRACT(DAYS FROM (CURRENT_TIMESTAMP - houseguest_relationships.first_detected))
        """, (hg1, hg2, rel_type, 50 + strength_change, strength_change))
    
    async def get_historical_context(self, houseguest: str, event_type: str = None) -> str:
        """Get historical context for a houseguest and event type"""
        try:
            # Check cache first
            cache_key = f"{houseguest}_{event_type or 'general'}"
            cached_context = await self._get_cached_context(cache_key)
            if cached_context:
                return cached_context
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            context_parts = []
            
            if event_type == 'hoh_win':
                # Get HOH win count
                cursor.execute("""
                    SELECT season_total FROM houseguest_stats 
                    WHERE houseguest_name = %s AND stat_type = 'hoh_win'
                """, (houseguest,))
                
                result = cursor.fetchone()
                if result and result['season_total'] > 1:
                    count = result['season_total']
                    ordinal = self._get_ordinal(count)
                    context_parts.append(f"This is {houseguest}'s {ordinal} HOH win this season")
                elif not result:
                    context_parts.append(f"This is {houseguest}'s first HOH win")
            
            elif event_type == 'nomination':
                # Get nomination count
                cursor.execute("""
                    SELECT season_total FROM houseguest_stats 
                    WHERE houseguest_name = %s AND stat_type = 'nomination'
                """, (houseguest,))
                
                result = cursor.fetchone()
                if result and result['season_total'] > 1:
                    count = result['season_total']
                    ordinal = self._get_ordinal(count)
                    context_parts.append(f"{houseguest} has been nominated {ordinal} time this season")
                elif not result:
                    context_parts.append(f"This is {houseguest}'s first nomination")
            
            # Get recent alliance context
            cursor.execute("""
                SELECT hr.houseguest_1, hr.houseguest_2, hr.relationship_type, hr.duration_days
                FROM houseguest_relationships hr
                WHERE (hr.houseguest_1 = %s OR hr.houseguest_2 = %s)
                  AND hr.status = 'active'
                  AND hr.relationship_type = 'alliance'
                ORDER BY hr.strength_score DESC
                LIMIT 2
            """, (houseguest, houseguest))
            
            relationships = cursor.fetchall()
            for rel in relationships:
                other_hg = rel['houseguest_2'] if rel['houseguest_1'] == houseguest else rel['houseguest_1']
                if rel['duration_days'] > 7:
                    context_parts.append(f"{houseguest} and {other_hg} have been allies for {rel['duration_days']} days")
            
            conn.close()
            
            # Combine context parts
            context = ". ".join(context_parts)
            
            # Cache the result
            if context:
                await self._cache_context(cache_key, context, 300)  # Cache for 5 minutes
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting historical context: {e}")
            return ""
    
    async def _get_cached_context(self, cache_key: str) -> Optional[str]:
        """Get cached context if still valid"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT context_text FROM context_cache 
                WHERE cache_key = %s AND expires_at > CURRENT_TIMESTAMP
            """, (cache_key,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result['context_text'] if result else None
            
        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")
            return None
    
    async def _cache_context(self, cache_key: str, context: str, ttl_seconds: int):
        """Cache context for performance"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            
            cursor.execute("""
                INSERT INTO context_cache (cache_key, context_text, expires_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (cache_key)
                DO UPDATE SET 
                    context_text = EXCLUDED.context_text,
                    expires_at = EXCLUDED.expires_at,
                    created_at = CURRENT_TIMESTAMP
            """, (cache_key, context, expires_at))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Context caching failed: {e}")
    
    def _get_ordinal(self, number: int) -> str:
        """Convert number to ordinal (1st, 2nd, 3rd, etc.)"""
        if 10 <= number % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
        return f"{number}{suffix}"
    
    def _get_current_week(self) -> int:
        """Calculate current BB week"""
        season_start = datetime(2025, 7, 8)  # Adjust for actual season
        current_date = datetime.now()
        week_number = ((current_date - season_start).days // 7) + 1
        return max(1, week_number)
    
    def _get_current_day(self) -> int:
        """Calculate current season day"""
        season_start = datetime(2025, 7, 8)  # Adjust for actual season
        current_date = datetime.now()
        day_number = (current_date - season_start).days + 1
        return max(1, day_number)

class AllianceTracker:
    """Tracks and analyzes Big Brother alliances"""
    
    # Alliance detection patterns
    ALLIANCE_FORMATION_PATTERNS = [
        (r"([\w\s]+) and ([\w\s]+) make a final (\d+)", "final_deal"),
        (r"([\w\s]+) forms? an? alliance with ([\w\s]+)", "alliance"),
        (r"([\w\s]+) and ([\w\s]+) agree to work together", "agreement"),
        (r"([\w\s]+) and ([\w\s]+) shake on it", "handshake"),
        (r"([\w\s]+), ([\w\s]+),? and ([\w\s]+) form an? alliance", "group_alliance"),
        (r"([\w\s]+) joins? forces with ([\w\s]+)", "joining_forces"),
        (r"([\w\s]+) wants? to work with ([\w\s]+)", "wants_work"),
        (r"([\w\s]+) trusts? ([\w\s]+) completely", "trust"),
    ]
    
    BETRAYAL_PATTERNS = [
        (r"([\w\s]+) wants? to backdoor ([\w\s]+)", "backdoor"),
        (r"([\w\s]+) throws? ([\w\s]+) under the bus", "bus"),
        (r"([\w\s]+) is now targeting ([\w\s]+)", "targeting"),
        (r"([\w\s]+) turns? on ([\w\s]+)", "turns"),
        (r"([\w\s]+) betrays? ([\w\s]+)", "betrays"),
        (r"([\w\s]+) flips? on ([\w\s]+)", "flips"),
        (r"([\w\s]+) wants? ([\w\s]+) out", "wants_out"),
    ]
    
    ALLIANCE_NAME_PATTERNS = [
        r"(?:The|the) ([\w\s]+) alliance",
        r"alliance (?:called|named) (?:The|the)? ?([\w\s]+)",
        r"(?:The|the) ([\w\s]+) \([\w\s,]+\)",  # The Core (Chelsie, Cam, etc)
    ]
    
    def __init__(self, db_path: str = "bb_updates.db", database_url: str = None, use_postgresql: bool = False):
        if use_postgresql and database_url:
            self.database_url = database_url
            self.use_postgresql = True
        else:
            self.db_path = db_path
            self.use_postgresql = False
        self.init_alliance_tables()
    
    def get_connection(self):
        """Get database connection with datetime support"""
        if self.use_postgresql:
            return psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            return conn
    
    def init_alliance_tables(self):
        """Initialize alliance tracking tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if self.use_postgresql:
            # PostgreSQL syntax
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alliances (
                    alliance_id SERIAL PRIMARY KEY,
                    name TEXT,
                    formed_date TIMESTAMP,
                    dissolved_date TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    confidence_level INTEGER DEFAULT 50,
                    last_activity TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alliance_members (
                    alliance_id INTEGER,
                    houseguest_name TEXT,
                    joined_date TIMESTAMP,
                    left_date TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (alliance_id) REFERENCES alliances(alliance_id),
                    UNIQUE(alliance_id, houseguest_name)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alliance_events (
                    event_id SERIAL PRIMARY KEY,
                    alliance_id INTEGER,
                    event_type TEXT,
                    description TEXT,
                    involved_houseguests TEXT,
                    timestamp TIMESTAMP,
                    update_hash TEXT,
                    FOREIGN KEY (alliance_id) REFERENCES alliances(alliance_id)
                )
            """)
        else:
            # SQLite syntax (your original code)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alliances (
                    alliance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    formed_date TIMESTAMP,
                    dissolved_date TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    confidence_level INTEGER DEFAULT 50,
                    last_activity TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alliance_members (
                    alliance_id INTEGER,
                    houseguest_name TEXT,
                    joined_date TIMESTAMP,
                    left_date TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (alliance_id) REFERENCES alliances(alliance_id),
                    UNIQUE(alliance_id, houseguest_name)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alliance_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alliance_id INTEGER,
                    event_type TEXT,
                    description TEXT,
                    involved_houseguests TEXT,
                    timestamp TIMESTAMP,
                    update_hash TEXT,
                    FOREIGN KEY (alliance_id) REFERENCES alliances(alliance_id)
                )
            """)
        
        # Create indexes (same for both)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alliance_status ON alliances(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alliance_members ON alliance_members(houseguest_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alliance_events_type ON alliance_events(event_type)")
        
        conn.commit()
        conn.close()
        
        logger.info("Alliance tracking tables initialized")
    
    def analyze_update_for_alliances(self, update: BBUpdate) -> List[Dict]:
        """Analyze an update for alliance information"""
        content = f"{update.title} {update.description}".strip()
        detected_events = []
        
        # Skip finale/voting/competition winner updates
        skip_phrases = [
            'votes for', 'voted for', 'to be the winner', 'winner of big brother',
            'jury vote', 'crown the winner', 'wins bb', 'wins hoh', 'wins pov',
            'wins the power', 'eviction vote', 'evicted', 'julie pulls the keys',
            'america\'s favorite', 'afp', 'finale', 'final vote', 'cast their vote',
            'announces the winner', 'wins big brother', 'jury votes', 'key to vote',
            'official with a vote', 'makes it official'
        ]
        
        content_lower = content.lower()
        if any(phrase in content_lower for phrase in skip_phrases):
            return []  # Don't process these updates for alliances
        
        # Also skip if it's clearly a competition/ceremony result
        if re.search(r'wins?\s+(the\s+)?(hoh|pov|veto|competition|challenge|big\s+brother)', content, re.IGNORECASE):
            return []
        
        # Skip eviction and ceremony updates
        if re.search(r'(eviction|nomination|veto)\s+ceremony', content, re.IGNORECASE):
            return []
        
        # Check for alliance formations
        for pattern, pattern_type in self.ALLIANCE_FORMATION_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                houseguests = [g.strip() for g in groups if g and not g.isdigit()]
                
                # Filter out common words that aren't houseguests
                houseguests = [hg for hg in houseguests if hg not in EXCLUDE_WORDS]
                
                # Additional filtering for common false positives
                houseguests = [hg for hg in houseguests if len(hg) > 2 and not hg.lower() in ['big', 'brother', 'julie', 'host', 'america']]
                
                if len(houseguests) >= 2:
                    detected_events.append({
                        'type': AllianceEventType.FORMED,
                        'houseguests': houseguests,
                        'pattern_type': pattern_type,
                        'confidence': self._calculate_confidence(pattern_type),
                        'update': update
                    })
        
        # Check for betrayals
        for pattern, pattern_type in self.BETRAYAL_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    betrayer = groups[0].strip()
                    betrayed = groups[1].strip()
                    
                    if betrayer not in EXCLUDE_WORDS and betrayed not in EXCLUDE_WORDS:
                        if len(betrayer) > 2 and len(betrayed) > 2:  # Additional length check
                            detected_events.append({
                                'type': AllianceEventType.BETRAYAL,
                                'betrayer': betrayer,
                                'betrayed': betrayed,
                                'pattern_type': pattern_type,
                                'update': update
                            })
        
        # Check for named alliances
        for pattern in self.ALLIANCE_NAME_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                alliance_name = match.group(1).strip()
                # Skip if alliance name contains finale/winner phrases
                if alliance_name and len(alliance_name) > 2:
                    if not any(skip in alliance_name.lower() for skip in ['winner', 'big brother', 'finale']):
                        # Try to extract members mentioned nearby
                        members = self._extract_nearby_houseguests(content, match.start(), match.end())
                        detected_events.append({
                            'type': AllianceEventType.FORMED,
                            'alliance_name': alliance_name,
                            'houseguests': members,
                            'pattern_type': 'named_alliance',
                            'confidence': 80,  # Named alliances have higher confidence
                            'update': update
                        })
        
        return detected_events
    
    def _calculate_confidence(self, pattern_type: str) -> int:
        """Calculate confidence level based on pattern type"""
        confidence_map = {
            'final_deal': 90,
            'alliance': 85,
            'handshake': 80,
            'group_alliance': 85,
            'agreement': 75,
            'joining_forces': 70,
            'wants_work': 50,
            'trust': 60,
            'named_alliance': 80
        }
        return confidence_map.get(pattern_type, 50)
    
    def _extract_nearby_houseguests(self, content: str, start: int, end: int, window: int = 100) -> List[str]:
        """Extract houseguest names near an alliance mention"""
        # Look in a window around the alliance name
        search_start = max(0, start - window)
        search_end = min(len(content), end + window)
        search_text = content[search_start:search_end]
        
        # Find capitalized words that could be names
        potential_names = re.findall(r'\b[A-Z][a-z]+\b', search_text)
        
        # Filter out common words
        houseguests = [name for name in potential_names if name not in EXCLUDE_WORDS]
        
        return list(set(houseguests))  # Remove duplicates
    
    def process_alliance_event(self, event: Dict) -> Optional[int]:
        """Process a detected alliance event and update database"""
        try:
            if event['type'] == AllianceEventType.FORMED:
                return self._handle_alliance_formation(event)
            elif event['type'] == AllianceEventType.BETRAYAL:
                return self._handle_betrayal(event)
            
        except Exception as e:
            logger.error(f"Error processing alliance event: {e}")
            return None
    
    def _handle_alliance_formation(self, event: Dict) -> Optional[int]:
        """Handle alliance formation event"""
        houseguests = event.get('houseguests', [])
        if len(houseguests) < 2:
            return None
        
        # Check if these houseguests already have an alliance together
        existing_alliance = self._find_existing_alliance(houseguests)
        
        if existing_alliance:
            # Update confidence and last activity
            self._update_alliance_confidence(existing_alliance['alliance_id'], event['confidence'])
            return existing_alliance['alliance_id']
        else:
            # Create new alliance
            alliance_name = event.get('alliance_name')
            if not alliance_name:
                # Generate a name if not provided
                alliance_name = f"{houseguests[0]}/{houseguests[1]}"
                if len(houseguests) > 2:
                    alliance_name += f" +{len(houseguests)-2}"
            
            return self._create_alliance(
                name=alliance_name,
                members=houseguests,
                confidence=event['confidence'],
                formed_date=event['update'].pub_date
            )
    
    def _handle_betrayal(self, event: Dict) -> None:
        """Handle betrayal event"""
        betrayer = event['betrayer']
        betrayed = event['betrayed']
        
        # Find alliances containing both houseguests
        shared_alliances = self._find_shared_alliances(betrayer, betrayed)
        
        for alliance in shared_alliances:
            # Record betrayal event
            self._record_alliance_event(
                alliance_id=alliance['alliance_id'],
                event_type=AllianceEventType.BETRAYAL,
                description=f"{betrayer} betrayed {betrayed}",
                involved=[betrayer, betrayed],
                timestamp=event['update'].pub_date,
                update_hash=event['update'].content_hash
            )
            
            # Consider marking alliance as broken if confidence drops
            self._update_alliance_confidence(alliance['alliance_id'], -30)
    
    def _create_alliance(self, name: str, members: List[str], confidence: int, formed_date: datetime) -> int:
        """Create a new alliance in the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.use_postgresql:
                # PostgreSQL syntax
                cursor.execute("""
                    INSERT INTO alliances (name, formed_date, status, confidence_level, last_activity)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING alliance_id
                """, (name, formed_date, AllianceStatus.ACTIVE.value, confidence, formed_date))
                
                result = cursor.fetchone()
                alliance_id = result['alliance_id'] if isinstance(result, dict) else result[0]
            else:
                # SQLite syntax
                cursor.execute("""
                    INSERT INTO alliances (name, formed_date, status, confidence_level, last_activity)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, formed_date, AllianceStatus.ACTIVE.value, confidence, formed_date))
                
                alliance_id = cursor.lastrowid
            
            # Insert members
            for member in members:
                if self.use_postgresql:
                    cursor.execute("""
                        INSERT INTO alliance_members (alliance_id, houseguest_name, joined_date)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (alliance_id, houseguest_name) DO NOTHING
                    """, (alliance_id, member, formed_date))
                else:
                    cursor.execute("""
                        INSERT OR IGNORE INTO alliance_members (alliance_id, houseguest_name, joined_date)
                        VALUES (?, ?, ?)
                    """, (alliance_id, member, formed_date))
            
            # Record formation event
            if self.use_postgresql:
                cursor.execute("""
                    INSERT INTO alliance_events (alliance_id, event_type, description, involved_houseguests, timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                """, (alliance_id, AllianceEventType.FORMED.value, 
                      f"Alliance '{name}' formed", ",".join(members), formed_date))
            else:
                cursor.execute("""
                    INSERT INTO alliance_events (alliance_id, event_type, description, involved_houseguests, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (alliance_id, AllianceEventType.FORMED.value, 
                      f"Alliance '{name}' formed", ",".join(members), formed_date))
            
            conn.commit()
            logger.info(f"Created new alliance: {name} with members {members}")
            
            return alliance_id
            
        except Exception as e:
            logger.error(f"Error creating alliance: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_active_alliances(self) -> List[Dict]:
        """Get all currently active alliances"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT a.alliance_id, a.name, a.confidence_level, a.last_activity,
                       GROUP_CONCAT(am.houseguest_name) as members
                FROM alliances a
                JOIN alliance_members am ON a.alliance_id = am.alliance_id
                WHERE a.status = ? AND am.is_active = 1
                GROUP BY a.alliance_id
                ORDER BY a.confidence_level DESC, a.last_activity DESC
            """, (AllianceStatus.ACTIVE.value,))
            
            alliances = []
            for row in cursor.fetchall():
                alliances.append({
                    'alliance_id': row[0],
                    'name': row[1],
                    'confidence': row[2],
                    'last_activity': row[3],
                    'members': row[4].split(',') if row[4] else []
                })
            
            return alliances
            
        except Exception as e:
            logger.error(f"Error getting active alliances: {e}")
            return []
        finally:
            conn.close()
    
    def create_alliance_map_embed(self) -> discord.Embed:
        """Create an embed showing current alliance relationships"""
        alliances = self.get_active_alliances()
        broken_alliances = self.get_recently_broken_alliances()
        
        if not alliances and not broken_alliances:
            embed = discord.Embed(
                title="🤝 Big Brother Alliance Map",
                description="No active alliances detected yet",
                color=0x95a5a6,
                timestamp=datetime.now()
            )
            return embed
        
        embed = discord.Embed(
            title="🤝 Big Brother Alliance Map",
            description=f"**{len(alliances)} Active Alliances**",
            color=0x3498db,
            timestamp=datetime.now()
        )
        
        # Group alliances by confidence
        high_conf = [a for a in alliances if a['confidence'] >= 70]
        med_conf = [a for a in alliances if 40 <= a['confidence'] < 70]
        low_conf = [a for a in alliances if a['confidence'] < 40]
        
        if high_conf:
            alliance_text = []
            for alliance in high_conf[:5]:  # Limit to 5
                members_str = " + ".join([f"**{m}**" for m in alliance['members']])
                alliance_text.append(f"🔗 {alliance['name']}\n{members_str}")
            
            embed.add_field(
                name="💪 Strong Alliances",
                value="\n\n".join(alliance_text),
                inline=False
            )
        
        if med_conf:
            alliance_text = []
            for alliance in med_conf[:3]:  # Limit to 3
                members_str = " + ".join(alliance['members'])
                alliance_text.append(f"🤝 {alliance['name']}: {members_str}")
            
            embed.add_field(
                name="🤔 Suspected Alliances",
                value="\n".join(alliance_text),
                inline=False
            )
        
        # Add recently broken strong alliances
        if broken_alliances:
            broken_text = []
            for alliance in broken_alliances[:3]:  # Limit to 3
                members_str = " + ".join(alliance['members'])
                days_ago = (datetime.now() - datetime.fromisoformat(alliance['broken_date'])).days
                broken_text.append(f"💔 {alliance['name']}: {members_str}\n   *Broke {days_ago}d ago after {alliance['days_strong']} days*")
            
            embed.add_field(
                name="⚰️ Recently Broken Alliances",
                value="\n\n".join(broken_text),
                inline=False
            )
        
        # Add recent betrayals
        recent_betrayals = self.get_recent_betrayals(days=3)
        if recent_betrayals:
            betrayal_text = []
            for betrayal in recent_betrayals[:3]:
                betrayal_text.append(f"⚡ {betrayal['description']}")
            
            embed.add_field(
                name="💥 Recent Betrayals",
                value="\n".join(betrayal_text),
                inline=False
            )
        
        embed.set_footer(text="Alliance confidence based on feed activity")
        
        return embed
    
    def get_recently_broken_alliances(self, days: int = 7) -> List[Dict]:
        """Get alliances that were strong but recently broke"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT DISTINCT a.alliance_id, a.name, a.formed_date, 
                       MAX(ae.timestamp) as broken_date,
                       GROUP_CONCAT(DISTINCT am.houseguest_name) as members
                FROM alliances a
                JOIN alliance_members am ON a.alliance_id = am.alliance_id
                JOIN alliance_events ae ON a.alliance_id = ae.alliance_id
                WHERE a.status IN (?, ?)
                  AND ae.event_type = ? 
                  AND datetime(ae.timestamp) > datetime(?)
                  AND a.confidence_level >= 70
                GROUP BY a.alliance_id
                ORDER BY broken_date DESC
            """, (AllianceStatus.BROKEN.value, AllianceStatus.DISSOLVED.value,
                  AllianceEventType.BETRAYAL.value, cutoff_date))
            
            broken_alliances = []
            for row in cursor.fetchall():
                alliance_id, name, formed_date, broken_date, members = row
                
                try:
                    formed = datetime.fromisoformat(formed_date) if isinstance(formed_date, str) else formed_date
                    broken = datetime.fromisoformat(broken_date) if isinstance(broken_date, str) else broken_date
                    days_strong = (broken - formed).days
                except:
                    days_strong = 0
                
                broken_alliances.append({
                    'alliance_id': alliance_id,
                    'name': name,
                    'members': members.split(',') if members else [],
                    'formed_date': formed_date,
                    'broken_date': broken_date,
                    'days_strong': days_strong
                })
            
            return broken_alliances
            
        except Exception as e:
            logger.error(f"Error getting broken alliances: {e}")
            return []
        finally:
            conn.close()
    
    def get_houseguest_loyalty_embed(self, houseguest: str) -> discord.Embed:
        """Create an embed showing a houseguest's alliance history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT a.name, a.status, am.joined_date, am.left_date, a.confidence_level,
                       GROUP_CONCAT(am2.houseguest_name) as all_members
                FROM alliance_members am
                JOIN alliances a ON am.alliance_id = a.alliance_id
                JOIN alliance_members am2 ON am2.alliance_id = a.alliance_id
                WHERE am.houseguest_name = ?
                GROUP BY a.alliance_id
                ORDER BY am.joined_date DESC
            """, (houseguest,))
            
            alliances = cursor.fetchall()
            
            cursor.execute("""
                SELECT COUNT(*) FROM alliance_events
                WHERE event_type = ? AND involved_houseguests LIKE ?
            """, (AllianceEventType.BETRAYAL.value, f"%{houseguest}%"))
            
            betrayal_count = cursor.fetchone()[0]
            
            embed = discord.Embed(
                title=f"🎭 {houseguest}'s Alliance History",
                color=0xe74c3c if betrayal_count > 2 else 0x2ecc71,
                timestamp=datetime.now()
            )
            
            if not alliances:
                embed.description = f"{houseguest} has not been detected in any alliances"
                return embed
            
            active_alliances = sum(1 for a in alliances if a[1] == AllianceStatus.ACTIVE.value)
            broken_alliances = sum(1 for a in alliances if a[1] in [AllianceStatus.BROKEN.value, AllianceStatus.DISSOLVED.value])
            
            loyalty_score = max(0, 100 - (betrayal_count * 20) - (broken_alliances * 10))
            loyalty_emoji = "🏆" if loyalty_score >= 80 else "⚠️" if loyalty_score >= 50 else "🚨"
            
            embed.description = f"**Loyalty Score**: {loyalty_emoji} {loyalty_score}/100\n"
            embed.description += f"**Betrayals**: {betrayal_count} | **Active Alliances**: {active_alliances}"
            
            alliance_text = []
            for alliance in alliances[:6]:
                name, status, joined, left, confidence, members = alliance
                status_emoji = "✅" if status == AllianceStatus.ACTIVE.value else "❌"
                
                other_members = [m for m in members.split(',') if m != houseguest]
                members_str = ", ".join(other_members[:3])
                if len(other_members) > 3:
                    members_str += f" +{len(other_members)-3}"
                
                alliance_text.append(f"{status_emoji} **{name}** (w/ {members_str})")
            
            embed.add_field(
                name="📋 Alliance History",
                value="\n".join(alliance_text) if alliance_text else "No alliances found",
                inline=False
            )
            
            return embed
            
        except Exception as e:
            logger.error(f"Error in loyalty embed: {e}")
            embed = discord.Embed(
                title=f"🎭 {houseguest}'s Alliance History",
                description="Error retrieving alliance data",
                color=0xe74c3c
            )
            return embed
        finally:
            conn.close()
    
    def get_recent_betrayals(self, days: int = 7) -> List[Dict]:
        """Get recent betrayal events"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT description, timestamp, involved_houseguests
                FROM alliance_events
                WHERE event_type = ? AND datetime(timestamp) > datetime(?)
                ORDER BY timestamp DESC
            """, (AllianceEventType.BETRAYAL.value, cutoff_date))
            
            betrayals = []
            for row in cursor.fetchall():
                betrayals.append({
                    'description': row[0],
                    'timestamp': row[1],
                    'involved': row[2].split(',') if row[2] else []
                })
            
            return betrayals
            
        except Exception as e:
            logger.error(f"Error getting betrayals: {e}")
            return []
        finally:
            conn.close()
    
    def _find_existing_alliance(self, houseguests: List[str]) -> Optional[Dict]:
        """Find if these houseguests already have an alliance"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.use_postgresql:
                # PostgreSQL syntax
                cursor.execute("""
                    SELECT DISTINCT a.alliance_id, a.name, a.confidence_level
                    FROM alliance_members am
                    JOIN alliances a ON am.alliance_id = a.alliance_id
                    WHERE am.houseguest_name = %s AND a.status = %s AND am.is_active = TRUE
                """, (houseguests[0], AllianceStatus.ACTIVE.value))
            else:
                # SQLite syntax
                cursor.execute("""
                    SELECT DISTINCT a.alliance_id, a.name, a.confidence_level
                    FROM alliance_members am
                    JOIN alliances a ON am.alliance_id = a.alliance_id
                    WHERE am.houseguest_name = ? AND a.status = ? AND am.is_active = 1
                """, (houseguests[0], AllianceStatus.ACTIVE.value))
            
            for row in cursor.fetchall():
                if self.use_postgresql:
                    alliance_id = row['alliance_id']
                    name = row['name']
                    confidence = row['confidence_level']
                else:
                    alliance_id, name, confidence = row
                
                # Check if all houseguests are in this alliance
                all_in = True
                for hg in houseguests[1:]:
                    if self.use_postgresql:
                        cursor.execute("""
                            SELECT 1 FROM alliance_members
                            WHERE alliance_id = %s AND houseguest_name = %s AND is_active = TRUE
                        """, (alliance_id, hg))
                    else:
                        cursor.execute("""
                            SELECT 1 FROM alliance_members
                            WHERE alliance_id = ? AND houseguest_name = ? AND is_active = 1
                        """, (alliance_id, hg))
                    
                    if not cursor.fetchone():
                        all_in = False
                        break
                
                if all_in:
                    return {
                        'alliance_id': alliance_id,
                        'name': name,
                        'confidence': confidence
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding existing alliance: {e}")
            return None
        finally:
            conn.close()
    
    
    def _find_shared_alliances(self, hg1: str, hg2: str) -> List[Dict]:
        """Find alliances containing both houseguests"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.use_postgresql:
                cursor.execute("""
                    SELECT DISTINCT a.alliance_id, a.name
                    FROM alliance_members am1
                    JOIN alliance_members am2 ON am1.alliance_id = am2.alliance_id
                    JOIN alliances a ON am1.alliance_id = a.alliance_id
                    WHERE am1.houseguest_name = %s AND am2.houseguest_name = %s
                          AND am1.is_active = TRUE AND am2.is_active = TRUE
                          AND a.status = %s
                """, (hg1, hg2, AllianceStatus.ACTIVE.value))
            else:
                cursor.execute("""
                    SELECT DISTINCT a.alliance_id, a.name
                    FROM alliance_members am1
                    JOIN alliance_members am2 ON am1.alliance_id = am2.alliance_id
                    JOIN alliances a ON am1.alliance_id = a.alliance_id
                    WHERE am1.houseguest_name = ? AND am2.houseguest_name = ?
                          AND am1.is_active = 1 AND am2.is_active = 1
                          AND a.status = ?
                """, (hg1, hg2, AllianceStatus.ACTIVE.value))
            
            alliances = []
            for row in cursor.fetchall():
                if self.use_postgresql:
                    alliances.append({
                        'alliance_id': row['alliance_id'],
                        'name': row['name']
                    })
                else:
                    alliances.append({
                        'alliance_id': row[0],
                        'name': row[1]
                    })
            
            return alliances
            
        except Exception as e:
            logger.error(f"Error finding shared alliances: {e}")
            return []
        finally:
            conn.close()
    
    def _update_alliance_confidence(self, alliance_id: int, change: int):
        """Update alliance confidence level"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get current confidence and status - Fixed for PostgreSQL
            if self.use_postgresql:
                cursor.execute("""
                    SELECT confidence_level, status, formed_date 
                    FROM alliances WHERE alliance_id = %s
                """, (alliance_id,))
            else:
                cursor.execute("""
                    SELECT confidence_level, status, formed_date 
                    FROM alliances WHERE alliance_id = ?
                """, (alliance_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Alliance {alliance_id} not found for confidence update")
                return
                
            if self.use_postgresql:
                current_conf = result['confidence_level']
                status = result['status']
                formed_date = result['formed_date']
            else:
                current_conf, status, formed_date = result
            
            # Track if this was a strong alliance before the change
            was_strong = current_conf >= 70 and status == AllianceStatus.ACTIVE.value
            
            # Calculate how long it's been active if it was strong
            if was_strong:
                if isinstance(formed_date, str):
                    formed = datetime.fromisoformat(formed_date)
                else:
                    formed = formed_date
                days_active = (datetime.now() - formed).days
            else:
                days_active = 0
            
            # Update confidence
            new_confidence = max(0, min(100, current_conf + change))
            
            if self.use_postgresql:
                cursor.execute("""
                    UPDATE alliances 
                    SET confidence_level = %s,
                        last_activity = %s
                    WHERE alliance_id = %s
                """, (new_confidence, datetime.now(), alliance_id))
            else:
                cursor.execute("""
                    UPDATE alliances 
                    SET confidence_level = ?,
                        last_activity = ?
                    WHERE alliance_id = ?
                """, (new_confidence, datetime.now(), alliance_id))
            
            # Check if confidence dropped too low
            if new_confidence < 20:
                # Mark alliance as broken
                new_status = AllianceStatus.BROKEN.value
                if self.use_postgresql:
                    cursor.execute("""
                        UPDATE alliances SET status = %s WHERE alliance_id = %s
                    """, (new_status, alliance_id))
                else:
                    cursor.execute("""
                        UPDATE alliances SET status = ? WHERE alliance_id = ?
                    """, (new_status, alliance_id))
                
                # If this was a strong alliance for over a week, record it as a major break
                if was_strong and days_active >= 7:
                    if self.use_postgresql:
                        cursor.execute("""
                            INSERT INTO alliance_events 
                            (alliance_id, event_type, description, timestamp)
                            VALUES (%s, %s, %s, %s)
                        """, (alliance_id, AllianceEventType.DISSOLVED.value,
                              f"Alliance dissolved after {days_active} days", datetime.now()))
                    else:
                        cursor.execute("""
                            INSERT INTO alliance_events 
                            (alliance_id, event_type, description, timestamp)
                            VALUES (?, ?, ?, ?)
                        """, (alliance_id, AllianceEventType.DISSOLVED.value,
                              f"Alliance dissolved after {days_active} days", datetime.now()))
                    
                    logger.info(f"Major alliance break: Alliance {alliance_id} was strong for {days_active} days")
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating alliance confidence: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _record_alliance_event(self, alliance_id: int, event_type: AllianceEventType, 
                             description: str, involved: List[str], 
                             timestamp: datetime, update_hash: str = None):
        """Record an alliance event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO alliance_events 
                (alliance_id, event_type, description, involved_houseguests, timestamp, update_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (alliance_id, event_type.value, description, 
                  ",".join(involved), timestamp, update_hash))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error recording alliance event: {e}")
            conn.rollback()
        finally:
            conn.close()

class PredictionManager:
    """Manages Big Brother prediction polls and leaderboards"""
    
    # Point values for each prediction type
    POINT_VALUES = {
        PredictionType.SEASON_WINNER: 20,
        PredictionType.FIRST_BOOT: 15,
        PredictionType.WEEKLY_HOH: 5,
        PredictionType.WEEKLY_VETO: 3,
        PredictionType.WEEKLY_EVICTION: 2
    }
    
    def __init__(self, db_path: str = "bb_updates.db", database_url: str = None, use_postgresql: bool = False):
        if use_postgresql and database_url:
            self.database_url = database_url
            self.use_postgresql = True
        else:
            self.db_path = db_path
            self.use_postgresql = False
        self.init_prediction_tables()
    
    def get_connection(self):
        """Get database connection with datetime support"""
        if self.use_postgresql:
            import psycopg2
            import psycopg2.extras
            return psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            return conn
    
    def _execute_query(self, cursor, query_sqlite, params, query_postgresql=None):
        """Execute query with proper syntax for current database"""
        if self.use_postgresql:
            if query_postgresql:
                # Use custom PostgreSQL query if provided
                cursor.execute(query_postgresql, params)
            else:
                # Convert SQLite ? to PostgreSQL %s
                pg_query = query_sqlite.replace('?', '%s')
                cursor.execute(pg_query, params)
        else:
            cursor.execute(query_sqlite, params)
    
    def init_prediction_tables(self):
        """Initialize prediction system database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if self.use_postgresql:
            # PostgreSQL syntax
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    prediction_type TEXT NOT NULL,
                    options TEXT NOT NULL,
                    created_by BIGINT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closes_at TIMESTAMP NOT NULL,
                    status TEXT DEFAULT 'active',
                    correct_option TEXT,
                    week_number INTEGER,
                    guild_id BIGINT NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_predictions (
                    user_id BIGINT NOT NULL,
                    prediction_id INTEGER NOT NULL,
                    option TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, prediction_id),
                    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_leaderboard (
                    user_id BIGINT NOT NULL,
                    guild_id BIGINT NOT NULL,
                    week_number INTEGER NOT NULL,
                    season_points INTEGER DEFAULT 0,
                    weekly_points INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    total_predictions INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, guild_id, week_number)
                )
            """)
        else:
            # SQLite syntax (your original code)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    prediction_type TEXT NOT NULL,
                    options TEXT NOT NULL,
                    created_by INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closes_at TIMESTAMP NOT NULL,
                    status TEXT DEFAULT 'active',
                    correct_option TEXT,
                    week_number INTEGER,
                    guild_id INTEGER NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_predictions (
                    user_id INTEGER NOT NULL,
                    prediction_id INTEGER NOT NULL,
                    option TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, prediction_id),
                    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_leaderboard (
                    user_id INTEGER NOT NULL,
                    guild_id INTEGER NOT NULL,
                    week_number INTEGER NOT NULL,
                    season_points INTEGER DEFAULT 0,
                    weekly_points INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    total_predictions INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, guild_id, week_number)
                )
            """)
        
        # Create indexes (same for both)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_closes_at ON predictions(closes_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_predictions_user ON user_predictions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_season ON prediction_leaderboard(guild_id, season_points DESC)")
        
        conn.commit()
        conn.close()
        
        logger.info("Prediction system tables initialized")
    
    def create_prediction(self, title: str, description: str, prediction_type: PredictionType,
                         options: List[str], created_by: int, guild_id: int,
                         duration_hours: int, week_number: Optional[int] = None) -> int:
        """Create a new prediction poll"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            closes_at = datetime.now() + timedelta(hours=duration_hours)
            options_json = json.dumps(options)
            
            if self.use_postgresql:
                cursor.execute("""
                    INSERT INTO predictions 
                    (title, description, prediction_type, options, created_by, closes_at, guild_id, week_number)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING prediction_id
                """, (title, description, prediction_type.value, options_json, 
                      created_by, closes_at, guild_id, week_number))
                
                result = cursor.fetchone()
                if result:
                    prediction_id = result[0] if isinstance(result, tuple) else result['prediction_id']
                else:
                    logger.error("No prediction_id returned from PostgreSQL insert")
                    raise Exception("Failed to get prediction ID")
            else:
                cursor.execute("""
                    INSERT INTO predictions 
                    (title, description, prediction_type, options, created_by, closes_at, guild_id, week_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (title, description, prediction_type.value, options_json, 
                      created_by, closes_at, guild_id, week_number))
                
                prediction_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Created prediction poll: {title} (ID: {prediction_id})")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error creating prediction: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def make_prediction(self, user_id: int, prediction_id: int, option: str) -> bool:
        """User makes or updates a prediction"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Check if prediction exists and is active
            self._execute_query(cursor, """
                SELECT status, closes_at, options FROM predictions 
                WHERE prediction_id = ?
            """, (prediction_id,))
            
            result = cursor.fetchone()
            if not result:
                return False
            
            # Handle result based on database type
            if self.use_postgresql:
                status = result['status']
                closes_at = result['closes_at']
                options_json = result['options']
            else:
                status, closes_at, options_json = result
            
            # Safely parse JSON options
            try:
                options = json.loads(options_json) if options_json else []
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Invalid JSON in options for prediction {prediction_id}: {options_json}")
                return False
            
            # Check if prediction is still active and not closed
            if status != PredictionStatus.ACTIVE.value:
                return False
            
            if datetime.now() >= closes_at:
                return False
            
            # Check if option is valid
            if option not in options:
                return False
            
            # Insert or update user prediction
            if self.use_postgresql:
                self._execute_query(cursor, "", (user_id, prediction_id, option), """
                    INSERT INTO user_predictions 
                    (user_id, prediction_id, option, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, prediction_id) 
                    DO UPDATE SET option = EXCLUDED.option, updated_at = CURRENT_TIMESTAMP
                """)
            else:
                cursor.execute("""
                    INSERT OR REPLACE INTO user_predictions 
                    (user_id, prediction_id, option, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (user_id, prediction_id, option))
            
            conn.commit()
            logger.info(f"User {user_id} predicted '{option}' for prediction {prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def close_prediction(self, prediction_id: int, admin_user_id: int) -> bool:
        """Manually close a prediction poll"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE predictions 
                SET status = ? 
                WHERE prediction_id = ? AND status = ?
            """, (PredictionStatus.CLOSED.value, prediction_id, PredictionStatus.ACTIVE.value))
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Prediction {prediction_id} closed by admin {admin_user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing prediction: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def resolve_prediction(self, prediction_id: int, correct_option: str, admin_user_id: int) -> Tuple[bool, int, List[int]]:
        """Resolve a prediction and award points, returning success, count, and user IDs"""
        conn = self.get_connection()
        
        try:
            # Set busy timeout for this connection
            if not self.use_postgresql:
                conn.execute("PRAGMA busy_timeout = 10000")
            cursor = conn.cursor()
            
            # Get prediction details
            self._execute_query(cursor, """
                SELECT prediction_type, options, status, guild_id, week_number
                FROM predictions WHERE prediction_id = ?
            """, (prediction_id,))
            
            result = cursor.fetchone()
            if not result:
                return False, 0, []
            
            # Handle result based on database type
            if self.use_postgresql:
                pred_type = result['prediction_type']
                options_json = result['options']
                status = result['status']
                guild_id = result['guild_id']
                week_number = result['week_number']
            else:
                pred_type, options_json, status, guild_id, week_number = result
            
            # Parse options safely
            try:
                options = json.loads(options_json) if options_json else []
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Invalid options JSON for prediction {prediction_id}")
                return False, 0, []
            
            # Validate correct option
            if correct_option not in options:
                return False, 0, []
            
            # Get users who predicted correctly BEFORE updating the prediction
            self._execute_query(cursor, """
                SELECT user_id FROM user_predictions 
                WHERE prediction_id = ? AND option = ?
            """, (prediction_id, correct_option))
            
            # Handle user IDs result
            if self.use_postgresql:
                correct_user_ids = [row['user_id'] for row in cursor.fetchall()]
            else:
                correct_user_ids = [row[0] for row in cursor.fetchall()]
            
            # Update prediction status
            self._execute_query(cursor, """
                UPDATE predictions 
                SET status = ?, correct_option = ?
                WHERE prediction_id = ?
            """, (PredictionStatus.RESOLVED.value, correct_option, prediction_id))
            
            # Get all user predictions for this poll
            self._execute_query(cursor, """
                SELECT user_id, option FROM user_predictions 
                WHERE prediction_id = ?
            """, (prediction_id,))
            
            # Handle user predictions result
            if self.use_postgresql:
                user_predictions = [(row['user_id'], row['option']) for row in cursor.fetchall()]
            else:
                user_predictions = cursor.fetchall()
            
            conn.commit()
            
            # Calculate points for correct predictions
            prediction_type = PredictionType(pred_type)
            points = self.POINT_VALUES[prediction_type]
            correct_users = 0
            
            current_week = week_number if week_number else self._get_current_week()
            
            # Process each user prediction
            for user_id, user_option in user_predictions:
                try:
                    if user_option == correct_option:
                        self._update_leaderboard(user_id, guild_id, current_week, points, True, True)
                        correct_users += 1
                    else:
                        self._update_leaderboard(user_id, guild_id, current_week, 0, False, True)
                except Exception as e:
                    logger.error(f"Error updating leaderboard for user {user_id}: {e}")
                    continue
            
            logger.info(f"Prediction {prediction_id} resolved. {correct_users} users got it right.")
            return True, correct_users, correct_user_ids
            
        except Exception as e:
            logger.error(f"Error resolving prediction: {e}")
            conn.rollback()
            return False, 0, []
        finally:
            conn.close()
        
            
    
    def _update_leaderboard(self, user_id: int, guild_id: int, week_number: int, 
                          points: int, was_correct: bool, participated: bool):
        """Update user's leaderboard stats with improved error handling"""
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            conn = None
            try:
                conn = self.get_connection()
                # Set a longer timeout and immediate lock behavior
                conn.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout
                conn.execute("BEGIN IMMEDIATE")  # Get exclusive lock immediately
                
                cursor = conn.cursor()
                
                # Get current stats
                cursor.execute("""
                    SELECT season_points, weekly_points, correct_predictions, total_predictions
                    FROM prediction_leaderboard 
                    WHERE user_id = ? AND guild_id = ? AND week_number = ?
                """, (user_id, guild_id, week_number))
                
                result = cursor.fetchone()
                
                if result:
                    # Update existing record
                    season_points, weekly_points, correct_preds, total_preds = result
                    new_season_points = season_points + points
                    new_weekly_points = weekly_points + points
                    new_correct = correct_preds + (1 if was_correct else 0)
                    new_total = total_preds + (1 if participated else 0)
                    
                    cursor.execute("""
                        UPDATE prediction_leaderboard 
                        SET season_points = ?, weekly_points = ?, 
                            correct_predictions = ?, total_predictions = ?,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND guild_id = ? AND week_number = ?
                    """, (new_season_points, new_weekly_points, new_correct, new_total,
                          user_id, guild_id, week_number))
                else:
                    # Create new record
                    cursor.execute("""
                        INSERT INTO prediction_leaderboard 
                        (user_id, guild_id, week_number, season_points, weekly_points, 
                         correct_predictions, total_predictions)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (user_id, guild_id, week_number, points, points,
                          1 if was_correct else 0, 1 if participated else 0))
                
                conn.commit()
                logger.info(f"Successfully updated leaderboard for user {user_id}")
                break  # Success, exit retry loop
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked on attempt {attempt + 1}, retrying in {retry_delay}s")
                    if conn:
                        try:
                            conn.rollback()
                        except:
                            pass
                        conn.close()
                    time.sleep(retry_delay)  # Use time.sleep instead of asyncio.sleep
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(f"Database lock error after {attempt + 1} attempts: {e}")
                    if conn:
                        conn.rollback()
                    raise
            except Exception as e:
                logger.error(f"Error updating leaderboard: {e}")
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    conn.close()

    
    def get_active_predictions(self, guild_id: int) -> List[Dict]:
        """Get all active predictions for a guild"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            self._execute_query(cursor, """
                SELECT prediction_id, title, description, prediction_type, 
                       options, closes_at, week_number
                FROM predictions 
                WHERE guild_id = ? AND status = ? AND closes_at > ?
                ORDER BY closes_at ASC
            """, (guild_id, PredictionStatus.ACTIVE.value, datetime.now()))
            
            predictions = []
            for row in cursor.fetchall():
                if self.use_postgresql:
                    # PostgreSQL returns RealDictCursor results
                    pred_id = row['prediction_id']
                    title = row['title']
                    desc = row['description']
                    pred_type = row['prediction_type']
                    options_json = row['options']
                    closes_at = row['closes_at']
                    week_num = row['week_number']
                else:
                    # SQLite returns tuple
                    pred_id, title, desc, pred_type, options_json, closes_at, week_num = row
                
                # Safely parse JSON
                try:
                    options = json.loads(options_json) if options_json else []
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Invalid JSON in options for prediction {pred_id}: {options_json}")
                    options = []
                
                # Handle datetime conversion
                if isinstance(closes_at, str):
                    try:
                        closes_at = datetime.fromisoformat(closes_at)
                    except ValueError:
                        logger.warning(f"Invalid date format for prediction {pred_id}: {closes_at}")
                        closes_at = datetime.now()
                
                predictions.append({
                    'id': pred_id,
                    'title': title,
                    'description': desc,
                    'type': pred_type,
                    'options': options,
                    'closes_at': closes_at,
                    'week_number': week_num
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting active predictions: {e}")
            return []
        finally:
            conn.close()
    
    def get_user_prediction(self, user_id: int, prediction_id: int) -> Optional[str]:
        """Get a user's prediction for a specific poll"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            self._execute_query(cursor, """
                SELECT option FROM user_predictions 
                WHERE user_id = ? AND prediction_id = ?
            """, (user_id, prediction_id))
            
            result = cursor.fetchone()
            if result:
                if self.use_postgresql:
                    return result['option']
                else:
                    return result[0]
            return None
            
        except Exception as e:
            logger.error(f"Error getting user prediction: {e}")
            return None
        finally:
            conn.close()
    
    def get_season_leaderboard(self, guild_id: int, limit: int = 10) -> List[Dict]:
        """Get season-long leaderboard"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            self._execute_query(cursor, """
                SELECT user_id, 
                       SUM(season_points) as total_points,
                       SUM(correct_predictions) as total_correct,
                       SUM(total_predictions) as total_predictions
                FROM prediction_leaderboard 
                WHERE guild_id = ?
                GROUP BY user_id
                ORDER BY total_points DESC, total_correct DESC
                LIMIT ?
            """, (guild_id, limit))
            
            leaderboard = []
            for row in cursor.fetchall():
                if self.use_postgresql:
                    user_id = row['user_id']
                    points = row['total_points'] or 0
                    correct = row['total_correct'] or 0
                    total = row['total_predictions'] or 0
                else:
                    user_id, points, correct, total = row
                    points = points or 0
                    correct = correct or 0
                    total = total or 0
                
                accuracy = (correct / total * 100) if total > 0 else 0
                leaderboard.append({
                    'user_id': user_id,
                    'points': points,
                    'correct': correct,
                    'total': total,
                    'accuracy': accuracy
                })
            
            return leaderboard
            
        except Exception as e:
            logger.error(f"Error getting season leaderboard: {e}")
            return []
        finally:
            conn.close()
    
    def get_weekly_leaderboard(self, guild_id: int, week_number: int = None, limit: int = 10) -> List[Dict]:
        """Get weekly leaderboard"""
        if week_number is None:
            week_number = self._get_current_week()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT user_id, weekly_points, correct_predictions, total_predictions
                FROM prediction_leaderboard 
                WHERE guild_id = ? AND week_number = ?
                ORDER BY weekly_points DESC, correct_predictions DESC
                LIMIT ?
            """, (guild_id, week_number, limit))
            
            leaderboard = []
            for row in cursor.fetchall():
                user_id, points, correct, total = row
                accuracy = (correct / total * 100) if total > 0 else 0
                leaderboard.append({
                    'user_id': user_id,
                    'points': points,
                    'correct': correct,
                    'total': total,
                    'accuracy': accuracy
                })
            
            return leaderboard
            
        except Exception as e:
            logger.error(f"Error getting weekly leaderboard: {e}")
            return []
        finally:
            conn.close()
    
    def get_user_predictions_history(self, user_id: int, guild_id: int) -> List[Dict]:
        """Get user's prediction history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            self._execute_query(cursor, """
                SELECT p.title, p.prediction_type, up.option, p.correct_option, 
                       p.status, p.created_at, p.week_number
                FROM user_predictions up
                JOIN predictions p ON up.prediction_id = p.prediction_id
                WHERE up.user_id = ? AND p.guild_id = ?
                ORDER BY p.created_at DESC
                LIMIT 20
            """, (user_id, guild_id))
            
            history = []
            for row in cursor.fetchall():
                if self.use_postgresql:
                    title = row['title']
                    pred_type = row['prediction_type']
                    user_option = row['option']
                    correct_option = row['correct_option']
                    status = row['status']
                    created_at = row['created_at']
                    week_num = row['week_number']
                else:
                    title, pred_type, user_option, correct_option, status, created_at, week_num = row
                
                is_correct = None
                if status == PredictionStatus.RESOLVED.value and correct_option:
                    is_correct = (user_option == correct_option)
                
                points_earned = 0
                if is_correct:
                    points_earned = self.POINT_VALUES[PredictionType(pred_type)]
                
                history.append({
                    'title': title,
                    'type': pred_type,
                    'user_option': user_option,
                    'correct_option': correct_option,
                    'is_correct': is_correct,
                    'points_earned': points_earned,
                    'status': status,
                    'created_at': created_at,
                    'week_number': week_num
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return []
        finally:
            conn.close()
    
    def _get_current_week(self) -> int:
        """Calculate current BB week (you may want to adjust this logic)"""
        # For now, using a simple calculation from season start
        # You can modify this based on actual BB season dates
        season_start = datetime(2025, 7, 8)  # Adjust for actual season
        current_date = datetime.now()
        week_number = ((current_date - season_start).days // 7) + 1
        return max(1, week_number)
    
    def auto_close_expired_predictions(self):
        """Close predictions that have passed their closing time"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.use_postgresql:
                cursor.execute("""
                    UPDATE predictions 
                    SET status = %s 
                    WHERE status = %s AND closes_at <= %s
                """, (PredictionStatus.CLOSED.value, PredictionStatus.ACTIVE.value, datetime.now()))
            else:
                cursor.execute("""
                    UPDATE predictions 
                    SET status = ? 
                    WHERE status = ? AND closes_at <= ?
                """, (PredictionStatus.CLOSED.value, PredictionStatus.ACTIVE.value, datetime.now()))
            
            closed_count = cursor.rowcount
            conn.commit()
            
            if closed_count > 0:
                logger.info(f"Auto-closed {closed_count} expired predictions")
            
            return closed_count
            
        except Exception as e:
            logger.error(f"Error auto-closing predictions: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def create_prediction_embed(self, prediction: Dict, user_prediction: Optional[str] = None) -> discord.Embed:
        """Create Discord embed for a prediction"""
        pred_type_names = {
            'season_winner': '👑 Season Winner',
            'first_boot': '👢 First Boot - Womp Womp',
            'weekly_hoh': '🏆 Weekly HOH',
            'weekly_veto': '💎 Weekly Veto',
            'weekly_eviction': '🚪 Weekly Eviction'
        }
        
        type_name = pred_type_names.get(prediction['type'], prediction['type'])
        points = self.POINT_VALUES[PredictionType(prediction['type'])]
        
        embed = discord.Embed(
            title=f"{type_name} - {prediction['title']}",
            description=prediction['description'],
            color=0xffd700 if prediction['type'] == 'season_winner' else 0x3498db,
            timestamp=datetime.now()
        )
        
        # Handle large number of options by splitting into multiple fields if needed
        options = prediction['options']
        
        if len(options) <= 10:
            # For 10 or fewer options, show them all with emojis
            options_text = []
            for i, option in enumerate(options, 1):
                emoji = "✅" if user_prediction == option else f"{i}️⃣" if i <= 10 else "▪️"
                options_text.append(f"{emoji} {option}")
            
            embed.add_field(
                name="📋 Options",
                value="\n".join(options_text),
                inline=False
            )
        else:
            # For more than 10 options, split into multiple columns
            mid_point = len(options) // 2
            
            # First half
            options_text_1 = []
            for i, option in enumerate(options[:mid_point], 1):
                emoji = "✅" if user_prediction == option else "▪️"
                options_text_1.append(f"{emoji} {option}")
            
            # Second half
            options_text_2 = []
            for i, option in enumerate(options[mid_point:], mid_point + 1):
                emoji = "✅" if user_prediction == option else "▪️"
                options_text_2.append(f"{emoji} {option}")
            
            embed.add_field(
                name="📋 Options (Part 1)",
                value="\n".join(options_text_1),
                inline=True
            )
            
            embed.add_field(
                name="📋 Options (Part 2)",
                value="\n".join(options_text_2),
                inline=True
            )
            
            # Add spacer
            embed.add_field(name="\u200b", value="\u200b", inline=True)
        
        # Add timing info
        closes_at = prediction['closes_at']
        if isinstance(closes_at, str):
            closes_at = datetime.fromisoformat(closes_at)
        
        time_left = closes_at - datetime.now()
        if time_left.total_seconds() > 0:
            hours_left = int(time_left.total_seconds() / 3600)
            minutes_left = int((time_left.total_seconds() % 3600) / 60)
            time_str = f"{hours_left}h {minutes_left}m remaining"
        else:
            time_str = "Closed"
        
        embed.add_field(name="⏰ Time Left", value=time_str, inline=True)
        embed.add_field(name="🎯 Points", value=f"{points} pts", inline=True)
        
        if prediction.get('week_number'):
            embed.add_field(name="📅 Week", value=f"Week {prediction['week_number']}", inline=True)
        
        if user_prediction:
            embed.add_field(
                name="✅ Your Prediction",
                value=user_prediction,
                inline=False
            )
        
        # Add instruction for making predictions
        embed.add_field(
            name="💡 How to Predict",
            value="Use `/predict` to make your prediction!\nThe command will guide you through selecting this poll and your choice.",
            inline=False
        )
        
        embed.set_footer(text=f"Prediction ID: {prediction['id']} • Use exact option names")
        
        return embed
    
    async def create_leaderboard_embed(self, leaderboard: List[Dict], guild, leaderboard_type: str = "Season") -> discord.Embed:
        """Create Discord embed for leaderboard"""
        embed = discord.Embed(
            title=f"🏆 {leaderboard_type} Prediction Leaderboard",
            description=f"Top predictors in {guild.name}",
            color=0xffd700,
            timestamp=datetime.now()
        )
        
        if not leaderboard:
            embed.add_field(
                name="No Data",
                value="No predictions have been made yet!",
                inline=False
            )
            return embed
        
        leaderboard_text = []
        medals = ["🥇", "🥈", "🥉"]
        
        for i, entry in enumerate(leaderboard[:10]):
            user = guild.get_member(entry['user_id'])
    
            if user:
                username = user.display_name
            else:
                # Show last 4 digits of user ID as fallback
                username = f"User#{str(entry['user_id'])[-4:]}"
            
            medal = medals[i] if i < 3 else f"{i+1}."
            accuracy_str = f"{entry['accuracy']:.1f}%" if entry['total'] > 0 else "0%"
            
            leaderboard_text.append(
                f"{medal} **{username}** - {entry['points']} pts "
                f"({entry['correct']}/{entry['total']} - {accuracy_str})"
            )
        
        embed.add_field(
            name="Rankings",
            value="\n".join(leaderboard_text),
            inline=False
        )
        
        embed.set_footer(text=f"{leaderboard_type} Leaderboard • Points = Correct Predictions × Point Values")
        
        return embed


class UpdateBatcher:
    """Enhanced batching system with dual rhythms - Highlights + Hourly Summary"""
    
    def __init__(self, analyzer: BBAnalyzer, config: Config):
        self.analyzer = analyzer
        self.config = config
        
        # Keep highlights queue for 25-update batches (remove hourly_queue)
        self.highlights_queue = []
        self.hourly_queue = []
        
        self.last_batch_time = datetime.now()
        self.last_hourly_summary = datetime.now()

        self.hourly_queue = []
        
        self.last_batch_time = datetime.now()
        self.last_hourly_summary = datetime.now()
        
        # Rest stays the same...
        max_hashes = config.get('max_processed_hashes', 10000)
        self.processed_hashes_cache = LRUCache(capacity=max_hashes)
        
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=config.get('llm_requests_per_minute', 10),
            max_requests_per_hour=config.get('llm_requests_per_hour', 100)
        )
        
        self.llm_client = None
        self.llm_model = config.get('llm_model', 'claude-3-haiku-20240307')
        self._init_llm_client()
        self.context_tracker = None
    
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
    
    def should_send_highlights(self) -> bool:
        """Check if we should send highlights (25 updates)"""
        if len(self.highlights_queue) >= 25:
            return True
        
        # Also send if we have some urgent updates and it's been a while
        has_urgent = any(self._is_urgent(update) for update in self.highlights_queue)
        time_elapsed = (datetime.now() - self.last_batch_time).total_seconds() / 60
        
        if has_urgent and len(self.highlights_queue) >= 10 and time_elapsed >= 15:
            return True
        
        return False
    
    def should_send_hourly_summary(self) -> bool:
        """Check if we should send hourly summary at the top of each hour"""
        now = datetime.now()
        
        # Check if we're at the beginning of a new hour (within first 5 minutes for reliability)
        if now.minute <= 5:
            # Get the hour we should be summarizing (previous hour)
            current_hour = now.replace(minute=0, second=0, microsecond=0)
            last_summary_hour = self.last_hourly_summary.replace(minute=0, second=0, microsecond=0)
            
            # Send if it's a new hour and we haven't sent summary for this hour yet
            if current_hour > last_summary_hour:
                return True
        
        return False
    
    def _is_urgent(self, update: BBUpdate) -> bool:
        """Check if update contains game-critical information"""
        content = f"{update.title} {update.description}".lower()
        return any(keyword in content for keyword in URGENT_KEYWORDS)
    
    async def add_update(self, update: BBUpdate):
        """Add update to both highlights and hourly queues if not already processed"""
        if not await self.processed_hashes_cache.contains(update.content_hash):
            # Add to highlights queue (for 25-update batches)
            self.highlights_queue.append(update)
            
            # Add to hourly queue (for hourly summaries)
            self.hourly_queue.append(update)
            
            # Mark as processed
            await self.processed_hashes_cache.add(update.content_hash)
            
            # Log cache stats periodically
            cache_stats = self.processed_hashes_cache.get_stats()
            if cache_stats['size'] % 1000 == 0:
                logger.info(f"Hash cache stats: {cache_stats}")
    
    async def _can_make_llm_request(self) -> bool:
        """Check if we can make an LLM request without hitting rate limits"""
        stats = self.rate_limiter.get_stats()
        return (stats['requests_this_minute'] < stats['minute_limit'] and 
                stats['requests_this_hour'] < stats['hour_limit'])
    
    async def create_highlights_batch(self) -> List[discord.Embed]:
        """Create highlights embed from current highlights queue"""
        if not self.highlights_queue:
            return []
        
        embeds = []
        
        # Use LLM if available and rate limits allow
        if self.llm_client and await self._can_make_llm_request():
            try:
                embeds = await self._create_llm_highlights_only()
            except Exception as e:
                logger.error(f"LLM highlights failed: {e}")
                embeds = [self._create_pattern_highlights_embed()]
        else:
            reason = "LLM unavailable" if not self.llm_client else "Rate limit reached"
            logger.info(f"Using pattern highlights: {reason}")
            embeds = [self._create_pattern_highlights_embed()]
        
        # Clear highlights queue after processing
        processed_count = len(self.highlights_queue)
        await self.save_queue_state()
        self.highlights_queue.clear()
        self.last_batch_time = datetime.now()
        
        logger.info(f"Created highlights batch from {processed_count} updates")
        return embeds
    
    async def create_hourly_summary(self) -> List[discord.Embed]:
        """Create hourly summary - FORCE LLM VERSION"""
        now = datetime.now()
        
        # Define the hour period
        summary_hour = now.replace(minute=0, second=0, microsecond=0)
        hour_start = summary_hour - timedelta(hours=1)
        hour_end = summary_hour
        
        logger.info(f"Creating hourly summary for {hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}")
        
        # Get updates from database
        if hasattr(self, 'db') and self.db:
            hourly_updates = self.db.get_updates_in_timeframe(hour_start, hour_end)
        else:
            hourly_updates = []
        
        if not hourly_updates:
            logger.info(f"No updates found for hour period {hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}")
            return []
        
        logger.info(f"Creating hourly summary with {len(hourly_updates)} updates from {hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}")
        
        # FORCE LLM STRUCTURED SUMMARY
        if self.llm_client and await self._can_make_llm_request():
            try:
                # Temporarily replace hourly_queue with the timeframe updates
                original_queue = self.hourly_queue.copy()
                self.hourly_queue = hourly_updates
                
                # Force structured summary
                embeds = await self._create_forced_structured_summary("hourly_summary")
                
                # Restore original queue
                self.hourly_queue = original_queue
                
                logger.info(f"Created LLM hourly summary from {len(hourly_updates)} updates")
                return embeds
                
            except Exception as e:
                logger.error(f"LLM hourly summary failed: {e}")
                # Restore original queue on error
                self.hourly_queue = original_queue
        
        # Fallback to pattern only if LLM completely fails
        logger.warning("Using pattern fallback for hourly summary")
        return self._create_pattern_hourly_summary_for_timeframe(hourly_updates, hour_start, hour_end)

    def _create_pattern_hourly_summary_for_timeframe(self, updates: List[BBUpdate], hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Pattern-based hourly summary for timeframe - LAST RESORT"""
        hour_display = hour_end.strftime("%I %p").lstrip('0')
        
        # Get top 8 most important updates
        top_updates = sorted(updates, key=lambda x: self.analyzer.analyze_strategic_importance(x), reverse=True)[:8]
        
        embed = discord.Embed(
            title=f"Chen Bot's House Summary - {hour_display} 🏠",
            description=f"**{len(updates)} updates this hour**",
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        if top_updates:
            summary_text = []
            for update in top_updates:
                time_str = self._extract_correct_time(update)
                title = update.title
                # Clean title
                title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*[-–]\s*', '', title)
                if len(title) > 100:
                    title = title[:97] + "..."
                summary_text.append(f"**{time_str}**: {title}")
            
            embed.add_field(
                name="🎯 Top Moments This Hour",
                value="\n".join(summary_text),
                inline=False
            )
        
        # Add importance rating
        if updates:
            avg_importance = sum(self.analyzer.analyze_strategic_importance(u) for u in updates) // len(updates)
            importance_icons = ["😴", "😴", "📝", "📈", "⭐", "⭐", "🔥", "🔥", "💥", "🚨"]
            importance_icon = importance_icons[min(avg_importance - 1, 9)] if avg_importance >= 1 else "📝"
            
            embed.add_field(
                name="📊 Hour Importance",
                value=f"{importance_icon} **{avg_importance}/10**",
                inline=False
            )
        
        embed.set_footer(text=f"Chen Bot's House Summary • {hour_display}")
        
        return [embed]

    async def _create_forced_structured_summary(self, summary_type: str) -> List[discord.Embed]:
        """Create structured summary with forced contextual format"""
        await self.rate_limiter.wait_if_needed()
        
        # Sort updates chronologically before formatting
        sorted_updates = sorted(self.hourly_queue, key=lambda x: x.pub_date)
        
        # Format updates in chronological order
        formatted_updates = []
        for i, update in enumerate(sorted_updates, 1):
            time_str = self._extract_correct_time(update)
            # Remove leading zero from time
            time_str = time_str.lstrip('0')
            formatted_updates.append(f"{i}. {time_str} - {update.title}")
            if update.description and update.description != update.title:
                # Truncate long descriptions
                desc = update.description[:150] + "..." if len(update.description) > 150 else update.description
                formatted_updates.append(f"   {desc}")
        
        updates_text = "\n".join(formatted_updates)
        
        # Calculate current day (simplified)
        current_day = max(1, (datetime.now().date() - datetime(2025, 7, 8).date()).days + 1)
        
        # Build structured prompt
        prompt = f"""You are a Big Brother superfan analyst creating an hourly summary for Day {current_day}.

NEW UPDATES TO ANALYZE (Day {current_day}) - IN CHRONOLOGICAL ORDER (earliest first):
{updates_text}

Create a comprehensive summary that presents events chronologically as they happened.

Provide your analysis in this EXACT JSON format:

{{
    "headline": "Brief headline summarizing the most important development this hour",
    "strategic_analysis": "Analysis of game moves, alliance discussions, targeting decisions, and strategic positioning. Only include if there are meaningful strategic developments - otherwise use null.",
    "alliance_dynamics": "Analysis of alliance formations, betrayals, trust shifts, and relationship changes. Only include if there are meaningful alliance developments - otherwise use null.",
    "entertainment_highlights": "Funny moments, drama, memorable interactions, and lighthearted content. Only include if there are entertaining moments - otherwise use null.",
    "showmance_updates": "Romance developments, flirting, relationship drama, and intimate moments. Only include if there are romance-related developments - otherwise use null.",
    "house_culture": "Daily routines, traditions, group dynamics, living situations, and house atmosphere. Only include if there are meaningful cultural/social developments - otherwise use null.",
    "key_players": ["List", "of", "houseguests", "who", "were", "central", "to", "this", "hour's", "developments"],
    "overall_importance": 8,
    "importance_explanation": "Brief explanation of why this hour received this importance score (1-10 scale)"
}}

CRITICAL INSTRUCTIONS:
- ONLY include sections where there are actual meaningful developments
- Use null for any section that doesn't have substantial content
- Present events chronologically (earliest to latest)
- Key players should be the houseguests most central to this hour's events
- Overall importance: 1-3 (quiet hour), 4-6 (moderate activity), 7-8 (high drama/strategy), 9-10 (explosive/game-changing)
- Don't force content into sections - be selective and only include what's truly noteworthy
- If a section would be empty or just say "nothing happened", use null instead"""

        # Call LLM
        response = await asyncio.to_thread(
            self.llm_client.messages.create,
            model="claude-3-haiku-20240307",
            max_tokens=1200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        try:
            # Parse JSON response
            analysis_data = self._parse_structured_llm_response(response_text)
            
            # Create structured embed
            embeds = self._create_structured_summary_embed(
                analysis_data, len(self.hourly_queue), summary_type
            )
            
            logger.info(f"Created forced structured {summary_type} summary with {len(self.hourly_queue)} updates")
            return embeds
            
        except Exception as e:
            logger.error(f"Failed to parse structured response: {e}")
            logger.error(f"Raw response: {response_text}")
            # Fallback to enhanced pattern-based
            return self._create_enhanced_pattern_hourly_summary()

    async def _create_llm_highlights_only(self) -> List[discord.Embed]:
        """Create just highlights using LLM - CHRONOLOGICAL ORDER"""
        await self.rate_limiter.wait_if_needed()
        
        # Sort updates chronologically BEFORE sending to LLM
        sorted_updates = sorted(self.highlights_queue, key=lambda x: x.pub_date)
        
        # Prepare update data in chronological order
        updates_text = "\n".join([
            f"{self._extract_correct_time(u)} - {re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*[-–]\s*', '', u.title)}"
            for u in sorted_updates
        ])
        
        prompt = f"""You are a Big Brother superfan curating the MOST IMPORTANT moments from these {len(self.highlights_queue)} recent updates.
    
    UPDATES IN CHRONOLOGICAL ORDER (earliest first):
    {updates_text}
    
    Select 6-10 updates that are TRUE HIGHLIGHTS - moments that stand out as particularly important, dramatic, funny, or game-changing.
    
    HIGHLIGHT-WORTHY updates include:
    - Competition wins (HOH, POV, etc.)
    - Major strategic moves or betrayals
    - Dramatic fights or confrontations  
    - Romantic moments (first kiss, breakup, etc.)
    - Hilarious or memorable incidents
    - Game-changing twists revealed
    - Eviction results or surprise votes
    - Alliance formations or breaks
    
    For each selected update, provide them in CHRONOLOGICAL ORDER (earliest to latest):
    
    {{
        "highlights": [
            {{
                "time": "exact time from update",
                "title": "exact title from update BUT REMOVE the time if it appears at the beginning",
                "importance_emoji": "🔥 for high, ⭐ for medium, 📝 for low",
                "reason": "ONLY add this field if the title needs crucial context that isn't obvious. Keep it VERY brief (under 10 words). Most updates won't need this."
            }}
        ]
    }}
    
    CRITICAL: Present the selected highlights in CHRONOLOGICAL ORDER from earliest to latest time.
    Be selective - these should be the updates that a superfan would want to know about from this batch."""
    
        response = await asyncio.to_thread(
            self.llm_client.messages.create,
            model=self.llm_model,
            max_tokens=800,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse and create embed
        try:
            highlights_data = self._parse_llm_response(response.content[0].text)
            
            if not highlights_data.get('highlights'):
                logger.warning("No highlights in LLM response, using pattern fallback")
                return [self._create_pattern_highlights_embed()]
            
            # SORT THE HIGHLIGHTS BY TIME after parsing (backup enforcement)
            highlights = highlights_data['highlights']
            
            # Extract time for sorting
            def extract_time_for_sorting(highlight):
                time_str = highlight.get('time', '00:00 AM')
                try:
                    # Convert to 24-hour format for proper sorting
                    time_obj = datetime.strptime(time_str, '%I:%M %p')
                    return time_obj.time()
                except:
                    return datetime.strptime('00:00', '%H:%M').time()
            
            # Sort highlights chronologically
            highlights.sort(key=extract_time_for_sorting)
            
            embed = discord.Embed(
                title="📹 Feed Highlights - What Just Happened",
                description=f"Key moments from the last {len(self.highlights_queue)} updates",
                color=0xe74c3c,
                timestamp=datetime.now()
            )
            
            for highlight in highlights[:10]:
                title = highlight.get('title', 'Update')
                title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*-\s*', '', title)
                
                if highlight.get('reason') and highlight['reason'].strip():
                    embed.add_field(
                        name=f"{highlight.get('importance_emoji', '📝')} {highlight.get('time', 'Time')}",
                        value=f"{title}\n*{highlight['reason']}*",
                        inline=False
                    )
                else:
                    embed.add_field(
                        name=f"{highlight.get('importance_emoji', '📝')} {highlight.get('time', 'Time')}",
                        value=title,
                        inline=False
                    )
            
            embed.set_footer(text=f"Highlights • {len(self.highlights_queue)} updates processed")
            return [embed]
            
        except Exception as e:
            logger.error(f"Failed to parse highlights response: {e}")
            return [self._create_pattern_highlights_embed()]
        
    async def save_queue_state(self):
        """Save current queue state to database"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                logger.debug("No database URL for queue persistence")
                return
                
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(database_url, cursor_factory=psycopg2.extras.RealDictCursor)
            cursor = conn.cursor()
            
            # Save highlights queue
            highlights_data = {
                'updates': [
                    {
                        'title': update.title,
                        'description': update.description,
                        'link': update.link,
                        'pub_date': update.pub_date.isoformat(),
                        'content_hash': update.content_hash,
                        'author': update.author
                    }
                    for update in self.highlights_queue
                ]
            }
            
            cursor.execute("""
                INSERT INTO summary_checkpoints 
                (summary_type, queue_state, queue_size, last_summary_time)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (summary_type) 
                DO UPDATE SET 
                    queue_state = EXCLUDED.queue_state,
                    queue_size = EXCLUDED.queue_size,
                    last_summary_time = EXCLUDED.last_summary_time,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                'highlights',
                json.dumps(highlights_data),
                len(self.highlights_queue),
                self.last_batch_time.isoformat()
            ))
            
            # Save hourly queue
            hourly_data = {
                'updates': [
                    {
                        'title': update.title,
                        'description': update.description,
                        'link': update.link,
                        'pub_date': update.pub_date.isoformat(),
                        'content_hash': update.content_hash,
                        'author': update.author
                    }
                    for update in self.hourly_queue
                ]
            }
            
            cursor.execute("""
                INSERT INTO summary_checkpoints 
                (summary_type, queue_state, queue_size, last_summary_time)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (summary_type) 
                DO UPDATE SET 
                    queue_state = EXCLUDED.queue_state,
                    queue_size = EXCLUDED.queue_size,
                    last_summary_time = EXCLUDED.last_summary_time,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                'hourly',
                json.dumps(hourly_data),
                len(self.hourly_queue),
                self.last_hourly_summary.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Saved queue state: {len(self.highlights_queue)} highlights, {len(self.hourly_queue)} hourly")
            
        except Exception as e:
            logger.error(f"Error saving queue state: {e}")
    
    async def _create_llm_hourly_summary_fallback(self) -> List[discord.Embed]:
        """Create narrative LLM hourly summary as fallback"""
        try:
            await self.rate_limiter.wait_if_needed()
            
            # Prepare update data
            updates_data = []
            for update in self.hourly_queue:
                time_str = self._extract_correct_time(update)
                updates_data.append({
                    'time': time_str,
                    'title': update.title,
                    'description': update.description[:200] if update.description != update.title else ""
                })
            
            updates_text = "\n".join([
                f"{u['time']} - {u['title']}" + (f" ({u['description']})" if u['description'] else "")
                for u in updates_data
            ])
            
            prompt = f"""You are a Big Brother superfan creating a comprehensive HOURLY SUMMARY.

Analyze these {len(self.hourly_queue)} updates from the past hour:

{updates_text}

Provide a thorough analysis covering both strategic and social aspects:

{{
    "headline": "Compelling headline that captures the hour's most significant development",
    "summary": "4-5 sentence summary of the hour's key developments and overall narrative",
    "strategic_analysis": "Strategic implications - targets, power shifts, competition positioning, voting plans",
    "social_dynamics": "Alliance formations, shifts, trust levels, betrayals, strategic partnerships",
    "entertainment_highlights": "Funny moments, drama, memorable quotes, personality clashes",
    "key_players": ["houseguests", "involved", "in", "major", "moments"],
    "game_phase": "one of: early_game, jury_phase, final_weeks, finale_night",
    "strategic_importance": 7,
    "house_culture": "Inside jokes, routines, traditions, or quirky moments from this hour",
    "relationship_updates": "Showmance developments, romantic connections, relationship changes"
}}

This is an HOURLY DIGEST so be comprehensive and analytical but not too wordy."""

            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.llm_model,
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response and create embed
            analysis = self._parse_llm_response(response.content[0].text)
            return self._create_hourly_summary_embed(analysis, len(self.hourly_queue))
            
        except Exception as e:
            logger.error(f"LLM hourly summary fallback failed: {e}")
            return self._create_enhanced_pattern_hourly_summary()

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

    def _parse_structured_llm_response(self, response_text: str) -> dict:
        """Parse structured LLM response with better error handling"""
        try:
            # Clean the response text first
            cleaned_text = response_text.strip()
            
            # Try to extract JSON - be more flexible with finding it
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = cleaned_text[json_start:json_end]
                
                # Clean up common JSON issues
                json_text = json_text.replace('\n', ' ')  # Remove newlines
                json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
                json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas in arrays
                
                try:
                    data = json.loads(json_text)
                    
                    # Validate and fix required fields
                    if not isinstance(data, dict):
                        raise ValueError("Response is not a dictionary")
                    
                    # Ensure required fields exist
                    required_fields = {
                        'headline': 'Big Brother Activity Continues',
                        'key_players': [],
                        'overall_importance': 5,
                        'importance_explanation': 'Standard house activity'
                    }
                    
                    for field, default_value in required_fields.items():
                        if field not in data or data[field] is None:
                            data[field] = default_value
                    
                    # Clean up null values in sections
                    section_fields = [
                        'strategic_analysis', 'alliance_dynamics', 
                        'entertainment_highlights', 'showmance_updates', 'house_culture'
                    ]
                    
                    for field in section_fields:
                        if field in data and (data[field] is None or str(data[field]).lower() in ['null', 'none', 'nothing']):
                            data[field] = None
                    
                    # Ensure key_players is a list
                    if not isinstance(data.get('key_players'), list):
                        data['key_players'] = []
                    
                    # Ensure overall_importance is an integer between 1-10
                    try:
                        importance = int(data.get('overall_importance', 5))
                        data['overall_importance'] = max(1, min(10, importance))
                    except (ValueError, TypeError):
                        data['overall_importance'] = 5
                    
                    logger.info("Successfully parsed structured LLM response")
                    return data
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    logger.error(f"Problematic JSON: {json_text[:200]}...")
                    raise ValueError("Invalid JSON structure")
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.warning(f"Structured JSON parsing failed: {e}, creating fallback")
            return self._create_fallback_structured_response(response_text)

    def _create_fallback_structured_response(self, response_text: str) -> dict:
        """Create fallback structured response when JSON parsing fails"""
        # Extract houseguest names from response
        names = re.findall(r'\b[A-Z][a-z]+\b', response_text)
        key_players = [name for name in names if name not in EXCLUDE_WORDS][:5]
        
        return {
            "headline": "Big Brother Activity Continues",
            "strategic_analysis": None,
            "alliance_dynamics": None,
            "entertainment_highlights": None,
            "showmance_updates": None,
            "house_culture": None,
            "key_players": key_players,
            "overall_importance": 5,
            "importance_explanation": "Standard house activity"
        }

    def _create_structured_summary_embed(self, analysis_data: dict, update_count: int, summary_type: str) -> List[discord.Embed]:
        """Create structured summary embed that only includes sections with content"""
        pacific_tz = pytz.timezone('US/Pacific')
        current_hour_pacific = datetime.now(pacific_tz).strftime("%I %p").lstrip('0')
        current_day = max(1, (datetime.now().date() - datetime(2025, 7, 1).date()).days + 1)
        
        # Determine embed color based on importance
        importance = analysis_data.get('overall_importance', 5)
        if importance >= 9:
            color = 0xff1744  # Red for explosive hours
        elif importance >= 7:
            color = 0xff9800  # Orange for high activity
        elif importance >= 4:
            color = 0x3498db  # Blue for moderate activity
        else:
            color = 0x95a5a6  # Gray for quiet hours
        
        # Create main embed
        # Create main embed with PACIFIC TIME
        if summary_type == "hourly_summary":
            title = f"Chen Bot's House Summary - {current_hour_pacific} 🏠"
            description = f""
            footer_text = f"Chen Bot's House Summary • {current_hour_pacific}"
        else:
            title = f"Chen Bot's House Summary - {current_hour_pacific} 🏠"
            description = f""
            footer_text = "Chen Bot's Summary"
        
        embed = discord.Embed(
            title=title,
            description=description,
            color=color,
            timestamp=datetime.now()
        )
        
        # Add headline as first field
        headline = analysis_data.get('headline', 'Big Brother Update')
        embed.add_field(
            name="📰 Headline",
            value=headline,
            inline=False
        )
        
        # Add structured sections ONLY if they have content
        sections = [
            ("🎯 Strategic Analysis", analysis_data.get('strategic_analysis')),
            ("🤝 Alliance Dynamics", analysis_data.get('alliance_dynamics')),
            ("🎬 Entertainment Highlights", analysis_data.get('entertainment_highlights')),
            ("💕 Showmance Updates", analysis_data.get('showmance_updates')),
            ("🏠 House Culture", analysis_data.get('house_culture'))
        ]
        
        for section_name, content in sections:
            # Only add section if it has actual content (not null, not empty, not "null" string)
            if content and content.strip() and content.lower() not in ['null', 'none', 'nothing']:
                # Split long content if needed
                if len(content) > 1000:
                    content = content[:997] + "..."
                embed.add_field(
                    name=section_name,
                    value=content,
                    inline=False
                )
        
        # ALWAYS add key players (required)
        key_players = analysis_data.get('key_players', [])
        if key_players:
            # Format key players nicely
            if len(key_players) <= 6:
                players_text = " • ".join([f"**{player}**" for player in key_players])
            else:
                players_text = " • ".join([f"**{player}**" for player in key_players[:6]]) + f" • +{len(key_players)-6} more"
        else:
            players_text = "No specific houseguests highlighted"
        
        embed.add_field(
            name="⭐ Key Players",
            value=players_text,
            inline=False
        )
        
        # ALWAYS add importance rating (required)
        custom_emoji = "<:chunky:1392638440582942974>"  # Replace this with your actual custom emoji
    
        # Create multiple icons based on the rating (like your second image)
        icon_count = min(importance, 10)  # Cap at 10 icons max
        importance_text = custom_emoji * icon_count + f" **{importance}/10**"
        
        explanation = analysis_data.get('importance_explanation', '')
        if explanation:
            importance_text += f"\n*{explanation}*"
        
        embed.add_field(
            name="📊 Overall Importance",
            value=importance_text,
            inline=False
        )
        
        # Set footer
        embed.set_footer(text=footer_text)
        
        return [embed]

    def _create_enhanced_pattern_hourly_summary(self) -> List[discord.Embed]:
        """Enhanced pattern-based summary as fallback when LLM unavailable"""
        logger.info("Creating enhanced pattern-based hourly summary")
        
        # Group updates by categories
        categories = defaultdict(list)
        for update in self.hourly_queue:
            update_categories = self.analyzer.categorize_update(update)
            for category in update_categories:
                categories[category].append(update)
        
        current_hour = datetime.now().strftime("%I %p").lstrip('0')
        
        custom_emoji = "<:takingnotes:916186747770130443>"
        title = f"Chen Bot's House Summary - {current_hour}"
        
        embed = discord.Embed(
            title=title,
            description=f"**{len(self.hourly_queue)} updates this hour** •{custom_emoji}",
            color=0x95a5a6,  # Gray for pattern-based
            timestamp=datetime.now()
        )
        
        # Add headline
        if self.hourly_queue:
            # Use the most important update as headline basis
            top_update = max(self.hourly_queue, key=lambda x: self.analyzer.analyze_strategic_importance(x))
            headline = self._create_pattern_headline(top_update, len(self.hourly_queue))
            embed.add_field(
                name="📰 Headline",
                value=headline,
                inline=False
            )
        
        # Create narrative summaries for each category (only if they have content)
        section_mapping = {
            "🎯 Strategy": "🎯 Strategic Analysis",
            "🤝 Alliance": "🤝 Alliance Dynamics", 
            "🎬 Entertainment": "🎬 Entertainment Highlights",
            "💕 Romance": "💕 Showmance Updates",
            "📝 General": "🏠 House Culture"
        }
        
        for category, updates in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
            if updates and len(updates) > 0:
                narrative_summary = self._create_category_narrative(category, updates)
                if narrative_summary and narrative_summary.strip():  # Only add if we have content
                    section_name = section_mapping.get(category, category)
                    embed.add_field(
                        name=section_name,
                        value=narrative_summary,
                        inline=False
                    )
        
        # ALWAYS add key players
        all_houseguests = set()
        for update in self.hourly_queue:
            hgs = self.analyzer.extract_houseguests(update.title + " " + update.description)
            all_houseguests.update(hgs[:3])  # Limit per update
        
        if all_houseguests:
            players_text = " • ".join([f"**{hg}**" for hg in list(all_houseguests)[:6]])
            if len(all_houseguests) > 6:
                players_text += f" • +{len(all_houseguests)-6} more"
        else:
            players_text = "No specific houseguests highlighted"
        
        embed.add_field(
            name="⭐ Key Players",
            value=players_text,
            inline=False
        )
        
        # ALWAYS add importance rating
        total_importance = sum(self.analyzer.analyze_strategic_importance(u) for u in self.hourly_queue)
        avg_importance = int(total_importance / len(self.hourly_queue)) if self.hourly_queue else 1
        
        importance_icons = ["😴", "😴", "📝", "📈", "⭐", "⭐", "🔥", "🔥", "💥", "🚨"]
        importance_icon = importance_icons[min(avg_importance - 1, 9)] if avg_importance >= 1 else "📝"
        
        if avg_importance >= 7:
            activity_desc = "High drama and strategic activity"
        elif avg_importance >= 5:
            activity_desc = "Moderate activity with notable moments"
        elif avg_importance >= 3:
            activity_desc = "Steady house activity"
        else:
            activity_desc = "Quiet hour with routine activities"
        
        embed.add_field(
            name="📊 Overall Importance",
            value=f"{importance_icon} **{avg_importance}/10**\n*{activity_desc}*",
            inline=False
        )
        
        embed.set_footer(text=f"Chen Bot's House Summary • {current_hour} • Enhanced Pattern Analysis")
        
        return [embed]

    def _create_pattern_headline(self, top_update: BBUpdate, total_updates: int) -> str:
        """Create a headline from the most important update"""
        # Clean the title
        title = top_update.title
        title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*[-–]\s*', '', title)
        title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*[-–]\s*', '', title)
        
        # Extract key themes
        content = title.lower()
        if any(word in content for word in ['winner', 'wins', 'victory', 'champion']):
            return f"Victory and Competition Results Highlight Hour"
        elif any(word in content for word in ['finale', 'final', 'crown']):
            return f"Finale Activities Dominate House Activity"
        elif any(word in content for word in ['vote', 'voting', 'jury']):
            return f"Voting Dynamics and Strategic Decisions Unfold"
        elif any(word in content for word in ['alliance', 'strategy', 'target']):
            return f"Strategic Gameplay and Alliance Activity"
        elif any(word in content for word in ['drama', 'fight', 'argument']):
            return f"House Drama and Interpersonal Tensions"
        else:
            return f"Big Brother House Activity Continues"

    def _create_category_narrative(self, category: str, updates: List) -> str:
        """Create narrative summary for a category"""
        if not updates:
            return ""
        
        # Get top 3 most important updates in this category
        top_updates = sorted(updates, key=lambda x: self.analyzer.analyze_strategic_importance(x), reverse=True)[:3]
        
        narratives = []
        for update in top_updates:
            time_str = self._extract_correct_time(update)
            # Clean title
            title = update.title
            title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*[-–]\s*', '', title)
            
            # Truncate if too long
            if len(title) > 120:
                title = title[:117] + "..."
            
            narratives.append(f"**{time_str}**: {title}")
        
        return "\n".join(narratives)

    def _create_pattern_highlights_embed(self) -> discord.Embed:
        """Create highlights embed using pattern matching when LLM unavailable"""
        try:
            # Sort updates by importance score
            updates_with_importance = [
                (update, self.analyzer.analyze_strategic_importance(update))
                for update in self.highlights_queue
            ]
            updates_with_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Select top 6-10 most important updates
            selected_updates = updates_with_importance[:min(10, len(updates_with_importance))]
            
            embed = discord.Embed(
                title="🎯 Feed Highlights - What Just Happened",
                description=f"Key moments from this batch ({len(selected_updates)} of {len(self.highlights_queue)} updates)",
                color=0x95a5a6,  # Gray for pattern-based
                timestamp=datetime.now()
            )
            
            # Add selected highlights
            for i, (update, importance) in enumerate(selected_updates, 1):
                # Extract correct time from content rather than pub_date
                time_str = self._extract_correct_time(update)
                
                # Clean the title - remove time if it appears at the beginning
                title = update.title
                title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*-\s*', '', title)
                
                # Only truncate if extremely long
                if len(title) > 1000:
                    title = title[:997] + "..."
                
                # Add importance indicators
                importance_emoji = "🔥" if importance >= 7 else "⭐" if importance >= 5 else "📝"
                
                embed.add_field(
                    name=f"{importance_emoji} {time_str}",
                    value=title,
                    inline=False
                )
            
            embed.set_footer(text=f"Pattern Analysis • {len(selected_updates)} key moments selected")
            
            return embed
            
        except Exception as e:
            logger.error(f"Pattern highlights creation failed: {e}")
            # Return a basic embed if everything fails
            return discord.Embed(
                title="🎯 Feed Highlights - What Just Happened",
                description=f"Recent updates from the feed ({len(self.highlights_queue)} updates)",
                color=0x95a5a6
            )

    def _create_hourly_summary_embed(self, analysis: dict, update_count: int) -> List[discord.Embed]:
        """Create hourly summary embed"""
        current_hour = datetime.now().strftime("%I %p")
        
        embed = discord.Embed(
            title=f"📊 Hourly Digest - {current_hour}",
            description=f"**{update_count} updates this hour** • {analysis.get('headline', 'Hourly Summary')}\n\n{analysis.get('summary', 'Summary not available')}",
            color=0x9b59b6,  # Purple for hourly summaries
            timestamp=datetime.now()
        )
        
        # Add comprehensive sections
        if analysis.get('strategic_analysis'):
            embed.add_field(
                name="🎯 Strategic Developments",
                value=analysis['strategic_analysis'],
                inline=False
            )
        
        if analysis.get('social_dynamics'):
            embed.add_field(
                name="🤝 Alliance & Social Dynamics",
                value=analysis['social_dynamics'],
                inline=False
            )
        
        if analysis.get('entertainment_highlights'):
            embed.add_field(
                name="🎬 Entertainment & Drama",
                value=analysis['entertainment_highlights'],
                inline=False
            )
        
        if analysis.get('relationship_updates'):
            embed.add_field(
                name="💕 Showmance Updates",
                value=analysis['relationship_updates'],
                inline=False
            )
        
        if analysis.get('house_culture'):
            embed.add_field(
                name="🏠 House Culture",
                value=analysis['house_culture'],
                inline=False
            )
        
        # Add key players and importance
        if analysis.get('key_players'):
            players = analysis['key_players'][:8]
            embed.add_field(
                name="⭐ Key Players This Hour",
                value=" • ".join(players),
                inline=False
            )
        
        importance = analysis.get('strategic_importance', 5)
        importance_bar = "🔥" * min(importance, 10)
        embed.add_field(
            name="📈 Hour Importance",
            value=f"{importance_bar} {importance}/10",
            inline=True
        )
        
        embed.set_footer(text=f"Hourly Digest • {current_hour} • Chen Bot Analysis")
        
        return [embed]

    def _extract_correct_time(self, update: BBUpdate) -> str:
        """Extract the correct time from the update content rather than pub_date"""
        # Look for time patterns in the title like "07:56 PM PST"
        time_pattern = r'(\d{1,2}:\d{2})\s*(PM|AM)\s*PST'
        
        # First try the title
        match = re.search(time_pattern, update.title)
        if match:
            time_str = match.group(1)
            ampm = match.group(2)
            return f"{time_str} {ampm}"
        
        # Then try the description
        match = re.search(time_pattern, update.description)
        if match:
            time_str = match.group(1)
            ampm = match.group(2)
            return f"{time_str} {ampm}"
        
        # Fallback to pub_date if no time found in content
        return update.pub_date.strftime("%I:%M %p")
    
    def get_rate_limit_stats(self) -> Dict[str, int]:
        """Get current rate limiting statistics"""
        return self.rate_limiter.get_stats()

    # Daily recap methods
    async def create_daily_recap(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
        """Create a comprehensive daily recap from all updates"""
        if not updates:
            return []
        
        logger.info(f"Creating daily recap for {len(updates)} updates")
        
        # Check if we need to chunk the updates (too many for single LLM call)
        if len(updates) > 50:
            return await self._create_chunked_daily_recap(updates, day_number)
        else:
            return await self._create_single_daily_recap(updates, day_number)
    
    async def _create_single_daily_recap(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
        """Create daily recap from manageable number of updates"""
        if not self.llm_client or not await self._can_make_llm_request():
            return self._create_pattern_daily_recap(updates, day_number)
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            # Prepare chronological update data
            updates_data = []
            for update in updates:
                time_str = self._extract_correct_time(update)
                updates_data.append({
                    'time': time_str,
                    'title': update.title,
                    'description': update.description[:150] if update.description != update.title else ""
                })
            
            # Create daily recap prompt
            prompt = f"""You are the ultimate Big Brother superfan creating a comprehensive DAILY RECAP for Day {day_number}.

Analyze these {len(updates)} updates from the entire day (chronological order):

{chr(10).join([f"{u['time']} - {u['title']}" + (f" ({u['description']})" if u['description'] else "") for u in updates_data])}

Create a comprehensive daily recap that tells the story of Day {day_number}:

{{
    "headline": "Day {day_number} headline capturing the most significant storyline",
    "summary": "4-5 sentence summary of the day's key developments and overall narrative",
    "strategic_analysis": "Strategic developments - key conversations, alliance shifts, target changes, power dynamics",
    "social_dynamics": "Alliance formations, trust shifts, betrayals, strategic partnerships throughout the day",
    "entertainment_highlights": "Memorable moments, drama, funny interactions, personality conflicts from the day",
    "key_players": ["houseguests", "who", "were", "central", "to", "the", "day"],
    "strategic_importance": 8,
    "house_culture": "Daily routines, inside jokes, traditions, or cultural moments that defined the day",
    "relationship_updates": "Showmance developments, romantic moments, or relationship changes",
    "day_timeline": "Brief chronological overview of how the day unfolded from morning to night"
}}

Focus on creating a comprehensive daily story that captures the full arc of Day {day_number}. This should read like a daily diary entry for superfans."""

            # Make LLM request
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.llm_model,
                max_tokens=1500,  # Longer for daily recap
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            analysis = self._parse_llm_response(response.content[0].text)
            
            # Create daily recap embed
            return self._create_daily_recap_embed(analysis, day_number, len(updates))
            
        except Exception as e:
            logger.error(f"Daily recap LLM failed: {e}")
            return self._create_pattern_daily_recap(updates, day_number)
    
    async def _create_chunked_daily_recap(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
        """Create daily recap by summarizing in chunks first"""
        chunk_size = 25
        chunks = [updates[i:i + chunk_size] for i in range(0, len(updates), chunk_size)]
        
        logger.info(f"Creating chunked daily recap: {len(chunks)} chunks of ~{chunk_size} updates each")
        
        # First pass: summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            if not await self._can_make_llm_request():
                # Fall back to pattern analysis if rate limited
                summary = self._summarize_chunk_pattern(chunk, i + 1)
            else:
                summary = await self._summarize_chunk_llm(chunk, i + 1)
            
            chunk_summaries.append(summary)
        
        # Second pass: create final daily recap from summaries
        return await self._create_final_daily_recap(chunk_summaries, day_number, len(updates))
    
    async def _summarize_chunk_llm(self, chunk: List[BBUpdate], chunk_number: int) -> str:
        """Summarize a chunk of updates using LLM"""
        try:
            await self.rate_limiter.wait_if_needed()
            
            updates_text = "\n".join([
                f"{self._extract_correct_time(u)} - {u.title}"
                for u in chunk
            ])
            
            prompt = f"""Summarize this chunk of Big Brother updates (Chunk {chunk_number}):

{updates_text}

Provide a 2-3 sentence summary capturing the key strategic and social developments in this time period. Focus on the most important events and conversations."""

            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.llm_model,
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return f"**Chunk {chunk_number}**: {response.content[0].text}"
            
        except Exception as e:
            logger.error(f"Chunk summarization failed: {e}")
            return self._summarize_chunk_pattern(chunk, chunk_number)
    
    def _summarize_chunk_pattern(self, chunk: List[BBUpdate], chunk_number: int) -> str:
        """Summarize a chunk using pattern matching"""
        # Group by importance and pick top updates
        important_updates = sorted(
            chunk, 
            key=lambda x: self.analyzer.analyze_strategic_importance(x), 
            reverse=True
        )[:3]
        
        titles = [u.title[:60] + "..." if len(u.title) > 60 else u.title for u in important_updates]
        return f"**Chunk {chunk_number}**: {' • '.join(titles)}"
    
    async def _create_final_daily_recap(self, chunk_summaries: List[str], day_number: int, total_updates: int) -> List[discord.Embed]:
        """Create final daily recap from chunk summaries"""
        if not self.llm_client or not await self._can_make_llm_request():
            return self._create_pattern_daily_recap_from_summaries(chunk_summaries, day_number, total_updates)
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            summaries_text = "\n\n".join(chunk_summaries)
            
            prompt = f"""Create a comprehensive Day {day_number} recap from these chunk summaries:

{summaries_text}

Create a daily recap that tells the story of Day {day_number} from these summaries:

{{
    "headline": "Day {day_number} headline capturing the most significant storyline",
    "summary": "4-5 sentence summary of the day's key developments and overall narrative",
    "strategic_analysis": "Strategic developments throughout the day",
    "social_dynamics": "Alliance and relationship dynamics",
    "entertainment_highlights": "Memorable moments and drama",
    "key_players": ["main", "houseguests", "from", "the", "day"],
    "strategic_importance": 8,
    "house_culture": "Daily culture and routine moments",
    "relationship_updates": "Showmance and relationship developments",
    "day_timeline": "How the day unfolded chronologically"
}}

Focus on creating a cohesive daily story from these summaries."""

            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.llm_model,
                max_tokens=1200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis = self._parse_llm_response(response.content[0].text)
            return self._create_daily_recap_embed(analysis, day_number, total_updates)
            
        except Exception as e:
            logger.error(f"Final daily recap failed: {e}")
            return self._create_pattern_daily_recap_from_summaries(chunk_summaries, day_number, total_updates)
    
    def _create_daily_recap_embed(self, analysis: dict, day_number: int, update_count: int) -> List[discord.Embed]:
        """Create the daily recap embed"""
        embed = discord.Embed(
            title=f"📅 Day {day_number} Recap",
            description=f"**{update_count} updates** • {analysis.get('headline', 'Daily Recap')}\n\n{analysis.get('summary', 'Daily summary not available')}",
            color=0x9b59b6,  # Purple for daily recaps
            timestamp=datetime.now()
        )
        
        # Add timeline if available
        if analysis.get('day_timeline'):
            embed.add_field(
                name="📖 Day Timeline",
                value=analysis['day_timeline'],
                inline=False
            )
        
        # Add strategic analysis
        if analysis.get('strategic_analysis'):
            embed.add_field(
                name="🎯 Strategic Developments",
                value=analysis['strategic_analysis'],
                inline=False
            )
        
        # Add other sections
        if analysis.get('social_dynamics'):
            embed.add_field(
                name="🤝 Alliance Dynamics",
                value=analysis['social_dynamics'],
                inline=False
            )
        
        if analysis.get('entertainment_highlights'):
            embed.add_field(
                name="🎬 Day Highlights",
                value=analysis['entertainment_highlights'],
                inline=False
            )
        
        if analysis.get('relationship_updates'):
            embed.add_field(
                name="💕 Showmance Updates",
                value=analysis['relationship_updates'],
                inline=False
            )
        
        if analysis.get('house_culture'):
            embed.add_field(
                name="🏠 House Culture",
                value=analysis['house_culture'],
                inline=False
            )
        
        # Add key players
        if analysis.get('key_players'):
            players = analysis['key_players'][:8]
            embed.add_field(
                name="⭐ Key Players of the Day",
                value=" • ".join(players),
                inline=False
            )
        
        # Add importance
        importance = analysis.get('strategic_importance', 5)
        importance_bar = "🔥" * min(importance, 10)
        embed.add_field(
            name="📊 Day Importance",
            value=f"{importance_bar} {importance}/10",
            inline=True
        )
        
        embed.set_footer(text=f"Daily Recap • Day {day_number} • BB Superfan AI")
        
        return [embed]
    
    def _create_pattern_daily_recap(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
        """Create daily recap using pattern matching"""
        # Analyze updates by importance
        important_updates = sorted(
            updates, 
            key=lambda x: self.analyzer.analyze_strategic_importance(x), 
            reverse=True
        )[:10]  # Top 10 most important
        
        # Group by categories
        categories = defaultdict(list)
        for update in important_updates:
            update_categories = self.analyzer.categorize_update(update)
            for category in update_categories:
                categories[category].append(update)
        
        embed = discord.Embed(
            title=f"📅 Day {day_number} Recap",
            description=f"**{len(updates)} updates** • Pattern-based daily summary",
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        # Add top moments
        top_moments = []
        for update in important_updates[:5]:
            time_str = self._extract_correct_time(update)
            title = update.title[:80] + "..." if len(update.title) > 80 else update.title
            top_moments.append(f"**{time_str}**: {title}")
        
        if top_moments:
            embed.add_field(
                name="🎯 Top Moments of the Day",
                value="\n".join(top_moments),
                inline=False
            )
        
        # Add categories
        for category, cat_updates in categories.items():
            if cat_updates:
                summary = f"{len(cat_updates)} updates in this category"
                embed.add_field(
                    name=f"{category}",
                    value=summary,
                    inline=True
                )
        
        embed.set_footer(text=f"Daily Recap • Day {day_number} • Pattern Analysis")
        
        return [embed]
    
    def _create_pattern_daily_recap_from_summaries(self, summaries: List[str], day_number: int, total_updates: int) -> List[discord.Embed]:
        """Create pattern-based daily recap from summaries"""
        embed = discord.Embed(
            title=f"📅 Day {day_number} Recap",
            description=f"**{total_updates} updates** • Chunked summary analysis",
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        # Add chunk summaries
        summary_text = "\n\n".join(summaries)
        if len(summary_text) > 1000:
            summary_text = summary_text[:997] + "..."
        
        embed.add_field(
            name="📝 Daily Summary",
            value=summary_text,
            inline=False
        )
        
        embed.set_footer(text=f"Daily Recap • Day {day_number} • Chunked Analysis")
        
        return [embed]

# Add these methods to your UpdateBatcher class

async def _create_llm_highlights_only(self) -> List[discord.Embed]:
    """Create just highlights using LLM"""
    await self.rate_limiter.wait_if_needed()
    
    # Prepare update data
    updates_text = "\n".join([
        f"{self._extract_correct_time(u)} - {u.title}"
        for u in self.highlights_queue
    ])
    
    prompt = f"""You are a Big Brother superfan curating the MOST IMPORTANT moments from these {len(self.highlights_queue)} recent updates.

{updates_text}

Select 6-10 updates that are TRUE HIGHLIGHTS - moments that stand out as particularly important, dramatic, funny, or game-changing.

HIGHLIGHT-WORTHY updates include:
- Competition wins (HOH, POV, etc.)
- Major strategic moves or betrayals
- Dramatic fights or confrontations  
- Romantic moments (first kiss, breakup, etc.)
- Hilarious or memorable incidents
- Game-changing twists revealed
- Eviction results or surprise votes
- Alliance formations or breaks

For each selected update, provide:
{{
    "highlights": [
        {{
            "time": "exact time from update",
            "title": "exact title from update BUT REMOVE the time if it appears at the beginning",
            "importance_emoji": "🔥 for high, ⭐ for medium, 📝 for low",
            "reason": "ONLY add this field if the title needs crucial context that isn't obvious. Keep it VERY brief (under 10 words). Most updates won't need this."
        }}
    ]
}}

Be selective - these should be the updates that a superfan would want to know about from this batch."""

    response = await asyncio.to_thread(
        self.llm_client.messages.create,
        model=self.llm_model,
        max_tokens=800,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse and create embed
    try:
        highlights_data = self._parse_llm_response(response.content[0].text)
        
        if not highlights_data.get('highlights'):
            logger.warning("No highlights in LLM response, using pattern fallback")
            return [self._create_pattern_highlights_embed()]
        
        embed = discord.Embed(
            title="🎯 Feed Highlights - What Just Happened",
            description=f"Key moments from the last {len(self.highlights_queue)} updates",
            color=0xe74c3c,
            timestamp=datetime.now()
        )
        
        for highlight in highlights_data['highlights'][:10]:
            title = highlight.get('title', 'Update')
            title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*-\s*', '', title)
            
            if highlight.get('reason') and highlight['reason'].strip():
                embed.add_field(
                    name=f"{highlight.get('importance_emoji', '📝')} {highlight.get('time', 'Time')}",
                    value=f"{title}\n*{highlight['reason']}*",
                    inline=False
                )
            else:
                embed.add_field(
                    name=f"{highlight.get('importance_emoji', '📝')} {highlight.get('time', 'Time')}",
                    value=title,
                    inline=False
                )
        
        embed.set_footer(text=f"Highlights • {len(self.highlights_queue)} updates processed")
        return [embed]
        
    except Exception as e:
        logger.error(f"Failed to parse highlights response: {e}")
        return [self._create_pattern_highlights_embed()]

def _create_pattern_highlights_embed(self) -> discord.Embed:
    """Create highlights embed using pattern matching when LLM unavailable"""
    try:
        # Sort updates by importance score
        updates_with_importance = [
            (update, self.analyzer.analyze_strategic_importance(update))
            for update in self.highlights_queue
        ]
        updates_with_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 6-10 most important updates
        selected_updates = updates_with_importance[:min(10, len(updates_with_importance))]
        
        embed = discord.Embed(
            title="🎯 Feed Highlights - What Just Happened",
            description=f"Key moments from this batch ({len(selected_updates)} of {len(self.highlights_queue)} updates)",
            color=0x95a5a6,  # Gray for pattern-based
            timestamp=datetime.now()
        )
        
        # Add selected highlights
        for i, (update, importance) in enumerate(selected_updates, 1):
            # Extract correct time from content rather than pub_date
            time_str = self._extract_correct_time(update)
            
            # Clean the title - remove time if it appears at the beginning
            title = update.title
            title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*-\s*', '', title)
            
            # Only truncate if extremely long
            if len(title) > 1000:
                title = title[:997] + "..."
            
            # Add importance indicators
            importance_emoji = "🔥" if importance >= 7 else "⭐" if importance >= 5 else "📝"
            
            embed.add_field(
                name=f"{importance_emoji} {time_str}",
                value=title,
                inline=False
            )
        
        embed.set_footer(text=f"Pattern Analysis • {len(selected_updates)} key moments selected")
        
        return embed
        
    except Exception as e:
        logger.error(f"Pattern highlights creation failed: {e}")
        # Return a basic embed if everything fails
        return discord.Embed(
            title="🎯 Feed Highlights - What Just Happened",
            description=f"Recent updates from the feed ({len(self.highlights_queue)} updates)",
            color=0x95a5a6
        )

async def _create_llm_hourly_summary_fallback(self) -> List[discord.Embed]:
    """Create narrative LLM hourly summary as fallback"""
    try:
        await self.rate_limiter.wait_if_needed()
        
        # Prepare update data
        updates_data = []
        for update in self.hourly_queue:
            time_str = self._extract_correct_time(update)
            updates_data.append({
                'time': time_str,
                'title': update.title,
                'description': update.description[:200] if update.description != update.title else ""
            })
        
        updates_text = "\n".join([
            f"{u['time']} - {u['title']}" + (f" ({u['description']})" if u['description'] else "")
            for u in updates_data
        ])
        
        prompt = f"""You are a Big Brother superfan creating a comprehensive HOURLY SUMMARY.

Analyze these {len(self.hourly_queue)} updates from the past hour:

{updates_text}

Provide a thorough analysis covering both strategic and social aspects:

{{
    "headline": "Compelling headline that captures the hour's most significant development",
    "summary": "4-5 sentence summary of the hour's key developments and overall narrative",
    "strategic_analysis": "Strategic implications - targets, power shifts, competition positioning, voting plans",
    "social_dynamics": "Alliance formations, shifts, trust levels, betrayals, strategic partnerships",
    "entertainment_highlights": "Funny moments, drama, memorable quotes, personality clashes",
    "key_players": ["houseguests", "involved", "in", "major", "moments"],
    "game_phase": "one of: early_game, jury_phase, final_weeks, finale_night",
    "strategic_importance": 7,
    "house_culture": "Inside jokes, routines, traditions, or quirky moments from this hour",
    "relationship_updates": "Showmance developments, romantic connections, relationship changes"
}}

This is an HOURLY DIGEST so be comprehensive and analytical but not too wordy."""

        response = await asyncio.to_thread(
            self.llm_client.messages.create,
            model=self.llm_model,
            max_tokens=1500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response and create embed
        analysis = self._parse_llm_response(response.content[0].text)
        return self._create_hourly_summary_embed(analysis, len(self.hourly_queue))
        
    except Exception as e:
        logger.error(f"LLM hourly summary fallback failed: {e}")
        return self._create_enhanced_pattern_hourly_summary()

def _parse_structured_llm_response(self, response_text: str) -> dict:
    """Parse structured LLM response with better error handling"""
    try:
        # Try to extract JSON
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_text = response_text[json_start:json_end]
            data = json.loads(json_text)
            
            # Validate required fields
            required_fields = ['headline', 'key_players', 'overall_importance']
            for field in required_fields:
                if field not in data:
                    if field == 'key_players':
                        data[field] = []
                    elif field == 'overall_importance':
                        data[field] = 5
                    else:
                        data[field] = f"No {field} provided"
            
            return data
        else:
            raise ValueError("No JSON found in response")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Structured JSON parsing failed: {e}, creating fallback")
        return self._create_fallback_structured_response(response_text)

def _create_fallback_structured_response(self, response_text: str) -> dict:
    """Create fallback structured response when JSON parsing fails"""
    # Extract houseguest names from response
    names = re.findall(r'\b[A-Z][a-z]+\b', response_text)
    key_players = [name for name in names if name not in EXCLUDE_WORDS][:5]
    
    return {
        "headline": "Big Brother Activity Continues",
        "strategic_analysis": None,
        "alliance_dynamics": None,
        "entertainment_highlights": None,
        "showmance_updates": None,
        "house_culture": None,
        "key_players": key_players,
        "overall_importance": 5,
        "importance_explanation": "Standard house activity"
    }

def _create_category_narrative(self, category: str, updates: List) -> str:
    """Create narrative summary for a category"""
    if not updates:
        return ""
    
    # Get top 3 most important updates in this category
    top_updates = sorted(updates, key=lambda x: self.analyzer.analyze_strategic_importance(x), reverse=True)[:3]
    
    narratives = []
    for update in top_updates:
        time_str = self._extract_correct_time(update)
        # Clean title
        title = update.title
        title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*[-–]\s*', '', title)
        
        # Truncate if too long
        if len(title) > 120:
            title = title[:117] + "..."
        
        narratives.append(f"**{time_str}**: {title}")
    
    return "\n".join(narratives)


    async def process_update_for_context(self, update: BBUpdate):
        """Process update for historical context tracking"""
        if not self.context_tracker:
            return
        
        try:
            # Detect and record events
            detected_events = await self.context_tracker.analyze_update_for_events(update)
            
            for event in detected_events:
                success = await self.context_tracker.record_event(event)
                if success:
                    logger.info(f"Recorded context event: {event['type']} - {event.get('description', 'No description')}")
        except Exception as e:
            logger.error(f"Error processing update for context: {e}")

    async def save_queue_state(self):
            """Save current queue state to database"""
            try:
                database_url = os.getenv('DATABASE_URL')
                if not database_url:
                    logger.debug("No database URL for queue persistence")
                    return
                    
                import psycopg2
                import psycopg2.extras
                conn = psycopg2.connect(database_url, cursor_factory=psycopg2.extras.RealDictCursor)
                cursor = conn.cursor()
                
                # Save highlights queue
                highlights_data = {
                    'updates': [
                        {
                            'title': update.title,
                            'description': update.description,
                            'link': update.link,
                            'pub_date': update.pub_date.isoformat(),
                            'content_hash': update.content_hash,
                            'author': update.author
                        }
                        for update in self.highlights_queue
                    ]
                }
                
                cursor.execute("""
                    INSERT INTO summary_checkpoints 
                    (summary_type, queue_state, queue_size, last_summary_time)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (summary_type) 
                    DO UPDATE SET 
                        queue_state = EXCLUDED.queue_state,
                        queue_size = EXCLUDED.queue_size,
                        last_summary_time = EXCLUDED.last_summary_time,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    'highlights',
                    json.dumps(highlights_data),
                    len(self.highlights_queue),
                    self.last_batch_time.isoformat()
                ))
                
                conn.commit()
                conn.close()
                
                logger.debug(f"Saved queue state: {len(self.highlights_queue)} highlights")
                
            except Exception as e:
                logger.error(f"Error saving queue state: {e}")
    
    async def clear_old_checkpoints(self, days_to_keep: int = 7):
        """Clean up old checkpoint data"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                return
                    
            import psycopg2
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()
                
            cursor.execute("""
                DELETE FROM summary_checkpoints 
                WHERE created_at < NOW() - INTERVAL '%s days'
            """, (days_to_keep,))
                
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
                
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old queue checkpoints")
                    
        except Exception as e:
            logger.error(f"Error cleaning checkpoints: {e}")

async def clear_old_checkpoints(self, days_to_keep: int = 7):
        """Clean up old checkpoint data"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                return
                
            import psycopg2
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM summary_checkpoints 
                WHERE created_at < NOW() - INTERVAL '%s days'
            """, (days_to_keep,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old queue checkpoints")
                
        except Exception as e:
            logger.error(f"Error cleaning checkpoints: {e}")

async def save_queue_state(self):
    """Save current queue state to database"""
    try:
        # Use the same database connection as prediction manager
        conn = None
        if hasattr(self, 'db') and hasattr(self.db, 'get_connection'):
            conn = self.db.get_connection()
        else:
            # If not available, use config to get connection
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                import psycopg2
                import psycopg2.extras
                conn = psycopg2.connect(database_url, cursor_factory=psycopg2.extras.RealDictCursor)
        
        if not conn:
            logger.warning("No database connection available for queue persistence")
            return
            
        cursor = conn.cursor()
        
        # Save highlights queue
        highlights_data = {
            'updates': [
                {
                    'title': update.title,
                    'description': update.description,
                    'link': update.link,
                    'pub_date': update.pub_date.isoformat(),
                    'content_hash': update.content_hash,
                    'author': update.author
                }
                for update in self.highlights_queue
            ]
        }
        
        cursor.execute("""
            INSERT INTO summary_checkpoints 
            (summary_type, queue_state, queue_size, last_summary_time)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (summary_type) 
            DO UPDATE SET 
                queue_state = EXCLUDED.queue_state,
                queue_size = EXCLUDED.queue_size,
                last_summary_time = EXCLUDED.last_summary_time,
                updated_at = CURRENT_TIMESTAMP
        """, (
            'highlights',
            json.dumps(highlights_data),
            len(self.highlights_queue),
            self.last_batch_time.isoformat()
        ))
        
        # Save hourly queue
        hourly_data = {
            'updates': [
                {
                    'title': update.title,
                    'description': update.description,
                    'link': update.link,
                    'pub_date': update.pub_date.isoformat(),
                    'content_hash': update.content_hash,
                    'author': update.author
                }
                for update in self.hourly_queue
            ]
        }
        
        cursor.execute("""
            INSERT INTO summary_checkpoints 
            (summary_type, queue_state, queue_size, last_summary_time)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (summary_type) 
            DO UPDATE SET 
                queue_state = EXCLUDED.queue_state,
                queue_size = EXCLUDED.queue_size,
                last_summary_time = EXCLUDED.last_summary_time,
                updated_at = CURRENT_TIMESTAMP
        """, (
            'hourly',
            json.dumps(hourly_data),
            len(self.hourly_queue),
            self.last_hourly_summary.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved queue state: {len(self.highlights_queue)} highlights, {len(self.hourly_queue)} hourly")
        
    except Exception as e:
        logger.error(f"Error saving queue state: {e}")


    async def restore_queue_state(self):
        """Restore queue state from database on startup"""
        try:
            # Get database connection
            conn = None
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                import psycopg2
                import psycopg2.extras
                conn = psycopg2.connect(database_url, cursor_factory=psycopg2.extras.RealDictCursor)
            
            if not conn:
                logger.info("No database connection for queue restoration")
                return
                
            cursor = conn.cursor()
            
            # Restore highlights queue
            cursor.execute("""
                SELECT queue_state, last_summary_time 
                FROM summary_checkpoints 
                WHERE summary_type = %s
                ORDER BY updated_at DESC 
                LIMIT 1
            """, ('highlights',))
            
            result = cursor.fetchone()
            if result and result['queue_state']:
                try:
                    highlights_data = json.loads(result['queue_state']) if isinstance(result['queue_state'], str) else result['queue_state']
                    
                    # Restore highlights queue
                    for update_data in highlights_data.get('updates', []):
                        update = BBUpdate(
                            title=update_data['title'],
                            description=update_data['description'],
                            link=update_data['link'],
                            pub_date=datetime.fromisoformat(update_data['pub_date']),
                            content_hash=update_data['content_hash'],
                            author=update_data['author']
                        )
                        self.highlights_queue.append(update)
                    
                    # Restore last batch time
                    if result['last_summary_time']:
                        self.last_batch_time = datetime.fromisoformat(result['last_summary_time'])
                    
                    logger.info(f"Restored {len(self.highlights_queue)} highlights from database")
                    
                except Exception as e:
                    logger.error(f"Error parsing highlights queue data: {e}")
            
            # Restore hourly queue
            cursor.execute("""
                SELECT queue_state, last_summary_time 
                FROM summary_checkpoints 
                WHERE summary_type = %s
                ORDER BY updated_at DESC 
                LIMIT 1
            """, ('hourly',))
            
            result = cursor.fetchone()
            if result and result['queue_state']:
                try:
                    hourly_data = json.loads(result['queue_state']) if isinstance(result['queue_state'], str) else result['queue_state']
                    
                    # Restore hourly queue
                    for update_data in hourly_data.get('updates', []):
                        update = BBUpdate(
                            title=update_data['title'],
                            description=update_data['description'],
                            link=update_data['link'],
                            pub_date=datetime.fromisoformat(update_data['pub_date']),
                            content_hash=update_data['content_hash'],
                            author=update_data['author']
                        )
                        self.hourly_queue.append(update)
                    
                    # Restore last hourly summary time
                    if result['last_summary_time']:
                        self.last_hourly_summary = datetime.fromisoformat(result['last_summary_time'])
                    
                    logger.info(f"Restored {len(self.hourly_queue)} hourly updates from database")
                    
                except Exception as e:
                    logger.error(f"Error parsing hourly queue data: {e}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error restoring queue state: {e}")
    
    async def clear_old_checkpoints(self, days_to_keep: int = 7):
        """Clean up old checkpoint data"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                return
                
            import psycopg2
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM summary_checkpoints 
                WHERE created_at < NOW() - INTERVAL '%s days'
            """, (days_to_keep,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old queue checkpoints")
                
        except Exception as e:
            logger.error(f"Error cleaning checkpoints: {e}")



    async def _create_context_aware_structured_summary(self, summary_type: str) -> List[discord.Embed]:
        """Create structured summary with historical context integration"""
        await self.rate_limiter.wait_if_needed()
        
        # Sort updates chronologically
        sorted_updates = sorted(self.hourly_queue, key=lambda x: x.pub_date)
        
        # STEP 1: Detect events and gather context
        all_detected_events = []
        contextual_info = {}
        
        for update in sorted_updates:
            # Detect events using context tracker
            if hasattr(self, 'context_tracker') and self.context_tracker:
                events = await self.context_tracker.analyze_update_for_events(update)
                all_detected_events.extend(events)
                
                # For each detected event, gather historical context
                for event in events:
                    if 'houseguest' in event:
                        hg = event['houseguest']
                        if hg not in contextual_info:
                            context = await self.context_tracker.get_historical_context(hg, event['type'])
                            contextual_info[hg] = context
        
        # STEP 2: Format updates with enhanced context awareness
        formatted_updates = []
        for i, update in enumerate(sorted_updates, 1):
            time_str = self._extract_correct_time(update)
            time_str = time_str.lstrip('0')
            
            # Enhanced title with context hints
            title = update.title
            enhanced_title = await self._enhance_title_with_context(title, contextual_info)
            
            formatted_updates.append(f"{i}. {time_str} - {enhanced_title}")
            if update.description and update.description != update.title:
                desc = update.description[:150] + "..." if len(update.description) > 150 else update.description
                formatted_updates.append(f"   {desc}")
        
        updates_text = "\n".join(formatted_updates)
        
        # STEP 3: Create context-enriched historical summary
        historical_context_summary = await self._create_historical_context_summary(all_detected_events, contextual_info)
        
        # Calculate current day
        current_day = max(1, (datetime.now().date() - datetime(2025, 7, 8).date()).days + 1)
        
        # STEP 4: Enhanced prompt with historical context
        prompt = f"""You are a Big Brother superfan analyst with access to HISTORICAL CONTEXT creating an hourly summary for Day {current_day}.
    
    NEW UPDATES TO ANALYZE (Day {current_day}) - IN CHRONOLOGICAL ORDER:
    {updates_text}
    
    HISTORICAL CONTEXT FOR THIS HOUR:
    {historical_context_summary}
    
    Create a comprehensive summary that leverages this historical context to tell a richer story.
    
    Provide your analysis in this EXACT JSON format:
    
    {{
        "headline": "Context-aware headline that references relevant history when appropriate",
        "strategic_analysis": "Analysis that connects current moves to past events and patterns. Reference historical context when relevant (e.g., 'This is X's 3rd HOH', 'After being betrayed last week by Y'). Use null if no strategic developments.",
        "alliance_dynamics": "Alliance analysis that references formation dates, past betrayals, and relationship history. Use null if no alliance developments.",
        "entertainment_highlights": "Entertaining moments with context about recurring dynamics or callbacks to past events. Use null if no entertainment moments.",
        "showmance_updates": "Romance developments with timeline context (how long they've been together, past drama, etc.). Use null if no romance developments.",
        "house_culture": "Daily culture moments that build on established house traditions or inside jokes. Use null if no cultural developments.",
        "key_players": ["List", "of", "houseguests", "central", "to", "developments"],
        "overall_importance": 8,
        "importance_explanation": "Explanation that considers both immediate events AND their historical significance",
        "context_integration": "Brief note on how historical context enhanced this summary"
    }}
    
    CRITICAL INSTRUCTIONS:
    - Weave in historical context naturally - don't force it if not relevant
    - When referencing past events, be specific (e.g., "3rd nomination" not "multiple nominations")
    - Connect current events to past patterns when meaningful
    - Use null for sections without substantial content
    - Overall importance should consider historical significance, not just immediate drama
    - Make the summary feel like it's written by someone who's been following all season"""
    
        # Call LLM
        response = await asyncio.to_thread(
            self.llm_client.messages.create,
            model="claude-3-haiku-20240307",
            max_tokens=1200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        try:
            # Parse JSON response
            analysis_data = self._parse_structured_llm_response(response_text)
            
            # Create enhanced structured embed
            embeds = self._create_context_aware_summary_embed(
                analysis_data, len(self.hourly_queue), summary_type, historical_context_summary
            )
            
            logger.info(f"Created context-aware {summary_type} summary with historical context")
            return embeds
            
        except Exception as e:
            logger.error(f"Failed to parse context-aware response: {e}")
            logger.error(f"Raw response: {response_text}")
            # Fallback to enhanced pattern-based
            return await self._create_enhanced_pattern_summary_with_context()
    
    async def _enhance_title_with_context(self, title: str, contextual_info: dict) -> str:
        """Enhance update title with brief context hints"""
        # Look for houseguest names in title and add context hints
        enhanced_title = title
        
        for houseguest, context in contextual_info.items():
            if houseguest.lower() in title.lower() and context:
                # Add brief context hint in parentheses
                context_hint = context[:30] + "..." if len(context) > 30 else context
                if context_hint and "first" not in context_hint.lower():
                    # Only add hint for non-first events
                    enhanced_title = title  # Keep original for now, LLM will handle context
        
        return enhanced_title
    
    async def _create_historical_context_summary(self, detected_events: List[dict], contextual_info: dict) -> str:
        """Create a summary of relevant historical context for this hour"""
        if not detected_events and not contextual_info:
            return "No significant historical context detected for this hour."
        
        context_parts = []
        
        # Summarize detected events
        if detected_events:
            event_summary = f"Detected {len(detected_events)} significant events this hour: "
            event_types = [event['type'] for event in detected_events]
            unique_types = list(set(event_types))
            event_summary += ", ".join(unique_types)
            context_parts.append(event_summary)
        
        # Add key historical context
        if contextual_info:
            for houseguest, context in list(contextual_info.items())[:3]:  # Limit to top 3
                if context and len(context) > 10:
                    context_parts.append(f"{houseguest}: {context}")
        
        return " | ".join(context_parts) if context_parts else "Standard house activity with no major historical patterns."
    
    def _create_context_aware_summary_embed(self, analysis_data: dict, update_count: int, 
                                           summary_type: str, historical_context: str) -> List[discord.Embed]:
        """Create context-aware summary embed with historical context integration"""
        pacific_tz = pytz.timezone('US/Pacific')
        current_hour = datetime.now(pacific_tz).strftime("%I %p").lstrip('0')
        current_day = max(1, (datetime.now().date() - datetime(2025, 7, 1).date()).days + 1)
        
        # Determine embed color based on importance
        importance = analysis_data.get('overall_importance', 5)
        if importance >= 9:
            color = 0xff1744  # Red for explosive hours
        elif importance >= 7:
            color = 0xff9800  # Orange for high activity
        elif importance >= 4:
            color = 0x3498db  # Blue for moderate activity
        else:
            color = 0x95a5a6  # Gray for quiet hours
        
        # Create main embed with context awareness indicator
        title = f"Chen Bot's Contextual Summary - {current_hour} 🧠"
        description = f"*Historical context integrated*"
        
        embed = discord.Embed(
            title=title,
            description=description,
            color=color,
            timestamp=datetime.now()
        )
        
        # Add headline
        headline = analysis_data.get('headline', 'Big Brother Update')
        embed.add_field(
            name="📰 Headline",
            value=headline,
            inline=False
        )
        
        # Add context-aware sections
        sections = [
            ("🎯 Strategic Analysis", analysis_data.get('strategic_analysis')),
            ("🤝 Alliance Dynamics", analysis_data.get('alliance_dynamics')),
            ("🎬 Entertainment Highlights", analysis_data.get('entertainment_highlights')),
            ("💕 Showmance Updates", analysis_data.get('showmance_updates')),
            ("🏠 House Culture", analysis_data.get('house_culture'))
        ]
        
        for section_name, content in sections:
            if content and content.strip() and content.lower() not in ['null', 'none', 'nothing']:
                if len(content) > 1000:
                    content = content[:997] + "..."
                embed.add_field(
                    name=section_name,
                    value=content,
                    inline=False
                )
        
        # Add key players
        key_players = analysis_data.get('key_players', [])
        if key_players:
            if len(key_players) <= 6:
                players_text = " • ".join([f"**{player}**" for player in key_players])
            else:
                players_text = " • ".join([f"**{player}**" for player in key_players[:6]]) + f" • +{len(key_players)-6} more"
        else:
            players_text = "No specific houseguests highlighted"
        
        embed.add_field(
            name="⭐ Key Players",
            value=players_text,
            inline=False
        )
        
        # Enhanced importance rating with context consideration
        custom_emoji = "<:chunky:1392638440582942974>"
        icon_count = min(importance, 10)
        importance_text = custom_emoji * icon_count + f" **{importance}/10**"
        
        explanation = analysis_data.get('importance_explanation', '')
        if explanation:
            importance_text += f"\n*{explanation}*"
        
        embed.add_field(
            name="📊 Overall Importance",
            value=importance_text,
            inline=False
        )
        
        # Add historical context integration note
        context_integration = analysis_data.get('context_integration', '')
        if context_integration:
            embed.add_field(
                name="🧠 Context Integration",
                value=f"*{context_integration}*",
                inline=False
            )
        
        # Enhanced footer
        embed.set_footer(text=f"Chen Bot's Contextual Summary • {current_hour} • Historical Context Enabled")
        
        return [embed]
    
    async def _create_enhanced_pattern_summary_with_context(self) -> List[discord.Embed]:
        """Enhanced pattern-based summary with basic context integration"""
        logger.info("Creating enhanced pattern-based summary with context")
        
        # Group updates by categories
        categories = defaultdict(list)
        for update in self.hourly_queue:
            update_categories = self.analyzer.categorize_update(update)
            for category in update_categories:
                categories[category].append(update)
        
        current_hour = datetime.now().strftime("%I %p").lstrip('0')
        
        custom_emoji = "<:takingnotes:916186747770130443>"
        title = f"Chen Bot's Enhanced Summary - {current_hour}"
        
        embed = discord.Embed(
            title=title,
            description=f"**{len(self.hourly_queue)} updates this hour** • {custom_emoji} *With basic context*",
            color=0x95a5a6,
            timestamp=datetime.now()
        )
        
        # Add headline with basic context awareness
        if self.hourly_queue:
            top_update = max(self.hourly_queue, key=lambda x: self.analyzer.analyze_strategic_importance(x))
            headline = await self._create_context_aware_headline(top_update, len(self.hourly_queue))
            embed.add_field(
                name="📰 Headline",
                value=headline,
                inline=False
            )
        
        # Create enhanced narratives for categories
        section_mapping = {
            "🎯 Strategy": "🎯 Strategic Analysis",
            "🤝 Alliance": "🤝 Alliance Dynamics", 
            "🎬 Entertainment": "🎬 Entertainment Highlights",
            "💕 Romance": "💕 Showmance Updates",
            "📝 General": "🏠 House Culture"
        }
        
        for category, updates in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
            if updates and len(updates) > 0:
                narrative_summary = await self._create_context_aware_category_narrative(category, updates)
                if narrative_summary and narrative_summary.strip():
                    section_name = section_mapping.get(category, category)
                    embed.add_field(
                        name=section_name,
                        value=narrative_summary,
                        inline=False
                    )
        
        # Add key players with basic context
        all_houseguests = set()
        for update in self.hourly_queue:
            hgs = self.analyzer.extract_houseguests(update.title + " " + update.description)
            all_houseguests.update(hgs[:3])
        
        if all_houseguests:
            players_text = " • ".join([f"**{hg}**" for hg in list(all_houseguests)[:6]])
            if len(all_houseguests) > 6:
                players_text += f" • +{len(all_houseguests)-6} more"
        else:
            players_text = "No specific houseguests highlighted"
        
        embed.add_field(
            name="⭐ Key Players",
            value=players_text,
            inline=False
        )
        
        # Enhanced importance rating
        total_importance = sum(self.analyzer.analyze_strategic_importance(u) for u in self.hourly_queue)
        avg_importance = int(total_importance / len(self.hourly_queue)) if self.hourly_queue else 1
        
        importance_icons = ["😴", "😴", "📝", "📈", "⭐", "⭐", "🔥", "🔥", "💥", "🚨"]
        importance_icon = importance_icons[min(avg_importance - 1, 9)] if avg_importance >= 1 else "📝"
        
        if avg_importance >= 7:
            activity_desc = "High drama and strategic activity with historical significance"
        elif avg_importance >= 5:
            activity_desc = "Moderate activity building on season patterns"
        elif avg_importance >= 3:
            activity_desc = "Steady house activity with some context"
        else:
            activity_desc = "Routine activities maintaining house dynamics"
        
        embed.add_field(
            name="📊 Overall Importance",
            value=f"{importance_icon} **{avg_importance}/10**\n*{activity_desc}*",
            inline=False
        )
        
        embed.set_footer(text=f"Chen Bot's Enhanced Summary • {current_hour} • Basic Context Integration")
        
        return [embed]
    
    async def _create_context_aware_headline(self, top_update: BBUpdate, total_updates: int) -> str:
        """Create a context-aware headline"""
        title = top_update.title
        title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*[-–]\s*', '', title)
        
        # Try to get basic context for the headline
        houseguests = self.analyzer.extract_houseguests(title)
        if houseguests and hasattr(self, 'context_tracker') and self.context_tracker:
            try:
                # Get quick context for first mentioned houseguest
                context = await self.context_tracker.get_historical_context(houseguests[0])
                if context and "first" not in context.lower():
                    return f"Continuing Season Patterns: {title}"
            except Exception as e:
                logger.debug(f"Context lookup failed for headline: {e}")
        
        # Fallback to pattern-based headlines
        content = title.lower()
        if any(word in content for word in ['winner', 'wins', 'victory', 'champion']):
            return f"Competition Results Shape Power Dynamics"
        elif any(word in content for word in ['alliance', 'strategy', 'target']):
            return f"Strategic Developments Continue Season Narrative" 
        else:
            return f"Big Brother House Activity Continues"
    
    async def _create_context_aware_category_narrative(self, category: str, updates: List) -> str:
        """Create context-aware narrative for a category"""
        if not updates:
            return ""
        
        # Get top 3 most important updates
        top_updates = sorted(updates, key=lambda x: self.analyzer.analyze_strategic_importance(x), reverse=True)[:3]
        
        narratives = []
        for update in top_updates:
            time_str = self._extract_correct_time(update)
            title = update.title
            title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*[-–]\s*', '', title)
            
            # Try to add basic context
            enhanced_title = await self._add_basic_context_to_title(title)
            
            if len(enhanced_title) > 120:
                enhanced_title = enhanced_title[:117] + "..."
            
            narratives.append(f"**{time_str}**: {enhanced_title}")
        
        return "\n".join(narratives)
    
    async def _add_basic_context_to_title(self, title: str) -> str:
        """Add basic context hints to a title"""
        if not hasattr(self, 'context_tracker') or not self.context_tracker:
            return title
        
        try:
            # Extract houseguests from title
            houseguests = self.analyzer.extract_houseguests(title)
            if not houseguests:
                return title
            
            # For performance, only check first houseguest
            hg = houseguests[0]
            
            # Quick context check
            if 'nomination' in title.lower():
                context = await self.context_tracker.get_historical_context(hg, 'nomination')
                if context and "first" not in context.lower():
                    return f"{title} ({context[:20]}...)"
            elif 'hoh' in title.lower():
                context = await self.context_tracker.get_historical_context(hg, 'hoh_win')
                if context and "first" not in context.lower():
                    return f"{title} ({context[:20]}...)"
            
            return title
            
        except Exception as e:
            logger.debug(f"Basic context addition failed: {e}")
            return title

    async def process_update_for_context(self, update: BBUpdate):
        """Process update for historical context tracking"""
        if not self.context_tracker:
            return
        
        try:
            # Detect and record events
            detected_events = await self.context_tracker.analyze_update_for_events(update)
            
            for event in detected_events:
                success = await self.context_tracker.record_event(event)
                if success:
                    logger.info(f"Recorded context event: {event['type']} - {event.get('description', 'No description')}")
        except Exception as e:
            logger.error(f"Error processing update for context: {e}")

    async def _create_llm_hourly_summary_for_timeframe(self, updates: List[BBUpdate], hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Create LLM-powered hourly summary for specific timeframe"""
        await self.rate_limiter.wait_if_needed()
        
        # Format the updates
        updates_text = "\n".join([
            f"{self._extract_correct_time(u)} - {u.title}"
            for u in updates
        ])
        
        hour_period = f"{hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}"
        
        prompt = f"""You are a Big Brother superfan creating an hourly summary.
    
    HOUR PERIOD: {hour_period}
    UPDATES FROM THIS HOUR ({len(updates)} total):
    {updates_text}
    
    Create a comprehensive summary for this specific hour period:
    
    {{
        "headline": "Headline for {hour_period}",
        "summary": "3-4 sentence summary of what happened during {hour_period}",
        "key_players": ["houseguests", "central", "to", "this", "hour"],
        "strategic_importance": 7,
        "hour_highlights": "Key moments that happened during {hour_period}"
    }}
    
    Focus specifically on what happened during {hour_period}."""
    
        response = await asyncio.to_thread(
            self.llm_client.messages.create,
            model=self.llm_model,
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        analysis = self._parse_llm_response(response.content[0].text)
        return self._create_timeframe_embed(analysis, len(updates), hour_start, hour_end)
    
    def _create_pattern_hourly_summary_for_timeframe(self, updates: List[BBUpdate], hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Pattern-based hourly summary for timeframe"""
        hour_period = f"{hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}"
        
        # Use your existing pattern logic
        categories = defaultdict(list)
        for update in updates:
            update_categories = self.analyzer.categorize_update(update)
            for category in update_categories:
                categories[category].append(update)
        
        analysis = {
            "headline": f"Big Brother Activity During {hour_period}",
            "summary": f"{len(updates)} updates occurred during {hour_period}",
            "key_players": [],
            "strategic_importance": 5,
            "categories": categories
        }
        
        return self._create_timeframe_embed(analysis, len(updates), hour_start, hour_end)
    
    def _create_basic_hourly_summary_for_timeframe(self, updates: List[BBUpdate], hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Basic fallback summary"""
        hour_period = f"{hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}"
        
        analysis = {
            "headline": f"Updates from {hour_period}",
            "summary": f"{len(updates)} updates during this hour",
            "key_players": [],
            "strategic_importance": 3
        }
        
        return self._create_timeframe_embed(analysis, len(updates), hour_start, hour_end)
    
    def _create_timeframe_embed(self, analysis: dict, update_count: int, hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Create embed for timeframe-based summary with original format"""
        pacific_tz = pytz.timezone('US/Pacific')
        hour_display = hour_end.astimezone(pacific_tz).strftime("%I %p").lstrip('0')
        
        # Keep your original title format exactly
        embed = discord.Embed(
            title=f"Chen Bot's House Summary - {hour_display} 🏠",
            description=f"",  # Your original description
            color=0x9b59b6,  # Keep your original color
            timestamp=datetime.now()
        )
        
        # Add your original content fields (same as before)
        embed.add_field(
            name="📰 Headline", 
            value=analysis.get('headline', 'Big Brother Update'),
            inline=False
        )
        
        # Add other fields exactly like your original format
        if analysis.get('strategic_analysis'):
            embed.add_field(
                name="🎯 Strategic Analysis",
                value=analysis['strategic_analysis'],
                inline=False
            )
        
        # Continue with your other original sections...
        # (Add the same fields you had before)
        
        # Keep your original footer
        embed.set_footer(text=f"Chen Bot's House Summary • {hour_display}")
        
        return [embed]

    async def _create_llm_hourly_summary_for_timeframe(self, updates: List[BBUpdate], hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Create LLM-powered hourly summary for specific timeframe"""
        await self.rate_limiter.wait_if_needed()
        
        # Format the updates
        updates_text = "\n".join([
            f"{self._extract_correct_time(u)} - {u.title}"
            for u in updates[:20]  # Limit to prevent token overflow
        ])
        
        hour_period = f"{hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}"
        
        prompt = f"""You are a Big Brother superfan creating an hourly summary.
    
    HOUR PERIOD: {hour_period}
    UPDATES FROM THIS HOUR ({len(updates)} total):
    {updates_text}
    
    Create a comprehensive summary for this specific hour period:
    
    {{
        "headline": "Headline for {hour_period}",
        "summary": "3-4 sentence summary of what happened during {hour_period}",
        "key_players": ["houseguests", "central", "to", "this", "hour"],
        "strategic_importance": 7,
        "hour_highlights": "Key moments that happened during {hour_period}"
    }}
    
    Focus specifically on what happened during {hour_period}."""
    
        response = await asyncio.to_thread(
            self.llm_client.messages.create,
            model=self.llm_model,
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        analysis = self._parse_llm_response(response.content[0].text)
        return self._create_timeframe_embed(analysis, len(updates), hour_start, hour_end)
    
    def _create_pattern_hourly_summary_for_timeframe(self, updates: List[BBUpdate], hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Pattern-based hourly summary for timeframe"""
        hour_period = f"{hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}"
        
        # Use your existing pattern logic
        categories = defaultdict(list)
        for update in updates:
            update_categories = self.analyzer.categorize_update(update)
            for category in update_categories:
                categories[category].append(update)
        
        # Get key players
        all_houseguests = set()
        for update in updates:
            hgs = self.analyzer.extract_houseguests(update.title + " " + update.description)
            all_houseguests.update(hgs[:3])
        
        analysis = {
            "headline": f"Big Brother Activity During {hour_period}",
            "summary": f"{len(updates)} updates occurred during {hour_period}",
            "key_players": list(all_houseguests)[:5],
            "strategic_importance": 5,
            "categories": categories
        }
        
        return self._create_timeframe_embed(analysis, len(updates), hour_start, hour_end)
    
    def _create_basic_hourly_summary_for_timeframe(self, updates: List[BBUpdate], hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Basic fallback summary"""
        hour_period = f"{hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}"
        
        analysis = {
            "headline": f"Updates from {hour_period}",
            "summary": f"{len(updates)} updates during this hour",
            "key_players": [],
            "strategic_importance": 3
        }
        
        return self._create_timeframe_embed(analysis, len(updates), hour_start, hour_end)
    
    def _create_timeframe_embed(self, analysis: dict, update_count: int, hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Create embed for timeframe-based summary"""
        hour_display = hour_end.strftime("%I %p").lstrip('0')
        hour_period = f"{hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}"
        
        embed = discord.Embed(
            title=f"📊 Hourly Summary - {hour_display}",
            description=f"**{hour_period}** • {update_count} updates",
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        embed.add_field(
            name="📰 Hour Headline", 
            value=analysis.get('headline', 'Activity during this hour'),
            inline=False
        )
        
        embed.add_field(
            name="📋 Summary",
            value=analysis.get('summary', 'Updates occurred during this period'),
            inline=False
        )
        
        if analysis.get('key_players'):
            embed.add_field(
                name="⭐ Key Players",
                value=" • ".join([f"**{player}**" for player in analysis['key_players'][:5]]),
                inline=False
            )
        
        # Add importance rating
        importance = analysis.get('strategic_importance', 5)
        importance_icons = ["😴", "😴", "📝", "📈", "⭐", "⭐", "🔥", "🔥", "💥", "🚨"]
        importance_icon = importance_icons[min(importance - 1, 9)] if importance >= 1 else "📝"
        
        embed.add_field(
            name="📊 Hour Importance",
            value=f"{importance_icon} **{importance}/10**",
            inline=False
        )
        
        embed.set_footer(text=f"Hourly Summary • {hour_period}")
        
        return [embed]

    async def process_update_for_context(self, update: BBUpdate):
        """Process update for historical context tracking"""
        if not self.context_tracker:
            return
        
        try:
            # Detect and record events
            detected_events = await self.context_tracker.analyze_update_for_events(update)
            
            for event in detected_events:
                success = await self.context_tracker.record_event(event)
                if success:
                    logger.info(f"Recorded context event: {event['type']} - {event.get('description', 'No description')}")
        except Exception as e:
            logger.error(f"Error processing update for context: {e}")


    async def _create_llm_hourly_summary_simple(self, updates: List[BBUpdate], hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Simple LLM hourly summary"""
        await self.rate_limiter.wait_if_needed()
        
        updates_text = "\n".join([f"{self._extract_correct_time(u)} - {u.title}" for u in updates[:15]])
        hour_period = f"{hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}"
        
        prompt = f"""Create a brief hourly summary for {hour_period}:
    
    {updates_text}
    
    Provide a 2-3 sentence summary of the key events during {hour_period}."""
    
        response = await asyncio.to_thread(
            self.llm_client.messages.create,
            model=self.llm_model,
            max_tokens=300,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        embed = discord.Embed(
            title=f"📊 Hourly Summary - {hour_end.strftime('%I %p').lstrip('0')}",
            description=response.content[0].text,
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        return [embed]
    
    def _create_pattern_hourly_summary_simple(self, updates: List[BBUpdate], hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Simple pattern-based hourly summary"""
        hour_display = hour_end.strftime("%I %p").lstrip('0')
        
        # Get top 5 most important updates
        top_updates = sorted(updates, key=lambda x: self.analyzer.analyze_strategic_importance(x), reverse=True)[:5]
        
        summary_text = "\n".join([f"• {update.title[:100]}..." if len(update.title) > 100 else f"• {update.title}" for update in top_updates])
        
        embed = discord.Embed(
            title=f"📊 Hourly Summary - {hour_display}",
            description=f"**{len(updates)} updates this hour**\n\n{summary_text}",
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        return [embed]
    
    def _create_basic_hourly_summary_simple(self, updates: List[BBUpdate], hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Basic fallback hourly summary"""
        hour_display = hour_end.strftime("%I %p").lstrip('0')
        
        embed = discord.Embed(
            title=f"📊 Hourly Summary - {hour_display}",
            description=f"**{len(updates)} updates occurred this hour**",
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        return [embed]

    async def clear_old_checkpoints(self, days_to_keep: int = 7):
        """Clean up old checkpoint data"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                return
                
            import psycopg2
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM summary_checkpoints 
                WHERE created_at < NOW() - INTERVAL '%s days'
            """, (days_to_keep,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old queue checkpoints")
                
        except Exception as e:
            logger.error(f"Error cleaning checkpoints: {e}")

    async def save_queue_state(self):
            """Save current queue state to database"""
            try:
                # Use the same database connection as prediction manager
                conn = None
                if hasattr(self, 'db') and hasattr(self.db, 'get_connection'):
                    conn = self.db.get_connection()
                else:
                    # If not available, use config to get connection
                    database_url = os.getenv('DATABASE_URL')
                    if database_url:
                        import psycopg2
                        import psycopg2.extras
                        conn = psycopg2.connect(database_url, cursor_factory=psycopg2.extras.RealDictCursor)
                
                if not conn:
                    logger.warning("No database connection available for queue persistence")
                    return
                    
                cursor = conn.cursor()
                
                # Save highlights queue
                highlights_data = {
                    'updates': [
                        {
                            'title': update.title,
                            'description': update.description,
                            'link': update.link,
                            'pub_date': update.pub_date.isoformat(),
                            'content_hash': update.content_hash,
                            'author': update.author
                        }
                        for update in self.highlights_queue
                    ]
                }
                
                cursor.execute("""
                    INSERT INTO summary_checkpoints 
                    (summary_type, queue_state, queue_size, last_summary_time)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (summary_type) 
                    DO UPDATE SET 
                        queue_state = EXCLUDED.queue_state,
                        queue_size = EXCLUDED.queue_size,
                        last_summary_time = EXCLUDED.last_summary_time,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    'highlights',
                    json.dumps(highlights_data),
                    len(self.highlights_queue),
                    self.last_batch_time.isoformat()
                ))
                
                # Save hourly queue
                hourly_data = {
                    'updates': [
                        {
                            'title': update.title,
                            'description': update.description,
                            'link': update.link,
                            'pub_date': update.pub_date.isoformat(),
                            'content_hash': update.content_hash,
                            'author': update.author
                        }
                        for update in self.hourly_queue
                    ]
                }
                
                cursor.execute("""
                    INSERT INTO summary_checkpoints 
                    (summary_type, queue_state, queue_size, last_summary_time)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (summary_type) 
                    DO UPDATE SET 
                        queue_state = EXCLUDED.queue_state,
                        queue_size = EXCLUDED.queue_size,
                        last_summary_time = EXCLUDED.last_summary_time,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    'hourly',
                    json.dumps(hourly_data),
                    len(self.hourly_queue),
                    self.last_hourly_summary.isoformat()
                ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Saved queue state: {len(self.highlights_queue)} highlights, {len(self.hourly_queue)} hourly")
                
            except Exception as e:
                logger.error(f"Error saving queue state: {e}")

class BBDatabase:
    """Handles database operations with connection pooling and error recovery"""
    
    def __init__(self, db_path: str = "bb_updates.db"):
        self.db_path = db_path
        self.connection_timeout = 30
        self.use_postgresql = False
        self.init_database()
    
    def get_connection(self):
        """Get database connection with proper error handling"""
        try:
            conn = sqlite3.connect(
                self.db_path, 
                timeout=self.connection_timeout,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
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
    
    def get_daily_updates(self, start_time: datetime, end_time: datetime) -> List[BBUpdate]:
        """Get all updates from a specific 24-hour period"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT title, description, link, pub_date, content_hash, author, importance_score
                FROM updates 
                WHERE pub_date >= ? AND pub_date < ?
                ORDER BY pub_date ASC
            """, (start_time, end_time))
            
            results = cursor.fetchall()
            conn.close()
            
            updates = []
            for row in results:
                update = BBUpdate(*row[:6])  # First 6 fields for BBUpdate
                updates.append(update)
            
            return updates
            
        except Exception as e:
            logger.error(f"Database daily query error: {e}")
    
    def get_recent_updates(self, hours: int) -> List[BBUpdate]:
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



    def get_updates_in_timeframe(self, start_time: datetime, end_time: datetime) -> List[BBUpdate]:
        """Get all updates within a specific timeframe"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if hasattr(self, 'use_postgresql') and self.use_postgresql:
                # PostgreSQL syntax
                cursor.execute("""
                    SELECT title, description, link, pub_date, content_hash, author
                    FROM updates 
                    WHERE pub_date >= %s AND pub_date < %s
                    ORDER BY pub_date ASC
                """, (start_time, end_time))
            else:
                # SQLite syntax
                cursor.execute("""
                    SELECT title, description, link, pub_date, content_hash, author
                    FROM updates 
                    WHERE pub_date >= ? AND pub_date < ?
                    ORDER BY pub_date ASC
                """, (start_time, end_time))
            
            results = cursor.fetchall()
            conn.close()
            
            updates = []
            for row in results:
                if hasattr(self, 'use_postgresql') and self.use_postgresql:
                    # PostgreSQL returns RealDictCursor results
                    update = BBUpdate(
                        title=row['title'],
                        description=row['description'], 
                        link=row['link'],
                        pub_date=row['pub_date'],
                        content_hash=row['content_hash'],
                        author=row['author']
                    )
                else:
                    # SQLite returns tuple
                    update = BBUpdate(*row)
                updates.append(update)
            
            return updates
            
        except Exception as e:
            logger.error(f"Database timeframe query error: {e}")
            return []

class PostgreSQLDatabase:
    """PostgreSQL database handler for Railway"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_timeout = 30
        self.use_postgresql = True
        self.init_database()
    
    def get_connection(self):
        """Get PostgreSQL connection"""
        try:
            conn = psycopg2.connect(
                self.database_url,
                connect_timeout=self.connection_timeout,
                cursor_factory=psycopg2.extras.RealDictCursor
            )
            return conn
        except Exception as e:
            logger.error(f"PostgreSQL connection error: {e}")
            raise
    
    def init_database(self):
        """Initialize PostgreSQL tables"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Updates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS updates (
                    id SERIAL PRIMARY KEY,
                    content_hash VARCHAR(32) UNIQUE,
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
                    prediction_id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    prediction_type VARCHAR(50) NOT NULL,
                    options TEXT NOT NULL,
                    created_by BIGINT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closes_at TIMESTAMP NOT NULL,
                    status VARCHAR(20) DEFAULT 'active',
                    correct_option TEXT,
                    week_number INTEGER,
                    guild_id BIGINT NOT NULL
                )
            """)
            
            # User predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_predictions (
                    user_id BIGINT NOT NULL,
                    prediction_id INTEGER NOT NULL,
                    option TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, prediction_id),
                    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
                )
            """)
            
            # Prediction leaderboard table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_leaderboard (
                    user_id BIGINT NOT NULL,
                    guild_id BIGINT NOT NULL,
                    week_number INTEGER NOT NULL,
                    season_points INTEGER DEFAULT 0,
                    weekly_points INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    total_predictions INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, guild_id, week_number)
                )
            """)
            
            # Alliance tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alliances (
                    alliance_id SERIAL PRIMARY KEY,
                    name TEXT,
                    formed_date TIMESTAMP,
                    dissolved_date TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    confidence_level INTEGER DEFAULT 50,
                    last_activity TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alliance_members (
                    alliance_id INTEGER,
                    houseguest_name TEXT,
                    joined_date TIMESTAMP,
                    left_date TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (alliance_id) REFERENCES alliances(alliance_id),
                    UNIQUE(alliance_id, houseguest_name)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alliance_events (
                    event_id SERIAL PRIMARY KEY,
                    alliance_id INTEGER,
                    event_type TEXT,
                    description TEXT,
                    involved_houseguests TEXT,
                    timestamp TIMESTAMP,
                    update_hash TEXT,
                    FOREIGN KEY (alliance_id) REFERENCES alliances(alliance_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_updates_hash ON updates(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_updates_date ON updates(pub_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_guild ON predictions(guild_id, status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alliance_status ON alliances(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alliance_members ON alliance_members(houseguest_name)")
            
            conn.commit()
            logger.info("PostgreSQL database initialized successfully")


            # Summary checkpoints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summary_checkpoints (
                    checkpoint_id SERIAL PRIMARY KEY,
                    summary_type VARCHAR(50) NOT NULL,
                    last_processed_update_id INTEGER,
                    queue_state JSONB,
                    queue_size INTEGER DEFAULT 0,
                    last_summary_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Summary metrics table (for tracking performance)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summary_metrics (
                    metric_id SERIAL PRIMARY KEY,
                    summary_type VARCHAR(50) NOT NULL,
                    update_count INTEGER,
                    llm_tokens_used INTEGER,
                    processing_time_ms INTEGER,
                    summary_quality_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_type ON summary_checkpoints(summary_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_type_date ON summary_metrics(summary_type, created_at)")
            
            logger.info("Summary persistence tables created")

            # Add these tables to your PostgreSQLDatabase.init_database() method
            # Place them after your existing table creation code
            
            # Houseguest events tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS houseguest_events (
                    event_id SERIAL PRIMARY KEY,
                    houseguest_name VARCHAR(100) NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    event_subtype VARCHAR(50),
                    description TEXT,
                    week_number INTEGER,
                    season_day INTEGER,
                    update_hash VARCHAR(32),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """)
            
            # Houseguest statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS houseguest_stats (
                    stat_id SERIAL PRIMARY KEY,
                    houseguest_name VARCHAR(100) NOT NULL,
                    stat_type VARCHAR(50) NOT NULL,
                    stat_value INTEGER DEFAULT 0,
                    season_total INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(houseguest_name, stat_type)
                )
            """)
            
            # Relationship tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS houseguest_relationships (
                    relationship_id SERIAL PRIMARY KEY,
                    houseguest_1 VARCHAR(100) NOT NULL,
                    houseguest_2 VARCHAR(100) NOT NULL,
                    relationship_type VARCHAR(50) NOT NULL,
                    strength_score INTEGER DEFAULT 50,
                    first_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'active',
                    duration_days INTEGER DEFAULT 0,
                    CHECK (houseguest_1 < houseguest_2)
                )
            """)
            
            # Context cache table (for performance)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_cache (
                    cache_id SERIAL PRIMARY KEY,
                    houseguest_name VARCHAR(100) NOT NULL,
                    context_type VARCHAR(50) NOT NULL,
                    context_text TEXT NOT NULL,
                    cache_key VARCHAR(100) NOT NULL,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(cache_key)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_houseguest ON houseguest_events(houseguest_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON houseguest_events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON houseguest_events(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_houseguest ON houseguest_stats(houseguest_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_hg1 ON houseguest_relationships(houseguest_1)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_hg2 ON houseguest_relationships(houseguest_2)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_cache_key ON context_cache(cache_key)")
            
            logger.info("Historical context tracking tables created")
            
        except Exception as e:
            logger.error(f"PostgreSQL initialization error: {e}")
            raise
        finally:
            conn.close()
    
    # Add the same methods as BBDatabase but with PostgreSQL syntax
    def is_duplicate(self, content_hash: str) -> bool:
        """Check if update already exists"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT 1 FROM updates WHERE content_hash = %s", (content_hash,))
            result = cursor.fetchone()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            logger.error(f"PostgreSQL duplicate check error: {e}")
            return False
    
    def store_update(self, update: BBUpdate, importance_score: int = 1, categories: List[str] = None):
        """Store a new update"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            categories_str = ",".join(categories) if categories else ""
            
            cursor.execute("""
                INSERT INTO updates (content_hash, title, description, link, pub_date, author, importance_score, categories)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (update.content_hash, update.title, update.description, 
                  update.link, update.pub_date, update.author, importance_score, categories_str))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"PostgreSQL store error: {e}")
            raise
    
    def get_recent_updates(self, hours: int) -> List[BBUpdate]:
        """Get updates from the last N hours"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT title, description, link, pub_date, content_hash, author
                FROM updates 
                WHERE pub_date > NOW() - INTERVAL '%s hours'
                ORDER BY pub_date DESC
            """, (hours,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [BBUpdate(*row) for row in results]
            
        except Exception as e:
            logger.error(f"PostgreSQL query error: {e}")
            return []

    def get_updates_in_timeframe(self, start_time: datetime, end_time: datetime) -> List[BBUpdate]:
        """Get all updates within a specific timeframe"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Use PostgreSQL syntax with %s placeholders
            cursor.execute("""
                SELECT title, description, link, pub_date, content_hash, author
                FROM updates 
                WHERE pub_date >= %s AND pub_date < %s
                ORDER BY pub_date ASC
            """, (start_time, end_time))
            
            results = cursor.fetchall()
            conn.close()
            
            updates = []
            for row in results:
                # PostgreSQL returns RealDictCursor results
                update = BBUpdate(
                    title=row['title'],
                    description=row['description'], 
                    link=row['link'],
                    pub_date=row['pub_date'],
                    content_hash=row['content_hash'],
                    author=row['author']
                )
                updates.append(update)
            
            return updates
            
        except Exception as e:
            logger.error(f"Database timeframe query error: {e}")
            return []

    def get_daily_updates(self, start_time: datetime, end_time: datetime) -> List[BBUpdate]:
        """Get all updates from a specific 24-hour period"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT title, description, link, pub_date, content_hash, author, importance_score
                FROM updates 
                WHERE pub_date >= %s AND pub_date < %s
                ORDER BY pub_date ASC
            """, (start_time, end_time))
            
            results = cursor.fetchall()
            conn.close()
            
            updates = []
            for row in results:
                update = BBUpdate(
                    title=row['title'],
                    description=row['description'],
                    link=row['link'],
                    pub_date=row['pub_date'],
                    content_hash=row['content_hash'],
                    author=row['author']
                )
                updates.append(update)
            
            return updates
            
        except Exception as e:
            logger.error(f"Database daily query error: {e}")
            return []

    

class HouseguestSelectionView(discord.ui.View):
    def __init__(self, prediction_type, duration_hours, week_number, title, description, prediction_manager, config):
        super().__init__(timeout=300)  # 5 minute timeout
        self.prediction_type = prediction_type
        self.duration_hours = duration_hours
        self.week_number = week_number
        self.poll_title = title
        self.poll_description = description
        self.prediction_manager = prediction_manager
        self.config = config
        self.selected_houseguests = []
        
        # Add the select dropdown
        self.add_item(HouseguestSelect())
    
    async def create_poll(self, interaction: discord.Interaction):
        """Create the poll with selected houseguests"""
        if len(self.selected_houseguests) < 2:
            await interaction.response.send_message(
                "Please select at least 2 houseguests for the poll.", 
                ephemeral=True
            )
            return
        
        try:
            pred_type = PredictionType(self.prediction_type)
            prediction_id = self.prediction_manager.create_prediction(
                title=self.poll_title,
                description=self.poll_description,
                prediction_type=pred_type,
                options=self.selected_houseguests,
                created_by=interaction.user.id,
                guild_id=interaction.guild.id,
                duration_hours=self.duration_hours,
                week_number=self.week_number
            )
            
            # Create success embed
            prediction_data = {
                'id': prediction_id,
                'title': self.poll_title,
                'description': self.poll_description,
                'type': self.prediction_type,
                'options': self.selected_houseguests,
                'closes_at': datetime.now() + timedelta(hours=self.duration_hours),
                'week_number': self.week_number
            }
            
            embed = self.prediction_manager.create_prediction_embed(prediction_data)
            embed.color = 0x2ecc71
            embed.title = f"✅ Poll Created - {embed.title}"
            
            await interaction.response.edit_message(
                content=f"Poll created successfully! ID: {prediction_id}",
                embed=embed,
                view=None  # Remove the view
            )
            
            # Announce in main channel
            if self.config.get('update_channel_id'):
                channel = interaction.client.get_channel(self.config.get('update_channel_id'))
                if channel:
                    announce_embed = self.prediction_manager.create_prediction_embed(prediction_data)
                    announce_embed.title = f"🗳️ New Prediction Poll - {self.poll_title}"
                    await channel.send("📢 **New Prediction Poll Created!**", embed=announce_embed)
            
        except Exception as e:
            logger.error(f"Error creating poll: {e}")
            await interaction.response.send_message("Error creating poll. Please try again.", ephemeral=True)

class HouseguestSelect(discord.ui.Select):
    def __init__(self):
        # Create options for the select menu (max 25 options)
        options = [
            discord.SelectOption(
                label=houseguest,
                value=houseguest,
                emoji="👤"
            ) for houseguest in BB27_HOUSEGUESTS
        ]
        
        super().__init__(
            placeholder="Select houseguests for the poll...",
            min_values=2,
            max_values=len(BB27_HOUSEGUESTS),  # Allow selecting all houseguests
            options=options
        )
    
    async def callback(self, interaction: discord.Interaction):
        # Update the selected houseguests
        self.view.selected_houseguests = self.values
        
        # Update the embed to show selected houseguests
        embed = discord.Embed(
            title=f"Creating Poll: {self.view.poll_title}",
            description=f"**Selected Houseguests ({len(self.values)}):**\n{', '.join(self.values)}",
            color=0x3498db
        )
        
        # Add create poll button
        if len(self.values) >= 2:
            create_button = CreatePollButton()
            # Check if button already exists
            button_exists = any(isinstance(child, CreatePollButton) for child in self.view.children)
            if not button_exists:
                self.view.add_item(create_button)
        
        await interaction.response.edit_message(embed=embed, view=self.view)

class CreatePollButton(discord.ui.Button):
    def __init__(self):
        super().__init__(
            style=discord.ButtonStyle.success,
            label="Create Poll",
            emoji="✅"
        )
    
    async def callback(self, interaction: discord.Interaction):
        await self.view.create_poll(interaction)

class PredictionSelectionView(discord.ui.View):
    def __init__(self, active_predictions, prediction_manager, user_id):
        super().__init__(timeout=300)  # 5 minute timeout
        self.active_predictions = active_predictions
        self.prediction_manager = prediction_manager
        self.user_id = user_id
        self.selected_prediction = None
        
        # Add the poll selection dropdown
        self.add_item(PollSelect(active_predictions))
    
    async def show_options_for_poll(self, interaction: discord.Interaction, selected_poll):
        """Show the options selection for the chosen poll"""
        self.selected_prediction = selected_poll
        
        # Get user's current prediction for this poll
        current_prediction = self.prediction_manager.get_user_prediction(
            self.user_id, selected_poll['id']
        )
        
        # Remove the poll selector and add the options selector
        self.clear_items()
        self.add_item(OptionsSelect(selected_poll['options'], current_prediction))
        self.add_item(ConfirmPredictionButton())
        self.add_item(BackToPollsButton(self.active_predictions))
        
        # Create embed showing the selected poll details
        embed = discord.Embed(
            title=f"🗳️ {selected_poll['title']}",
            description=selected_poll['description'],
            color=0x3498db
        )
        
        # Add timing info
        closes_at = selected_poll['closes_at']
        if isinstance(closes_at, str):
            closes_at = datetime.fromisoformat(closes_at)
        
        time_left = closes_at - datetime.now()
        if time_left.total_seconds() > 0:
            hours_left = int(time_left.total_seconds() / 3600)
            minutes_left = int((time_left.total_seconds() % 3600) / 60)
            time_str = f"{hours_left}h {minutes_left}m remaining"
        else:
            time_str = "Closed"
        
        embed.add_field(name="⏰ Time Left", value=time_str, inline=True)
        
        pred_type_names = {
            'season_winner': '👑 Season Winner',
            'weekly_hoh': '🏆 Weekly HOH',
            'weekly_veto': '💎 Weekly Veto',
            'weekly_eviction': '🚪 Weekly Eviction'
        }
        type_name = pred_type_names.get(selected_poll['type'], selected_poll['type'])
        embed.add_field(name="📊 Type", value=type_name, inline=True)
        
        if selected_poll.get('week_number'):
            embed.add_field(name="📅 Week", value=f"Week {selected_poll['week_number']}", inline=True)
        
        if current_prediction:
            embed.add_field(
                name="✅ Current Prediction",
                value=current_prediction,
                inline=False
            )
        
        embed.add_field(
            name="💡 Instructions",
            value="Select your prediction from the dropdown below, then click 'Confirm Prediction'",
            inline=False
        )
        
        await interaction.response.edit_message(embed=embed, view=self)

class PollSelect(discord.ui.Select):
    def __init__(self, active_predictions):
        # Create options for available polls (max 25)
        options = []
        
        for pred in active_predictions[:25]:  # Discord limit of 25 options
            # Calculate time left
            closes_at = pred['closes_at']
            if isinstance(closes_at, str):
                closes_at = datetime.fromisoformat(closes_at)
            
            time_left = closes_at - datetime.now()
            if time_left.total_seconds() > 0:
                hours_left = int(time_left.total_seconds() / 3600)
                time_desc = f"({hours_left}h left)"
            else:
                time_desc = "(Closed)"
            
            # Create option label and description
            label = f"{pred['title']}"
            if len(label) > 100:  # Discord limit
                label = label[:97] + "..."
            
            description = f"ID: {pred['id']} • {time_desc}"
            if len(description) > 100:
                description = description[:97] + "..."
            
            # Add emoji based on prediction type
            emoji_map = {
                'season_winner': '👑',
                'weekly_hoh': '🏆',
                'weekly_veto': '💎',
                'weekly_eviction': '🚪'
            }
            emoji = emoji_map.get(pred['type'], '📊')
            
            options.append(discord.SelectOption(
                label=label,
                value=str(pred['id']),
                description=description,
                emoji=emoji
            ))
        
        super().__init__(
            placeholder="Select a poll to make your prediction...",
            options=options,
            min_values=1,
            max_values=1
        )
    
    async def callback(self, interaction: discord.Interaction):
        selected_poll_id = int(self.values[0])
        
        # Find the selected poll
        selected_poll = None
        for pred in self.view.active_predictions:
            if pred['id'] == selected_poll_id:
                selected_poll = pred
                break
        
        if selected_poll:
            await self.view.show_options_for_poll(interaction, selected_poll)
        else:
            await interaction.response.send_message("Error: Poll not found", ephemeral=True)

class OptionsSelect(discord.ui.Select):
    def __init__(self, options, current_prediction=None):
        # Create options for houseguests
        select_options = []
        
        for option in options[:25]:  # Discord limit of 25 options
            # Check if this is the user's current prediction
            is_current = (option == current_prediction)
            
            select_options.append(discord.SelectOption(
                label=option,
                value=option,
                emoji="✅" if is_current else "👤",
                description="Current prediction" if is_current else None
            ))
        
        super().__init__(
            placeholder="Select your prediction...",
            options=select_options,
            min_values=1,
            max_values=1
        )
        
        self.selected_option = current_prediction  # Store current selection
    
    async def callback(self, interaction: discord.Interaction):
        self.selected_option = self.values[0]
        
        # Update the embed to show selection
        embed = interaction.message.embeds[0]
        
        # Update or add the "Selected Prediction" field
        field_found = False
        for i, field in enumerate(embed.fields):
            if field.name == "🎯 Selected Prediction":
                embed.set_field_at(i, name="🎯 Selected Prediction", value=self.selected_option, inline=False)
                field_found = True
                break
        
        if not field_found:
            embed.add_field(
                name="🎯 Selected Prediction",
                value=self.selected_option,
                inline=False
            )
        
        await interaction.response.edit_message(embed=embed, view=self.view)

class ConfirmPredictionButton(discord.ui.Button):
    def __init__(self):
        super().__init__(
            style=discord.ButtonStyle.success,
            label="Confirm Prediction",
            emoji="✅"
        )
    
    async def callback(self, interaction: discord.Interaction):
        # Find the options selector to get the selected option
        options_select = None
        for item in self.view.children:
            if isinstance(item, OptionsSelect):
                options_select = item
                break
        
        if not options_select or not options_select.selected_option:
            await interaction.response.send_message(
                "Please select a prediction option first!", 
                ephemeral=True
            )
            return
        
        # Make the prediction
        success = self.view.prediction_manager.make_prediction(
            user_id=self.view.user_id,
            prediction_id=self.view.selected_prediction['id'],
            option=options_select.selected_option
        )
        
        if success:
            embed = discord.Embed(
                title="✅ Prediction Confirmed!",
                description=f"**Poll:** {self.view.selected_prediction['title']}\n"
                           f"**Your Prediction:** {options_select.selected_option}\n\n"
                           f"You can change your prediction anytime before the poll closes.",
                color=0x2ecc71
            )
            
            # Remove the view (disable buttons)
            await interaction.response.edit_message(embed=embed, view=None)
        else:
            await interaction.response.send_message(
                "❌ Failed to record prediction. The poll may be closed or there was an error.",
                ephemeral=True
            )

class BackToPollsButton(discord.ui.Button):
    def __init__(self, active_predictions):
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label="Back to Polls",
            emoji="⬅️"
        )
        self.active_predictions = active_predictions
    
    async def callback(self, interaction: discord.Interaction):
        # Reset view to poll selection
        self.view.clear_items()
        self.view.add_item(PollSelect(self.active_predictions))
        self.view.selected_prediction = None
        
        embed = discord.Embed(
            title="🗳️ Make Your Prediction",
            description=f"**{len(self.active_predictions)} active polls** available\nSelect a poll to make your prediction:",
            color=0x3498db
        )
        
        # Show poll summary again
        poll_list = []
        for pred in self.active_predictions[:5]:
            closes_at = pred['closes_at']
            if isinstance(closes_at, str):
                closes_at = datetime.fromisoformat(closes_at)
            
            time_left = closes_at - datetime.now()
            if time_left.total_seconds() > 0:
                hours_left = int(time_left.total_seconds() / 3600)
                time_str = f"{hours_left}h left"
            else:
                time_str = "Closed"
            
            poll_list.append(f"**{pred['title']}** - {time_str}")
        
        if poll_list:
            embed.add_field(
                name="Available Polls",
                value="\n".join(poll_list),
                inline=False
            )
        
        await interaction.response.edit_message(embed=embed, view=self.view)

class ResolvePollView(discord.ui.View):
    def __init__(self, active_predictions, prediction_manager, admin_user_id, config):
        super().__init__(timeout=300)  # 5 minute timeout
        self.active_predictions = active_predictions
        self.prediction_manager = prediction_manager
        self.admin_user_id = admin_user_id
        self.config = config
        self.selected_prediction = None
        
        # Add the poll selection dropdown
        self.add_item(ResolvePollSelect(active_predictions))
    
    async def show_options_for_resolution(self, interaction: discord.Interaction, selected_poll):
        """Show the answer options for the chosen poll"""
        self.selected_prediction = selected_poll
        
        # Remove the poll selector and add the answer options selector
        self.clear_items()
        self.add_item(AnswerOptionsSelect(selected_poll['options']))
        self.add_item(ResolveConfirmButton())
        self.add_item(BackToResolvePollsButton(self.active_predictions))
        
        # Create embed showing the selected poll details
        embed = discord.Embed(
            title=f"🎯 Resolve Poll: {selected_poll['title']}",
            description=f"**Poll ID:** {selected_poll['id']}\n{selected_poll['description']}",
            color=0xff9800
        )
        
        # Get poll statistics
        try:
            conn = self.prediction_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM user_predictions 
                WHERE prediction_id = ?
            """, (selected_poll['id'],))
            
            total_predictions = cursor.fetchone()[0]
            conn.close()
            
        except Exception as e:
            total_predictions = "Unknown"
        
        # Add poll info
        embed.add_field(name="📊 Total Predictions", value=str(total_predictions), inline=True)
        
        pred_type_names = {
            'season_winner': '👑 Season Winner',
            'weekly_hoh': '🏆 Weekly HOH',
            'weekly_veto': '💎 Weekly Veto',
            'weekly_eviction': '🚪 Weekly Eviction'
        }
        type_name = pred_type_names.get(selected_poll['type'], selected_poll['type'])
        embed.add_field(name="📋 Type", value=type_name, inline=True)
        
        if selected_poll.get('week_number'):
            embed.add_field(name="📅 Week", value=f"Week {selected_poll['week_number']}", inline=True)
        
        # Show all options
        options_text = "\n".join([f"• {option}" for option in selected_poll['options']])
        embed.add_field(
            name="🎯 Available Options",
            value=options_text,
            inline=False
        )
        
        embed.add_field(
            name="💡 Instructions",
            value="Select the correct answer from the dropdown below, then click 'Resolve Poll'",
            inline=False
        )
        
        await interaction.response.edit_message(embed=embed, view=self)

class ResolvePollSelect(discord.ui.Select):
    def __init__(self, active_predictions):
        # Create options for polls that can be resolved (closed or active)
        options = []
        
        for pred in active_predictions[:25]:  # Discord limit of 25 options
            # Calculate time left or status
            closes_at = pred['closes_at']
            if isinstance(closes_at, str):
                closes_at = datetime.fromisoformat(closes_at)
            
            time_left = closes_at - datetime.now()
            if time_left.total_seconds() > 0:
                hours_left = int(time_left.total_seconds() / 3600)
                time_desc = f"({hours_left}h left)"
            else:
                time_desc = "(Closed)"
            
            # Create option label and description
            label = f"{pred['title']}"
            if len(label) > 100:  # Discord limit
                label = label[:97] + "..."
            
            description = f"ID: {pred['id']} • {time_desc}"
            if len(description) > 100:
                description = description[:97] + "..."
            
            # Add emoji based on prediction type
            emoji_map = {
                'season_winner': '👑',
                'weekly_hoh': '🏆',
                'weekly_veto': '💎',
                'weekly_eviction': '🚪'
            }
            emoji = emoji_map.get(pred['type'], '📊')
            
            options.append(discord.SelectOption(
                label=label,
                value=str(pred['id']),
                description=description,
                emoji=emoji
            ))
        
        super().__init__(
            placeholder="Select a poll to resolve...",
            options=options,
            min_values=1,
            max_values=1
        )
    
    async def callback(self, interaction: discord.Interaction):
        selected_poll_id = int(self.values[0])
        
        # Find the selected poll
        selected_poll = None
        for pred in self.view.active_predictions:
            if pred['id'] == selected_poll_id:
                selected_poll = pred
                break
        
        if selected_poll:
            await self.view.show_options_for_resolution(interaction, selected_poll)
        else:
            await interaction.response.send_message("Error: Poll not found", ephemeral=True)

class AnswerOptionsSelect(discord.ui.Select):
    def __init__(self, options):
        # Create options for the correct answers
        select_options = []
        
        for option in options[:25]:  # Discord limit of 25 options
            select_options.append(discord.SelectOption(
                label=option,
                value=option,
                emoji="🎯"
            ))
        
        super().__init__(
            placeholder="Select the correct answer...",
            options=select_options,
            min_values=1,
            max_values=1
        )
        
        self.selected_answer = None
    
    async def callback(self, interaction: discord.Interaction):
        self.selected_answer = self.values[0]
        
        # Update the embed to show selection
        embed = interaction.message.embeds[0]
        
        # Update or add the "Selected Answer" field
        field_found = False
        for i, field in enumerate(embed.fields):
            if field.name == "✅ Selected Correct Answer":
                embed.set_field_at(i, name="✅ Selected Correct Answer", value=self.selected_answer, inline=False)
                field_found = True
                break
        
        if not field_found:
            embed.add_field(
                name="✅ Selected Correct Answer",
                value=self.selected_answer,
                inline=False
            )
        
        await interaction.response.edit_message(embed=embed, view=self.view)

class ResolveConfirmButton(discord.ui.Button):
    def __init__(self):
        super().__init__(
            style=discord.ButtonStyle.danger,
            label="Resolve Poll",
            emoji="✅"
        )
    
    async def callback(self, interaction: discord.Interaction):
        # Find the answer selector to get the selected answer
        answer_select = None
        for item in self.view.children:
            if isinstance(item, AnswerOptionsSelect):
                answer_select = item
                break
        
        if not answer_select or not answer_select.selected_answer:
            await interaction.response.send_message(
                "Please select the correct answer first!", 
                ephemeral=True
            )
            return
        
        # Get correct users BEFORE resolving the poll
        correct_users_data = self._get_correct_users(
            self.view.selected_prediction['id'], 
            answer_select.selected_answer,
            interaction.guild
        )
        
        # Resolve the poll (now returns 3 values)
        success, correct_users_count, correct_user_ids = self.view.prediction_manager.resolve_prediction(
            prediction_id=self.view.selected_prediction['id'],
            correct_option=answer_select.selected_answer,
            admin_user_id=self.view.admin_user_id
        )
        
        if success:
            embed = discord.Embed(
                title="✅ Poll Resolved Successfully!",
                description=f"**Poll:** {self.view.selected_prediction['title']}\n"
                           f"**Correct Answer:** {answer_select.selected_answer}",
                color=0x2ecc71
            )
            
            # Calculate points awarded
            point_values = {
                'season_winner': 20,
                'weekly_hoh': 5,
                'weekly_veto': 3,
                'weekly_eviction': 2
            }
            points_per_user = point_values.get(self.view.selected_prediction['type'], 5)
            total_points = correct_users_count * points_per_user
            
            # Add winners section
            if correct_users_data:
                winners_text = self._format_winners_list(correct_users_data)
                embed.add_field(
                    name=f"🎉 Winners ({len(correct_users_data)} users)",
                    value=winners_text,
                    inline=False
                )
            else:
                embed.add_field(
                    name="😢 No Winners",
                    value="No one predicted correctly this time!",
                    inline=False
                )
            
            embed.add_field(
                name="🏆 Points Awarded",
                value=f"{points_per_user} points per correct prediction\n{total_points} total points distributed",
                inline=False
            )
            
            # Remove the view (disable buttons)
            await interaction.response.edit_message(embed=embed, view=None)
            
            # Announce resolution in main channel with winners
            if self.view.config.get('update_channel_id'):
                channel = interaction.client.get_channel(self.view.config.get('update_channel_id'))
                if channel:
                    public_embed = self._create_public_results_embed(
                        self.view.selected_prediction,
                        answer_select.selected_answer,
                        correct_users_data,
                        points_per_user
                    )
                    await channel.send(embed=public_embed)
        else:
            await interaction.response.send_message(
                "❌ Failed to resolve poll. There may have been an error.",
                ephemeral=True
            )

    def _get_correct_users(self, prediction_id, correct_answer, guild):
        """Get list of users who predicted correctly with their display names"""
        try:
            conn = self.view.prediction_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT user_id FROM user_predictions 
                WHERE prediction_id = ? AND option = ?
            """, (prediction_id, correct_answer))
            
            user_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # Get display names for users
            correct_users = []
            for user_id in user_ids:
                member = guild.get_member(user_id)
                if member:
                    correct_users.append({
                        'user_id': user_id,
                        'display_name': member.display_name,
                        'mention': member.mention
                    })
                else:
                    # User no longer in server, show partial ID
                    correct_users.append({
                        'user_id': user_id,
                        'display_name': f"User#{str(user_id)[-4:]}",
                        'mention': f"User#{str(user_id)[-4:]}"
                    })
            
            return correct_users
            
        except Exception as e:
            logger.error(f"Error getting correct users: {e}")
            return []
    
    def _format_winners_list(self, correct_users_data):
        """Format the winners list for display"""
        if not correct_users_data:
            return "No winners"
        
        if len(correct_users_data) <= 10:
            # Show all winners if 10 or fewer
            winners = [user['display_name'] for user in correct_users_data]
            return " • ".join(winners)
        else:
            # Show first 8 winners + count of remaining
            displayed_winners = [user['display_name'] for user in correct_users_data[:8]]
            remaining_count = len(correct_users_data) - 8
            winners_text = " • ".join(displayed_winners)
            winners_text += f" • +{remaining_count} more"
            return winners_text
    
    def _create_public_results_embed(self, prediction, correct_answer, correct_users_data, points_per_user):
        """Create public results embed for main channel announcement"""
        pred_type_names = {
            'season_winner': '👑 Season Winner',
            'weekly_hoh': '🏆 Weekly HOH',
            'weekly_veto': '💎 Weekly Veto',
            'weekly_eviction': '🚪 Weekly Eviction'
        }
        
        type_name = pred_type_names.get(prediction['type'], prediction['type'])
        
        embed = discord.Embed(
            title=f"🎉 Poll Results - {prediction['title']}",
            description=f"**{type_name}**\n✅ **Correct Answer:** {correct_answer}",
            color=0x2ecc71,
            timestamp=datetime.now()
        )
        
        # Add winners section
        if correct_users_data:
            winners_text = self._format_winners_list(correct_users_data)
            embed.add_field(
                name=f"🏆 Winners ({len(correct_users_data)} users)",
                value=winners_text,
                inline=False
            )
            
            # Add points info
            total_points = len(correct_users_data) * points_per_user
            embed.add_field(
                name="💎 Points Awarded",
                value=f"{points_per_user} points each • {total_points} total points distributed",
                inline=False
            )
        else:
            embed.add_field(
                name="😢 No Winners",
                value="No one predicted correctly this time!",
                inline=False
            )
        
        embed.set_footer(text="Prediction Poll Results • Check your points with /leaderboard")
        
        return embed

class BackToResolvePollsButton(discord.ui.Button):
    def __init__(self, active_predictions):
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label="Back to Polls",
            emoji="⬅️"
        )
        self.active_predictions = active_predictions
    
    async def callback(self, interaction: discord.Interaction):
        # Reset view to poll selection
        self.view.clear_items()
        self.view.add_item(ResolvePollSelect(self.active_predictions))
        self.view.selected_prediction = None
        
        embed = discord.Embed(
            title="🎯 Resolve Prediction Poll",
            description=f"**{len(self.active_predictions)} polls** available to resolve\nSelect a poll to resolve:",
            color=0xff9800
        )
        
        # Show poll summary again
        poll_list = []
        for pred in self.active_predictions[:5]:
            closes_at = pred['closes_at']
            if isinstance(closes_at, str):
                closes_at = datetime.fromisoformat(closes_at)
            
            time_left = closes_at - datetime.now()
            if time_left.total_seconds() > 0:
                hours_left = int(time_left.total_seconds() / 3600)
                time_str = f"{hours_left}h left"
            else:
                time_str = "Closed"
            
            poll_list.append(f"**{pred['title']}** - {time_str}")
        
        if poll_list:
            embed.add_field(
                name="Available Polls",
                value="\n".join(poll_list),
                inline=False
            )
        
        await interaction.response.edit_message(embed=embed, view=self.view)

class ClosePollView(discord.ui.View):
    def __init__(self, active_predictions, prediction_manager, admin_user_id):
        super().__init__(timeout=300)  # 5 minute timeout
        self.active_predictions = active_predictions
        self.prediction_manager = prediction_manager
        self.admin_user_id = admin_user_id
        self.selected_prediction = None
        
        # Add the poll selection dropdown
        self.add_item(ClosePollSelect(active_predictions))
    
    async def show_close_confirmation(self, interaction: discord.Interaction, selected_poll):
        """Show confirmation for closing the selected poll"""
        self.selected_prediction = selected_poll
        
        # Remove the poll selector and add confirmation button
        self.clear_items()
        self.add_item(ConfirmCloseButton())
        self.add_item(BackToClosePollsButton(self.active_predictions))
        
        # Get poll statistics
        try:
            conn = self.prediction_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM user_predictions 
                WHERE prediction_id = ?
            """, (selected_poll['id'],))
            
            total_predictions = cursor.fetchone()[0]
            conn.close()
            
        except Exception as e:
            total_predictions = "Unknown"
        
        # Create embed showing the selected poll details
        embed = discord.Embed(
            title=f"🔒 Close Poll: {selected_poll['title']}",
            description=f"**Poll ID:** {selected_poll['id']}\n{selected_poll['description']}",
            color=0xff6b35
        )
        
        # Add timing info
        closes_at = selected_poll['closes_at']
        if isinstance(closes_at, str):
            closes_at = datetime.fromisoformat(closes_at)
        
        time_left = closes_at - datetime.now()
        if time_left.total_seconds() > 0:
            hours_left = int(time_left.total_seconds() / 3600)
            minutes_left = int((time_left.total_seconds() % 3600) / 60)
            time_str = f"{hours_left}h {minutes_left}m remaining"
        else:
            time_str = "Already expired"
        
        embed.add_field(name="⏰ Original Time Left", value=time_str, inline=True)
        embed.add_field(name="📊 Total Predictions", value=str(total_predictions), inline=True)
        
        pred_type_names = {
            'season_winner': '👑 Season Winner',
            'weekly_hoh': '🏆 Weekly HOH',
            'weekly_veto': '💎 Weekly Veto',
            'weekly_eviction': '🚪 Weekly Eviction'
        }
        type_name = pred_type_names.get(selected_poll['type'], selected_poll['type'])
        embed.add_field(name="📋 Type", value=type_name, inline=True)
        
        if selected_poll.get('week_number'):
            embed.add_field(name="📅 Week", value=f"Week {selected_poll['week_number']}", inline=True)
        
        # Show warning about what closing does
        embed.add_field(
            name="⚠️ What happens when you close this poll:",
            value="• Users can no longer make or change predictions\n"
                  "• The poll will be locked until you resolve it\n"
                  "• You can still resolve it later with `/resolvepoll`\n"
                  "• This action cannot be undone",
            inline=False
        )
        
        embed.add_field(
            name="💡 Instructions",
            value="Click 'Close Poll' to confirm, or 'Back to Polls' to cancel",
            inline=False
        )
        
        await interaction.response.edit_message(embed=embed, view=self)

class ClosePollSelect(discord.ui.Select):
    def __init__(self, active_predictions):
        # Create options for active polls only
        options = []
        
        for pred in active_predictions[:25]:  # Discord limit of 25 options
            # Calculate time left
            closes_at = pred['closes_at']
            if isinstance(closes_at, str):
                closes_at = datetime.fromisoformat(closes_at)
            
            time_left = closes_at - datetime.now()
            if time_left.total_seconds() > 0:
                hours_left = int(time_left.total_seconds() / 3600)
                time_desc = f"({hours_left}h left)"
            else:
                time_desc = "(Expired)"
            
            # Create option label and description
            label = f"{pred['title']}"
            if len(label) > 100:  # Discord limit
                label = label[:97] + "..."
            
            description = f"ID: {pred['id']} • {time_desc}"
            if len(description) > 100:
                description = description[:97] + "..."
            
            # Add emoji based on prediction type
            emoji_map = {
                'season_winner': '👑',
                'weekly_hoh': '🏆',
                'weekly_veto': '💎',
                'weekly_eviction': '🚪'
            }
            emoji = emoji_map.get(pred['type'], '📊')
            
            options.append(discord.SelectOption(
                label=label,
                value=str(pred['id']),
                description=description,
                emoji=emoji
            ))
        
        super().__init__(
            placeholder="Select a poll to close...",
            options=options,
            min_values=1,
            max_values=1
        )
    
    async def callback(self, interaction: discord.Interaction):
        selected_poll_id = int(self.values[0])
        
        # Find the selected poll
        selected_poll = None
        for pred in self.view.active_predictions:
            if pred['id'] == selected_poll_id:
                selected_poll = pred
                break
        
        if selected_poll:
            await self.view.show_close_confirmation(interaction, selected_poll)
        else:
            await interaction.response.send_message("Error: Poll not found", ephemeral=True)

class ConfirmCloseButton(discord.ui.Button):
    def __init__(self):
        super().__init__(
            style=discord.ButtonStyle.danger,
            label="Close Poll",
            emoji="🔒"
        )
    
    async def callback(self, interaction: discord.Interaction):
        # Close the poll
        success = self.view.prediction_manager.close_prediction(
            prediction_id=self.view.selected_prediction['id'],
            admin_user_id=self.view.admin_user_id
        )
        
        if success:
            embed = discord.Embed(
                title="🔒 Poll Closed Successfully!",
                description=f"**Poll:** {self.view.selected_prediction['title']}\n\n"
                           f"✅ The poll has been closed and no longer accepts predictions.\n"
                           f"📝 You can resolve it later with `/resolvepoll` to award points.",
                color=0x2ecc71
            )
            
            embed.add_field(
                name="🎯 Next Steps",
                value="1. Wait for the actual result (who won HOH, got evicted, etc.)\n"
                      "2. Use `/resolvepoll` to set the correct answer\n"
                      "3. Points will be awarded to correct predictors",
                inline=False
            )
            
            # Remove the view (disable buttons)
            await interaction.response.edit_message(embed=embed, view=None)
            
        else:
            await interaction.response.send_message(
                "❌ Failed to close poll. It may not exist or already be closed.",
                ephemeral=True
            )

class BackToClosePollsButton(discord.ui.Button):
    def __init__(self, active_predictions):
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label="Back to Polls",
            emoji="⬅️"
        )
        self.active_predictions = active_predictions
    
    async def callback(self, interaction: discord.Interaction):
        # Reset view to poll selection
        self.view.clear_items()
        self.view.add_item(ClosePollSelect(self.active_predictions))
        self.view.selected_prediction = None
        
        embed = discord.Embed(
            title="🔒 Close Prediction Poll",
            description=f"**{len(self.active_predictions)} active polls** available\nSelect a poll to close:",
            color=0xff6b35
        )
        
        # Show poll summary again
        poll_list = []
        for pred in self.active_predictions[:5]:
            closes_at = pred['closes_at']
            if isinstance(closes_at, str):
                closes_at = datetime.fromisoformat(closes_at)
            
            time_left = closes_at - datetime.now()
            if time_left.total_seconds() > 0:
                hours_left = int(time_left.total_seconds() / 3600)
                time_str = f"{hours_left}h left"
            else:
                time_str = "Expired"
            
            poll_list.append(f"**{pred['title']}** - {time_str}")
        
        if poll_list:
            embed.add_field(
                name="Available Polls",
                value="\n".join(poll_list),
                inline=False
            )
        
        embed.add_field(
            name="💡 About Closing Polls",
            value="Closing a poll stops users from making new predictions but doesn't award points yet. "
                  "Use this when you want to 'lock in' predictions before the result is known.",
            inline=False
        )
        
        await interaction.response.edit_message(embed=embed, view=self.view)

class BBChatAnalyzer:
    """Analyzes current game state to answer user questions"""
    
    def __init__(self, db: BBDatabase, alliance_tracker: AllianceTracker, analyzer: BBAnalyzer, llm_client=None):
        self.db = db
        self.alliance_tracker = alliance_tracker
        self.analyzer = analyzer
        self.llm_client = llm_client
    
    async def answer_question(self, question: str) -> dict:
        """Answer a question about the current game state"""
        question_lower = question.lower()
        
        # Get recent updates (last 24 hours)
        recent_updates = self.db.get_recent_updates(24)
        
        # Determine question type and route to appropriate handler
        if any(keyword in question_lower for keyword in ['control', 'power', 'hoh', 'head of household']):
            return await self._analyze_power_structure(recent_updates, question)
        
        elif any(keyword in question_lower for keyword in ['danger', 'target', 'eviction', 'nominated', 'block']):
            return await self._analyze_danger_level(recent_updates, question)
        
        elif any(keyword in question_lower for keyword in ['alliance', 'working together', 'team', 'group']):
            return await self._analyze_alliances(question)
        
        elif any(keyword in question_lower for keyword in ['showmance', 'romance', 'relationship', 'dating', 'couple']):
            return await self._analyze_relationships(recent_updates, question)
        
        elif any(keyword in question_lower for keyword in ['winning', 'winner', 'favorite', 'best positioned']):
            return await self._analyze_win_chances(recent_updates, question)
        
        elif any(keyword in question_lower for keyword in ['drama', 'fight', 'argument', 'tension']):
            return await self._analyze_drama(recent_updates, question)
        
        elif any(keyword in question_lower for keyword in ['competition', 'comp', 'challenge', 'veto', 'pov']):
            return await self._analyze_competitions(recent_updates, question)
        
        else:
            return await self._general_analysis(recent_updates, question)
    
    async def _analyze_power_structure(self, updates: List[BBUpdate], question: str) -> dict:
        """Analyze who's in control of the house"""
        
        # Look for HOH wins and power-related updates
        power_updates = []
        current_hoh = None
        
        for update in updates[:20]:  # Check recent updates
            content = f"{update.title} {update.description}".lower()
            
            if 'hoh' in content or 'head of household' in content:
                if 'wins' in content or 'winner' in content:
                    # Extract potential HOH winner
                    houseguests = self.analyzer.extract_houseguests(update.title + " " + update.description)
                    if houseguests:
                        current_hoh = houseguests[0]
                power_updates.append(update)
        
        # Get active alliances to see power structure
        active_alliances = self.alliance_tracker.get_active_alliances()
        
        if self.llm_client:
            return await self._llm_power_analysis(power_updates, active_alliances, current_hoh, question)
        else:
            return self._pattern_power_analysis(power_updates, active_alliances, current_hoh)
    
    async def _analyze_danger_level(self, updates: List[BBUpdate], question: str) -> dict:
        """Analyze who's in danger of eviction"""
        
        danger_updates = []
        nominees = []
        targets = []
        
        for update in updates[:15]:
            content = f"{update.title} {update.description}".lower()
            
            if any(word in content for word in ['nominate', 'nomination', 'block', 'target', 'backdoor']):
                danger_updates.append(update)
                
                # Extract potential nominees/targets
                houseguests = self.analyzer.extract_houseguests(update.title + " " + update.description)
                if 'nominate' in content or 'block' in content:
                    nominees.extend(houseguests[:2])
                elif 'target' in content or 'backdoor' in content:
                    targets.extend(houseguests[:1])
        
        if self.llm_client:
            return await self._llm_danger_analysis(danger_updates, nominees, targets, question)
        else:
            return self._pattern_danger_analysis(danger_updates, nominees, targets)
    
    async def _analyze_alliances(self, question: str) -> dict:
        """Analyze current alliance structure"""
        
        active_alliances = self.alliance_tracker.get_active_alliances()
        broken_alliances = self.alliance_tracker.get_recently_broken_alliances(days=3)
        recent_betrayals = self.alliance_tracker.get_recent_betrayals(days=3)
        
        if self.llm_client:
            return await self._llm_alliance_analysis(active_alliances, broken_alliances, recent_betrayals, question)
        else:
            return self._pattern_alliance_analysis(active_alliances, broken_alliances, recent_betrayals)
    
    async def _analyze_relationships(self, updates: List[BBUpdate], question: str) -> dict:
        """Analyze showmances and relationships"""
        
        relationship_updates = []
        
        for update in updates[:20]:
            categories = self.analyzer.categorize_update(update)
            if "💕 Romance" in categories:
                relationship_updates.append(update)
        
        if self.llm_client:
            return await self._llm_relationship_analysis(relationship_updates, question)
        else:
            return self._pattern_relationship_analysis(relationship_updates)
    
    async def _analyze_win_chances(self, updates: List[BBUpdate], question: str) -> dict:
        """Analyze who's best positioned to win"""
        
        # Get strategic updates and alliance data
        strategic_updates = []
        for update in updates[:25]:
            if self.analyzer.analyze_strategic_importance(update) >= 6:
                strategic_updates.append(update)
        
        active_alliances = self.alliance_tracker.get_active_alliances()
        
        if self.llm_client:
            return await self._llm_winner_analysis(strategic_updates, active_alliances, question)
        else:
            return self._pattern_winner_analysis(strategic_updates, active_alliances)
    
    async def _analyze_drama(self, updates: List[BBUpdate], question: str) -> dict:
        """Analyze current drama and conflicts"""
        
        drama_updates = []
        
        for update in updates[:15]:
            categories = self.analyzer.categorize_update(update)
            if "💥 Drama" in categories:
                drama_updates.append(update)
        
        if self.llm_client:
            return await self._llm_drama_analysis(drama_updates, question)
        else:
            return self._pattern_drama_analysis(drama_updates)
    
    async def _analyze_competitions(self, updates: List[BBUpdate], question: str) -> dict:
        """Analyze recent competition results and upcoming comps"""
        
        comp_updates = []
        
        for update in updates[:20]:
            categories = self.analyzer.categorize_update(update)
            if "🏆 Competition" in categories:
                comp_updates.append(update)
        
        if self.llm_client:
            return await self._llm_competition_analysis(comp_updates, question)
        else:
            return self._pattern_competition_analysis(comp_updates)
    
    async def _general_analysis(self, updates: List[BBUpdate], question: str) -> dict:
        """General analysis for questions that don't fit specific categories"""
        
        if self.llm_client:
            return await self._llm_general_analysis(updates[:20], question)
        else:
            return self._pattern_general_analysis(updates[:20])
    
    # LLM-powered analysis methods
    async def _llm_power_analysis(self, power_updates: List[BBUpdate], alliances: List[dict], current_hoh: str, question: str) -> dict:
        """Use LLM to analyze power structure"""
        
        updates_text = "\n".join([f"• {update.title}" for update in power_updates[:10]])
        alliances_text = "\n".join([f"• {alliance['name']}: {', '.join(alliance['members'])}" for alliance in alliances[:5]])
        
        prompt = f"""Based on recent Big Brother updates, analyze the current power structure in the house.

RECENT POWER-RELATED UPDATES:
{updates_text}

CURRENT ALLIANCES:
{alliances_text}

CURRENT HOH: {current_hoh or "Unknown"}

USER QUESTION: {question}

Provide analysis in this format:
{{
    "main_answer": "Direct answer to the user's question",
    "current_hoh": "Current HOH if known",
    "power_players": ["list", "of", "houseguests", "with", "influence"],
    "power_analysis": "2-3 sentence analysis of who controls the house and why",
    "next_targets": ["potential", "targets", "for", "eviction"],
    "confidence": "high/medium/low based on available information"
}}

Focus on answering their specific question about power and control."""

        try:
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=800,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_llm_chat_response(response.content[0].text, "power")
            
        except Exception as e:
            logger.error(f"LLM power analysis failed: {e}")
            return self._pattern_power_analysis(power_updates, alliances, current_hoh)
    
    async def _llm_danger_analysis(self, danger_updates: List[BBUpdate], nominees: List[str], targets: List[str], question: str) -> dict:
        """Use LLM to analyze danger levels"""
        
        updates_text = "\n".join([f"• {update.title}" for update in danger_updates[:8]])
        nominees_text = ", ".join(list(set(nominees))) if nominees else "Unknown"
        targets_text = ", ".join(list(set(targets))) if targets else "Unknown"
        
        prompt = f"""Based on recent Big Brother updates, analyze who is in danger of eviction.

RECENT DANGER/TARGETING UPDATES:
{updates_text}

KNOWN NOMINEES: {nominees_text}
KNOWN TARGETS: {targets_text}

USER QUESTION: {question}

Provide analysis in this format:
{{
    "main_answer": "Direct answer to the user's question",
    "current_nominees": ["current", "nominees", "if", "known"],
    "biggest_threats": ["houseguests", "in", "most", "danger"],
    "danger_analysis": "2-3 sentence explanation of who's in danger and why",
    "safe_players": ["houseguests", "who", "seem", "safe"],
    "confidence": "high/medium/low based on available information"
}}

Focus on answering their specific question about danger and eviction threats."""

        try:
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=800,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_llm_chat_response(response.content[0].text, "danger")
            
        except Exception as e:
            logger.error(f"LLM danger analysis failed: {e}")
            return self._pattern_danger_analysis(danger_updates, nominees, targets)
    
    # Pattern-based fallback methods
    def _pattern_power_analysis(self, power_updates: List[BBUpdate], alliances: List[dict], current_hoh: str) -> dict:
        """Pattern-based power analysis"""
        
        power_players = []
        
        # Current HOH has power
        if current_hoh:
            power_players.append(current_hoh)
        
        # Strong alliance members have power
        for alliance in alliances:
            if alliance['confidence'] >= 70:
                power_players.extend(alliance['members'][:2])  # Top 2 from strong alliances
        
        power_players = list(set(power_players))[:5]  # Remove duplicates, limit to 5
        
        main_answer = f"Based on recent updates, "
        if current_hoh:
            main_answer += f"{current_hoh} currently holds HOH power. "
        
        if power_players:
            main_answer += f"Key power players include: {', '.join(power_players[:3])}"
        else:
            main_answer += "power structure is unclear from recent updates"
        
        return {
            "response_type": "power",
            "main_answer": main_answer,
            "current_hoh": current_hoh,
            "power_players": power_players,
            "confidence": "medium",
            "data_source": "pattern_analysis"
        }
    
    def _pattern_danger_analysis(self, danger_updates: List[BBUpdate], nominees: List[str], targets: List[str]) -> dict:
        """Pattern-based danger analysis"""
        
        all_threatened = list(set(nominees + targets))
        
        main_answer = "Based on recent targeting discussions, "
        if all_threatened:
            main_answer += f"the following houseguests appear to be in danger: {', '.join(all_threatened[:3])}"
        else:
            main_answer += "no clear eviction targets have emerged from recent updates"
        
        return {
            "response_type": "danger",
            "main_answer": main_answer,
            "threatened_players": all_threatened[:5],
            "nominees": list(set(nominees)) if nominees else [],
            "confidence": "medium",
            "data_source": "pattern_analysis"
        }
    
    def _pattern_alliance_analysis(self, active_alliances: List[dict], broken_alliances: List[dict], betrayals: List[dict]) -> dict:
        """Pattern-based alliance analysis"""
        
        strong_alliances = [a for a in active_alliances if a['confidence'] >= 70]
        
        main_answer = f"Currently tracking {len(active_alliances)} active alliances. "
        if strong_alliances:
            main_answer += f"Strongest alliance appears to be '{strong_alliances[0]['name']}' with members: {', '.join(strong_alliances[0]['members'])}"
        
        if betrayals:
            main_answer += f". Recent betrayals detected: {len(betrayals)} in the last 3 days"
        
        return {
            "response_type": "alliances",
            "main_answer": main_answer,
            "strong_alliances": strong_alliances[:3],
            "recent_betrayals": len(betrayals),
            "confidence": "high",
            "data_source": "alliance_tracker"
        }
    
    def _pattern_relationship_analysis(self, relationship_updates: List[BBUpdate]) -> dict:
        """Pattern-based relationship analysis"""
        
        if relationship_updates:
            main_answer = f"Recent showmance/relationship activity detected. Latest update: {relationship_updates[0].title}"
        else:
            main_answer = "No significant showmance or relationship developments in recent updates"
        
        return {
            "response_type": "relationships",
            "main_answer": main_answer,
            "recent_updates_count": len(relationship_updates),
            "confidence": "medium",
            "data_source": "pattern_analysis"
        }
    
    def _pattern_winner_analysis(self, strategic_updates: List[BBUpdate], alliances: List[dict]) -> dict:
        """Pattern-based winner analysis"""
        
        # Find houseguests mentioned most in strategic contexts
        houseguest_mentions = {}
        for update in strategic_updates:
            houseguests = self.analyzer.extract_houseguests(update.title + " " + update.description)
            for hg in houseguests:
                houseguest_mentions[hg] = houseguest_mentions.get(hg, 0) + 1
        
        top_strategic_players = sorted(houseguest_mentions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        main_answer = "Based on strategic activity, "
        if top_strategic_players:
            main_answer += f"the most active strategic players appear to be: {', '.join([hg[0] for hg in top_strategic_players])}"
        else:
            main_answer += "it's difficult to assess winner potential from recent updates"
        
        return {
            "response_type": "winners",
            "main_answer": main_answer,
            "strategic_players": [hg[0] for hg in top_strategic_players],
            "confidence": "low",
            "data_source": "pattern_analysis"
        }
    
    def _pattern_drama_analysis(self, drama_updates: List[BBUpdate]) -> dict:
        """Pattern-based drama analysis"""
        
        if drama_updates:
            main_answer = f"Recent drama detected: {len(drama_updates)} conflict-related updates. Latest: {drama_updates[0].title}"
        else:
            main_answer = "House appears relatively calm with no major drama in recent updates"
        
        return {
            "response_type": "drama",
            "main_answer": main_answer,
            "drama_count": len(drama_updates),
            "confidence": "medium",
            "data_source": "pattern_analysis"
        }
    
    def _pattern_competition_analysis(self, comp_updates: List[BBUpdate]) -> dict:
        """Pattern-based competition analysis"""
        
        if comp_updates:
            main_answer = f"Recent competition activity: {len(comp_updates)} competition-related updates. Latest: {comp_updates[0].title}"
        else:
            main_answer = "No recent competition results or announcements detected"
        
        return {
            "response_type": "competitions",
            "main_answer": main_answer,
            "comp_count": len(comp_updates),
            "confidence": "medium",
            "data_source": "pattern_analysis"
        }
    
    def _pattern_general_analysis(self, updates: List[BBUpdate]) -> dict:
        """Pattern-based general analysis"""
        
        if updates:
            # Get most important recent update
            top_update = max(updates, key=lambda x: self.analyzer.analyze_strategic_importance(x))
            main_answer = f"Most significant recent development: {top_update.title}"
        else:
            main_answer = "No recent significant updates to analyze"
        
        return {
            "response_type": "general",
            "main_answer": main_answer,
            "total_updates": len(updates),
            "confidence": "medium",
            "data_source": "pattern_analysis"
        }
    
    def _parse_llm_chat_response(self, response_text: str, response_type: str) -> dict:
        """Parse LLM response for chat feature"""
        try:
            # Try to extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start:json_end]
                data = json.loads(json_text)
                data["response_type"] = response_type
                data["data_source"] = "llm_analysis"
                return data
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"LLM chat JSON parsing failed: {e}")
            # Return basic response with the text
            return {
                "response_type": response_type,
                "main_answer": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "confidence": "medium",
                "data_source": "llm_analysis"
            }
    
    def create_chat_response_embed(self, analysis_result: dict, question: str) -> discord.Embed:
        """Create Discord embed for chat response"""
        
        response_type = analysis_result.get("response_type", "general")
        confidence = analysis_result.get("confidence", "medium")
        data_source = analysis_result.get("data_source", "analysis")
        
        # Set embed color based on response type
        colors = {
            "power": 0xffd700,      # Gold for power
            "danger": 0xff1744,     # Red for danger
            "alliances": 0x3498db,  # Blue for alliances
            "relationships": 0xe91e63,  # Pink for showmances
            "winners": 0x9c27b0,    # Purple for winners
            "drama": 0xff5722,      # Orange for drama
            "competitions": 0x4caf50,   # Green for comps
            "general": 0x607d8b     # Blue-gray for general
        }
        
        color = colors.get(response_type, 0x607d8b)
        
        # Set title based on response type
        titles = {
            "power": "🏛️ Power Structure Analysis",
            "danger": "⚠️ Eviction Danger Analysis", 
            "alliances": "🤝 Alliance Analysis",
            "relationships": "💕 Showmance Analysis",
            "winners": "👑 Winner Potential Analysis",
            "drama": "💥 Drama Analysis",
            "competitions": "🏆 Competition Analysis",
            "general": "📊 Game Analysis"
        }
        
        title = titles.get(response_type, "📊 Game Analysis")
        
        embed = discord.Embed(
            title=title,
            description=f"**Your Question:** {question}",
            color=color,
            timestamp=datetime.now()
        )
        
        # Main answer
        main_answer = analysis_result.get("main_answer", "Unable to provide analysis")
        embed.add_field(
            name="🎯 Analysis",
            value=main_answer,
            inline=False
        )
        
        # Add specific fields based on response type
        if response_type == "power":
            if analysis_result.get("current_hoh"):
                embed.add_field(
                    name="👑 Current HOH",
                    value=analysis_result["current_hoh"],
                    inline=True
                )
            
            if analysis_result.get("power_players"):
                players = analysis_result["power_players"][:5]
                embed.add_field(
                    name="💪 Power Players",
                    value=" • ".join(players),
                    inline=True
                )
        
        elif response_type == "danger":
            if analysis_result.get("threatened_players"):
                threatened = analysis_result["threatened_players"][:5]
                embed.add_field(
                    name="⚠️ In Danger",
                    value=" • ".join(threatened),
                    inline=True
                )
            
            if analysis_result.get("safe_players"):
                safe = analysis_result["safe_players"][:5]
                embed.add_field(
                    name="✅ Likely Safe",
                    value=" • ".join(safe),
                    inline=True
                )
        
        elif response_type == "alliances":
            if analysis_result.get("strong_alliances"):
                alliances = analysis_result["strong_alliances"][:3]
                alliance_text = []
                for alliance in alliances:
                    members = " + ".join(alliance['members'][:4])
                    alliance_text.append(f"**{alliance['name']}**: {members}")
                
                embed.add_field(
                    name="💪 Strong Alliances",
                    value="\n".join(alliance_text),
                    inline=False
                )
        
        # Add confidence and data source
        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        source_emoji = {"llm_analysis": "🤖", "pattern_analysis": "📊", "alliance_tracker": "🤝"}
        
        embed.add_field(
            name="📈 Analysis Quality",
            value=f"{confidence_emoji.get(confidence, '🟡')} {confidence.title()} confidence\n"
                  f"{source_emoji.get(data_source, '📊')} {data_source.replace('_', ' ').title()}",
            inline=True
        )
        
        embed.set_footer(text="Ask me about power, danger, alliances, showmances, winners, drama, or competitions!")
        
        return embed

class BBDiscordBot(commands.Bot):
    """Main Discord bot class with 24/7 reliability features"""
    
    def __init__(self):
        self.config = Config()
        
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        super().__init__(command_prefix='!bb', intents=intents)
        
        self.rss_url = "https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php"
        
        # Use PostgreSQL if DATABASE_URL exists, otherwise SQLite
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            logger.info("Using PostgreSQL database")
            self.db = PostgreSQLDatabase(database_url)
            self.alliance_tracker = AllianceTracker(database_url=database_url, use_postgresql=True)
            self.prediction_manager = PredictionManager(database_url=database_url, use_postgresql=True)
            self.context_tracker = HistoricalContextTracker(database_url=database_url, use_postgresql=True)
        else:
            logger.info("Using SQLite database")
            self.db = BBDatabase(self.config.get('database_path', 'bb_updates.db'))
            self.alliance_tracker = AllianceTracker(self.config.get('database_path', 'bb_updates.db'))
            self.prediction_manager = PredictionManager(self.config.get('database_path', 'bb_updates.db'))
            self.context_tracker = None
            logger.warning("Historical context tracking disabled - no database URL")
        
        self.analyzer = BBAnalyzer()
        self.update_batcher = UpdateBatcher(self.analyzer, self.config)
        
        # ADD THIS LINE: Set the database reference for the batcher
        self.update_batcher.db = self.db
        
        if self.context_tracker:
            self.update_batcher.context_tracker = self.context_tracker
        else:
            logger.warning("Context tracker not available - summaries will not include historical context")
        

        
        
        # Rest of your existing initialization code stays the same...
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.remove_command('help')
        self.setup_commands()
            
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        self.total_updates_processed = 0
        self.consecutive_errors = 0
            
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
            
    
    async def setup_hook(self):
        """Setup hook called when bot is ready"""
        try:
            # Give the batcher access to database
            self.update_batcher.db = self.db
            
            # Ensure summary tables exist first
            await self.ensure_summary_tables_exist()
            
            # Now restore queue state
            await self.restore_queues_inline()
            
            logger.info("Bot setup completed with queue restoration")
            
        except Exception as e:
            logger.error(f"Error in setup hook: {e}")
    
    async def restore_queues_inline(self):
        """Restore queue state inline"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                logger.info("No database URL for queue restoration")
                return
                
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(database_url, cursor_factory=psycopg2.extras.RealDictCursor)
            cursor = conn.cursor()
            
            # Restore highlights queue
            cursor.execute("""
                SELECT queue_state, last_summary_time 
                FROM summary_checkpoints 
                WHERE summary_type = %s
                ORDER BY updated_at DESC 
                LIMIT 1
            """, ('highlights',))
            
            result = cursor.fetchone()
            if result and result['queue_state']:
                try:
                    highlights_data = json.loads(result['queue_state']) if isinstance(result['queue_state'], str) else result['queue_state']
                    
                    # Clear and restore highlights queue
                    self.update_batcher.highlights_queue.clear()
                    for update_data in highlights_data.get('updates', []):
                        update = BBUpdate(
                            title=update_data['title'],
                            description=update_data['description'],
                            link=update_data['link'],
                            pub_date=datetime.fromisoformat(update_data['pub_date']),
                            content_hash=update_data['content_hash'],
                            author=update_data['author']
                        )
                        self.update_batcher.highlights_queue.append(update)
                    
                    # Restore last batch time
                    if result['last_summary_time']:
                        self.update_batcher.last_batch_time = datetime.fromisoformat(result['last_summary_time'])
                    
                    logger.info(f"Restored {len(self.update_batcher.highlights_queue)} highlights from database")
                    
                except Exception as e:
                    logger.error(f"Error parsing highlights queue data: {e}")
            else:
                logger.info("No highlights queue state to restore")
            
            # Restore hourly queue
            cursor.execute("""
                SELECT queue_state, last_summary_time 
                FROM summary_checkpoints 
                WHERE summary_type = %s
                ORDER BY updated_at DESC 
                LIMIT 1
            """, ('hourly',))
            
            result = cursor.fetchone()
            if result and result['queue_state']:
                try:
                    hourly_data = json.loads(result['queue_state']) if isinstance(result['queue_state'], str) else result['queue_state']
                    
                    # Clear and restore hourly queue
                    self.update_batcher.hourly_queue.clear()
                    for update_data in hourly_data.get('updates', []):
                        update = BBUpdate(
                            title=update_data['title'],
                            description=update_data['description'],
                            link=update_data['link'],
                            pub_date=datetime.fromisoformat(update_data['pub_date']),
                            content_hash=update_data['content_hash'],
                            author=update_data['author']
                        )
                        self.update_batcher.hourly_queue.append(update)
                    
                    # Restore last hourly summary time
                    if result['last_summary_time']:
                        self.update_batcher.last_hourly_summary = datetime.fromisoformat(result['last_summary_time'])
                    
                    logger.info(f"Restored {len(self.update_batcher.hourly_queue)} hourly updates from database")
                    
                except Exception as e:
                    logger.error(f"Error parsing hourly queue data: {e}")
            else:
                logger.info("No hourly queue state to restore")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error restoring queue state: {e}")
    
    def is_owner_or_admin(self, user: discord.User, interaction: discord.Interaction = None) -> bool:
        """Check if user is bot owner or has admin permissions"""
        # Check if user is bot owner
        owner_id = self.config.get('owner_id')
        if owner_id and user.id == owner_id:
            return True
        
        # Check if user has admin permissions in the guild (if in a guild)
        if interaction and interaction.guild:
            member = interaction.guild.get_member(user.id)
            if member and member.guild_permissions.administrator:
                return True
        
        return False
    
    def setup_commands(self):
        """Setup all slash commands"""
        
        # Prevent duplicate registration
        if hasattr(self, '_commands_setup') and self._commands_setup:
            logger.warning("Commands already setup, skipping duplicate registration")
            return
        
        @self.tree.command(name="status", description="Show bot status and statistics")
        async def status_slash(interaction: discord.Interaction):
            """Show bot status"""
            try:
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
                    return
                
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
                
                highlights_queue_size = len(self.update_batcher.highlights_queue)
                hourly_queue_size = len(self.update_batcher.hourly_queue)

                embed.add_field(name="Highlights Queue", value=f"{highlights_queue_size}/25", inline=True)
                embed.add_field(name="Hourly Queue", value=str(hourly_queue_size), inline=True)

                # Show time until next hourly summary
                time_since_hourly = datetime.now() - self.update_batcher.last_hourly_summary
                minutes_until_hourly = 60 - (time_since_hourly.total_seconds() / 60)
                if minutes_until_hourly <= 0:
                    hourly_status = "Ready to send!"
                else:
                    hourly_status = f"{minutes_until_hourly:.0f} min"

                embed.add_field(name="Next Hourly Summary", value=hourly_status, inline=True)
                
                llm_status = "✅ Enabled" if self.update_batcher.llm_client else "❌ Disabled"
                embed.add_field(name="LLM Summaries", value=llm_status, inline=True)
                
                # Add cache statistics
                cache_stats = self.update_batcher.processed_hashes_cache.get_stats()
                embed.add_field(
                    name="Hash Cache",
                    value=f"Size: {cache_stats['size']}/{cache_stats['capacity']}\n"
                          f"Hit Rate: {cache_stats['hit_rate']}\n"
                          f"Evictions: {cache_stats['evictions']}",
                    inline=True
                )
                
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
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
                    return
                
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
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
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
                ("/summary", "Get a summary of recent updates (Admin only)"),
                ("/status", "Show bot status and statistics (Admin only)"),
                ("/setchannel", "Set update channel (Admin only)"),
                ("/commands", "Show this help message"),
                ("/forcebatch", "Force send any queued updates (Admin only)"),
                ("/testllm", "Test LLM connection (Admin only)"),
                ("/sync", "Sync slash commands (Owner only)"),
                ("/alliances", "Show current Big Brother alliances"),
                ("/loyalty", "Show a houseguest's alliance history"),
                ("/betrayals", "Show recent alliance betrayals"),
                ("/removebadalliance", "Remove incorrectly detected alliance (Admin only)"),
                ("/clearalliances", "Clear all alliance data (Owner only)"),
                ("/zing", "Deliver a BB-style zing! (target someone, random, or self-zing)"),
                # Prediction commands
                ("/createpoll", "Create a prediction poll (Admin only)"),
                ("/predict", "Make a prediction on an active poll"),
                ("/polls", "View active prediction polls"),
                ("/closepoll", "Manually close a prediction poll (Admin only)"),
                ("/resolvepoll", "Resolve a poll and award points (Admin only)"),
                ("/leaderboard", "View prediction leaderboards"),
                ("/mypredictions", "View your prediction history"),
                ("/pollsummary", "Show a summary of all active polls (Admin/Owner only)"),
                ("/ask", "Ask questions about the current Big Brother game state")
            ]
            
            for name, description in commands_list:
                embed.add_field(name=name, value=description, inline=False)
            
            embed.set_footer(text="All commands are ephemeral (only you can see the response)")
            
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @self.tree.command(name="forcebatch", description="Force send any queued updates")
        async def forcebatch_slash(interaction: discord.Interaction):
            try:
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                highlights_size = len(self.update_batcher.highlights_queue)
                
                if highlights_size == 0:
                    await interaction.followup.send("No updates in highlights queue to send.", ephemeral=True)
                    return
                
                sent_embeds = 0
                
                # Force send highlights if any exist
                if highlights_size > 0:
                    await self.send_highlights_batch()
                    sent_embeds += 1
                
                response_msg = f"Force sent highlights batch: {highlights_size} updates\n"
                response_msg += f"Total embeds sent: {sent_embeds}"
                
                await interaction.followup.send(response_msg, ephemeral=True)
                
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

        @self.tree.command(name="alliances", description="Show current Big Brother alliances")
        async def alliances_slash(interaction: discord.Interaction):
            """Show current alliance map"""
            try:
                await interaction.response.defer()
                
                logger.info(f"Alliance command called by {interaction.user}")
                
                embed = self.alliance_tracker.create_alliance_map_embed()
                
                logger.info(f"Alliance embed created with {len(embed.fields)} fields")
                
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error showing alliances: {e}")
                logger.error(traceback.format_exc())
                await interaction.followup.send("Error generating alliance map.")

        @self.tree.command(name="loyalty", description="Show a houseguest's alliance history")
        async def loyalty_slash(interaction: discord.Interaction, houseguest: str):
            """Show loyalty information for a houseguest"""
            try:
                await interaction.response.defer()
                
                # Capitalize the name properly
                houseguest = houseguest.strip().title()
                
                embed = self.alliance_tracker.get_houseguest_loyalty_embed(houseguest)
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error showing loyalty: {e}")
                logger.error(traceback.format_exc())
                await interaction.followup.send("Error generating loyalty information.")

        @self.tree.command(name="betrayals", description="Show recent alliance betrayals")
        async def betrayals_slash(interaction: discord.Interaction, days: int = 7):
            """Show recent betrayals"""
            try:
                await interaction.response.defer()
                
                if days < 1 or days > 30:
                    await interaction.followup.send("Days must be between 1 and 30")
                    return
                
                betrayals = self.alliance_tracker.get_recent_betrayals(days)
                
                embed = discord.Embed(
                    title="💔 Recent Alliance Betrayals",
                    description=f"Betrayals in the last {days} days",
                    color=0xe74c3c,
                    timestamp=datetime.now()
                )
                
                if not betrayals:
                    embed.add_field(
                        name="No Betrayals",
                        value="No alliance betrayals detected in this time period",
                        inline=False
                    )
                else:
                    for i, betrayal in enumerate(betrayals[:10], 1):
                        time_ago = datetime.now() - datetime.fromisoformat(betrayal['timestamp'])
                        hours_ago = int(time_ago.total_seconds() / 3600)
                        time_str = f"{hours_ago}h ago" if hours_ago < 24 else f"{hours_ago//24}d ago"
                        
                        embed.add_field(
                            name=f"⚡ {time_str}",
                            value=betrayal['description'],
                            inline=False
                        )
                
                embed.set_footer(text="Based on live feed updates")
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error showing betrayals: {e}")
                logger.error(traceback.format_exc())
                await interaction.followup.send("Error generating betrayal list.")

        @self.tree.command(name="removebadalliance", description="Remove incorrectly detected alliance (Admin only)")
        async def remove_bad_alliance(interaction: discord.Interaction, alliance_name: str):
            """Remove a bad alliance"""
            try:
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
                    return
                    
                await interaction.response.defer(ephemeral=True)
                
                # Mark alliance as dissolved
                conn = self.alliance_tracker.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE alliances 
                    SET status = 'dissolved', confidence_level = 0 
                    WHERE name = ?
                """, (alliance_name,))
                
                affected = cursor.rowcount
                conn.commit()
                conn.close()
                
                if affected > 0:
                    await interaction.followup.send(f"✅ Removed alliance: **{alliance_name}**", ephemeral=True)
                    logger.info(f"Removed bad alliance: {alliance_name}")
                else:
                    await interaction.followup.send(f"❌ Alliance '{alliance_name}' not found", ephemeral=True)
                    
            except Exception as e:
                logger.error(f"Error removing alliance: {e}")
                await interaction.followup.send("Error removing alliance", ephemeral=True)

        @self.tree.command(name="clearalliances", description="Clear all alliance data (Owner only)")
        async def clear_alliances(interaction: discord.Interaction):
            """Clear all alliance data - owner only"""
            try:
                owner_id = self.config.get('owner_id')
                if not owner_id or interaction.user.id != owner_id:
                    await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                conn = self.alliance_tracker.get_connection()
                cursor = conn.cursor()
                
                # Clear all alliance data
                cursor.execute("DELETE FROM alliance_events")
                cursor.execute("DELETE FROM alliance_members")
                cursor.execute("DELETE FROM alliances")
                
                conn.commit()
                conn.close()
                
                await interaction.followup.send("✅ All alliance data has been cleared", ephemeral=True)
                logger.info("Alliance data cleared by owner")
                
            except Exception as e:
                logger.error(f"Error clearing alliances: {e}")
                await interaction.followup.send("Error clearing alliance data", ephemeral=True)

        @self.tree.command(name="zing", description="Deliver a Big Brother style zing!")
        @discord.app_commands.describe(
            zing_type="Choose your zing type",
            target="Only required for targeted zings (leave blank for random/self)"
        )
        @discord.app_commands.choices(zing_type=[
            discord.app_commands.Choice(name="Targeted Zing", value="targeted"),
            discord.app_commands.Choice(name="Random Zing", value="random"),
            discord.app_commands.Choice(name="Self Zing", value="self")
        ])
        async def zing_slash(interaction: discord.Interaction, 
                            zing_type: discord.app_commands.Choice[str],
                            target: discord.Member = None):
            """Deliver a zing to someone"""
            try:
                zing_choice = zing_type.value
                
                # Determine the actual target based on choice
                if zing_choice == "targeted":
                    if not target:
                        await interaction.response.send_message("Please specify a target for a targeted zing!", ephemeral=True)
                        return
                    final_target = target
                elif zing_choice == "random":
                    if target:
                        await interaction.response.send_message("Random zing doesn't need a target - it will pick someone automatically!", ephemeral=True)
                        return
                    # Get all members in the server (excluding bots) - INCLUDE the command user
                    members = [m for m in interaction.guild.members if not m.bot]
                    if not members:
                        await interaction.response.send_message("No valid members to zing!", ephemeral=True)
                        return
                    import random as rand_module
                    final_target = rand_module.choice(members)
                elif zing_choice == "self":
                    if target:
                        await interaction.response.send_message("Self zing doesn't need a target - you're zinging yourself!", ephemeral=True)
                        return
                    # Self-zing
                    final_target = interaction.user
                else:
                    await interaction.response.send_message("Invalid zing type!", ephemeral=True)
                    return
                
                # Select a random zing
                import random as rand_module
                zing = rand_module.choice(ALL_ZINGS)
                
                # Replace {target} with the actual mention
                zing_text = zing.replace("{target}", final_target.mention)
                
                # If the zing contains [name], replace it with a random other member's name
                if "[name]" in zing_text:
                    other_members = [m for m in interaction.guild.members if not m.bot and m.id != final_target.id]
                    if other_members:
                        other_member = rand_module.choice(other_members)
                        zing_text = zing_text.replace("[name]", other_member.display_name)
                    else:
                        # Fallback if no other members
                        zing_text = zing_text.replace("[name]", "someone")
                
                # Create the embed
                embed = discord.Embed(
                    title=":robot: ZING!",
                    description=zing_text,
                    color=0xff1744 if final_target == interaction.user else 0xff9800
                )
                
                # Add Zingbot style footer
                zingbot_phrases = [
                    "ZING! That's what I'm programmed for!",
                    "Another successful zing delivered!",
                    "Zingbot 3000 strikes again!",
                    "I came, I saw, I ZINGED!",
                    "My circuits are buzzing with that zing!",
                    "Zing executed successfully!",
                    "Target acquired and zinged!",
                    "Maximum zing efficiency achieved!"
                ]
                
                embed.set_footer(text=rand_module.choice(zingbot_phrases))
                
                # Add special effects for self-zings
                if final_target == interaction.user:
                    embed.add_field(
                        name="😅 Self-Zing Award",
                        value="You zinged yourself! That takes guts... or poor decision making!",
                        inline=False
                    )
                
                await interaction.response.send_message(embed=embed)
                
                # Log the zing
                logger.info(f"Zing delivered: {interaction.user} zinged {final_target}")
                
            except Exception as e:
                logger.error(f"Error delivering zing: {e}")
                await interaction.response.send_message("Error delivering zing. My circuits must be malfunctioning!", ephemeral=True)

        # PREDICTION SYSTEM COMMANDS
        @self.tree.command(name="createpoll", description="Create a prediction poll (Admin only)")
        @discord.app_commands.describe(
            prediction_type="Type of prediction",
            duration_hours="How long the poll stays open (hours)",
            week_number="Week number (required for weekly predictions, ignored for season winner)"
        )
        @discord.app_commands.choices(prediction_type=[
            discord.app_commands.Choice(name="👑 Season Winner", value="season_winner"),
            discord.app_commands.Choice(name="👢 First Boot - Womp Womp", value="first_boot"),
            discord.app_commands.Choice(name="🏆 Weekly HOH", value="weekly_hoh"),
            discord.app_commands.Choice(name="💎 Weekly Veto", value="weekly_veto"),
            discord.app_commands.Choice(name="🚪 Weekly Eviction", value="weekly_eviction")
        ])
        async def createpoll_slash(interaction: discord.Interaction, 
                                  prediction_type: discord.app_commands.Choice[str],
                                  duration_hours: int,
                                  week_number: int = None):
            """Create a prediction poll with interactive houseguest selection"""
            try:
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
                    return
                
                if duration_hours < 1 or duration_hours > 168:
                    await interaction.response.send_message("Duration must be between 1 and 168 hours.", ephemeral=True)
                    return
                
                # Validate week_number for weekly predictions
                pred_type_value = prediction_type.value
                if pred_type_value not in ["season_winner", "first_boot"]:  # Update this line
                    if week_number is None or week_number < 1:
                        await interaction.response.send_message("Week number is required for weekly predictions and must be 1 or greater.", ephemeral=True)
                        return
                
                # Auto-generate title and description
                if pred_type_value == "season_winner":
                    title = "Season Winner"
                    description = "Season 27"
                    week_number = None
                elif pred_type_value == "first_boot":  # Add this block
                    title = "First Boot - Womp Womp"
                    description = "Season 27"
                    week_number = None
                elif pred_type_value == "weekly_hoh":
                    title = f"Week {week_number} HOH"
                    description = f"Week {week_number}"
                elif pred_type_value == "weekly_veto":
                    title = f"Week {week_number} Veto"
                    description = f"Week {week_number}"
                elif pred_type_value == "weekly_eviction":
                    title = f"Week {week_number} Eviction"
                    description = f"Week {week_number}"
                else:
                    await interaction.response.send_message("Invalid prediction type.", ephemeral=True)
                    return
                
                # Create the view with houseguest selection
                view = HouseguestSelectionView(
                    prediction_type=pred_type_value,
                    duration_hours=duration_hours,
                    week_number=week_number,
                    title=title,
                    description=description,
                    prediction_manager=self.prediction_manager,
                    config=self.config
                )
                
                embed = discord.Embed(
                    title=f"Creating Poll: {title}",
                    description="Please select the houseguests for this poll using the dropdown below.",
                    color=0x3498db
                )
                
                embed.add_field(name="Poll Type", value=prediction_type.name, inline=True)
                embed.add_field(name="Duration", value=f"{duration_hours} hours", inline=True)
                if week_number:
                    embed.add_field(name="Week", value=str(week_number), inline=True)
                
                await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error in createpoll command: {e}")
                if not interaction.response.is_done():
                    await interaction.response.send_message("Error creating poll.", ephemeral=True)

        @self.tree.command(name="predict", description="Make a prediction on an active poll")
        async def predict_slash(interaction: discord.Interaction):
            """Make a prediction using interactive selection"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                # Get active predictions for this guild
                active_predictions = self.prediction_manager.get_active_predictions(interaction.guild.id)
                
                if not active_predictions:
                    embed = discord.Embed(
                        title="📊 No Active Polls",
                        description="There are no active prediction polls right now.\nAsk an admin to create one with `/createpoll`!",
                        color=0x95a5a6
                    )
                    await interaction.followup.send(embed=embed, ephemeral=True)
                    return
                
                # Create the prediction selection view
                view = PredictionSelectionView(
                    active_predictions=active_predictions,
                    prediction_manager=self.prediction_manager,
                    user_id=interaction.user.id
                )
                
                embed = discord.Embed(
                    title="🗳️ Make Your Prediction",
                    description=f"**{len(active_predictions)} active polls** available\nSelect a poll to make your prediction:",
                    color=0x3498db
                )
                
                # Show a summary of active polls
                poll_list = []
                for pred in active_predictions[:5]:  # Show up to 5 polls
                    closes_at = pred['closes_at']
                    if isinstance(closes_at, str):
                        closes_at = datetime.fromisoformat(closes_at)
                    
                    time_left = closes_at - datetime.now()
                    if time_left.total_seconds() > 0:
                        hours_left = int(time_left.total_seconds() / 3600)
                        time_str = f"{hours_left}h left"
                    else:
                        time_str = "Closed"
                    
                    poll_list.append(f"**{pred['title']}** - {time_str}")
                
                if poll_list:
                    embed.add_field(
                        name="Available Polls",
                        value="\n".join(poll_list),
                        inline=False
                    )
                
                await interaction.followup.send(embed=embed, view=view, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error in predict command: {e}")
                await interaction.followup.send("Error loading prediction polls.", ephemeral=True)

        @self.tree.command(name="polls", description="View active prediction polls")
        async def polls_slash(interaction: discord.Interaction):
            """View active polls"""
            try:
                await interaction.response.defer(ephemeral=True)  # Changed to ephemeral=True
                
                active_predictions = self.prediction_manager.get_active_predictions(interaction.guild.id)
                
                if not active_predictions:
                    embed = discord.Embed(
                        title="📊 Active Prediction Polls",
                        description="No active polls right now.",
                        color=0x95a5a6
                    )
                    await interaction.followup.send(embed=embed, ephemeral=True)  # Added ephemeral=True
                    return
                
                # Show up to 5 active polls
                for prediction in active_predictions[:5]:
                    user_prediction = self.prediction_manager.get_user_prediction(
                        interaction.user.id, prediction['id']
                    )
                    
                    embed = self.prediction_manager.create_prediction_embed(prediction, user_prediction)
                    await interaction.followup.send(embed=embed, ephemeral=True)  # Added ephemeral=True
                
                if len(active_predictions) > 5:
                    await interaction.followup.send(
                        f"*Showing 5 of {len(active_predictions)} active polls. Use `/predict` to make your predictions.*",
                        ephemeral=True  # Added ephemeral=True
                    )
                
            except Exception as e:
                logger.error(f"Error showing polls: {e}")
                await interaction.followup.send("Error retrieving polls.", ephemeral=True)  # Added ephemeral=True

        @self.tree.command(name="closepoll", description="Manually close a prediction poll (Admin only)")
        async def closepoll_slash(interaction: discord.Interaction):
            """Close a poll using interactive selection"""
            try:
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                # Get active predictions that can be closed
                active_predictions = self.prediction_manager.get_active_predictions(interaction.guild.id)
                
                if not active_predictions:
                    embed = discord.Embed(
                        title="📊 No Active Polls",
                        description="There are no active polls that can be closed right now.",
                        color=0x95a5a6
                    )
                    await interaction.followup.send(embed=embed, ephemeral=True)
                    return
                
                # Create the close poll view
                view = ClosePollView(
                    active_predictions=active_predictions,
                    prediction_manager=self.prediction_manager,
                    admin_user_id=interaction.user.id
                )
                
                embed = discord.Embed(
                    title="🔒 Close Prediction Poll",
                    description=f"**{len(active_predictions)} active polls** available\nSelect a poll to close:",
                    color=0xff6b35
                )
                
                # Show a summary of active polls
                poll_list = []
                for pred in active_predictions[:5]:  # Show up to 5 polls
                    closes_at = pred['closes_at']
                    if isinstance(closes_at, str):
                        closes_at = datetime.fromisoformat(closes_at)
                    
                    time_left = closes_at - datetime.now()
                    if time_left.total_seconds() > 0:
                        hours_left = int(time_left.total_seconds() / 3600)
                        time_str = f"{hours_left}h left"
                    else:
                        time_str = "Expired"
                    
                    poll_list.append(f"**{pred['title']}** - {time_str}")
                
                if poll_list:
                    embed.add_field(
                        name="Available Polls",
                        value="\n".join(poll_list),
                        inline=False
                    )
                
                embed.add_field(
                    name="💡 About Closing Polls",
                    value="Closing a poll stops users from making new predictions but doesn't award points yet. "
                          "Use this when you want to 'lock in' predictions before the result is known.",
                    inline=False
                )
                
                await interaction.followup.send(embed=embed, view=view, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error in closepoll command: {e}")
                await interaction.followup.send("Error loading polls to close.", ephemeral=True)

        @self.tree.command(name="resolvepoll", description="Resolve a poll and award points (Admin only)")
        async def resolvepoll_slash(interaction: discord.Interaction):
            """Resolve a poll using interactive selection"""
            try:
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                # Get polls that can be resolved (both active and closed)
                conn = self.prediction_manager.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT prediction_id, title, description, prediction_type, 
                           options, closes_at, week_number, status
                    FROM predictions 
                    WHERE guild_id = ? AND status IN ('active', 'closed')
                    ORDER BY closes_at DESC
                """, (interaction.guild.id,))
                
                results = cursor.fetchall()
                conn.close()
                
                if not results:
                    embed = discord.Embed(
                        title="📊 No Polls to Resolve",
                        description="There are no active or closed polls that can be resolved.",
                        color=0x95a5a6
                    )
                    await interaction.followup.send(embed=embed, ephemeral=True)
                    return
                
                # Convert to the format expected by the view
                resolvable_predictions = []
                for row in results:
                    pred_id, title, desc, pred_type, options_json, closes_at, week_num, status = row
                    resolvable_predictions.append({
                        'id': pred_id,
                        'title': title,
                        'description': desc,
                        'type': pred_type,
                        'options': json.loads(options_json),
                        'closes_at': closes_at,
                        'week_number': week_num,
                        'status': status
                    })
                
                # Create the resolution view
                view = ResolvePollView(
                    active_predictions=resolvable_predictions,
                    prediction_manager=self.prediction_manager,
                    admin_user_id=interaction.user.id,
                    config=self.config
                )
                
                embed = discord.Embed(
                    title="🎯 Resolve Prediction Poll",
                    description=f"**{len(resolvable_predictions)} polls** available to resolve\nSelect a poll to resolve:",
                    color=0xff9800
                )
                
                # Show a summary of resolvable polls
                poll_list = []
                for pred in resolvable_predictions[:5]:  # Show up to 5 polls
                    closes_at = pred['closes_at']
                    if isinstance(closes_at, str):
                        closes_at = datetime.fromisoformat(closes_at)
                    
                    time_left = closes_at - datetime.now()
                    if time_left.total_seconds() > 0:
                        hours_left = int(time_left.total_seconds() / 3600)
                        time_str = f"{hours_left}h left"
                    else:
                        time_str = "Closed"
                    
                    status_emoji = "🔴" if pred['status'] == 'closed' else "🟢"
                    poll_list.append(f"{status_emoji} **{pred['title']}** - {time_str}")
                
                if poll_list:
                    embed.add_field(
                        name="Available Polls",
                        value="\n".join(poll_list),
                        inline=False
                    )
                
                embed.add_field(
                    name="💡 Note",
                    value="🟢 = Active polls | 🔴 = Closed polls\nBoth can be resolved to award points.",
                    inline=False
                )
                
                await interaction.followup.send(embed=embed, view=view, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error in resolvepoll command: {e}")
                await interaction.followup.send("Error loading polls for resolution.", ephemeral=True)

        @self.tree.command(name="leaderboard", description="View prediction leaderboards")
        async def leaderboard_slash(interaction: discord.Interaction):
            """View leaderboards"""
            try:
                await interaction.response.defer()
                    
                # Only show season leaderboard
                leaderboard = self.prediction_manager.get_season_leaderboard(interaction.guild.id)
                embed = await self.prediction_manager.create_leaderboard_embed(
                    leaderboard, interaction.guild, "Season"
                )
                    
                await interaction.followup.send(embed=embed)
                    
            except Exception as e:
                logger.error(f"Error showing leaderboard: {e}")
                await interaction.followup.send("Error retrieving leaderboard.")

        @self.tree.command(name="mypredictions", description="View your prediction history")
        async def mypredictions_slash(interaction: discord.Interaction):
            """View user's prediction history"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                history = self.prediction_manager.get_user_predictions_history(
                    interaction.user.id, interaction.guild.id
                )
                
                if not history:
                    embed = discord.Embed(
                        title="📊 Your Prediction History",
                        description="You haven't made any predictions yet!",
                        color=0x95a5a6
                    )
                    await interaction.followup.send(embed=embed, ephemeral=True)
                    return
                
                # Calculate user stats
                total_predictions = len(history)
                correct_predictions = sum(1 for h in history if h['is_correct'])
                total_points = sum(h['points_earned'] for h in history)
                accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                
                embed = discord.Embed(
                    title="📊 Your Prediction History",
                    description=f"**Total Points:** {total_points}\n"
                               f"**Accuracy:** {correct_predictions}/{total_predictions} ({accuracy:.1f}%)",
                    color=0x3498db,
                    timestamp=datetime.now()
                )
                
                # Show recent predictions
                history_text = []
                for pred in history[:10]:  # Show last 10
                    status_emoji = "✅" if pred['is_correct'] else "❌" if pred['is_correct'] is False else "⏳"
                    points_text = f"(+{pred['points_earned']} pts)" if pred['points_earned'] > 0 else ""
                    
                    history_text.append(
                        f"{status_emoji} **{pred['title']}**\n"
                        f"   Your pick: {pred['user_option']} {points_text}"
                    )
                
                if history_text:
                    embed.add_field(
                        name="Recent Predictions",
                        value="\n\n".join(history_text),
                        inline=False
                    )
                
                embed.set_footer(text="Only showing last 10 predictions")
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error showing user predictions: {e}")
                await interaction.followup.send("Error retrieving your predictions.", ephemeral=True)

        @self.tree.command(name="pollsummary", description="Show a summary of all active polls (Admin/Owner only)")
        async def pollsummary_slash(interaction: discord.Interaction):
            """Show a summary of all active polls publicly"""
            try:
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
                    return
                    
                await interaction.response.defer()
                
                active_predictions = self.prediction_manager.get_active_predictions(interaction.guild.id)
                
                if not active_predictions:
                    embed = discord.Embed(
                        title="📊 Active Prediction Polls",
                        description="No active polls right now.",
                        color=0x95a5a6
                    )
                    await interaction.followup.send(embed=embed)
                    return
                
                embed = discord.Embed(
                    title="🗳️ Active Prediction Polls",
                    description=f"**{len(active_predictions)} active polls** - Use `/predict` to participate!",
                    color=0x3498db,
                    timestamp=datetime.now()
                )
                
                for prediction in active_predictions:
                    # Calculate time left
                    closes_at = prediction['closes_at']
                    if isinstance(closes_at, str):
                        closes_at = datetime.fromisoformat(closes_at)
                    
                    time_left = closes_at - datetime.now()
                    if time_left.total_seconds() > 0:
                        hours_left = int(time_left.total_seconds() / 3600)
                        minutes_left = int((time_left.total_seconds() % 3600) / 60)
                        time_str = f"{hours_left}h {minutes_left}m remaining"
                    else:
                        time_str = "Closed"
                    
                    # Get point value
                    point_values = {
                        'season_winner': 20,
                        'first_boot': 15,
                        'weekly_hoh': 5,
                        'weekly_veto': 3,
                        'weekly_eviction': 2
                    }
                    points = point_values.get(prediction['type'], 5)
                    
                    # Add field for each poll
                    embed.add_field(
                        name=f"🎯 {prediction['title']}",
                        value=f"**{time_str}** • {points} points • ID: {prediction['id']}\n"
                              f"{len(prediction['options'])} options available",
                        inline=False
                    )
                
                embed.add_field(
                    name="💡 How to Participate",
                    value="Use `/predict` to make your predictions privately!\n"
                          "Use `/polls` to see detailed poll information.",
                    inline=False
                )
                
                embed.set_footer(text="Prediction System • Points awarded for correct predictions")
                
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error showing poll summary: {e}")
                await interaction.followup.send("Error retrieving poll summary.")

        
        @self.tree.command(name="ask", description="Ask questions about the current Big Brother game")
        @discord.app_commands.describe(question="Your question about the current game state")
        async def ask_slash(interaction: discord.Interaction, question: str):
            """Chat feature for game analysis"""
            try:
                await interaction.response.defer()
                
                # Initialize chat analyzer
                chat_analyzer = BBChatAnalyzer(
                    db=self.db,
                    alliance_tracker=self.alliance_tracker,
                    analyzer=self.analyzer,
                    llm_client=self.update_batcher.llm_client
                )
                
                # Get analysis
                analysis_result = await chat_analyzer.answer_question(question)
                
                # Create response embed
                embed = chat_analyzer.create_chat_response_embed(analysis_result, question)
                
                await interaction.followup.send(embed=embed)
                
                logger.info(f"Chat question answered: {interaction.user} asked '{question[:50]}...'")
                
            except Exception as e:
                logger.error(f"Error in ask command: {e}")
                logger.error(traceback.format_exc())
                
                error_embed = discord.Embed(
                    title="❌ Analysis Error",
                    description="Sorry, I couldn't analyze the current game state. Please try again later.",
                    color=0xe74c3c
                )
                
                await interaction.followup.send(embed=error_embed)

        @self.tree.command(name="testcontext", description="Test historical context integration (Admin only)")
        async def test_context_slash(interaction: discord.Interaction, houseguest: str = ""):
            """Test context integration"""
            try:
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                if not self.context_tracker:
                    await interaction.followup.send("❌ Context tracker not available", ephemeral=True)
                    return
                
                embed = discord.Embed(
                    title="🧠 Context Integration Test",
                    color=0x3498db,
                    timestamp=datetime.now()
                )
                
                if houseguest:
                    # Test specific houseguest context
                    houseguest = houseguest.strip().title()
                    context = await self.context_tracker.get_historical_context(houseguest)
                    
                    embed.add_field(
                        name=f"Context for {houseguest}",
                        value=context if context else "No historical context found",
                        inline=False
                    )
                else:
                    # Test recent events
                    recent_updates = self.db.get_recent_updates(2)
                    if recent_updates:
                        test_update = recent_updates[0]
                        detected_events = await self.context_tracker.analyze_update_for_events(test_update)
                        
                        embed.add_field(
                            name="Recent Update Analysis",
                            value=f"**Update**: {test_update.title[:100]}...\n**Events Detected**: {len(detected_events)}",
                            inline=False
                        )
                        
                        if detected_events:
                            events_text = "\n".join([f"• {event['type']}: {event.get('description', 'No description')}" for event in detected_events[:3]])
                            embed.add_field(
                                name="Detected Events",
                                value=events_text,
                                inline=False
                            )
                    else:
                        embed.add_field(
                            name="Test Status",
                            value="No recent updates to analyze",
                            inline=False
                        )
                
                # Show integration status
                integration_status = "✅ Active" if hasattr(self.update_batcher, 'context_tracker') and self.update_batcher.context_tracker else "❌ Not Connected"
                embed.add_field(
                    name="Integration Status",
                    value=f"Context Tracker: {integration_status}",
                    inline=False
                )
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error testing context: {e}")
                await interaction.followup.send("Error testing context integration.", ephemeral=True)
    
        @self.tree.command(name="recreatecontexttables", description="Recreate context tables (Owner only)")
        async def recreate_context_tables(interaction: discord.Interaction):
            """Recreate context tables"""
            try:
                owner_id = self.config.get('owner_id')
                if not owner_id or interaction.user.id != owner_id:
                    await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                database_url = os.getenv('DATABASE_URL')
                if not database_url:
                    await interaction.followup.send("No database URL available", ephemeral=True)
                    return
                
                import psycopg2
                conn = psycopg2.connect(database_url)
                cursor = conn.cursor()
                
                # Drop and recreate the missing tables
                cursor.execute("DROP TABLE IF EXISTS context_cache")
                cursor.execute("DROP TABLE IF EXISTS houseguest_relationships")
                cursor.execute("DROP TABLE IF EXISTS houseguest_stats") 
                cursor.execute("DROP TABLE IF EXISTS houseguest_events")
                
                # Recreate them with the correct schema
                cursor.execute("""
                    CREATE TABLE houseguest_events (
                        event_id SERIAL PRIMARY KEY,
                        houseguest_name VARCHAR(100) NOT NULL,
                        event_type VARCHAR(50) NOT NULL,
                        description TEXT,
                        week_number INTEGER,
                        season_day INTEGER,
                        update_hash VARCHAR(32),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE houseguest_stats (
                        stat_id SERIAL PRIMARY KEY,
                        houseguest_name VARCHAR(100) NOT NULL,
                        stat_type VARCHAR(50) NOT NULL,
                        stat_value INTEGER DEFAULT 0,
                        season_total INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(houseguest_name, stat_type)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE houseguest_relationships (
                        relationship_id SERIAL PRIMARY KEY,
                        houseguest_1 VARCHAR(100) NOT NULL,
                        houseguest_2 VARCHAR(100) NOT NULL,
                        relationship_type VARCHAR(50) NOT NULL,
                        strength_score INTEGER DEFAULT 50,
                        first_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status VARCHAR(20) DEFAULT 'active',
                        duration_days INTEGER DEFAULT 0,
                        CHECK (houseguest_1 < houseguest_2)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE context_cache (
                        cache_id SERIAL PRIMARY KEY,
                        cache_key VARCHAR(100) NOT NULL UNIQUE,
                        context_text TEXT NOT NULL,
                        expires_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                conn.close()
                
                await interaction.followup.send("✅ Context tables recreated successfully!", ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error recreating tables: {e}")
                await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)
        # I'll continue with the rest of the commands in the next step...
        # For now, this should fix your immediate startup error

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
            self.daily_recap_task.start()
            self.auto_close_predictions_task.start()
            self.hourly_summary_task.start()  # ADD THIS LINE
            logger.info("RSS feed monitoring, daily recap, prediction auto-close, and hourly summary tasks started")
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
        if not hasattr(self, '_cleanup_task_started'):
            self._cleanup_task_started = True
            self.loop.create_task(self._periodic_cleanup())
        
    async def _periodic_cleanup(self):
        """Periodic cleanup of old checkpoints"""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.update_batcher.clear_old_checkpoints()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
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
    
    async def filter_duplicates(self, updates: List[BBUpdate]) -> List[BBUpdate]:
        """Filter out duplicate updates"""
        new_updates = []
        
        for update in updates:
            # Check both database and cache
            if not self.db.is_duplicate(update.content_hash):
                if not await self.update_batcher.processed_hashes_cache.contains(update.content_hash):
                    new_updates.append(update)
        
        return new_updates
    

    @tasks.loop(hours=24)
    async def daily_recap_task(self):
        """Daily recap task that runs at 8:00 AM Pacific Time"""
        if self.is_shutting_down:
            return
        
        # ADD THIS BULLETPROOF TIME CHECK:
        try:
            # BULLETPROOF TIME CHECK - Only run at 8:00 AM Pacific
            pacific_tz = pytz.timezone('US/Pacific')
            now_pacific = datetime.now(pacific_tz)
            
            # Only run between 7:55 AM and 8:05 AM Pacific (10-minute window)
            if not (7 <= now_pacific.hour <= 8):
                logger.info(f"Daily recap skipped - wrong hour: {now_pacific.strftime('%I:%M %p Pacific')} (need 8:00 AM Pacific)")
                return
            
            if now_pacific.hour == 7 and now_pacific.minute < 55:
                logger.info(f"Daily recap skipped - too early: {now_pacific.strftime('%I:%M %p Pacific')}")
                return
                
            if now_pacific.hour == 8 and now_pacific.minute > 5:
                logger.info(f"Daily recap skipped - too late: {now_pacific.strftime('%I:%M %p Pacific')}")
                return
            
            # If we get here, it's the right time!
            logger.info(f"Daily recap starting at correct time: {now_pacific.strftime('%I:%M %p Pacific')}")
            
            # THEN continue with your existing code starting from line 9171:
            # Only proceed if we have an update channel configured
            if not self.config.get('update_channel_id'):
                logger.debug("No update channel configured for daily recap")
                return
            
            logger.info("Starting daily recap generation")        
            
            # Only proceed if we have an update channel configured
            if not self.config.get('update_channel_id'):
                logger.debug("No update channel configured for daily recap")
                return
            
            logger.info("Starting daily recap generation")
            
            
            # Calculate the day period (previous 8:00 AM to current 8:00 AM)
            end_time = now_pacific.replace(hour=8, minute=0, second=0, microsecond=0, tzinfo=None)
            start_time = end_time - timedelta(hours=24)  # 24 hours ago
            
            # Get all updates from the day
            daily_updates = self.db.get_daily_updates(start_time, end_time)
            
            if not daily_updates:
                logger.info("No updates found for daily recap")
                # Still send a "quiet day" recap
                await self.send_quiet_day_recap()
                return
            
            # Calculate day number (days since season start)
            season_start = datetime(2025, 7, 8)  # Adjust this date for actual season start
            day_number = (end_time.date() - season_start.date()).days + 1
            
            # Create daily recap
            recap_embeds = await self.update_batcher.create_daily_recap(daily_updates, day_number)
            
            # Send daily recap
            await self.send_daily_recap(recap_embeds)
            
            logger.info(f"Daily recap sent for Day {day_number} with {len(daily_updates)} updates")
            
        except Exception as e:
            logger.error(f"Error in daily recap task: {e}")
            logger.error(traceback.format_exc())
    
    @daily_recap_task.before_loop
    async def before_daily_recap_task(self):
        """Wait for bot to be ready and sync to 8:00 AM Pacific"""
        await self.wait_until_ready()
        
        try:
            # Calculate wait time until next 8:00 AM Pacific
            pacific_tz = pytz.timezone('US/Pacific')
            now_pacific = datetime.now(pacific_tz)
            
            # Get next 8:00 AM Pacific
            next_recap = now_pacific.replace(hour=8, minute=0, second=0, microsecond=0)
            
            # If it's already past 8:00 AM today, schedule for tomorrow
            if now_pacific.hour >= 8:
                next_recap += timedelta(days=1)
            
            wait_seconds = (next_recap - now_pacific).total_seconds()
            
            logger.info(f"Daily recap task will start in {wait_seconds:.0f} seconds (at {next_recap.strftime('%A, %B %d at %I:%M %p Pacific')})")
            
            # Wait until the scheduled time
            await asyncio.sleep(wait_seconds)
            
        except Exception as e:
            logger.error(f"Error in daily recap before_loop: {e}")
            # If there's an error, wait 1 hour and try again
            await asyncio.sleep(3600)
    
    @tasks.loop(minutes=60)  # Run every 60 minutes
    async def hourly_summary_task(self):
        """Send hourly summary at the top of each hour"""
        if self.is_shutting_down:
            return
        
        try:
            # Only proceed if we have an update channel configured
            if not self.config.get('update_channel_id'):
                logger.debug("No update channel configured for hourly summary")
                return
            
            # Send the hourly summary
            await self.send_hourly_summary()
            logger.info("Hourly summary task completed successfully")
                
        except Exception as e:
            logger.error(f"Error in hourly summary task: {e}")
    
    @hourly_summary_task.before_loop
    async def before_hourly_summary_task(self):
        """Wait for bot to be ready and sync to top of hour"""
        await self.wait_until_ready()
        
        # Calculate wait time until next top of hour
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        wait_seconds = (next_hour - now).total_seconds()
        
        logger.info(f"Hourly summary task will start in {wait_seconds:.0f} seconds (at {next_hour.strftime('%I:%M %p')})")
        await asyncio.sleep(wait_seconds)
    
    @daily_recap_task.before_loop
    async def before_daily_recap_task(self):
        """Wait for bot to be ready before starting daily recap task"""
        await self.wait_until_ready()
    
    @tasks.loop(minutes=30)
    async def auto_close_predictions_task(self):
        """Auto-close expired predictions every 30 minutes"""
        if self.is_shutting_down:
            return
        
        try:
            closed_count = self.prediction_manager.auto_close_expired_predictions()
            if closed_count > 0:
                logger.info(f"Auto-closed {closed_count} expired predictions")
        except Exception as e:
            logger.error(f"Error in auto-close predictions task: {e}")

    @auto_close_predictions_task.before_loop
    async def before_auto_close_predictions_task(self):
        """Wait for bot to be ready before starting auto-close task"""
        await self.wait_until_ready()
        
    async def send_daily_recap(self, embeds: List[discord.Embed]):
        """Send daily recap to the configured channel"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            logger.warning("Update channel not configured for daily recap")
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found for daily recap")
                return
            
            # Send all embeds
            for embed in embeds[:5]:  # Limit to 5 embeds max
                await channel.send(embed=embed)
            
            logger.info(f"Daily recap sent with {len(embeds)} embeds")
            
        except Exception as e:
            logger.error(f"Error sending daily recap: {e}")

    async def send_highlights_batch(self):
        """Send highlights batch (every 25 updates)"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            logger.warning("Update channel not configured for highlights")
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found")
                return
            
            embeds = await self.update_batcher.create_highlights_batch()
            
            for embed in embeds:
                await channel.send(embed=embed)
            
            logger.info(f"Sent highlights batch with {len(embeds)} embeds")
            
        except Exception as e:
            logger.error(f"Error sending highlights batch: {e}")

    async def send_hourly_summary(self):
        """Send hourly comprehensive summary"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            logger.warning("Update channel not configured for hourly summary")
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found")
                return
            
            # Create hourly summary (this will automatically get the right timeframe)
            embeds = await self.update_batcher.create_hourly_summary()
            
            if embeds:  
                # Send the normal summary with content
                for embed in embeds:
                    await channel.send(embed=embed)
                logger.info(f"Sent hourly summary with {len(embeds)} embeds")
            else:
                # CREATE AND SEND A "QUIET HOUR" EMBED
                quiet_embed = self._create_quiet_hour_embed()
                await channel.send(embed=quiet_embed)
                logger.info("Sent quiet hour summary (no updates)")
            
        except Exception as e:
            logger.error(f"Error sending hourly summary: {e}")

    def _create_quiet_hour_embed(self):
        """Create embed for quiet hours with no updates"""
        import pytz
        
        pacific_tz = pytz.timezone('US/Pacific')
        current_hour = datetime.now(pacific_tz).strftime("%I %p").lstrip('0')
        
        # Fun random messages for quiet hours
        quiet_messages = [ 
            "Not even the ants were causing drama this hour. 🐜",
        ]
        
        import random
        message = random.choice(quiet_messages)
        
        embed = discord.Embed(
            title=f"Chen Bot's House Summary - {current_hour} 🏠",
            description=f"**{message}**",
            color=0x95a5a6,  # Gray for quiet hours
            timestamp=datetime.now()
        )
        
        embed.add_field(
            name="📊 Hour Activity Level",
            value="😴 **Quiet Hour**\n*No significant updates detected*",
            inline=False
        )
        
        embed.add_field(
            name="🏠 House Status", 
            value="All houseguests accounted for and... doing very little apparently.",
            inline=False
        )
        
        embed.set_footer(text=f"Chen Bot's House Summary • {current_hour} • Even quiet hours need reporting!")
        
        return embed
    
    async def send_quiet_day_recap(self):
        """Send a recap for days with no updates"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found for quiet day recap")
                return
            
            # Calculate day number
            season_start = datetime(2025, 7, 8)  # Adjust for actual season
            current_date = datetime.now().date()
            day_number = (current_date - season_start.date()).days + 1
            
            quiet_messages = [
                "Not even the ants are causing drama 🐜",

            ]
            
            import random
            message = random.choice(quiet_messages)
            
            embed = discord.Embed(
                title=f"📅 Day {day_number} Recap",
                description=f"**{message}**",
                color=0x95a5a6,  # Gray for quiet days
                timestamp=datetime.now()
            )
            
            embed.add_field(
                name="📊 Day Summary",
                value="No significant updates detected on the live feeds today.",
                inline=False
            )
            
            embed.add_field(
                name="🏠 House Status",
                value="All houseguests accounted for and apparently living very quiet lives.",
                inline=False
            )
            
            embed.add_field(
                name="📺 Feed Activity",
                value="The cameras were probably more active than the houseguests today.",
                inline=False
            )
            
            embed.set_footer(text=f"Daily Recap • Day {day_number} • Even quiet days make history!")
            
            await channel.send(embed=embed)
            logger.info(f"Sent quiet day recap for Day {day_number}")
            
        except Exception as e:
            logger.error(f"Error sending quiet day recap: {e}")
    
    @tasks.loop(minutes=2)
    async def check_rss_feed(self):
        """Check RSS feed for new updates with dual batching system"""
        if self.is_shutting_down:
            return

        try:
            feed = feedparser.parse(self.rss_url)
        
            if not feed.entries:
                logger.warning("No entries returned from RSS feed")
                return
        
            updates = self.process_rss_entries(feed.entries)
            new_updates = await self.filter_duplicates(updates)
        
            for update in new_updates:
                try:
                    categories = self.analyzer.categorize_update(update)
                    importance = self.analyzer.analyze_strategic_importance(update)
                
                    self.db.store_update(update, importance, categories)
                    await self.update_batcher.add_update(update)  # Adds to both queues
                
                    # Check for alliance information
                    alliance_events = self.alliance_tracker.analyze_update_for_alliances(update)
                    for event in alliance_events:
                        alliance_id = self.alliance_tracker.process_alliance_event(event)
                        if alliance_id:
                            logger.info(f"Alliance event processed: {event['type'].value}")
                    
                    # Add context tracking here (INSIDE the try block)
                    
                    self.total_updates_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing update: {e}")
                    self.consecutive_errors += 1
        
            # Check for highlights batch (25 updates or urgent conditions)
            if self.update_batcher.should_send_highlights():
                await self.send_highlights_batch()
        
        
            self.last_successful_check = datetime.now()
            self.consecutive_errors = 0
        
            if new_updates:
                logger.info(f"Added {len(new_updates)} updates to both batching queues")
                logger.info(f"Queue status - Highlights: {len(self.update_batcher.highlights_queue)}, Hourly: {len(self.update_batcher.hourly_queue)}")
            
        except Exception as e:
            logger.error(f"Error in RSS check: {e}")
            self.consecutive_errors += 1

    async def ensure_summary_tables_exist(self):
        """Ensure summary tables exist before trying to use them"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                return
                
            import psycopg2
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()
            
            # Create summary_checkpoints table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summary_checkpoints (
                    checkpoint_id SERIAL PRIMARY KEY,
                    summary_type VARCHAR(50) NOT NULL UNIQUE,
                    last_processed_update_id INTEGER,
                    queue_state JSONB,
                    queue_size INTEGER DEFAULT 0,
                    last_summary_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create summary_metrics table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summary_metrics (
                    metric_id SERIAL PRIMARY KEY,
                    summary_type VARCHAR(50) NOT NULL,
                    update_count INTEGER,
                    llm_tokens_used INTEGER,
                    processing_time_ms INTEGER,
                    summary_quality_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_type ON summary_checkpoints(summary_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_type_date ON summary_metrics(summary_type, created_at)")
            
            conn.commit()
            conn.close()
            
            logger.info("Summary persistence tables ensured to exist")
            
        except Exception as e:
            logger.error(f"Error ensuring summary tables exist: {e}")

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
