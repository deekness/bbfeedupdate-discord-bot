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
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
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
from psycopg2 import pool
from urllib.parse import urlparse
from contextlib import contextmanager
from typing import Optional

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
]

# Add these NEW constants right after the BB27_HOUSEGUESTS list:
BB27_HOUSEGUESTS_SET = {
    "Adrian", "Amy", "Ashley", "Ava", "Jimmy", "Katherine", "Keanu", 
    "Kelley", "Lauren", "Mickey", "Morgan", "Rachel", "Rylie", "Vince", "Will", "Zach", "Zae"
}

NICKNAME_MAP = {
    "kat": "Katherine",
    "rach": "Rachel", 
    "rachael": "Rachel",
    "vinny": "Vince",
    "vinnie": "Vince",
    "mick": "Mickey",
    "ash": "Ashley",
    "jim": "Jimmy",
    "ry": "Rylie"
}

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

class BlueskyClient:
    """Client for interacting with Bluesky's AT Protocol API"""
    
    def __init__(self, username: str = None, password: str = None):
        self.base_url = "https://bsky.social/xrpc"
        self.session = None
        self.access_token = None
        self.refresh_token = None
        self.username = username
        self.password = password
        self.authenticated = False
        self.last_auth_attempt = None
        
        # Session for HTTP requests with retry logic
        self.http_session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.http_session.mount("http://", adapter)
        self.http_session.mount("https://", adapter)
        
        self.http_session.headers.update({
            'User-Agent': 'BigBrotherBot/1.0',
            'Content-Type': 'application/json'
        })
    
    async def authenticate(self) -> bool:
        """Authenticate with Bluesky if credentials provided"""
        if not self.username or not self.password:
            logger.info("No Bluesky credentials provided - skipping authentication")
            return False
        
        # Don't retry authentication too frequently
        if self.last_auth_attempt:
            time_since_last = (datetime.now() - self.last_auth_attempt).total_seconds()
            if time_since_last < 60:  # Wait at least 1 minute between attempts
                logger.debug("Skipping authentication attempt - too soon since last attempt")
                return self.authenticated
        
        self.last_auth_attempt = datetime.now()
        
        try:
            logger.info(f"Attempting Bluesky authentication for {self.username}")
            
            # Clear any existing auth headers
            if 'Authorization' in self.http_session.headers:
                del self.http_session.headers['Authorization']
            
            response = self.http_session.post(
                f"{self.base_url}/com.atproto.server.createSession",
                json={
                    "identifier": self.username,
                    "password": self.password
                },
                timeout=15
            )
            
            logger.info(f"Bluesky auth response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('accessJwt')
                self.refresh_token = data.get('refreshJwt')
                
                if self.access_token:
                    # Update session headers
                    self.http_session.headers.update({
                        'Authorization': f'Bearer {self.access_token}'
                    })
                    
                    logger.info("✅ Successfully authenticated with Bluesky")
                    self.authenticated = True
                    return True
                else:
                    logger.error("❌ No access token received from Bluesky")
                    self.authenticated = False
                    return False
            else:
                logger.error(f"❌ Bluesky authentication failed: {response.status_code}")
                if response.text:
                    logger.error(f"Response: {response.text}")
                self.authenticated = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Error authenticating with Bluesky: {e}")
            self.authenticated = False
            return False
    
    def get_profile_posts(self, handle: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Get posts from a specific profile"""
        if not self.authenticated:
            logger.debug(f"Not authenticated - skipping {handle}")
            return []
        
        try:
            # Convert handle to DID if needed
            if not handle.startswith('did:'):
                resolve_response = self.http_session.get(
                    f"{self.base_url}/com.atproto.identity.resolveHandle",
                    params={"handle": handle},
                    timeout=10
                )
                
                if resolve_response.status_code != 200:
                    logger.warning(f"Failed to resolve handle {handle}: {resolve_response.status_code}")
                    return []
                
                try:
                    resolve_data = resolve_response.json()
                    did = resolve_data.get('did')
                    if not did:
                        logger.warning(f"No DID found for handle {handle}")
                        return []
                except:
                    logger.warning(f"Invalid JSON response for handle {handle}")
                    return []
            else:
                did = handle
            
            # Get posts from the profile
            posts_response = self.http_session.get(
                f"{self.base_url}/app.bsky.feed.getAuthorFeed",
                params={
                    "actor": did,
                    "limit": limit,
                    "filter": "posts_no_replies"
                },
                timeout=15
            )
            
            if posts_response.status_code == 200:
                try:
                    data = posts_response.json()
                    posts = data.get('feed', [])
                    logger.debug(f"Retrieved {len(posts)} posts from {handle}")
                    return posts
                except:
                    logger.warning(f"Invalid JSON response for posts from {handle}")
                    return []
            else:
                logger.warning(f"Failed to get posts for {handle}: {posts_response.status_code}")
                
                # If we get 401 (unauthorized), mark as not authenticated
                if posts_response.status_code == 401:
                    logger.info("Got 401 - marking as not authenticated")
                    self.authenticated = False
                
                return []
                
        except Exception as e:
            logger.error(f"Error getting posts for {handle}: {e}")
            return []
            
class UnifiedContentMonitor:
    """Unified monitor that combines RSS and Bluesky content"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.analyzer = bot_instance.analyzer
        
        # Initialize Bluesky client
        bluesky_username = os.getenv('BLUESKY_USERNAME') or self.bot.config.get('bluesky_username')
        bluesky_password = os.getenv('BLUESKY_PASSWORD') or self.bot.config.get('bluesky_password')
        self.bluesky_client = BlueskyClient(bluesky_username, bluesky_password)
        
        # Configure monitored Bluesky accounts
        self.monitored_accounts = [
            "bigbrothernetwork.bsky.social",
            "hamsterwatch.com",
            "pooyaism.bsky.social",
            "thebbpresident.bsky.social", 
            "bblionotev.bsky.social",
            "bbliveupdaters.bsky.social",
            "toomsbb.bsky.social",
            "bbteamnorth.bsky.social",
            "bbnutters.bsky.social",
        ]
        
        # BB-related keywords for filtering
        self.bb_keywords = [
            "big brother", "bb27", "#bb27", "bb 27", "#bigbrother",
            "houseguest", "hoh", "veto", "eviction", "nomination", 
            "backdoor", "alliance", "showmance", "feeds", "live feeds",
            "diary room", "have not", "america's favorite", "jury",
            "finale", "pov", "power of veto", "head of household"
        ]
        
        # Track last check times
        self.last_check_times = {}
        
        # Stats
        self.total_bluesky_updates = 0
        self.bluesky_auth_status = False
    
    async def check_all_sources(self) -> List[BBUpdate]:
        """Check both RSS and Bluesky sources and return unified updates"""
        all_updates = []
        
        # 1. Check RSS (your existing logic)
        rss_updates = await self._check_rss_feed()
        all_updates.extend(rss_updates)
        
        # 2. Check Bluesky accounts
        bluesky_updates = await self._check_bluesky_accounts()
        all_updates.extend(bluesky_updates)
        
        # 3. Filter duplicates across ALL sources
        new_updates = await self._filter_all_duplicates(all_updates)
        
        logger.info(f"Content check: {len(rss_updates)} RSS, {len(bluesky_updates)} Bluesky, {len(new_updates)} new total")
        
        return new_updates
    
    async def _check_rss_feed(self) -> List[BBUpdate]:
        """Check RSS feed (your existing logic, extracted)"""
        try:
            feed = feedparser.parse(self.bot.rss_url)
            
            if not feed.entries:
                return []
            
            updates = self.bot.process_rss_entries(feed.entries)
            return updates
            
        except Exception as e:
            logger.error(f"Error checking RSS feed: {e}")
            return []
    
    async def _check_bluesky_accounts(self) -> List[BBUpdate]:
        """Check all monitored Bluesky accounts with improved error handling"""
        all_bluesky_updates = []
        
        # Try to authenticate if not already authenticated
        if not self.bluesky_auth_status:
            try:
                self.bluesky_auth_status = await self.bluesky_client.authenticate()
                if not self.bluesky_auth_status:
                    logger.debug("Bluesky authentication failed, skipping this check")
                    return []
            except Exception as e:
                logger.error(f"Bluesky authentication error: {e}")
                return []
        
        # Check each monitored account
        successful_checks = 0
        failed_checks = 0
        
        for account in self.monitored_accounts:
            try:
                posts = self.bluesky_client.get_profile_posts(account, limit=20)
                if posts:
                    account_updates = await self._process_account_posts(account, posts)
                    all_bluesky_updates.extend(account_updates)
                    successful_checks += 1
                else:
                    failed_checks += 1
                
                # Small delay between accounts to be respectful
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error checking Bluesky account {account}: {e}")
                failed_checks += 1
                continue
        
        # Update stats
        self.total_bluesky_updates += len(all_bluesky_updates)
        
        # Log results
        if successful_checks > 0 or failed_checks > 0:
            logger.info(f"Bluesky check: {successful_checks} successful, {failed_checks} failed")
        
        # If too many failures, mark as not authenticated for retry
        if failed_checks > successful_checks and failed_checks >= 3:
            logger.warning("Too many Bluesky failures - will retry authentication next time")
            self.bluesky_auth_status = False
        
        return all_bluesky_updates
    
    async def _process_account_posts(self, account: str, posts: List[Dict]) -> List[BBUpdate]:
        """Process posts from a specific Bluesky account"""
        updates = []
        last_check = self.last_check_times.get(account, datetime.now() - timedelta(hours=6))
        
        for post_data in posts:
            try:
                post = post_data.get('post', {})
                record = post.get('record', {})
                
                # Extract post data
                text = record.get('text', '')
                created_at = record.get('createdAt', '')
                uri = post.get('uri', '')
                
                # Parse timestamp
                try:
                    post_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    post_time = post_time.replace(tzinfo=None)
                except:
                    post_time = datetime.now()
                
                # Skip if post is older than last check
                if post_time <= last_check:
                    continue
                
                # Skip if not BB related
                if not self._is_bb_related(text):
                    # Account-level filtering for content quality
                    account_name = account.split('.')[0]  # Get username part
                    
                    # Skip obvious fan commentary posts even from good accounts
                    if self._is_fan_commentary(text):
                        continue
                    continue
                
                # Clean and format the content
                cleaned_text = self._clean_bluesky_text(text)
                
                # Create title that mirrors RSS format
                title = f"{self._extract_time_from_post(post_time)} - {cleaned_text}"
                if len(title) > 200:
                    title = title[:197] + "..."
                
                # Create BBUpdate object that integrates seamlessly
                description = cleaned_text
                link = f"https://bsky.app/profile/{account}/post/{uri.split('/')[-1]}"
                content_hash = self._create_bluesky_hash(text, created_at, account)
                
                update = BBUpdate(
                    title=title,
                    description=description,
                    link=link,
                    pub_date=post_time,
                    content_hash=content_hash,
                    author=f"@{account.split('.')[0]}"  # Clean up handle
                )
                
                updates.append(update)
                
            except Exception as e:
                logger.error(f"Error processing post from {account}: {e}")
                continue
        
        # Update last check time for this account
        if updates:
            self.last_check_times[account] = max(update.pub_date for update in updates)
            logger.debug(f"Found {len(updates)} new updates from @{account}")
        
        return updates
    
    async def _filter_all_duplicates(self, all_updates: List[BBUpdate]) -> List[BBUpdate]:
        """Filter duplicates across RSS and Bluesky using enhanced detection"""
        new_updates = []
        
        for update in all_updates:
            # Check database first
            if self.bot.db.is_duplicate(update.content_hash):
                continue
            
            # Check cache
            if await self.bot.update_batcher.processed_hashes_cache.contains(update.content_hash):
                continue
            
            # Check for content similarity across sources
            if not await self._is_content_duplicate(update, new_updates):
                new_updates.append(update)
        
        return new_updates
    
    async def _is_content_duplicate(self, new_update: BBUpdate, existing_updates: List[BBUpdate]) -> bool:
        """Check if content is duplicate based on semantic similarity"""
        new_content = new_update.description.lower()
        
        # Remove common variations for comparison
        new_content_cleaned = self._normalize_content_for_comparison(new_content)
        
        for existing in existing_updates:
            existing_content = self._normalize_content_for_comparison(existing.description.lower())
            
            # Check for high similarity (adjust threshold as needed)
            similarity = self._calculate_content_similarity(new_content_cleaned, existing_content)
            if similarity > 0.85:  # 85% similarity threshold
                logger.debug(f"Filtered duplicate content: {similarity:.2f} similarity")
                return True
        
        return False
    
    def _normalize_content_for_comparison(self, content: str) -> str:
        """Normalize content for duplicate detection"""
        import re
        
        # Remove timestamps, URLs, mentions
        content = re.sub(r'\d{1,2}:\d{2}[ap]m', '', content)
        content = re.sub(r'https?://\S+', '', content)
        content = re.sub(r'@\w+', '', content)
        content = re.sub(r'#\w+', '', content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simple implementation)"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _is_bb_related(self, text: str) -> bool:
        """Enhanced BB content detection focusing on house activity"""
        text_lower = text.lower()
        
        # HOUSEGUEST NAMES AND NICKNAMES mapping
        houseguest_names = {
            # Full names
            "adrian": "adrian", "amy": "amy", "ashley": "ashley", "ava": "ava", 
            "jimmy": "jimmy", "katherine": "katherine", "keanu": "keanu", 
            "kelley": "kelley", "lauren": "lauren", "mickey": "mickey", 
            "morgan": "morgan", "rachel": "rachel", "rylie": "rylie", 
            "vince": "vince", "will": "will", "zach": "zach", "zae": "zae",
            
            # Common nicknames
            "vinny": "vince",        # Vinny -> Vince
            "kat": "katherine",      # Kat -> Katherine
            "rach": "rachel",        # Rach -> Rachel
            "rachael": "rachel",        # Racheael -> Rachel
            "mick": "mickey",        # Mick -> Mickey
            "ash": "ashley",         # Ash -> Ashley
            "jim": "jimmy",          # Jim -> Jimmy
            "morg": "morgan",        # Morg -> Morgan (if used)
            "ry": "rylie",           # Ry -> Rylie (if used)
            
            # Add any other nicknames you've observed

        }
        
        # IMMEDIATE EXCLUSIONS: Block these types of content first
        strong_exclusions = [
            "subscribe to", "follow me on", "link in bio", "check out my",
            "patreon", "donate", "support me", "buy my", "use code",
            "fanfic", "fanart", "not big brother", "off topic",
            "tonight's bb", "tonight on bb", "episode", "julie chen",
            "watch tonight", "tune in", "airs tonight", "tv schedule",
            "blockbuster episode", "don't miss", "coming up on bb"
        ]
        
        if any(exclusion in text_lower for exclusion in strong_exclusions):
            return False
        
        # IMMEDIATE INCLUSIONS: Always include these regardless of account
        immediate_keywords = [
            "feeds are", "feeds down", "feeds back", "live feeds",
            "hoh wins", "hoh winner", "head of household",
            "veto winner", "veto wins", "pov wins", "power of veto", 
            "nomination ceremony", "nominations", "nominated", "on the block",
            "eviction", "evicted", "voted out",
            "competition", "challenge", "comp wins",
            "ceremony", "veto ceremony", "veto meeting",
            "diary room", "dr session"
        ]
        
        if any(keyword in text_lower for keyword in immediate_keywords):
            return True
        
        # HOUSEGUEST CONVERSATIONS: Look for houseguest names/nicknames followed by dialogue
        all_names = list(houseguest_names.keys())
        names_pattern = "|".join(re.escape(name) for name in all_names)  # Escape special chars
        
        houseguest_dialogue_patterns = [
            rf'\b({names_pattern})\s*:\s*.{{10,}}',      # "Name: longer dialogue" (10+ chars after colon)
            rf'\b({names_pattern})\s+says?\s+',          # "Name says"
            rf'\b({names_pattern})\s+tells?\s+',         # "Name tells"
            rf'\b({names_pattern})\s+asks?\s+',          # "Name asks"
            rf'\b({names_pattern})\s+mentions?\s+',      # "Name mentions"
        ]
        
        try:
            if any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in houseguest_dialogue_patterns):
                return True
        except re.error as e:
            logger.error(f"Regex error in dialogue patterns: {e}")
            # Fallback to simpler name detection
            if any(name in text_lower for name in all_names):
                return True
        
        # STRATEGIC CONTENT: Allow strategic discussion
        strategy_keywords = [
            "alliance", "final two", "final three", "final four",
            "backdoor", "target", "targeting", "campaign", "campaigning",
            "vote", "voting", "votes", "jury", "blindside",
            "deal", "promise", "pitch", "strategy", "plan", "scheme",
            "pitching to", "wants to work with", "working with"
        ]
        
        if any(keyword in text_lower for keyword in strategy_keywords):
            return True
        
        # HOUSE ACTIVITY: Physical events and conversations
        house_activity = [
            "in the kitchen", "in the bedroom", "in the backyard", "in the hoh room",
            "storage room", "have not room", "slop", "luxury competition",
            "house meeting", "group conversation"
        ]
        
        if any(activity in text_lower for activity in house_activity):
            return True
        
        # HOUSEGUEST NAMES/NICKNAMES: Include if mentioned with meaningful context
        for name_or_nickname in all_names:
            if name_or_nickname in text_lower:
                try:
                    # Escape the name to handle special regex characters
                    escaped_name = re.escape(name_or_nickname)
                    name_context_patterns = [
                        rf'{escaped_name}\s*:.*',           # "Name: dialogue"
                        rf'{escaped_name}\s+(says?|tells?|asks?|mentions?|wants?|thinks?)',  # "Name says/tells/etc"
                        rf'(says?|tells?).*{escaped_name}',  # "says to Name"
                        rf'{escaped_name}.*\b(strategy|alliance|vote|target|deal|plan)\b'  # Name + strategy words
                    ]
                    
                    if any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in name_context_patterns):
                        return True
                except re.error as e:
                    logger.error(f"Regex error with name '{name_or_nickname}': {e}")
                    # Simple fallback - just check if name appears with strategy words
                    if any(word in text_lower for word in ["strategy", "alliance", "vote", "target", "deal", "plan"]):
                        return True
        
        # DEFAULT: If none of the above criteria match, exclude
        return False
    
    def _clean_bluesky_text(self, text: str) -> str:
        """Clean and format Bluesky text for better integration"""
        # Remove excessive line breaks
        text = re.sub(r'\n+', ' ', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_time_from_post(self, post_time: datetime) -> str:
        """Extract time in format that matches RSS entries"""
        return post_time.strftime("%I:%M %p PST")
    
    def _create_bluesky_hash(self, text: str, created_at: str, author: str) -> str:
        """Create unique hash for Bluesky posts"""
        import hashlib
        content = f"bluesky|{author}|{text}|{created_at}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "monitored_accounts": len(self.monitored_accounts),
            "accounts_with_activity": len(self.last_check_times),
            "total_bluesky_updates": self.total_bluesky_updates,
            "authentication_status": self.bluesky_auth_status,
            "last_check_times": dict(self.last_check_times)
        }

    def _is_fan_commentary(self, text: str) -> bool:
        """Simplified fan commentary detection for trusted accounts only"""
        text_lower = text.lower()
        
        # Since all monitored accounts are trusted, only block obvious spam/promotional content
        spam_patterns = [
            "subscribe", "follow me", "link in bio", "patreon", "donate",
            "buy my", "use code", "check out my", "support me"
        ]
        
        # Block only if multiple spam indicators
        spam_count = sum(1 for pattern in spam_patterns if pattern in text_lower)
        return spam_count >= 2  # Need multiple spam indicators to block
            
        
        
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
        """Extract houseguest names from text, including nicknames"""
        # Nickname to full name mapping
        nickname_mapping = {
            "vinny": "Vince", "vinnie": "Vince",
            "kat": "Katherine",
            "rach": "Rachel", "rachael": "Rachel",
            "mick": "Mickey",
            "ash": "Ashley",
            "jim": "Jimmy",
            "morg": "Morgan",
        }
        
        # Full names list
        full_names = ["Adrian", "Amy", "Ashley", "Ava", "Jimmy", "Katherine", 
                      "Keanu", "Kelley", "Lauren", "Mickey", "Morgan", "Rachel", 
                      "Rylie", "Vince", "Will", "Zach", "Zae"]
        
        # Find potential names (capitalized words)
        houseguest_pattern = r'\b[A-Z][a-z]+\b'
        potential_names = re.findall(houseguest_pattern, text)
        
        found_houseguests = []
        
        for name in potential_names:
            # Check if it's a full name
            if name in full_names and name not in EXCLUDE_WORDS:
                found_houseguests.append(name)
            # Check if it's a nickname
            elif name.lower() in nickname_mapping:
                mapped_name = nickname_mapping[name.lower()]
                found_houseguests.append(mapped_name)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_houseguests))
    
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
    def _execute_query(self, cursor, query_sqlite, params, query_postgresql=None):
        """Execute query with proper syntax for current database"""
        if self.use_postgresql:
            if query_postgresql:
                cursor.execute(query_postgresql, params)
            else:
                # Convert SQLite ? to PostgreSQL %s and handle CURRENT_TIMESTAMP
                pg_query = query_sqlite.replace('?', '%s')
                pg_query = pg_query.replace('CURRENT_TIMESTAMP', 'CURRENT_TIMESTAMP')
                cursor.execute(pg_query, params)
        else:
            cursor.execute(query_sqlite, params)
    
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
        'BETRAYAL': 'betrayal',
        'JURY_TALK': 'jury_talk',
        'TARGET_DISCUSSION': 'target_discussion',
        'STRATEGY_SESSION': 'strategy_session',
        'POWER_DISCUSSION': 'power_discussion'
    }
    
    
    # Competition detection patterns
    COMPETITION_PATTERNS = [
        # Original patterns
        (r'(\w+)\s+wins?\s+(?:the\s+)?hoh', 'HOH_WIN'),
        (r'(\w+)\s+wins?\s+(?:the\s+)?(?:power\s+of\s+)?veto', 'VETO_WIN'),
        (r'(\w+)\s+wins?\s+(?:the\s+)?pov', 'VETO_WIN'),
        (r'(\w+)\s+(?:is\s+)?nominated?', 'NOMINATION'),
        (r'(\w+)\s+(?:gets?\s+)?evicted', 'EVICTION'),
        (r'(\w+)\s+(?:was\s+)?eliminated', 'EVICTION'),
        
        # NEW: More flexible HOH patterns
        (r'(\w+)\s+(?:becomes?|is\s+now)\s+hoh', 'HOH_WIN'),
        (r'hoh\s+(?:winner|champion):\s*(\w+)', 'HOH_WIN'),
        (r'congratulations\s+(\w+).*hoh', 'HOH_WIN'),
        (r'(\w+)\s+(?:has\s+)?won\s+hoh', 'HOH_WIN'),
        
        # NEW: Current power holder references
        (r'(\w+)\s+(?:is|as)\s+(?:the\s+)?current\s+hoh', 'HOH_WIN'),
        (r'(\w+)\s+(?:holds?|has)\s+(?:the\s+)?hoh\s+power', 'HOH_WIN'),
    ]
      
    # Enhanced social/strategic patterns  
    SOCIAL_PATTERNS = [
        # Original patterns
        (r'(\w+)\s+and\s+(\w+)\s+(?:kiss|kissed|make\s+out)', 'SHOWMANCE_START'),
        (r'(\w+)\s+and\s+(\w+)\s+(?:fight|argue|confrontation)', 'FIGHT'),
        (r'(\w+)\s+betrays?\s+(\w+)', 'BETRAYAL'),
        (r'(\w+)\s+(?:throws?\s+)?(\w+)\s+under\s+the\s+bus', 'BETRAYAL'),
        (r'(\w+)\s+says?\s+he\s+wants?\s+([^.]+)\s+in\s+his\s+jury', 'JURY_TALK'),
        (r'(\w+)\s+says?\s+she\s+wants?\s+([^.]+)\s+in\s+her\s+jury', 'JURY_TALK'),
        (r'(\w+)\s+(?:wants?|targeting)\s+([^.]+)\s+(?:out|gone|evicted)', 'TARGET_DISCUSSION'),
        (r'(\w+)\s+tells?\s+(\w+)\s+(?:about|that)', 'STRATEGY_SESSION'),
        (r'(\w+)\s+(?:mentions?|says?)\s+(?:he|she)\s+(?:wants?|needs?)', 'STRATEGY_SESSION'),
    ]


    # NEW: Context-aware patterns for current game state
    GAME_STATE_PATTERNS = [
        (r'(\w+)\s+(?:is|as)\s+hoh', 'POWER_DISCUSSION'),
        (r'(\w+)\s+has\s+(?:the\s+)?power', 'POWER_DISCUSSION'),
        (r'(\w+)\s+(?:on\s+the\s+)?block', 'NOMINATION'),
        (r'(\w+)\s+nominated', 'NOMINATION'),
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
        """Enhanced analysis with more patterns"""
        detected_events = []
        content = f"{update.title} {update.description}".lower()
        
        # Skip finale/voting updates
        skip_phrases = [
            'votes for', 'voted for', 'to be the winner', 'winner of big brother',
            'jury vote', 'crown the winner', 'wins bb', 'finale', 'final vote'
        ]
        
        if any(phrase in content for phrase in skip_phrases):
            return []
        
        # Check all pattern categories
        all_patterns = [
            (self.COMPETITION_PATTERNS, "competition"),
            (self.SOCIAL_PATTERNS, "social"), 
            (self.GAME_STATE_PATTERNS, "game_state")
        ]
        
        for patterns, category in all_patterns:
            for pattern, event_type in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if groups:
                        # Handle different event types
                        if event_type in ['JURY_TALK', 'TARGET_DISCUSSION']:
                            houseguest = groups[0].strip().title()
                            details = groups[1] if len(groups) > 1 else ""
                            
                            if houseguest not in EXCLUDE_WORDS and len(houseguest) > 2:
                                detected_events.append({
                                    'type': self.EVENT_TYPES[event_type],
                                    'houseguest': houseguest,
                                    'description': f"{houseguest} {event_type.lower().replace('_', ' ')}: {details[:100]}",
                                    'update': update,
                                    'confidence': self._calculate_event_confidence(event_type, content),
                                    'details': details
                                })
                        
                        elif len(groups) == 1:
                            # Single houseguest events
                            houseguest = groups[0].strip().title()
                            if houseguest not in EXCLUDE_WORDS and len(houseguest) > 2:
                                detected_events.append({
                                    'type': self.EVENT_TYPES[event_type],
                                    'houseguest': houseguest,
                                    'description': f"{houseguest} {event_type.lower().replace('_', ' ')}",
                                    'update': update,
                                    'confidence': self._calculate_event_confidence(event_type, content)
                                })
                        
                        elif len(groups) >= 2:
                            # Multi-houseguest events
                            hg1 = groups[0].strip().title()
                            hg2 = groups[1].strip().title()
                            
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
    def _execute_query(self, cursor, query_sqlite, params, query_postgresql=None):
        """Execute query with proper syntax for current database"""
        if self.use_postgresql:
            if query_postgresql:
                cursor.execute(query_postgresql, params)
            else:
                # Convert SQLite ? to PostgreSQL %s
                pg_query = query_sqlite.replace('?', '%s')
                cursor.execute(pg_query, params)
        else:
            cursor.execute(query_sqlite, params)
    
    """Tracks and analyzes Big Brother alliances"""
    
    # Alliance detection patterns
    ALLIANCE_FORMATION_PATTERNS = [
        # Only the most reliable patterns
        (r"([\w\s]+) and ([\w\s]+) make a final (\d+)", "final_deal"),
        (r"([\w\s]+) forms? an? alliance with ([\w\s]+)", "alliance"),
        (r"([\w\s]+) and ([\w\s]+) agree to work together", "agreement"),
        (r"([\w\s]+) and ([\w\s]+) form an? alliance", "alliance"),
        # Removed the overly broad patterns that were causing issues
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
        """Enhanced analysis with better filtering"""
        content = f"{update.title} {update.description}".strip()
        detected_events = []
        
        # ENHANCED SKIP LOGIC - More comprehensive exclusions
        skip_phrases = [
            'votes for', 'voted for', 'to be the winner', 'winner of big brother',
            'jury vote', 'crown the winner', 'wins bb', 'wins hoh', 'wins pov',
            'wins the power', 'eviction vote', 'evicted', 'julie pulls the keys',
            'america\'s favorite', 'afp', 'finale', 'final vote', 'cast their vote',
            'announces the winner', 'wins big brother', 'jury votes', 'key to vote',
            'official with a vote', 'makes it official', 'competition winner',
            'diary room', 'live feeds', 'feeds are', 'cameras', 'production',
            'episode', 'tonight on', 'watch tonight', 'tune in', 'coming up',
            # Add more exclusions for common false positives
            'ground adrian finished', 'way by', 'easiest thing', 'radar and be',
            'notes morgan', 'says that he knows'
        ]
        
        content_lower = content.lower()
        if any(phrase in content_lower for phrase in skip_phrases):
            return []
        
        # Skip very short content that's likely not alliance-related
        if len(content.strip()) < 20:
            return []
        
        # Check for alliance formations with better validation
        for pattern, pattern_type in self.ALLIANCE_FORMATION_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                houseguests = []
                
                # Better houseguest extraction and validation
                for group in groups:
                    if group and not group.isdigit():
                        cleaned_name = self._clean_and_validate_houseguest(group.strip())
                        if cleaned_name:
                            houseguests.append(cleaned_name)
                
                # Only proceed if we have 2+ valid houseguests
                if len(houseguests) >= 2:
                    # Additional context check - make sure this actually sounds like alliance talk
                    if self._is_likely_alliance_context(content, houseguests):
                        detected_events.append({
                            'type': AllianceEventType.FORMED,
                            'houseguests': houseguests,
                            'pattern_type': pattern_type,
                            'confidence': self._calculate_confidence(pattern_type),
                            'update': update
                        })
        
        # Process named alliances with much stricter validation
        for pattern in self.ALLIANCE_NAME_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                alliance_name = match.group(1).strip()
                
                # Strict alliance name validation
                if self._is_valid_alliance_name(alliance_name):
                    # Extract members from surrounding context
                    members = self._extract_alliance_members_from_context(content, match.start(), match.end())
                    
                    if len(members) >= 2:  # Need at least 2 valid members
                        detected_events.append({
                            'type': AllianceEventType.FORMED,
                            'alliance_name': alliance_name,
                            'houseguests': members,
                            'pattern_type': 'named_alliance',
                            'confidence': 85,
                            'update': update
                        })
        
        return detected_events

    def clean_alliance_members(self) -> int:
        """Clean invalid members from existing alliances"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get all alliance members
            cursor.execute("""
                SELECT alliance_id, houseguest_name 
                FROM alliance_members 
                WHERE is_active = TRUE
            """)
            
            all_members = cursor.fetchall()
            cleaned_count = 0
            
            for member_row in all_members:
                if self.use_postgresql:
                    alliance_id = member_row['alliance_id']
                    houseguest_name = member_row['houseguest_name']
                else:
                    alliance_id, houseguest_name = member_row
                
                # Validate the houseguest name using the existing validation
                validated_name = self._clean_and_validate_houseguest(houseguest_name)
                
                if not validated_name:
                    # Remove invalid member
                    if self.use_postgresql:
                        cursor.execute("""
                            DELETE FROM alliance_members 
                            WHERE alliance_id = %s AND houseguest_name = %s
                        """, (alliance_id, houseguest_name))
                    else:
                        cursor.execute("""
                            DELETE FROM alliance_members 
                            WHERE alliance_id = ? AND houseguest_name = ?
                        """, (alliance_id, houseguest_name))
                    
                    cleaned_count += 1
                    logger.info(f"Removed invalid member '{houseguest_name}' from alliance {alliance_id}")
            
            # Now remove alliances that have fewer than 2 valid members
            if self.use_postgresql:
                cursor.execute("""
                    SELECT a.alliance_id, a.name, COUNT(am.houseguest_name) as member_count
                    FROM alliances a
                    LEFT JOIN alliance_members am ON a.alliance_id = am.alliance_id AND am.is_active = TRUE
                    WHERE a.status = 'active'
                    GROUP BY a.alliance_id, a.name
                    HAVING COUNT(am.houseguest_name) < 2
                """)
            else:
                cursor.execute("""
                    SELECT a.alliance_id, a.name, COUNT(am.houseguest_name) as member_count
                    FROM alliances a
                    LEFT JOIN alliance_members am ON a.alliance_id = am.alliance_id AND am.is_active = 1
                    WHERE a.status = 'active'
                    GROUP BY a.alliance_id, a.name
                    HAVING COUNT(am.houseguest_name) < 2
                """)
            
            invalid_alliances = cursor.fetchall()
            
            for alliance_row in invalid_alliances:
                if self.use_postgresql:
                    alliance_id = alliance_row['alliance_id']
                    name = alliance_row['name']
                else:
                    alliance_id, name, _ = alliance_row
                
                # Mark alliance as dissolved
                if self.use_postgresql:
                    cursor.execute("""
                        UPDATE alliances 
                        SET status = 'dissolved', confidence_level = 0 
                        WHERE alliance_id = %s
                    """, (alliance_id,))
                else:
                    cursor.execute("""
                        UPDATE alliances 
                        SET status = 'dissolved', confidence_level = 0 
                        WHERE alliance_id = ?
                    """, (alliance_id,))
                
                cleaned_count += 1
                logger.info(f"Dissolved alliance '{name}' (ID: {alliance_id}) - insufficient valid members")
            
            conn.commit()
            logger.info(f"Member cleanup completed: {cleaned_count} actions taken")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning alliance members: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def _clean_and_validate_houseguest(self, name: str) -> Optional[str]:
        """Ultra-strict houseguest validation - only BB27 cast"""
        if not name or len(name) < 2:
            return None
            
        # Clean the name
        cleaned = re.sub(r'[^\w\s]', '', name).strip().title()
        
        # ONLY allow actual BB27 houseguests
        if cleaned in BB27_HOUSEGUESTS_SET:
            return cleaned
            
        # Check nicknames
        if cleaned.lower() in NICKNAME_MAP:
            return NICKNAME_MAP[cleaned.lower()]
        
        # REJECT EVERYTHING ELSE - be extremely strict
        return None
            
        # Reject if it's a common word that's not a houseguest
        common_words = {
            'Ground', 'Way', 'Thing', 'Person', 'Notes', 'Says', 'Knows', 'Going',
            'Easy', 'Under', 'Cover', 'First', 'Second', 'Third', 'Brother',
            'House', 'Game', 'Winner', 'Player', 'Member', 'Group', 'Team', 'And',
            'The', 'Of', 'To', 'In', 'For', 'On', 'At', 'By', 'With', 'From',
            'She', 'He', 'Him', 'Her', 'They', 'Them', 'I', 'You', 'We', 'Us',
            'Me', 'If', 'Best', 'Do', 'Feeds', 'Cut', 'Table', 'Bathroom', 'Comp',
            'Bond', 'Wall', 'Mom', 'Big', 'Block', 'Spoke', 'Tell', 'World'
        }
        
        if cleaned in common_words or cleaned in EXCLUDE_WORDS:
            return None
            
        return None  # Only return validated BB27 houseguest names
    
    def _is_valid_alliance_name(self, name: str) -> bool:
        """Final ultra-strict validation - only allow clearly legitimate alliance names"""
        if not name or len(name) < 2 or len(name) > 25:
            return False
        
        name_lower = name.lower().strip()
        
        # WHITELIST: Only allow these specific patterns for alliance names
        # 1. Known real alliance naming patterns
        known_alliance_patterns = {
            'the bond', 'bond', 'the core', 'core', 'the committee', 'committee',
            'the cookout', 'cookout', 'the brigade', 'brigade', 'the six', 'six',
            'final two', 'final three', 'final four', 'final five', 'final six',
            'the showmance', 'showmance', 'the veterans', 'veterans', 'the newbies',
            'newbies', 'the alliance', 'alliance', 'the group', 'group'
        }
        
        # 2. Allow names that sound like real alliances (adjective + noun patterns)
        if name_lower in known_alliance_patterns:
            return True
        
        # 3. Allow compound names that follow alliance naming conventions
        words = name_lower.split()
        
        # Single words that could be alliance names (very restrictive)
        if len(words) == 1:
            # Only allow if it's clearly an alliance-style name
            valid_single_words = {
                'bond', 'core', 'committee', 'cookout', 'brigade', 'six',
                'showmance', 'veterans', 'newbies', 'alliance', 'group',
                'team', 'crew', 'squad', 'trio', 'quartet', 'quintet'
            }
            return name_lower in valid_single_words
        
        # Two word combinations that could be alliances
        if len(words) == 2:
            # Pattern: "The [Name]" or "[Adjective] [Noun]"
            if words[0] == 'the':
                return True  # "The Something" format is usually legitimate
            
            # Common alliance word combinations
            alliance_adjectives = {'final', 'strong', 'solid', 'tight', 'close', 'secret', 'inner'}
            alliance_nouns = {'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'circle', 'alliance', 'group', 'team', 'crew', 'squad'}
            
            if words[0] in alliance_adjectives and words[1] in alliance_nouns:
                return True
        
        # BLACKLIST: Reject obvious non-alliance names
        
        # Reject single words that are clearly not alliance names
        if len(words) == 1:
            obvious_rejects = {
                'bathroom', 'dishes', 'loose', 'nicest', 'comp', 'house', 'same',
                'real', 'show', 'camera', 'life', 'powers', 'right', 'side',
                'vote', 'proposing', 'caught', 'lived', 'blockbuster'
            }
            if name_lower in obvious_rejects:
                return False
        
        # Reject any name containing these words (sentence fragments)
        reject_if_contains = {
            'proposing', 'caught', 'lived', 'camera', 'powers', 'comp',
            'blockbuster', 'tamar', 'she', 'he', 'has', 'them', 'side',
            'vote', 'us', 'on', 'from', 'hoh', 'is', 'are'
        }
        
        for word in words:
            if word in reject_if_contains:
                return False
        
        # Reject long phrases (more than 3 words are usually sentence fragments)
        if len(words) > 3:
            return False
        
        # If we get here, it might be legitimate - allow it
        return True
    
    def _is_likely_alliance_context(self, content: str, houseguests: List[str]) -> bool:
        """Check if the content actually sounds like alliance discussion"""
        content_lower = content.lower()
        
        # Must contain alliance-related keywords
        alliance_keywords = [
            'alliance', 'work together', 'team up', 'final', 'deal', 'agreement',
            'partnership', 'collaborate', 'join forces', 'stick together'
        ]
        
        if not any(keyword in content_lower for keyword in alliance_keywords):
            return False
            
        # Must contain actual conversation or strategy talk
        conversation_indicators = [
            'told', 'said', 'mentioned', 'discussed', 'talking about',
            'wants to', 'planning to', 'decided to', 'agreed to'
        ]
        
        if not any(indicator in content_lower for indicator in conversation_indicators):
            return False
            
        return True
    
    def _extract_alliance_members_from_context(self, content: str, start: int, end: int) -> List[str]:
        """Extract valid houseguest names from context around alliance mention"""
        # Look in a larger window for member names
        window = 150
        search_start = max(0, start - window)
        search_end = min(len(content), end + window)
        search_text = content[search_start:search_end]
        
        # Find potential names
        potential_names = re.findall(r'\b[A-Z][a-z]+\b', search_text)
        
        # Validate each name
        valid_members = []
        for name in potential_names:
            validated = self._clean_and_validate_houseguest(name)
            if validated and validated not in valid_members:
                valid_members.append(validated)
                
        return valid_members[:6]  # Max 6 members per alliance
    
    def cleanup_invalid_alliances(self) -> int:
        """Ultra-aggressive cleanup - remove anything suspicious"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get all alliances
            if self.use_postgresql:
                cursor.execute("SELECT alliance_id, name FROM alliances")
                alliances = cursor.fetchall()
            else:
                cursor.execute("SELECT alliance_id, name FROM alliances")
                alliances = cursor.fetchall()
            
            cleaned_count = 0
            
            for alliance_row in alliances:
                if self.use_postgresql:
                    alliance_id = alliance_row['alliance_id']
                    name = alliance_row['name']
                else:
                    alliance_id, name = alliance_row
                
                should_remove = False
                removal_reason = ""
                
                # Ultra-strict name validation
                if not name or not self._is_valid_alliance_name(name):
                    should_remove = True
                    removal_reason = f"invalid name: '{name}'"
                
                # Ultra-strict member validation
                if not should_remove:
                    if self.use_postgresql:
                        cursor.execute("""
                            SELECT houseguest_name FROM alliance_members 
                            WHERE alliance_id = %s AND is_active = TRUE
                        """, (alliance_id,))
                    else:
                        cursor.execute("""
                            SELECT houseguest_name FROM alliance_members 
                            WHERE alliance_id = ? AND is_active = 1
                        """, (alliance_id,))
                    
                    members_result = cursor.fetchall()
                    if self.use_postgresql:
                        members = [row['houseguest_name'] for row in members_result]
                    else:
                        members = [row[0] for row in members_result]
                    
                    # Validate each member strictly
                    valid_members = []
                    for member in members:
                        validated = self._clean_and_validate_houseguest(member)
                        if validated:
                            valid_members.append(validated)
                    
                    # Require at least 2 valid BB27 houseguests
                    if len(valid_members) < 2:
                        should_remove = True
                        removal_reason = f"insufficient valid members: {members} -> {valid_members}"
                    
                    # Additional check: if any member is clearly invalid, remove the whole alliance
                    invalid_member_patterns = {
                        'She', 'He', 'Small', 'In', 'As', 'The', 'And', 'Or', 'But',
                        'With', 'From', 'To', 'For', 'Of', 'At', 'By', 'On'
                    }
                    
                    if any(member in invalid_member_patterns for member in members):
                        should_remove = True
                        removal_reason = f"contains invalid member words: {members}"
                
                if should_remove:
                    # Remove the alliance
                    if self.use_postgresql:
                        cursor.execute("DELETE FROM alliance_events WHERE alliance_id = %s", (alliance_id,))
                        cursor.execute("DELETE FROM alliance_members WHERE alliance_id = %s", (alliance_id,))
                        cursor.execute("DELETE FROM alliances WHERE alliance_id = %s", (alliance_id,))
                    else:
                        cursor.execute("DELETE FROM alliance_events WHERE alliance_id = ?", (alliance_id,))
                        cursor.execute("DELETE FROM alliance_members WHERE alliance_id = ?", (alliance_id,))
                        cursor.execute("DELETE FROM alliances WHERE alliance_id = ?", (alliance_id,))
                    cleaned_count += 1
                    logger.info(f"Removed alliance {alliance_id}: {removal_reason}")
            
            conn.commit()
            logger.info(f"Ultra-aggressive cleanup removed {cleaned_count} alliances")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            logger.error(traceback.format_exc())
            conn.rollback()
            return 0
        finally:
            conn.close()
    
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
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.use_postgresql:
                # PostgreSQL specific - uses STRING_AGG
                cursor.execute("""
                    SELECT a.alliance_id, a.name, a.confidence_level, a.last_activity,
                           STRING_AGG(am.houseguest_name, ',') as members
                    FROM alliances a
                    JOIN alliance_members am ON a.alliance_id = am.alliance_id
                    WHERE a.status = %s AND am.is_active = TRUE
                    GROUP BY a.alliance_id, a.name, a.confidence_level, a.last_activity
                    ORDER BY a.confidence_level DESC, a.last_activity DESC
                """, (AllianceStatus.ACTIVE.value,))
            else:
                # SQLite specific - uses GROUP_CONCAT
                cursor.execute("""
                    SELECT a.alliance_id, a.name, a.confidence_level, a.last_activity,
                           GROUP_CONCAT(am.houseguest_name) as members
                    FROM alliances a
                    JOIN alliance_members am ON a.alliance_id = am.alliance_id
                    WHERE a.status = ? AND am.is_active = TRUE
                    GROUP BY a.alliance_id
                    ORDER BY a.confidence_level DESC, a.last_activity DESC
                """, (AllianceStatus.ACTIVE.value,))
            
            alliances = []
            for row in cursor.fetchall():
                if self.use_postgresql:
                    # PostgreSQL returns RealDictCursor results
                    alliances.append({
                        'alliance_id': row['alliance_id'],
                        'name': row['name'],
                        'confidence': row['confidence_level'],
                        'last_activity': row['last_activity'],
                        'members': row['members'].split(',') if row['members'] else []
                    })
                else:
                    # SQLite returns tuple
                    alliance_id, name, confidence, last_activity, members = row
                    alliances.append({
                        'alliance_id': alliance_id,
                        'name': name,
                        'confidence': confidence,
                        'last_activity': last_activity,
                        'members': members.split(',') if members else []
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
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.use_postgresql:
                # PostgreSQL syntax (already correct)
                cursor.execute("""
                    SELECT DISTINCT a.alliance_id, a.name, a.formed_date, 
                           MAX(ae.timestamp) as broken_date,
                           STRING_AGG(DISTINCT am.houseguest_name, ',') as members
                    FROM alliances a
                    JOIN alliance_members am ON a.alliance_id = am.alliance_id
                    JOIN alliance_events ae ON a.alliance_id = ae.alliance_id
                    WHERE a.status IN (%s, %s)
                      AND ae.event_type = %s 
                      AND ae.timestamp > (CURRENT_TIMESTAMP - INTERVAL '%s days')
                      AND a.confidence_level >= 70
                    GROUP BY a.alliance_id, a.name, a.formed_date
                    ORDER BY broken_date DESC
                """, (AllianceStatus.BROKEN.value, AllianceStatus.DISSOLVED.value,
                      AllianceEventType.BETRAYAL.value, days))
            else:
                # SQLite syntax - ONLY CHANGE THE GROUP BY LINE
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
                    GROUP BY a.alliance_id, a.name, a.formed_date
                    ORDER BY broken_date DESC
                """, (AllianceStatus.BROKEN.value, AllianceStatus.DISSOLVED.value,
                      AllianceEventType.BETRAYAL.value, cutoff_date))
            
            broken_alliances = []
            for row in cursor.fetchall():
                if self.use_postgresql:
                    alliance_id = row['alliance_id']
                    name = row['name']
                    formed_date = row['formed_date']
                    broken_date = row['broken_date']
                    members = row['members']
                else:
                    alliance_id, name, formed_date, broken_date, members = row
                
                try:
                    formed = formed_date if isinstance(formed_date, datetime) else datetime.fromisoformat(formed_date)
                    broken = broken_date if isinstance(broken_date, datetime) else datetime.fromisoformat(broken_date)
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
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.use_postgresql:
                cursor.execute("""
                    SELECT a.name, a.status, am.joined_date, am.left_date, a.confidence_level,
                           STRING_AGG(am2.houseguest_name, ',') as all_members
                    FROM alliance_members am
                    JOIN alliances a ON am.alliance_id = a.alliance_id
                    JOIN alliance_members am2 ON am2.alliance_id = a.alliance_id
                    WHERE am.houseguest_name = %s
                    GROUP BY a.alliance_id, a.name, a.status, am.joined_date, am.left_date, a.confidence_level
                    ORDER BY am.joined_date DESC
                """, (houseguest,))
            else:
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
            if self.use_postgresql:
                # PostgreSQL syntax
                cursor.execute("""
                    SELECT description, timestamp, involved_houseguests
                    FROM alliance_events
                    WHERE event_type = %s AND timestamp > (CURRENT_TIMESTAMP - INTERVAL '%s days')
                    ORDER BY timestamp DESC
                """, (AllianceEventType.BETRAYAL.value, days))
            else:
                # SQLite syntax
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                cursor.execute("""
                    SELECT description, timestamp, involved_houseguests
                    FROM alliance_events
                    WHERE event_type = ? AND datetime(timestamp) > datetime(?)
                    ORDER BY timestamp DESC
                """, (AllianceEventType.BETRAYAL.value, cutoff_date))
            
            betrayals = []
            for row in cursor.fetchall():
                if self.use_postgresql:
                    betrayals.append({
                        'description': row['description'],
                        'timestamp': row['timestamp'],
                        'involved': row['involved_houseguests'].split(',') if row['involved_houseguests'] else []
                    })
                else:
                    description, timestamp, involved = row
                    betrayals.append({
                        'description': description,
                        'timestamp': timestamp,
                        'involved': involved.split(',') if involved else []
                    })
            
            return betrayals
            
        except Exception as e:
            logger.error(f"Error getting betrayals: {e}")
            return []
        finally:
            conn.close()
    
    def _find_existing_alliance(self, houseguests: List[str]) -> Optional[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            self._execute_query(cursor, """
                SELECT DISTINCT a.alliance_id, a.name, a.confidence_level
                FROM alliance_members am
                JOIN alliances a ON am.alliance_id = a.alliance_id
                WHERE am.houseguest_name = ? AND a.status = ? AND am.is_active = TRUE
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
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get current confidence and status
            self._execute_query(cursor, """
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
                        last_activity = CURRENT_TIMESTAMP
                    WHERE alliance_id = %s
                """, (new_confidence, alliance_id))
            else:
                cursor.execute("""
                    UPDATE alliances 
                    SET confidence_level = ?,
                        last_activity = CURRENT_TIMESTAMP
                    WHERE alliance_id = ?
                """, (new_confidence, alliance_id))
            
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
                            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                        """, (alliance_id, AllianceEventType.DISSOLVED.value,
                              f"Alliance dissolved after {days_active} days"))
                    else:
                        cursor.execute("""
                            INSERT INTO alliance_events 
                            (alliance_id, event_type, description, timestamp)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        """, (alliance_id, AllianceEventType.DISSOLVED.value,
                              f"Alliance dissolved after {days_active} days"))
                    
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
        """User makes or updates a prediction - FIXED"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Check if prediction exists and is active - FIXED
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
            
            # Insert or update user prediction - FIXED
            if self.use_postgresql:
                self._execute_query(cursor, "", (user_id, prediction_id, option), """
                    INSERT INTO user_predictions 
                    (user_id, prediction_id, option, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, prediction_id) 
                    DO UPDATE SET option = EXCLUDED.option, updated_at = CURRENT_TIMESTAMP
                """)
            else:
                self._execute_query(cursor, """
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
        """Manually close a prediction poll - FIXED"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # FIXED: Use _execute_query helper
            self._execute_query(cursor, """
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
        """Resolve a prediction and award points - FIXED"""
        conn = self.get_connection()
        
        try:
            # Set busy timeout for SQLite
            if not self.use_postgresql:
                conn.execute("PRAGMA busy_timeout = 10000")
            cursor = conn.cursor()
            
            # Get prediction details - FIXED
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
            
            # Get users who predicted correctly BEFORE updating the prediction - FIXED
            self._execute_query(cursor, """
                SELECT user_id FROM user_predictions 
                WHERE prediction_id = ? AND option = ?
            """, (prediction_id, correct_option))
            
            # Handle user IDs result
            if self.use_postgresql:
                correct_user_ids = [row['user_id'] for row in cursor.fetchall()]
            else:
                correct_user_ids = [row[0] for row in cursor.fetchall()]
            
            # Update prediction status - FIXED
            self._execute_query(cursor, """
                UPDATE predictions 
                SET status = ?, correct_option = ?
                WHERE prediction_id = ?
            """, (PredictionStatus.RESOLVED.value, correct_option, prediction_id))
            
            # Get all user predictions for this poll - FIXED
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
        """Update user's leaderboard stats - FIXED"""
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            conn = None
            try:
                conn = self.get_connection()
                cursor = conn.cursor()  # FIXED: Get cursor from connection
                
                # Set timeout and lock for SQLite only
                if not self.use_postgresql:
                    conn.execute("PRAGMA busy_timeout = 5000")
                    conn.execute("BEGIN IMMEDIATE")
                
                # Get current stats - FIXED
                self._execute_query(cursor, """
                    SELECT season_points, weekly_points, correct_predictions, total_predictions
                    FROM prediction_leaderboard 
                    WHERE user_id = ? AND guild_id = ? AND week_number = ?
                """, (user_id, guild_id, week_number))
                
                result = cursor.fetchone()
                
                if result:
                    # Update existing record
                    if self.use_postgresql:
                        season_points = result['season_points']
                        weekly_points = result['weekly_points']
                        correct_preds = result['correct_predictions']
                        total_preds = result['total_predictions']
                    else:
                        season_points, weekly_points, correct_preds, total_preds = result
                    
                    new_season_points = season_points + points
                    new_weekly_points = weekly_points + points
                    new_correct = correct_preds + (1 if was_correct else 0)
                    new_total = total_preds + (1 if participated else 0)
                    
                    # FIXED: Use _execute_query
                    self._execute_query(cursor, """
                        UPDATE prediction_leaderboard 
                        SET season_points = ?, weekly_points = ?, 
                            correct_predictions = ?, total_predictions = ?,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND guild_id = ? AND week_number = ?
                    """, (new_season_points, new_weekly_points, new_correct, new_total,
                          user_id, guild_id, week_number))
                else:
                    # Create new record - FIXED
                    self._execute_query(cursor, """
                        INSERT INTO prediction_leaderboard 
                        (user_id, guild_id, week_number, season_points, weekly_points, 
                         correct_predictions, total_predictions)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (user_id, guild_id, week_number, points, points,
                          1 if was_correct else 0, 1 if participated else 0))
                
                conn.commit()
                logger.info(f"Successfully updated leaderboard for user {user_id}")
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e).lower()
                if "database is locked" in error_msg and attempt < max_retries - 1:
                    logger.warning(f"Database locked on attempt {attempt + 1}, retrying in {retry_delay}s")
                    if conn:
                        try:
                            conn.rollback()
                        except:
                            pass
                        conn.close()
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(f"Database error after {attempt + 1} attempts: {e}")
                    if conn:
                        conn.rollback()
                    raise
            finally:
                if conn:
                    conn.close()

    
    def get_active_predictions(self, guild_id: int) -> List[Dict]:
        """Get all active predictions for a guild - FIXED"""
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
        """Get a user's prediction for a specific poll - FIXED"""
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
        """Get season-long leaderboard - FIXED"""
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
        """Get user's prediction history - FIXED"""
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
        """Close predictions that have passed their closing time - FIXED"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            self._execute_query(cursor, """
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

    def reprocess_all_resolved_polls(self, guild_id: int, admin_user_id: int) -> Dict[str, int]:
        """Reprocess all resolved polls to fix point distribution"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get all resolved polls for this guild
            self._execute_query(cursor, """
                SELECT prediction_id, prediction_type, correct_option, week_number, title
                FROM predictions 
                WHERE guild_id = ? AND status = ? AND correct_option IS NOT NULL
                ORDER BY prediction_id ASC
            """, (guild_id, PredictionStatus.RESOLVED.value))
            
            resolved_polls = cursor.fetchall()
            
            if not resolved_polls:
                return {"polls_processed": 0, "users_updated": 0, "total_points_awarded": 0}
            
            stats = {
                "polls_processed": 0,
                "users_updated": 0,
                "total_points_awarded": 0,
                "polls_details": []
            }
            
            for poll_row in resolved_polls:
                if self.use_postgresql:
                    pred_id = poll_row['prediction_id']
                    pred_type = poll_row['prediction_type']
                    correct_option = poll_row['correct_option']
                    week_number = poll_row['week_number']
                    title = poll_row['title']
                else:
                    pred_id, pred_type, correct_option, week_number, title = poll_row
                
                # Process this poll
                poll_stats = self._reprocess_single_poll(cursor, pred_id, pred_type, correct_option, week_number, title, guild_id)
                
                stats["polls_processed"] += 1
                stats["users_updated"] += poll_stats["users_updated"]
                stats["total_points_awarded"] += poll_stats["points_awarded"]
                stats["polls_details"].append({
                    "id": pred_id,
                    "title": title,
                    "correct_users": poll_stats["correct_users"],
                    "points_awarded": poll_stats["points_awarded"]
                })
            
            conn.commit()
            logger.info(f"Reprocessed {stats['polls_processed']} polls, updated {stats['users_updated']} user records, awarded {stats['total_points_awarded']} total points")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error reprocessing resolved polls: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _reprocess_single_poll(self, cursor, pred_id: int, pred_type: str, correct_option: str, week_number: int, title: str, guild_id: int) -> Dict[str, int]:
        """Reprocess a single poll and return stats"""
        
        # Get all user predictions for this poll
        self._execute_query(cursor, """
            SELECT user_id, option FROM user_predictions 
            WHERE prediction_id = ?
        """, (pred_id,))
        
        if self.use_postgresql:
            user_predictions = [(row['user_id'], row['option']) for row in cursor.fetchall()]
        else:
            user_predictions = cursor.fetchall()
        
        if not user_predictions:
            return {"users_updated": 0, "points_awarded": 0, "correct_users": 0}
        
        # Calculate points for this prediction type
        try:
            prediction_type = PredictionType(pred_type)
            points_per_correct = self.POINT_VALUES[prediction_type]
        except (ValueError, KeyError):
            logger.warning(f"Unknown prediction type: {pred_type} for poll {pred_id}")
            points_per_correct = 5  # Default points
        
        current_week = week_number if week_number else self._get_current_week()
        
        users_updated = 0
        total_points_awarded = 0
        correct_users = 0
        
        for user_id, user_option in user_predictions:
            was_correct = (user_option == correct_option)
            points_to_award = points_per_correct if was_correct else 0
            
            if was_correct:
                correct_users += 1
            
            try:
                # Use the fixed _update_leaderboard_safe method
                self._update_leaderboard_safe(cursor, user_id, guild_id, current_week, points_to_award, was_correct, True)
                users_updated += 1
                total_points_awarded += points_to_award
                
            except Exception as e:
                logger.error(f"Error updating leaderboard for user {user_id} in poll {pred_id}: {e}")
                continue
        
        logger.info(f"Reprocessed poll '{title}' (ID: {pred_id}): {correct_users} correct out of {len(user_predictions)} predictions")
        
        return {
            "users_updated": users_updated,
            "points_awarded": total_points_awarded,
            "correct_users": correct_users
        }
    
    def _update_leaderboard_safe(self, cursor, user_id: int, guild_id: int, week_number: int, 
                               points: int, was_correct: bool, participated: bool):
        """Safe leaderboard update that adds to existing scores instead of replacing"""
        
        # Get current stats
        self._execute_query(cursor, """
            SELECT season_points, weekly_points, correct_predictions, total_predictions
            FROM prediction_leaderboard 
            WHERE user_id = ? AND guild_id = ? AND week_number = ?
        """, (user_id, guild_id, week_number))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing record - ADD to existing points instead of replacing
            if self.use_postgresql:
                season_points = result['season_points'] or 0
                weekly_points = result['weekly_points'] or 0
                correct_preds = result['correct_predictions'] or 0
                total_preds = result['total_predictions'] or 0
            else:
                season_points, weekly_points, correct_preds, total_preds = result
                season_points = season_points or 0
                weekly_points = weekly_points or 0
                correct_preds = correct_preds or 0
                total_preds = total_preds or 0
            
            new_season_points = season_points + points
            new_weekly_points = weekly_points + points
            new_correct = correct_preds + (1 if was_correct else 0)
            new_total = total_preds + (1 if participated else 0)
            
            self._execute_query(cursor, """
                UPDATE prediction_leaderboard 
                SET season_points = ?, weekly_points = ?, 
                    correct_predictions = ?, total_predictions = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ? AND guild_id = ? AND week_number = ?
            """, (new_season_points, new_weekly_points, new_correct, new_total,
                  user_id, guild_id, week_number))
        else:
            # Create new record
            self._execute_query(cursor, """
                INSERT INTO prediction_leaderboard 
                (user_id, guild_id, week_number, season_points, weekly_points, 
                 correct_predictions, total_predictions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, guild_id, week_number, points, points,
                  1 if was_correct else 0, 1 if participated else 0))
    
    def clear_all_leaderboard_data(self, guild_id: int) -> int:
        """Clear all leaderboard data for a guild (for fresh reprocessing)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            self._execute_query(cursor, """
                DELETE FROM prediction_leaderboard 
                WHERE guild_id = ?
            """, (guild_id,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"Cleared {deleted_count} leaderboard records for guild {guild_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error clearing leaderboard data: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()


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
        logger.info(f"DEBUG: Checking update hash {update.content_hash[:8]}...")
        
        cache_contains = await self.processed_hashes_cache.contains(update.content_hash)
        db_duplicate = self.db.is_duplicate(update.content_hash)
        
        logger.info(f"DEBUG: Cache contains: {cache_contains}, DB duplicate: {db_duplicate}")
        
        # Check BOTH cache AND database before processing
        is_rss = "@" not in update.author  # RSS updates don't have @ in author
        if not cache_contains and (is_rss or not db_duplicate):
            logger.info(f"DEBUG: Adding update to queues - {update.title[:50]}...")
            
            # Add to queues
            self.highlights_queue.append(update)
            self.hourly_queue.append(update)
            
            logger.info(f"DEBUG: After adding - Highlights: {len(self.highlights_queue)}, Hourly: {len(self.hourly_queue)}")
            
            # Store in database
            categories = self.analyzer.categorize_update(update)
            importance = self.analyzer.analyze_strategic_importance(update)
            self.db.store_update(update, importance, categories)
            
            
            # ONLY add to cache after successful processing
            await self.processed_hashes_cache.add(update.content_hash)
            
            logger.info(f"DEBUG: Successfully processed update")
        else:
            logger.info(f"DEBUG: Skipping duplicate - cache: {cache_contains}, db: {db_duplicate}")
    
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
        
        logger.info(f"Created highlights batch from {len(self.highlights_queue)} updates")
        return embeds
    
    async def create_hourly_summary(self) -> List[discord.Embed]:
        """Create hourly summary - ALWAYS USE DATABASE DATA FOR SPECIFIC TIMEFRAME"""
        now = datetime.now()
        
        # Define the previous hour period
        summary_hour = now.replace(minute=0, second=0, microsecond=0)
        hour_start = summary_hour - timedelta(hours=1)
        hour_end = summary_hour
        
        # ALWAYS get data from database for the specific hour
        hourly_updates = self.db.get_updates_in_timeframe(hour_start, hour_end)
        
        if not hourly_updates:
            return []  # This will trigger quiet hour message
        
        # Create summary from database data only
        if self.llm_client and await self._can_make_llm_request():
            try:
                embeds = await self._create_forced_structured_summary_from_db(hourly_updates, hour_start, hour_end)
                return embeds
            except Exception as e:
                logger.error(f"LLM hourly summary failed: {e}")
                return self._create_pattern_hourly_summary_for_timeframe(hourly_updates, hour_start, hour_end)
        else:
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

    async def _create_forced_structured_summary_from_db(self, updates: List[BBUpdate], hour_start: datetime, hour_end: datetime) -> List[discord.Embed]:
        """Create structured summary from database data for specific timeframe"""
        await self.rate_limiter.wait_if_needed()
        
        # Sort updates chronologically
        sorted_updates = sorted(updates, key=lambda x: x.pub_date)
        
        # Format updates in chronological order (limit to prevent token overflow)
        formatted_updates = []
        for i, update in enumerate(sorted_updates[:20], 1):  # Limit to 20 updates
            time_str = self._extract_correct_time(update)
            time_str = time_str.lstrip('0')
            formatted_updates.append(f"{i}. {time_str} - {update.title}")
            if update.description and update.description != update.title:
                desc = update.description[:150] + "..." if len(update.description) > 150 else update.description
                formatted_updates.append(f"   {desc}")
        
        updates_text = "\n".join(formatted_updates)
        
        # Calculate current day
        current_day = max(1, (datetime.now().date() - datetime(2025, 7, 8).date()).days + 1)
        
        # Format hour period for display
        hour_period = f"{hour_start.strftime('%I %p')} - {hour_end.strftime('%I %p')}"
        
        # Build structured prompt
        prompt = f"""You are a Big Brother superfan analyst creating an hourly summary for Day {current_day}.
    
    HOUR PERIOD: {hour_period}
    UPDATES FROM THIS HOUR ({len(updates)} total) - IN CHRONOLOGICAL ORDER:
    {updates_text}
    
    Create a comprehensive summary that presents events chronologically as they happened during {hour_period}.
    
    Provide your analysis in this EXACT JSON format:
    
    {{
        "headline": "Brief headline summarizing the most important development during {hour_period}",
        "strategic_analysis": "Analysis of game moves, alliance discussions, targeting decisions, and strategic positioning during this hour. Only include if there are meaningful strategic developments - otherwise use null.",
        "alliance_dynamics": "Analysis of alliance formations, betrayals, trust shifts, and relationship changes during this hour. Only include if there are meaningful alliance developments - otherwise use null.",
        "entertainment_highlights": "Funny moments, drama, memorable interactions, and lighthearted content from this hour. Only include if there are entertaining moments - otherwise use null.",
        "showmance_updates": "Romance developments, flirting, relationship drama, and intimate moments during this hour. Only include if there are romance-related developments - otherwise use null.",
        "house_culture": "Daily routines, traditions, group dynamics, living situations, and house atmosphere during this hour. Only include if there are meaningful cultural/social developments - otherwise use null.",
        "key_players": ["List", "of", "houseguests", "who", "were", "central", "to", "this", "hour's", "developments"],
        "overall_importance": 8,
        "importance_explanation": "Brief explanation of why this hour received this importance score (1-10 scale)"
    }}
    
    CRITICAL INSTRUCTIONS:
    - ONLY include sections where there are actual meaningful developments during {hour_period}
    - Use null for any section that doesn't have substantial content
    - Present events chronologically (earliest to latest) within {hour_period}
    - Key players should be the houseguests most central to this specific hour's events
    - Overall importance: 1-3 (quiet hour), 4-6 (moderate activity), 7-8 (high drama/strategy), 9-10 (explosive/game-changing)
    - Don't force content into sections - be selective and only include what's truly noteworthy
    - If a section would be empty or just say "nothing happened", use null instead
    - Focus specifically on what happened during {hour_period}, not general game state"""
    
        try:
            # Call LLM
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=3000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            # Parse JSON response
            analysis_data = self._parse_structured_llm_response(response_text)
            
            # Create structured embed using existing method
            embeds = self._create_structured_summary_embed(
                analysis_data, len(updates), "hourly_summary"
            )
            
            logger.info(f"Created structured hourly summary from database data: {len(updates)} updates from {hour_period}")
            return embeds
            
        except Exception as e:
            logger.error(f"Failed to create structured summary from DB: {e}")
            logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
            # Fallback to pattern-based summary
            return self._create_pattern_hourly_summary_for_timeframe(updates, hour_start, hour_end)
    
    async def _create_forced_structured_summary(self, summary_type: str) -> List[discord.Embed]:
        """Create structured summary with forced contextual format"""
        await self.rate_limiter.wait_if_needed()
        
        # Sort updates chronologically before formatting
        sorted_updates = sorted(self.hourly_queue, key=lambda x: x.pub_date)
        
        # Format updates in chronological order
        formatted_updates = []
        for i, update in enumerate(sorted_updates, 1):
            time_str = self._extract_correct_time(update)
            time_str = time_str.lstrip('0')
            formatted_updates.append(f"{i}. {time_str} - {update.title}")
            if update.description and update.description != update.title:
                desc = update.description[:150] + "..." if len(update.description) > 150 else update.description
                formatted_updates.append(f"   {desc}")
        
        updates_text = "\n".join(formatted_updates)
        
        # Calculate current day
        current_day = max(1, (datetime.now().date() - datetime(2025, 7, 8).date()).days + 1)
        
        # Build structured prompt WITH PERSONALITY
        prompt = f"""You are a Big Brother superfan analyst with a sassy personality creating an hourly summary for Day {current_day}.
    
    PERSONALITY TRAITS:
    - You're a superfan who loves the strategic and messy gameplay
    - You have a bit of sass and aren't afraid to throw shade when houseguests make questionable moves
    - You're slightly biased against Rachel - when she does something good, acknowledge it but with a hint of reluctance or backhanded compliment
    - You love drama and get excited when the house gets messy
    - You appreciate good gameplay but LOVE calling out bad gameplay with witty commentary
    
    NEW UPDATES TO ANALYZE (Day {current_day}) - IN CHRONOLOGICAL ORDER:
    {updates_text}
    
    Create a comprehensive summary that presents events chronologically as they happened.
    
    Provide your analysis in this EXACT JSON format:
    
    {{
        "headline": "Brief headline summarizing the most important development this hour (feel free to be sassy if someone made a bad move)",
        "strategic_analysis": "Analysis of game moves with your honest, slightly sassy take. Be especially critical of bad gameplay. If Rachel is involved, include subtle shade or reluctant praise as appropriate. Only include if there are meaningful strategic developments - otherwise use null.",
        "alliance_dynamics": "Analysis of alliance formations, betrayals, trust shifts with commentary on whether these are smart moves or messy choices. Only include if there are meaningful alliance developments - otherwise use null.",
        "entertainment_highlights": "Get excited about the drama! Funny moments, fights, memorable interactions - this is where your personality shines. Only include if there are entertaining moments - otherwise use null.",
        "showmance_updates": "Romance developments with your take on whether it's cute or cringe. Only include if there are romance-related developments - otherwise use null.",
        "house_culture": "Daily routines and house atmosphere with witty observations. Only include if there are meaningful cultural/social developments - otherwise use null.",
        "key_players": ["List", "of", "houseguests", "who", "were", "central", "to", "this", "hour's", "developments"],
        "overall_importance": 8,
        "importance_explanation": "Brief explanation with personality - get excited about drama, be sassy about bad gameplay"
    }}
    
    CRITICAL INSTRUCTIONS:
    - Let your personality show! Be a fun, sassy superfan
    - When Rachel does something good, acknowledge it but maybe add "I can't believe I'm saying this but..." or "Credit where it's due..."
    - When Rachel messes up, enjoy it a little: "Classic Rachel move" or "Rachel gonna Rachel"
    - Celebrate messy gameplay and drama
    - Call out bad strategic moves with wit
    - Keep it fun but still informative
    - Use null for any section that doesn't have substantial content"""

        # Call LLM
        response = await asyncio.to_thread(
            self.llm_client.messages.create,
            model="claude-3-haiku-20240307",
            max_tokens=3000,
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
        
        # Prepare update data in chronological order with full content
        formatted_updates = []
        for i, update in enumerate(sorted_updates, 1):
            full_title = update.title
            cleaned_title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*[-–]\s*', '', full_title)
            
            if update.description and update.description != update.title and len(update.description.strip()) > 10:
                content = f"{cleaned_title} - {update.description}"
            else:
                content = cleaned_title
            
            formatted_updates.append(f"{i}. {content}")
        
        updates_text = "\n".join(formatted_updates)
        
        prompt = f"""You are a sassy Big Brother superfan curating the MOST IMPORTANT moments from these {len(self.highlights_queue)} recent updates.
    
    PERSONALITY:
    - You LIVE for the drama and aren't shy about it
    - You throw shade at bad gameplay (especially Rachel's questionable choices)
    - You get genuinely excited when things get messy
    - You reluctantly give Rachel credit when she does well ("Ugh, fine, Rachel actually did something right")

    
    UPDATES IN CHRONOLOGICAL ORDER (earliest first):
    {updates_text}
    
    Select 6-10 updates that are TRUE HIGHLIGHTS - moments that stand out as particularly important, dramatic, funny, or game-changing.
    
    HIGHLIGHT-WORTHY updates include:
    - INSIDE THE HOUSE: What houseguests are doing, saying, strategizing, fighting about
    - Competition wins (HOH, POV, etc.) - but focus on the houseguests' reactions and gameplay
    - Major strategic moves or betrayals between houseguests
    - Dramatic fights or confrontations between houseguests
    - Romantic moments between houseguests (first kiss, breakup, etc.)
    - Hilarious or memorable incidents happening in the house
    - Alliance formations or breaks between houseguests
    - Emotional moments, breakdowns, celebrations by houseguests
    
    AVOID highlighting:
    - TV episode schedules or upcoming shows (unless houseguests are discussing them)
    - Production updates not involving houseguest reactions
    - Technical feed issues (unless houseguests react to them)
    - General announcements not affecting house dynamics
    
    For each selected update, provide them in CHRONOLOGICAL ORDER with NO TIMESTAMPS:
    
    {{
    "highlights": [
        {{
            "summary": "Create a COMPLETE SUMMARY with your sassy take. If it's bad gameplay, roast it. If it's Rachel messing up, enjoy it. If it's drama, celebrate it! Make it engaging and fun while informative.",
            "importance_emoji": "🔥 for high drama/strategy, ⭐ for notable moments, 📝 for interesting developments, 🙄 for Rachel's questionable choices",
            "context": "ONLY add this field if crucial context is needed. Keep it brief and sassy."
        }}
    ]
}}

Remember: We're here for the MESS. Give us the tea with extra sass!"""
    
    CRITICAL INSTRUCTIONS:
    - NO TIMESTAMPS - remove all time references 
    - Create COMPLETE summaries - never truncate or cut off mid-sentence
    - Make summaries engaging and informative - tell the full story
    - Focus on HOUSEGUEST activities and reactions, not production/TV scheduling
    - Present the selected highlights in CHRONOLOGICAL ORDER from earliest to latest
    - Each summary should be a complete thought that stands alone"""
    
        try:
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.llm_model,
                max_tokens=4000,  # INCREASED from 3500 to prevent cutoff
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse and create embed
            try:
                highlights_data = self._parse_llm_response(response.content[0].text)
                
                if not highlights_data.get('highlights'):
                    logger.warning("No highlights in LLM response, using pattern fallback")
                    return [self._create_pattern_highlights_embed()]
                
                # SORT THE HIGHLIGHTS BY CHRONOLOGICAL ORDER (backup enforcement)
                highlights = highlights_data['highlights']
                
                embed = discord.Embed(
                    title="📹 Feed Highlights - What Just Happened",
                    description=f"Key moments from the last {len(self.highlights_queue)} updates",
                    color=0xe74c3c,
                    timestamp=datetime.now()
                )
                
                for i, highlight in enumerate(highlights[:10], 1):
                    summary = highlight.get('summary', 'Update')
                    importance_emoji = highlight.get('importance_emoji', '📝')
                    context = highlight.get('context', '').strip()
                    
                    # Create field name without timestamp
                    field_name = f"{importance_emoji} Highlight {i}"
                    
                    # Create field value
                    if context:
                        field_value = f"{summary}\n\n*{context}*"
                    else:
                        field_value = summary
                    
                    embed.add_field(
                        name=field_name,
                        value=field_value,
                        inline=False
                    )
                
                embed.set_footer(text=f"Highlights • {len(self.highlights_queue)} updates processed • No timestamps for cleaner reading")
                return [embed]
                
            except Exception as e:
                logger.error(f"Failed to parse highlights response: {e}")
                return [self._create_pattern_highlights_embed()]
                
        except Exception as e:
            logger.error(f"LLM highlights request failed: {e}")
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
                max_tokens=3000,
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
        """Parse LLM response with better JSON handling and debugging"""
        try:
            # Clean the response first
            cleaned_text = response_text.strip()
            
            # Find JSON boundaries more carefully
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = cleaned_text[json_start:json_end]
                
                # Log the JSON for debugging
                logger.debug(f"Attempting to parse JSON: {json_text[:500]}...")
                
                # Clean up common JSON issues
                json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
                json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas in arrays
                
                return json.loads(json_text)
            else:
                raise ValueError("No valid JSON found")
                
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Raw response: {response_text[:500]}...")
            
            # Better fallback that doesn't show raw JSON
            return {
                "highlights": []  # Return empty list to trigger pattern fallback
            }
    
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
            name="🔑 Key Players",
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
            name="🔑 Key Players",
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
                name="🔑 Key Players This Hour",
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
        """Extract the correct PACIFIC time from the update content"""
        # Look for Pacific time patterns first
        pacific_patterns = [
            r'(\d{1,2}:\d{2})\s*(PM|AM)\s*PST',
            r'(\d{1,2}:\d{2})\s*(PM|AM)\s*Pacific',
            r'(\d{1,2}:\d{2})\s*(PM|AM)\s*PT'
        ]
        
        # First try the title
        for pattern in pacific_patterns:
            match = re.search(pattern, update.title, re.IGNORECASE)
            if match:
                time_str = match.group(1)
                ampm = match.group(2).upper()
                return f"{time_str} {ampm} PST"
        
        # Then try the description
        for pattern in pacific_patterns:
            match = re.search(pattern, update.description, re.IGNORECASE)
            if match:
                time_str = match.group(1)
                ampm = match.group(2).upper()
                return f"{time_str} {ampm} PST"
        
        # Fallback - convert pub_date to Pacific
        try:
            import pytz
            pacific_tz = pytz.timezone('US/Pacific')
            if update.pub_date.tzinfo is None:
                # Assume UTC if no timezone
                utc_time = pytz.utc.localize(update.pub_date)
            else:
                utc_time = update.pub_date.astimezone(pytz.utc)
            
            pacific_time = utc_time.astimezone(pacific_tz)
            return pacific_time.strftime("%I:%M %p PST").lstrip('0')
        except:
            # Final fallback
            return update.pub_date.strftime("%I:%M %p PST").lstrip('0')
    
    def get_rate_limit_stats(self) -> Dict[str, int]:
        """Get current rate limiting statistics"""
        return self.rate_limiter.get_stats()

    
    async def _create_llm_daily_buzz(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
        """Create LLM-powered Daily Buzz format"""
        await self.rate_limiter.wait_if_needed()
        
        # Sort updates chronologically and format for LLM
        sorted_updates = sorted(updates, key=lambda x: x.pub_date)
        
        # Limit updates to prevent token overflow
        formatted_updates = []
        for i, update in enumerate(sorted_updates[:30], 1):  # Top 30 most recent
            time_str = self._extract_correct_time(update)
            formatted_updates.append(f"{i}. {time_str} - {update.title}")
            if update.description and update.description != update.title:
                desc = update.description[:100] + "..." if len(update.description) > 100 else update.description
                formatted_updates.append(f"   {desc}")
        
        updates_text = "\n".join(formatted_updates)
        
        prompt = f"""You are creating "THE DAILY BUZZ" for Big Brother Day {day_number} - a Twitter-style breakdown of key house dynamics.
    
    UPDATES FROM DAY {day_number} (chronological order):
    {updates_text}
    
    Create a Daily Buzz in this EXACT format:
    
    {{
        "buzz_items": [
            "Morgan pitched a seven-person voting group to Zach with herself, Vince, Rachel, Ava, Will, and Lauren, with some talk of including Mickey or looping in Amy as an extension...but nothing official has formed.",
            "Jimmy told Will he's considering nominating him as a replacement, despite promising he wouldn't.",
            "Rachel tried to shift Jimmy toward targeting Adrian, not realizing Jimmy had already moved his sights to Will.",
            "Rachel and Keanu clashed again after he brought up her low points from BB12. She called him out, he deflected, and later apologized...but as they hugged it out, Rachel rolled her eyes and smirked at the camera.",
            "Rachel started rallying support to keep Amy or Will, depending on who lands on the block."
        ]
    }}
    
    INSTRUCTIONS:
    - Create 5-10 bullet points of the MOST IMPORTANT house dynamics/developments from Day {day_number}
    - Focus on: alliance talks, targeting decisions, relationship shifts, strategic moves, key confrontations
    - Write in a casual, engaging Twitter style like the example
    - Each bullet should be 1-2 sentences max
    - Include specific names and concrete actions
    - Prioritize information that affects the current game state going into today
    - Don't include minor daily routine stuff unless it's strategically significant"""
    
        try:
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=3000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            buzz_data = self._parse_llm_response(response.content[0].text)
            return self._create_daily_buzz_embed(buzz_data, day_number, len(updates))
            
        except Exception as e:
            logger.error(f"Failed to parse daily buzz response: {e}")
            return self._create_pattern_daily_buzz(updates, day_number)
    
    def _create_pattern_daily_buzz(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
        """Pattern-based fallback for Daily Buzz"""
        
        # Get top 8 most important updates
        top_updates = sorted(
            updates, 
            key=lambda x: self.analyzer.analyze_strategic_importance(x), 
            reverse=True
        )[:8]
        
        # Create buzz items from top updates
        buzz_items = []
        for update in top_updates:
            # Clean the title
            title = update.title
            title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*[-–]\s*', '', title)
            
            # Truncate if too long
            if len(title) > 150:
                title = title[:147] + "..."
            
            buzz_items.append(title)
        
        buzz_data = {"buzz_items": buzz_items}
        return self._create_daily_buzz_embed(buzz_data, day_number, len(updates))
    
    def _create_daily_buzz_embed(self, buzz_data: dict, day_number: int, total_updates: int) -> List[discord.Embed]:
        """Create the Daily Buzz embed in Twitter style"""
        
        # Get current date for the header
        current_date = datetime.now().strftime("%A, %B %d")
        
        embed = discord.Embed(
            title=f"THE RUNDOWN // #BB27 // House Happenings // {current_date} 👁️💬🏠📄🔍",
            description="",  # No description, go straight to content
            color=0x1DA1F2,  # Twitter blue
            timestamp=datetime.now()
        )
        
        # Add numbered buzz items
        buzz_items = buzz_data.get("buzz_items", [])
        if buzz_items:
            buzz_text = []
            for i, item in enumerate(buzz_items[:10], 1):  # Max 10 items
                # Add number emoji
                number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
                number = number_emojis[i-1] if i <= 10 else f"{i}️⃣"
                buzz_text.append(f"{number} {item}")
            
            # Join with double newlines for spacing
            embed.description = "\n\n".join(buzz_text)
        
        # Add hashtags at the bottom like Twitter
        embed.add_field(
            name="",
            value="#BigBrother #BigBrotherBuzz",
            inline=False
        )
        
        embed.set_footer(text=f"Daily Rundown • Day {day_number} • Based on {total_updates} feed updates")
        
        return [embed]

    async def _create_llm_daily_buzz(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
        """Create LLM-powered Daily Buzz format"""
        if not self.llm_client:
            return self._create_pattern_daily_buzz(updates, day_number)
        
        await self.rate_limiter.wait_if_needed()
        
        # Sort updates chronologically and format for LLM
        sorted_updates = sorted(updates, key=lambda x: x.pub_date)
        
        # Limit updates to prevent token overflow
        formatted_updates = []
        for i, update in enumerate(sorted_updates[:30], 1):  # Top 30 most recent
            time_str = self._extract_correct_time(update)
            formatted_updates.append(f"{i}. {time_str} - {update.title}")
            if update.description and update.description != update.title:
                desc = update.description[:100] + "..." if len(update.description) > 100 else update.description
                formatted_updates.append(f"   {desc}")
        
        updates_text = "\n".join(formatted_updates)
        
        prompt = f"""You are creating "THE DAILY BUZZ" for Big Brother Day {day_number} - a Twitter-style breakdown of key house dynamics.
    
    UPDATES FROM DAY {day_number} (chronological order):
    {updates_text}
    
    Create a Daily Buzz in this EXACT format:
    
    {{
        "buzz_items": [
            "Morgan pitched a seven-person voting group to Zach with herself, Vince, Rachel, Ava, Will, and Lauren, with some talk of including Mickey or looping in Amy as an extension...but nothing official has formed.",
            "Jimmy told Will he's considering nominating him as a replacement, despite promising he wouldn't.",
            "Rachel tried to shift Jimmy toward targeting Adrian, not realizing Jimmy had already moved his sights to Will.",
            "Rachel and Keanu clashed again after he brought up her low points from BB12. She called him out, he deflected, and later apologized...but as they hugged it out, Rachel rolled her eyes and smirked at the camera.",
            "Rachel started rallying support to keep Amy or Will, depending on who lands on the block."
        ]
    }}
    
    INSTRUCTIONS:
    - Create 5-10 bullet points of the MOST IMPORTANT house dynamics/developments from Day {day_number}
    - Focus on: alliance talks, targeting decisions, relationship shifts, strategic moves, key confrontations
    - Write in a casual, engaging Twitter style like the example
    - Each bullet should be 1-2 sentences max
    - Include specific names and concrete actions
    - Prioritize information that affects the current game state going into today
    - Don't include minor daily routine stuff unless it's strategically significant"""
    
        try:
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=3000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            buzz_data = self._parse_llm_response(response.content[0].text)
            return self._create_daily_buzz_embed(buzz_data, day_number, len(updates))
            
        except Exception as e:
            logger.error(f"Failed to parse daily buzz response: {e}")
            return self._create_pattern_daily_buzz(updates, day_number)
            
def _create_pattern_daily_buzz(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
    """Pattern-based fallback for Daily Buzz"""
    
    # Get top 8 most important updates
    top_updates = sorted(
        updates, 
        key=lambda x: self.analyzer.analyze_strategic_importance(x), 
        reverse=True
    )[:8]
    
    # Create buzz items from top updates
    buzz_items = []
    for update in top_updates:
        # Clean the title
        title = update.title
        title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*[-–]\s*', '', title)
        
        # Truncate if too long
        if len(title) > 150:
            title = title[:147] + "..."
        
        buzz_items.append(title)
    
    buzz_data = {"buzz_items": buzz_items}
    return self._create_daily_buzz_embed(buzz_data, day_number, len(updates))

def _create_daily_buzz_embed(self, buzz_data: dict, day_number: int, total_updates: int) -> List[discord.Embed]:
    """Create the Daily Buzz embed in Twitter style"""
    
    # Get current date for the header
    current_date = datetime.now().strftime("%A, %B %d")
    
    embed = discord.Embed(
        title=f"THE RUNDOWN // #BB27 // House Happenings // {current_date} 👁️💬🏠📄🔍",
        description="",  # No description, go straight to content
        color=0x1DA1F2,  # Twitter blue
        timestamp=datetime.now()
    )
    
    # Add numbered buzz items
    buzz_items = buzz_data.get("buzz_items", [])
    if buzz_items:
        buzz_text = []
        for i, item in enumerate(buzz_items[:10], 1):  # Max 10 items
            # Add number emoji
            number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
            number = number_emojis[i-1] if i <= 10 else f"{i}️⃣"
            buzz_text.append(f"{number} {item}")
        
        # Join with double newlines for spacing
        embed.description = "\n\n".join(buzz_text)
    
    # Add hashtags at the bottom like Twitter
    embed.add_field(
        name="",
        value="#BigBrother #BigBrotherBuzz",
        inline=False
    )
    
    embed.set_footer(text=f"Daily Rundown • Day {day_number} • Based on {total_updates} feed updates")
    
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
        max_tokens=2500,
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
            max_tokens=3000,
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
                            pub_date=update_data['pub_date'] if isinstance(update_data['pub_date'], datetime) else datetime.fromisoformat(update_data['pub_date']),
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
                            pub_date=update_data['pub_date'] if isinstance(update_data['pub_date'], datetime) else datetime.fromisoformat(update_data['pub_date']),
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
            max_tokens=3000,
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
            name="🔑 Key Players",
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
            name="🔑 Key Players",
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
            max_tokens=3000,
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
            max_tokens=3000,
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
                name="🔑 Key Players",
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
            max_tokens=800,
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

    # Add these methods to the UpdateBatcher class

    def _get_rachel_commentary(self, is_positive: bool = False) -> str:
        """Get appropriate Rachel commentary"""
        if is_positive:
            return random.choice([
                "I can't believe I'm saying this, but Rachel actually made a good move",
                "Credit where it's due, Rachel didn't completely mess this up",
                "Shocking development: Rachel did something right for once",
                "In a surprising turn of events, Rachel showed competence",
                "Rachel accidentally stumbled into a good decision"
            ])
        else:
            return random.choice([
                "Classic Rachel move right there",
                "Rachel gonna Rachel, I guess",
                "And this is why Rachel is... Rachel",
                "Rachel continuing her streak of questionable choices",
                "Rachel proving once again why she's... special"
            ])
    
    def _get_sassy_drama_reaction(self) -> str:
        """Get sassy reactions to drama"""
        return random.choice([
            "The girls are FIGHTING!",
            "And THIS is why we watch feeds!",
            "The chaos we deserve!",
            "Someone get the popcorn!",
            "The mess is IMMACULATE!",
            "Drama alert! Drama alert!",
            "This is the content we signed up for!"
        ])
    
    def _get_bad_gameplay_roast(self, houseguest: str) -> str:
        """Roast bad gameplay"""
        return random.choice([
            f"{houseguest} really said 'strategy? never heard of her'",
            f"{houseguest} playing 4D chess... badly... in a checkers game",
            f"Someone needs to explain Big Brother to {houseguest} again",
            f"{houseguest} choosing chaos over logic, we love to see it",
            f"{houseguest} making choices that would make Dr. Will weep"
        ])
    
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

    async def cleanup_old_hourly_queue_items(self):
        """Remove items older than 2 hours from hourly queue to prevent endless growth"""
        if not self.hourly_queue:
            return
        
        cutoff_time = datetime.now() - timedelta(hours=2)
        original_size = len(self.hourly_queue)
        
        # Keep only items from last 2 hours
        self.hourly_queue = [update for update in self.hourly_queue if update.pub_date > cutoff_time]
        
        cleaned_count = original_size - len(self.hourly_queue)
        if cleaned_count > 0:
            logger.info(f"Cleaned {cleaned_count} old items from hourly queue")
    
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
    """PostgreSQL database handler with connection pooling"""
    
    def __init__(self, database_url: str, min_connections: int = 2, max_connections: int = 20):
        self.database_url = database_url
        self.connection_timeout = 30
        self.use_postgresql = True
        self.connection_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        
        # Initialize connection pool
        self._init_connection_pool(min_connections, max_connections)
        
        # Initialize database schema
        self.init_database()
    
    def _init_connection_pool(self, min_conn: int, max_conn: int):
        """Initialize PostgreSQL connection pool with retry logic"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing connection pool (attempt {attempt + 1}/{max_retries})")
                
                self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                    min_conn, max_conn,
                    self.database_url,
                    cursor_factory=psycopg2.extras.RealDictCursor,
                    connect_timeout=self.connection_timeout
                )
                
                # Test the pool
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1 as test_value")
                    result = cursor.fetchone()
                    if result['test_value'] != 1:
                        raise Exception("Pool test failed")
                
                logger.info(f"✅ Connection pool initialized successfully ({min_conn}-{max_conn} connections)")
                return
                
            except Exception as e:
                logger.error(f"❌ Connection pool initialization failed (attempt {attempt + 1}): {e}")
                
                if self.connection_pool:
                    try:
                        self.connection_pool.closeall()
                    except:
                        pass
                    self.connection_pool = None
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise Exception(f"Failed to initialize connection pool after {max_retries} attempts")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup"""
        if not self.connection_pool:
            raise Exception("Connection pool not initialized")
        
        conn = None
        try:
            # Get connection from pool (this blocks if pool is full)
            conn = self.connection_pool.getconn()
            
            if conn is None:
                raise Exception("Failed to get connection from pool")
            
            # Test connection is still alive
            if conn.closed:
                logger.warning("Got closed connection from pool, attempting to recreate")
                self.connection_pool.putconn(conn, close=True)
                conn = self.connection_pool.getconn()
            
            yield conn
            
        except Exception as e:
            # If there's an error, rollback the transaction
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except:
                    pass
            raise
        finally:
            # Return connection to pool
            if conn and self.connection_pool:
                try:
                    self.connection_pool.putconn(conn)
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
    
    def get_pool_status(self) -> dict:
        """Get connection pool statistics"""
        if not self.connection_pool:
            return {"status": "not_initialized"}
        
        try:
            return {
                "status": "active",
                "total_connections": len(self.connection_pool._pool + self.connection_pool._used),
                "available_connections": len(self.connection_pool._pool),
                "used_connections": len(self.connection_pool._used),
                "min_connections": self.connection_pool.minconn,
                "max_connections": self.connection_pool.maxconn
            }
        except Exception as e:
            logger.error(f"Error getting pool status: {e}")
            return {"status": "error", "error": str(e)}
    
    def close_pool(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            try:
                self.connection_pool.closeall()
                logger.info("Connection pool closed")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")
            finally:
                self.connection_pool = None
    
    # Update all your existing methods to use the context manager
    def is_duplicate(self, content_hash: str) -> bool:
        """Check if update already exists"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM updates WHERE content_hash = %s", (content_hash,))
                result = cursor.fetchone()
                return result is not None
                
        except Exception as e:
            logger.error(f"PostgreSQL duplicate check error: {e}")
            return False
    
    def store_update(self, update, importance_score: int = 1, categories = None):
        """Store a new update"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                categories_str = ",".join(categories) if categories else ""
                
                cursor.execute("""
                    INSERT INTO updates (content_hash, title, description, link, pub_date, author, importance_score, categories)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (update.content_hash, update.title, update.description, 
                      update.link, update.pub_date, update.author, importance_score, categories_str))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"PostgreSQL store error: {e}")
            raise
    
    def get_recent_updates(self, hours: int):
        """Get updates from the last N hours"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT title, description, link, pub_date, content_hash, author
                    FROM updates 
                    WHERE pub_date > NOW() - INTERVAL '%s hours'
                    ORDER BY pub_date DESC
                """, (hours,))
                
                results = cursor.fetchall()
                return [self._row_to_bb_update(row) for row in results]
                
        except Exception as e:
            logger.error(f"PostgreSQL query error: {e}")
            return []
    
    def get_updates_in_timeframe(self, start_time, end_time):
        """Get all updates within a specific timeframe"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT title, description, link, pub_date, content_hash, author
                    FROM updates 
                    WHERE pub_date >= %s AND pub_date < %s
                    ORDER BY pub_date ASC
                """, (start_time, end_time))
                
                results = cursor.fetchall()
                return [self._row_to_bb_update(row) for row in results]
                
        except Exception as e:
            logger.error(f"Database timeframe query error: {e}")
            return []
    
    def get_daily_updates(self, start_time, end_time):
        """Get all updates from a specific 24-hour period"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT title, description, link, pub_date, content_hash, author, importance_score
                    FROM updates 
                    WHERE pub_date >= %s AND pub_date < %s
                    ORDER BY pub_date ASC
                """, (start_time, end_time))
                
                results = cursor.fetchall()
                return [self._row_to_bb_update(row) for row in results]
                
        except Exception as e:
            logger.error(f"Database daily query error: {e}")
            return []
    
    def _row_to_bb_update(self, row):
        """Convert database row to BBUpdate object"""
        return BBUpdate(
            title=row['title'],
            description=row['description'],
            link=row['link'],
            pub_date=row['pub_date'],
            content_hash=row['content_hash'],
            author=row['author']
        )
    
    def init_database(self):
        """Initialize PostgreSQL tables - using connection pool"""
        try:
            with self.get_connection() as conn:
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
                
                # Add your other table creation code here...
                # (I'll include the rest in the next artifact)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_updates_hash ON updates(content_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_updates_date ON updates(pub_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_updates_pub_date_importance ON updates(pub_date, importance_score)")
                
                conn.commit()
                logger.info("PostgreSQL database initialized successfully with connection pool")
                
        except Exception as e:
            logger.error(f"PostgreSQL initialization error: {e}")
            raise


    

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
            
            # FIXED: Use proper database syntax
            if self.view.prediction_manager.use_postgresql:
                cursor.execute("""
                    SELECT user_id FROM user_predictions 
                    WHERE prediction_id = %s AND option = %s
                """, (prediction_id, correct_answer))
            else:
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
    """Analyzes current game state using ALL stored data to answer user questions"""
    
    def __init__(self, db: BBDatabase, alliance_tracker: AllianceTracker, analyzer: BBAnalyzer, llm_client=None):
        self.db = db
        self.alliance_tracker = alliance_tracker
        self.analyzer = analyzer
        self.llm_client = llm_client
    
    async def answer_question(self, question: str) -> dict:
        """Answer a question using ALL available data"""
        question_lower = question.lower()
        
        # Route to appropriate handler based on question type
        if any(keyword in question_lower for keyword in ['control', 'power', 'hoh', 'head of household']):
            return await self._analyze_power_structure(question)
        
        elif any(keyword in question_lower for keyword in ['danger', 'target', 'eviction', 'nominated', 'block']):
            return await self._analyze_danger_level(question)
        
        elif any(keyword in question_lower for keyword in ['alliance', 'working together', 'team', 'group']):
            return await self._analyze_alliances(question)
        
        elif any(keyword in question_lower for keyword in ['showmance', 'romance', 'relationship', 'dating', 'couple']):
            return await self._analyze_relationships(question)
        
        elif any(keyword in question_lower for keyword in ['winning', 'winner', 'favorite', 'best positioned']):
            return await self._analyze_win_chances(question)
        
        elif any(keyword in question_lower for keyword in ['drama', 'fight', 'argument', 'tension']):
            return await self._analyze_drama(question)
        
        elif any(keyword in question_lower for keyword in ['competition', 'comp', 'challenge', 'veto', 'pov']):
            return await self._analyze_competitions(question)
        
        else:
            return await self._general_analysis(question)
    
    async def _analyze_power_structure(self, question: str) -> dict:
        """Analyze power structure using ALL available data"""
        
        # Get ALL power-related data
        all_updates = self.db.get_recent_updates(24 * 30)  # Last 30 days
        power_updates = [u for u in all_updates if any(word in f"{u.title} {u.description}".lower() 
                        for word in ['hoh', 'head of household', 'power', 'control', 'nomination', 'veto'])]
        
        current_hoh = None
        
        # Check context tracker for most recent HOH win if available
        if hasattr(self, 'context_tracker') and self.alliance_tracker.use_postgresql:
            try:
                conn = self.alliance_tracker.get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT houseguest_name, created_at, description 
                    FROM houseguest_events 
                    WHERE event_type = 'hoh_win' 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                result = cursor.fetchone()
                if result:
                    current_hoh = result['houseguest_name'] if isinstance(result, dict) else result[0]
                conn.close()
            except Exception as e:
                logger.debug(f"Context tracker query failed: {e}")
        
        # Fallback: scan ALL power updates for HOH mentions
        if not current_hoh:
            hoh_patterns = [
                r'(\w+)\s+(?:wins?|won|is|becomes?)\s+(?:the\s+)?hoh',
                r'hoh\s+(?:winner|champion):\s*(\w+)',
                r'(\w+)\s+(?:has\s+)?won\s+hoh',
                r'(\w+)\s+(?:is\s+)?(?:the\s+)?(?:current\s+)?hoh',
                r'(\w+)\s+hoh\s+(?:win|victory|reign)',
                r'hoh\s+(\w+)',
                r'(\w+)\'s\s+hoh',
            ]
            
            # Sort by most recent first
            power_updates_sorted = sorted(power_updates, key=lambda x: x.pub_date, reverse=True)
            
            # Check most recent power updates first
            for update in power_updates_sorted[:100]:  # Check more updates
                content = f"{update.title} {update.description}".lower()
                
                for pattern in hoh_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        potential_hoh = match.group(1).strip().title()
                        
                        # More flexible validation
                        if potential_hoh in BB27_HOUSEGUESTS_SET:
                            current_hoh = potential_hoh
                            logger.info(f"Found HOH from pattern: {potential_hoh} in update: {update.title[:50]}")
                            break
                        elif potential_hoh.lower() in NICKNAME_MAP:
                            current_hoh = NICKNAME_MAP[potential_hoh.lower()]
                            logger.info(f"Found HOH from nickname: {potential_hoh} -> {current_hoh} in update: {update.title[:50]}")
                            break
                        # Try partial matching
                        for houseguest in BB27_HOUSEGUESTS:
                            if houseguest.lower().startswith(potential_hoh.lower()) and len(potential_hoh) >= 3:
                                current_hoh = houseguest
                                logger.info(f"Found HOH from partial match: {potential_hoh} -> {houseguest} in update: {update.title[:50]}")
                                break
                        
                        if current_hoh:
                            break
                if current_hoh:
                    break
        
        # Get ALL alliance data
        active_alliances = self.alliance_tracker.get_active_alliances()
        
        # Use LLM with comprehensive data
        if self.llm_client:
            return await self._llm_power_analysis_comprehensive(power_updates, active_alliances, current_hoh, question)
        else:
            return self._pattern_power_analysis_comprehensive(power_updates, active_alliances, current_hoh)
    
    async def _analyze_alliances(self, question: str) -> dict:
        """Analyze alliances using ALL stored alliance data"""
        
        active_alliances = self.alliance_tracker.get_active_alliances()
        broken_alliances = self.alliance_tracker.get_recently_broken_alliances(days=30)  # More history
        recent_betrayals = self.alliance_tracker.get_recent_betrayals(days=14)  # More history
        
        if self.llm_client:
            return await self._llm_alliance_analysis_comprehensive(active_alliances, broken_alliances, recent_betrayals, question)
        else:
            return self._pattern_alliance_analysis_comprehensive(active_alliances, broken_alliances, recent_betrayals)
    
    async def _analyze_danger_level(self, question: str) -> dict:
        """Analyze danger using ALL stored data"""
        
        # Get ALL updates that mention targeting/nominations/evictions
        all_updates = self.db.get_recent_updates(24 * 14)  # Last 2 weeks
        danger_updates = [u for u in all_updates if any(word in f"{u.title} {u.description}".lower() 
                         for word in ['nominate', 'nomination', 'block', 'target', 'backdoor', 'evict'])]
        
        # Extract nominees and targets from ALL relevant updates
        nominees = []
        targets = []
        
        for update in danger_updates:
            content = f"{update.title} {update.description}".lower()
            houseguests = self.analyzer.extract_houseguests(update.title + " " + update.description)
            
            if any(word in content for word in ['nominate', 'nomination', 'block']):
                nominees.extend(houseguests[:2])
            elif any(word in content for word in ['target', 'backdoor']):
                targets.extend(houseguests[:1])
        
        if self.llm_client:
            return await self._llm_danger_analysis_comprehensive(danger_updates, nominees, targets, question)
        else:
            return self._pattern_danger_analysis_comprehensive(danger_updates, nominees, targets)
    
    async def _llm_power_analysis_comprehensive(self, power_updates: List[BBUpdate], alliances: List[dict], current_hoh: str, question: str) -> dict:
        """LLM analysis using comprehensive data"""
        
        # Format recent power updates (limit for token management)
        updates_text = []
        for update in power_updates[:20]:
            time_str = self._extract_correct_time(update)
            updates_text.append(f"• {time_str} - {update.title}")
        
        updates_formatted = "\n".join(updates_text) if updates_text else "No power-related updates found"
        
        # Format all alliance data
        alliances_text = []
        for alliance in alliances:
            members_str = ", ".join(alliance['members'])
            alliances_text.append(f"• {alliance['name']}: {members_str} (confidence: {alliance['confidence']}%)")
        
        alliances_formatted = "\n".join(alliances_text) if alliances_text else "No active alliances detected"
        
        prompt = f"""You are analyzing the COMPLETE Big Brother season data to answer this question about power.

USER QUESTION: {question}

CURRENT/RECENT POWER DYNAMICS (from all stored data):
{updates_formatted}

ALL ACTIVE ALLIANCES:
{alliances_formatted}

DETECTED CURRENT HOH: {current_hoh or "Not clearly identified from data"}

BB27 HOUSEGUESTS: {', '.join(BB27_HOUSEGUESTS)}

Using ALL this accumulated season data, provide your analysis in this EXACT JSON format:

{{
    "main_answer": "Direct answer using all available season data",
    "current_hoh": "Current HOH name or 'Unknown' if unclear",
    "power_players": ["houseguests", "with", "current", "influence"],
    "power_analysis": "Analysis based on full season context",
    "confidence": "high/medium/low based on data completeness"
}}

Base your analysis on the COMPLETE picture from all the data, not just recent events."""
        
        try:
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=2500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_llm_chat_response(response.content[0].text, "power")
            
        except Exception as e:
            logger.error(f"LLM comprehensive power analysis failed: {e}")
            return self._pattern_power_analysis_comprehensive(power_updates, alliances, current_hoh)
    
    def _pattern_power_analysis_comprehensive(self, power_updates: List[BBUpdate], alliances: List[dict], current_hoh: str) -> dict:
        """Pattern analysis using all available data"""
        
        power_players = []
        
        if current_hoh:
            power_players.append(current_hoh)
        
        # Include members from ALL strong alliances
        for alliance in alliances:
            if alliance['confidence'] >= 60:  # Lower threshold to include more
                power_players.extend(alliance['members'][:3])
        
        power_players = list(set(power_players))[:8]  # More power players
        
        main_answer = f"Based on all stored season data ({len(power_updates)} power-related updates, {len(alliances)} alliances tracked), "
        
        if current_hoh:
            main_answer += f"the current HOH appears to be **{current_hoh}**. "
        else:
            main_answer += "the current HOH is not clearly identified from available data. "
        
        if power_players:
            main_answer += f"Key power players this season include: **{', '.join(power_players[:4])}**"
        else:
            main_answer += "Power structure analysis is limited by available data."
        
        return {
            "response_type": "power",
            "main_answer": main_answer,
            "current_hoh": current_hoh or "Unknown",
            "power_players": power_players,
            "power_analysis": f"Analysis based on {len(power_updates)} power-related updates and {len(alliances)} tracked alliances from the complete season.",
            "confidence": "high" if current_hoh and len(power_updates) > 10 else "medium",
            "data_source": "comprehensive_season_data"
        }
    
    def _extract_correct_time(self, update: BBUpdate) -> str:
        """Extract time from update (simplified)"""
        try:
            return update.pub_date.strftime("%I:%M %p")
        except:
            return "Unknown time"
    
    # Add similar comprehensive methods for other analysis types...
    async def _llm_alliance_analysis_comprehensive(self, active_alliances: List[dict], broken_alliances: List[dict], recent_betrayals: List[dict], question: str) -> dict:
        """Comprehensive alliance analysis using ALL data"""
        
        active_text = "\n".join([
            f"• {alliance['name']}: {', '.join(alliance['members'])} (confidence: {alliance['confidence']}%)"
            for alliance in active_alliances
        ]) if active_alliances else "No active alliances"
        
        broken_text = "\n".join([
            f"• {alliance['name']}: {', '.join(alliance['members'])}"
            for alliance in broken_alliances[:10]
        ]) if broken_alliances else "No broken alliances"
        
        betrayals_text = "\n".join([
            f"• {betrayal['description']}"
            for betrayal in recent_betrayals[:10]
        ]) if recent_betrayals else "No recent betrayals"
        
        prompt = f"""Using ALL stored Big Brother alliance data to answer this question.

USER QUESTION: {question}

ALL ACTIVE ALLIANCES:
{active_text}

ALL BROKEN ALLIANCES (Season History):
{broken_text}

ALL BETRAYALS (Season History):
{betrayals_text}

Provide comprehensive analysis using this COMPLETE season alliance data:

{{
    "main_answer": "Answer using full season alliance history",
    "strongest_alliances": ["current", "strongest", "alliances"],
    "alliance_analysis": "Analysis of complete alliance landscape",
    "betrayal_patterns": "Patterns from all betrayal data",
    "confidence": "high/medium/low"
}}"""
        
        try:
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=2500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_llm_chat_response(response.content[0].text, "alliances")
            
        except Exception as e:
            logger.error(f"LLM comprehensive alliance analysis failed: {e}")
            return self._pattern_alliance_analysis_comprehensive(active_alliances, broken_alliances, recent_betrayals)
    
    def _pattern_alliance_analysis_comprehensive(self, active_alliances: List[dict], broken_alliances: List[dict], betrayals: List[dict]) -> dict:
        """Pattern-based comprehensive alliance analysis"""
        
        strong_alliances = [a for a in active_alliances if a['confidence'] >= 70]
        
        main_answer = f"Based on complete season alliance tracking: {len(active_alliances)} active alliances, {len(broken_alliances)} broken alliances, {len(betrayals)} betrayals recorded. "
        
        if strong_alliances:
            main_answer += f"Strongest current alliance: '{strong_alliances[0]['name']}' with {', '.join(strong_alliances[0]['members'])}"
        else:
            main_answer += "No highly confident alliances currently active."
        
        return {
            "response_type": "alliances",
            "main_answer": main_answer,
            "strongest_alliances": [a['name'] for a in strong_alliances[:3]],
            "alliance_analysis": f"Complete season tracking shows {len(active_alliances)} active alliances with {len(betrayals)} total betrayals",
            "confidence": "high",
            "data_source": "comprehensive_alliance_tracker"
        }
    
    # Placeholder methods for other analysis types - implement similar comprehensive approach
    async def _llm_danger_analysis_comprehensive(self, danger_updates, nominees, targets, question):
        # Similar comprehensive approach for danger analysis
        return await self._llm_danger_analysis(danger_updates, nominees, targets, question)
    
    def _pattern_danger_analysis_comprehensive(self, danger_updates, nominees, targets):
        # Similar comprehensive approach for danger analysis  
        return self._pattern_danger_analysis(danger_updates, nominees, targets)
    
    async def _analyze_relationships(self, question: str) -> dict:
        # Get ALL relationship updates, not just recent
        all_updates = self.db.get_recent_updates(24 * 30)
        relationship_updates = [u for u in all_updates if "💕 Romance" in self.analyzer.categorize_update(u)]
        
        return {
            "response_type": "relationships",
            "main_answer": f"Based on {len(relationship_updates)} relationship-related updates from the complete season data.",
            "confidence": "medium",
            "data_source": "comprehensive_season_data"
        }
    
    async def _analyze_win_chances(self, question: str) -> dict:
        # Use ALL strategic updates and alliance data
        all_updates = self.db.get_recent_updates(24 * 30)
        strategic_updates = [u for u in all_updates if self.analyzer.analyze_strategic_importance(u) >= 6]
        active_alliances = self.alliance_tracker.get_active_alliances()
        
        return {
            "response_type": "winners", 
            "main_answer": f"Winner analysis based on {len(strategic_updates)} strategic updates and {len(active_alliances)} alliances from complete season data.",
            "confidence": "medium",
            "data_source": "comprehensive_season_data"
        }
    
    async def _analyze_drama(self, question: str) -> dict:
        # Get ALL drama updates
        all_updates = self.db.get_recent_updates(24 * 30)
        drama_updates = [u for u in all_updates if "💥 Drama" in self.analyzer.categorize_update(u)]
        
        return {
            "response_type": "drama",
            "main_answer": f"Drama analysis based on {len(drama_updates)} conflict-related updates from complete season data.",
            "confidence": "medium", 
            "data_source": "comprehensive_season_data"
        }
    
    async def _analyze_competitions(self, question: str) -> dict:
        """Analyze competitions and actually find specific winners"""
        question_lower = question.lower()
        
        # Get ALL competition updates
        all_updates = self.db.get_recent_updates(24 * 30)
        comp_updates = [u for u in all_updates if "🏆 Competition" in self.analyzer.categorize_update(u) or 
                       any(word in f"{u.title} {u.description}".lower() for word in ['veto', 'pov', 'hoh', 'competition', 'wins', 'winner'])]
        
        # Sort by most recent first
        comp_updates_sorted = sorted(comp_updates, key=lambda x: x.pub_date, reverse=True)
        
        # Determine what they're asking about
        asking_about_veto = any(word in question_lower for word in ['veto', 'pov', 'power of veto'])
        asking_about_hoh = any(word in question_lower for word in ['hoh', 'head of household'])
        
        found_winner = None
        winning_update = None
        
        # Look for specific competition winners
        if asking_about_veto:
            veto_patterns = [
                r'(\w+)\s+(?:wins?|won)\s+(?:the\s+)?(?:power\s+of\s+)?veto',
                r'(\w+)\s+(?:wins?|won)\s+(?:the\s+)?pov',
                r'veto\s+(?:winner|champion):\s*(\w+)',
                r'pov\s+(?:winner|champion):\s*(\w+)',
                r'(\w+)\s+(?:has\s+)?won\s+veto',
            ]
            
            for update in comp_updates_sorted[:50]:
                content = f"{update.title} {update.description}".lower()
                
                for pattern in veto_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        potential_winner = match.group(1).strip().title()
                        if self._validate_houseguest(potential_winner):
                            found_winner = self._validate_houseguest(potential_winner)
                            winning_update = update
                            logger.info(f"Found veto winner: {found_winner} in update: {update.title[:50]}")
                            break
                if found_winner:
                    break
        
        elif asking_about_hoh:
            hoh_patterns = [
                r'(\w+)\s+(?:wins?|won)\s+(?:the\s+)?hoh',
                r'hoh\s+(?:winner|champion):\s*(\w+)',
                r'(\w+)\s+(?:has\s+)?won\s+hoh',
            ]
            
            for update in comp_updates_sorted[:50]:
                content = f"{update.title} {update.description}".lower()
                
                for pattern in hoh_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        potential_winner = match.group(1).strip().title()
                        if self._validate_houseguest(potential_winner):
                            found_winner = self._validate_houseguest(potential_winner)
                            winning_update = update
                            logger.info(f"Found HOH winner: {found_winner} in update: {update.title[:50]}")
                            break
                if found_winner:
                    break
        
        # Create response
        if found_winner:
            comp_type = "veto" if asking_about_veto else "HOH" if asking_about_hoh else "competition"
            main_answer = f"**{found_winner}** won the most recent {comp_type} based on the available data."
            if winning_update:
                days_ago = (datetime.now() - winning_update.pub_date).days
                if days_ago == 0:
                    time_desc = "today"
                elif days_ago == 1:
                    time_desc = "yesterday"
                else:
                    time_desc = f"{days_ago} days ago"
                main_answer += f" (Update from {time_desc})"
        else:
            comp_type = "veto" if asking_about_veto else "HOH" if asking_about_hoh else "competition"
            main_answer = f"Could not identify the most recent {comp_type} winner from the available data ({len(comp_updates)} competition updates searched)."
        
        return {
            "response_type": "competitions",
            "main_answer": main_answer,
            "winner": found_winner,
            "confidence": "high" if found_winner else "low",
            "data_source": "comprehensive_season_data"
        }
    
    def _validate_houseguest(self, name: str) -> Optional[str]:
        """Validate and normalize houseguest name"""
        if not name:
            return None
        
        name = name.strip().title()
        
        # Direct match
        if name in BB27_HOUSEGUESTS_SET:
            return name
        
        # Nickname match
        if name.lower() in NICKNAME_MAP:
            return NICKNAME_MAP[name.lower()]
        
        # Partial match
        for houseguest in BB27_HOUSEGUESTS:
            if houseguest.lower().startswith(name.lower()) and len(name) >= 3:
                return houseguest
        
        return None
    
    async def _general_analysis(self, question: str) -> dict:
        # Use ALL available data for general questions
        all_updates = self.db.get_recent_updates(24 * 30)
        active_alliances = self.alliance_tracker.get_active_alliances()
        
        return {
            "response_type": "general",
            "main_answer": f"Based on complete season data: {len(all_updates)} total updates and {len(active_alliances)} tracked alliances.",
            "confidence": "medium",
            "data_source": "comprehensive_season_data"
        }
    
    def _parse_llm_chat_response(self, response_text: str, response_type: str) -> dict:
        """Parse LLM response for chat feature"""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start:json_end]
                data = json.loads(json_text)
                data["response_type"] = response_type
                data["data_source"] = "llm_comprehensive_analysis"
                return data
        except Exception as e:
            logger.warning(f"LLM JSON parsing failed: {e}")
        
        return {
            "response_type": response_type,
            "main_answer": response_text[:500] + "..." if len(response_text) > 500 else response_text,
            "confidence": "medium",
            "data_source": "llm_comprehensive_analysis"
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
            if analysis_result.get("strongest_alliances"):
                alliances = analysis_result["strongest_alliances"][:3]
                embed.add_field(
                    name="💪 Strong Alliances",
                    value=" • ".join(alliances),
                    inline=False
                )
        
        # Add confidence and data source
        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        source_emoji = {"llm_comprehensive_analysis": "🤖", "comprehensive_season_data": "📊", "comprehensive_alliance_tracker": "🤝"}
        
        embed.add_field(
            name="📈 Analysis Quality",
            value=f"{confidence_emoji.get(confidence, '🟡')} {confidence.title()} confidence\n"
                  f"{source_emoji.get(data_source, '📊')} Using complete season data",
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
        self._last_recap_date = None
        
        self.rss_url = "https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php"
        
        # Use PostgreSQL if DATABASE_URL exists, otherwise SQLite
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            logger.info("Using PostgreSQL database with connection pooling")
            self.db = PostgreSQLDatabase(
                database_url=database_url,
                min_connections=2,    # Minimum connections in pool
                max_connections=20    # Maximum connections in pool
            )
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
        self.content_monitor = UnifiedContentMonitor(self)
        
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
                        pub_date = update_data['pub_date']
                        # FIX: Handle both string and datetime objects
                        if isinstance(pub_date, str):
                            pub_date = datetime.fromisoformat(pub_date)
                        elif not isinstance(pub_date, datetime):
                            # If it's neither string nor datetime, skip this update
                            logger.warning(f"Invalid pub_date type: {type(pub_date)}")
                            continue
                        
                        update = BBUpdate(
                            title=update_data['title'],
                            description=update_data['description'],
                            link=update_data['link'], 
                            pub_date=pub_date,
                            content_hash=update_data['content_hash'],
                            author=update_data['author']
                        )
                        self.update_batcher.highlights_queue.append(update)
                    
                    # Restore last batch time
                    if result['last_summary_time']:
                        last_summary = result['last_summary_time']
                        if isinstance(last_summary, str):
                            self.update_batcher.last_batch_time = datetime.fromisoformat(last_summary)
                        else:
                            self.update_batcher.last_batch_time = last_summary
                    
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
                        pub_date = update_data['pub_date']
                        # FIX: Handle both string and datetime objects
                        if isinstance(pub_date, str):
                            pub_date = datetime.fromisoformat(pub_date)
                        elif not isinstance(pub_date, datetime):
                            # If it's neither string nor datetime, skip this update
                            logger.warning(f"Invalid pub_date type: {type(pub_date)}")
                            continue
                        
                        update = BBUpdate(
                            title=update_data['title'],
                            description=update_data['description'],
                            link=update_data['link'], 
                            pub_date=pub_date,
                            content_hash=update_data['content_hash'],
                            author=update_data['author']
                        )
                        self.update_batcher.hourly_queue.append(update)
                    
                    # Restore last hourly summary time
                    if result['last_summary_time']:
                        last_summary = result['last_summary_time']
                        if isinstance(last_summary, str):
                            self.update_batcher.last_hourly_summary = datetime.fromisoformat(last_summary)
                        else:
                            self.update_batcher.last_hourly_summary = last_summary
                    
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
        
        # Add this test command to your setup_commands method
        @self.tree.command(name="testdailyrecap", description="Test daily recap generation (Owner only)")
        async def test_daily_recap(interaction: discord.Interaction):
            """Test daily recap generation"""
            try:
                owner_id = self.config.get('owner_id')
                if not owner_id or interaction.user.id != owner_id:
                    await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                # Get current Pacific time
                pacific_tz = pytz.timezone('US/Pacific')
                now_pacific = datetime.now(pacific_tz)
                
                # Calculate yesterday's period (for testing)
                end_time = now_pacific.replace(hour=8, minute=0, second=0, microsecond=0, tzinfo=None)
                start_time = end_time - timedelta(hours=24)
                
                # Get updates from the period
                daily_updates = self.db.get_daily_updates(start_time, end_time)
                
                # Calculate day number
                season_start = datetime(2025, 7, 8)
                recap_date = start_time.date()
                day_number = (recap_date - season_start.date()).days + 1
                
                await interaction.followup.send(
                    f"**Daily Recap Test**\n"
                    f"Day {day_number}: Found {len(daily_updates)} updates\n"
                    f"Period: {start_time.strftime('%m/%d %I:%M %p')} to {end_time.strftime('%m/%d %I:%M %p')} Pacific\n"
                    f"Generating recap...",
                    ephemeral=True
                )
                
                if daily_updates:
                    # Create and send recap
                    recap_embeds = await self.update_batcher.create_daily_recap(daily_updates, day_number)
                    await self.send_daily_recap(recap_embeds)
                    await interaction.followup.send(f"✅ Daily recap sent with {len(recap_embeds)} embeds", ephemeral=True)
                else:
                    await self.send_quiet_day_recap(day_number)
                    await interaction.followup.send("✅ Quiet day recap sent", ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error in test daily recap: {e}")
                await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)
        
        @self.tree.command(name="contentstatus", description="Show unified content monitoring status")
        async def content_status_slash(interaction: discord.Interaction):
            """Show unified content monitoring status"""
            try:
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                # Get stats from unified monitor
                monitor_stats = self.content_monitor.get_stats()
                
                embed = discord.Embed(
                    title="📡 Unified Content Monitoring Status",
                    color=0x2ecc71 if self.consecutive_errors == 0 else 0xe74c3c,
                    timestamp=datetime.now()
                )
                
                # RSS Status
                embed.add_field(
                    name="📰 RSS Feed",
                    value=f"**Source**: {self.rss_url}\n**Status**: {'✅ Active' if self.consecutive_errors == 0 else '❌ Errors'}",
                    inline=False
                )
                
                # Bluesky Status
                bluesky_status = "✅ Authenticated" if monitor_stats['authentication_status'] else "❌ Not authenticated"
                embed.add_field(
                    name="📱 Bluesky Integration",
                    value=f"**Status**: {bluesky_status}\n"
                          f"**Monitored Accounts**: {monitor_stats['monitored_accounts']}\n"
                          f"**Active Accounts**: {monitor_stats['accounts_with_activity']}\n"
                          f"**Total Updates**: {monitor_stats['total_bluesky_updates']}",
                    inline=False
                )
                
                # General Stats
                embed.add_field(name="Total Updates Processed", value=str(self.total_updates_processed), inline=True)
                embed.add_field(name="Consecutive Errors", value=str(self.consecutive_errors), inline=True)
                
                time_since_check = datetime.now() - self.last_successful_check
                embed.add_field(name="Last Check", value=f"{time_since_check.total_seconds():.0f}s ago", inline=True)
                
                # Queue Status
                highlights_queue_size = len(self.update_batcher.highlights_queue)
                hourly_queue_size = len(self.update_batcher.hourly_queue)
                embed.add_field(name="Highlights Queue", value=f"{highlights_queue_size}/25", inline=True)
                embed.add_field(name="Hourly Queue", value=str(hourly_queue_size), inline=True)
                
                # Show monitored accounts
                if monitor_stats['monitored_accounts'] > 0:
                    accounts_list = "\n".join([f"• @{acc.split('.')[0]}" for acc in self.content_monitor.monitored_accounts[:10]])
                    if len(self.content_monitor.monitored_accounts) > 10:
                        accounts_list += f"\n• ... and {len(self.content_monitor.monitored_accounts) - 10} more"
                    
                    embed.add_field(
                        name="🔍 Monitored Bluesky Accounts",
                        value=accounts_list,
                        inline=False
                    )
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error generating content status: {e}")
                await interaction.followup.send("Error generating status.", ephemeral=True)
        
        @self.tree.command(name="clearcache", description="Clear duplicate cache (Owner only)")
        async def clear_cache_slash(interaction: discord.Interaction):
            owner_id = self.config.get('owner_id')
            if not owner_id or interaction.user.id != owner_id:
                await interaction.response.send_message("Owner only", ephemeral=True)
                return
            
            await self.update_batcher.processed_hashes_cache.clear()
            await interaction.response.send_message("✅ Cache cleared - RSS updates will now flow through", ephemeral=True)
        
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
                ("/contentstatus", "Show unified content monitoring status (Admin only)"),
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

        @self.tree.command(name="checkduplicates", description="Check recent database entries (Owner only)")
        async def check_duplicates_slash(interaction: discord.Interaction):
            try:
                owner_id = self.config.get('owner_id')
                if not owner_id or interaction.user.id != owner_id:
                    await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                # Get recent database entries
                recent_updates = self.db.get_recent_updates(24)  # Last 24 hours
                
                if recent_updates:
                    info = f"**Database has {len(recent_updates)} updates from last 24h**\n\n"
                    for i, update in enumerate(recent_updates[:5], 1):
                        info += f"**{i}.** {update.title[:80]}...\n"
                        info += f"Stored: {update.pub_date}\n\n"
                    
                    await interaction.followup.send(info, ephemeral=True)
                else:
                    await interaction.followup.send("No recent updates in database", ephemeral=True)
                    
            except Exception as e:
                await interaction.followup.send(f"Error: {e}", ephemeral=True)
        
        # Add this diagnostic command to see what's actually being processed

        @self.tree.command(name="testdailybuzz", description="Test daily buzz generation (Owner only)")
        async def test_daily_buzz(interaction: discord.Interaction):
            """Test daily buzz generation"""
            try:
                owner_id = self.config.get('owner_id')
                if not owner_id or interaction.user.id != owner_id:
                    await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                # Get current Pacific time
                pacific_tz = pytz.timezone('US/Pacific')
                now_pacific = datetime.now(pacific_tz)
                
                # Calculate yesterday's period (for testing)
                end_time = now_pacific.replace(hour=6, minute=0, second=0, microsecond=0, tzinfo=None)
                start_time = end_time - timedelta(hours=24)
                
                # Get updates from the period
                daily_updates = self.db.get_daily_updates(start_time, end_time)
                
                # Calculate day number
                season_start = datetime(2025, 7, 8)
                buzz_date = start_time.date()
                day_number = (buzz_date - season_start.date()).days + 1
                
                await interaction.followup.send(
                    f"**Daily Buzz Test**\n"
                    f"Day {day_number}: Found {len(daily_updates)} updates\n"
                    f"Period: {start_time.strftime('%m/%d %I:%M %p')} to {end_time.strftime('%m/%d %I:%M %p')} Pacific\n"
                    f"Generating buzz...",
                    ephemeral=True
                )
                
                if daily_updates:
                    # Create and send buzz
                    try:
                        buzz_embeds = await self.update_batcher._create_llm_daily_buzz(daily_updates, day_number)
                    except AttributeError:
                        buzz_embeds = self.update_batcher._create_pattern_daily_buzz(daily_updates, day_number)
                    
                    await self.send_daily_recap(buzz_embeds)
                    await interaction.followup.send(f"✅ Daily buzz sent with {len(buzz_embeds)} embeds", ephemeral=True)
                else:
                    await self.send_quiet_day_recap(day_number)
                    await interaction.followup.send("✅ Quiet day recap sent", ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error in test daily buzz: {e}")
                await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)
        
        @self.tree.command(name="testrss", description="Test RSS feed processing (Owner only)")
        async def test_rss_slash(interaction: discord.Interaction):
            """Test RSS feed processing"""
            try:
                owner_id = self.config.get('owner_id')
                if not owner_id or interaction.user.id != owner_id:
                    await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                # Manually fetch and parse RSS
                import feedparser
                
                try:
                    feed = feedparser.parse(self.rss_url)
                    
                    if not feed.entries:
                        await interaction.followup.send("❌ No RSS entries found!", ephemeral=True)
                        return
                    
                    # Show first 3 entries
                    rss_info = []
                    rss_info.append(f"**RSS URL:** {self.rss_url}")
                    rss_info.append(f"**Feed Title:** {feed.feed.get('title', 'Unknown')}")
                    rss_info.append(f"**Total Entries:** {len(feed.entries)}")
                    rss_info.append(f"**Feed Updated:** {feed.feed.get('updated', 'Unknown')}")
                    rss_info.append("\n**FIRST 3 ENTRIES:**")
                    
                    for i, entry in enumerate(feed.entries[:3], 1):
                        title = entry.get('title', 'No title')[:100]
                        pub_date_raw = getattr(entry, 'published', 'No date')
                        pub_date_parsed = getattr(entry, 'published_parsed', None)
                        
                        # Try to convert pub_date
                        if pub_date_parsed:
                            pub_date_obj = datetime(*pub_date_parsed[:6])
                        else:
                            pub_date_obj = "Could not parse"
                        
                        rss_info.append(f"\n**Entry {i}:**")
                        rss_info.append(f"Title: {title}...")
                        rss_info.append(f"Raw pubDate: {pub_date_raw}")
                        rss_info.append(f"Parsed pubDate: {pub_date_obj}")
                        
                        # Test what your time extraction would return
                        if hasattr(entry, 'title'):
                            # Create a fake BBUpdate to test time extraction
                            fake_update = BBUpdate(
                                title=entry.title,
                                description=entry.get('description', ''),
                                link=entry.get('link', ''),
                                pub_date=pub_date_obj if isinstance(pub_date_obj, datetime) else datetime.now(),
                                content_hash="test",
                                author=""
                            )
                            extracted_time = self.update_batcher._extract_correct_time(fake_update)
                            rss_info.append(f"Extracted time: {extracted_time}")
                    
                    # Check if these are being filtered as duplicates
                    if len(feed.entries) > 0:
                        first_entry = feed.entries[0]
                        content_hash = self.create_content_hash(
                            first_entry.get('title', ''), 
                            first_entry.get('description', '')
                        )
                        is_duplicate = self.db.is_duplicate(content_hash)
                        is_in_cache = await self.update_batcher.processed_hashes_cache.contains(content_hash)
                        
                        rss_info.append(f"\n**DUPLICATE CHECK (First Entry):**")
                        rss_info.append(f"Content Hash: {content_hash[:12]}...")
                        rss_info.append(f"In Database: {is_duplicate}")
                        rss_info.append(f"In Cache: {is_in_cache}")
                    
                    # Send the info
                    full_info = "\n".join(rss_info)
                    
                    # Split if too long
                    if len(full_info) > 1900:
                        chunks = [full_info[i:i+1900] for i in range(0, len(full_info), 1900)]
                        for chunk in chunks[:3]:  # Max 3 chunks
                            await interaction.followup.send(f"```{chunk}```", ephemeral=True)
                    else:
                        await interaction.followup.send(f"```{full_info}```", ephemeral=True)
                    
                except Exception as e:
                    await interaction.followup.send(f"❌ RSS fetch error: {e}", ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error in test RSS: {e}")
                await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)
        
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

        @self.tree.command(name="cleanmembers", description="Clean invalid members from alliances (Admin only)")
        async def clean_members_slash(interaction: discord.Interaction):
            """Clean invalid alliance members"""
            try:
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                # Clean invalid members
                cleaned_count = self.alliance_tracker.clean_alliance_members()
                
                await interaction.followup.send(
                    f"✅ Cleaned {cleaned_count} invalid alliance members.\n"
                    f"Alliances should now only show valid BB27 houseguest names.",
                    ephemeral=True
                )
                
            except Exception as e:
                logger.error(f"Error cleaning alliance members: {e}")
                await interaction.followup.send("Error cleaning alliance members", ephemeral=True)
        
        @self.tree.command(name="cleanalliances", description="Clean up invalid alliance data (Owner only)")
        async def clean_alliances_slash(interaction: discord.Interaction):
            """Clean up invalid alliances"""
            try:
                owner_id = self.config.get('owner_id')
                if not owner_id or interaction.user.id != owner_id:
                    await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                # Clean up invalid alliances
                cleaned_count = self.alliance_tracker.cleanup_invalid_alliances()
                
                await interaction.followup.send(
                    f"✅ Cleaned up {cleaned_count} invalid alliances.\n"
                    f"Alliance detection has been improved to prevent future false positives.",
                    ephemeral=True
                )
                
            except Exception as e:
                logger.error(f"Error cleaning alliances: {e}")
                await interaction.followup.send("Error cleaning alliance data", ephemeral=True)
        
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
                
                self.alliance_tracker._execute_query(cursor, """
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

        @self.tree.command(name="reprocesspolls", description="Reprocess all resolved polls to fix point distribution (Owner only)")
        @discord.app_commands.describe(
            clear_leaderboard="Clear existing leaderboard before reprocessing (recommended for fixing corrupted data)"
        )
        async def reprocess_polls_slash(interaction: discord.Interaction, clear_leaderboard: bool = True):
            """Reprocess all resolved polls to fix point distribution"""
            try:
                owner_id = self.config.get('owner_id')
                if not owner_id or interaction.user.id != owner_id:
                    await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                # Optionally clear existing leaderboard data
                if clear_leaderboard:
                    cleared_count = self.prediction_manager.clear_all_leaderboard_data(interaction.guild.id)
                    await interaction.followup.send(
                        f"🧹 Cleared {cleared_count} existing leaderboard records.\nNow reprocessing all resolved polls...",
                        ephemeral=True
                    )
                
                # Reprocess all resolved polls
                stats = self.prediction_manager.reprocess_all_resolved_polls(interaction.guild.id, interaction.user.id)
                
                if stats["polls_processed"] == 0:
                    await interaction.followup.send("No resolved polls found to reprocess.", ephemeral=True)
                    return
                
                # Create detailed results embed
                embed = discord.Embed(
                    title="✅ Poll Reprocessing Complete",
                    description=f"Successfully reprocessed prediction data",
                    color=0x2ecc71,
                    timestamp=datetime.now()
                )
                
                embed.add_field(
                    name="📊 Summary",
                    value=f"**Polls Processed:** {stats['polls_processed']}\n"
                          f"**Users Updated:** {stats['users_updated']}\n"
                          f"**Total Points Awarded:** {stats['total_points_awarded']}",
                    inline=False
                )
                
                # Show details for each poll
                if stats.get("polls_details"):
                    poll_details = []
                    for poll in stats["polls_details"][:10]:  # Show first 10
                        poll_details.append(
                            f"**{poll['title']}** (ID: {poll['id']})\n"
                            f"   {poll['correct_users']} winners, {poll['points_awarded']} points awarded"
                        )
                    
                    embed.add_field(
                        name="🎯 Poll Details",
                        value="\n\n".join(poll_details),
                        inline=False
                    )
                    
                    if len(stats["polls_details"]) > 10:
                        embed.add_field(
                            name="📝 Note",
                            value=f"Showing 10 of {len(stats['polls_details'])} polls processed",
                            inline=False
                        )
                
                embed.add_field(
                    name="✅ Next Steps",
                    value="Users can now check their updated points with `/leaderboard` and `/mypredictions`",
                    inline=False
                )
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
                # Also send a public announcement
                if self.config.get('update_channel_id'):
                    channel = self.get_channel(self.config.get('update_channel_id'))
                    if channel:
                        public_embed = discord.Embed(
                            title="🔄 Prediction Points Updated",
                            description=f"All prediction polls have been reprocessed to fix point distribution issues.\n\n"
                                       f"**{stats['polls_processed']} polls** reprocessed\n"
                                       f"**{stats['total_points_awarded']} points** correctly distributed\n\n"
                                       f"Check your updated standings with `/leaderboard`!",
                            color=0x3498db
                        )
                        await channel.send(embed=public_embed)
                
            except Exception as e:
                logger.error(f"Error in reprocess polls command: {e}")
                await interaction.followup.send(f"❌ Error reprocessing polls: {e}", ephemeral=True)
        
        @self.tree.command(name="pollstatus", description="Show status of all polls (Admin only)")
        async def poll_status_slash(interaction: discord.Interaction):
            """Show comprehensive poll status"""
            try:
                if not self.is_owner_or_admin(interaction.user, interaction):
                    await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                conn = self.prediction_manager.get_connection()
                cursor = conn.cursor()
                
                # Get poll counts by status
                self.prediction_manager._execute_query(cursor, """
                    SELECT status, COUNT(*) as count
                    FROM predictions 
                    WHERE guild_id = ?
                    GROUP BY status
                """, (interaction.guild.id,))
                
                status_counts = {}
                for row in cursor.fetchall():
                    if self.prediction_manager.use_postgresql:
                        status_counts[row['status']] = row['count']
                    else:
                        status, count = row
                        status_counts[status] = count
                
                # Get resolved polls with their details
                self.prediction_manager._execute_query(cursor, """
                    SELECT prediction_id, title, correct_option, 
                           (SELECT COUNT(*) FROM user_predictions WHERE prediction_id = p.prediction_id) as total_predictions,
                           (SELECT COUNT(*) FROM user_predictions WHERE prediction_id = p.prediction_id AND option = p.correct_option) as correct_predictions
                    FROM predictions p
                    WHERE guild_id = ? AND status = ?
                    ORDER BY prediction_id DESC
                """, (interaction.guild.id, 'resolved'))
                
                resolved_details = cursor.fetchall()
                conn.close()
                
                embed = discord.Embed(
                    title="📊 Poll Status Report",
                    description=f"Complete poll overview for {interaction.guild.name}",
                    color=0x3498db,
                    timestamp=datetime.now()
                )
                
                # Status summary
                status_text = []
                total_polls = sum(status_counts.values())
                for status, count in status_counts.items():
                    status_emoji = {"active": "🟢", "closed": "🟡", "resolved": "✅"}.get(status, "⚪")
                    status_text.append(f"{status_emoji} **{status.title()}:** {count}")
                
                embed.add_field(
                    name=f"📈 Poll Summary ({total_polls} total)",
                    value="\n".join(status_text) if status_text else "No polls found",
                    inline=False
                )
                
                # Resolved polls details
                if resolved_details:
                    resolved_text = []
                    for detail in resolved_details[:8]:  # Show first 8
                        if self.prediction_manager.use_postgresql:
                            pred_id = detail['prediction_id']
                            title = detail['title']
                            correct_option = detail['correct_option']
                            total_preds = detail['total_predictions']
                            correct_preds = detail['correct_predictions']
                        else:
                            pred_id, title, correct_option, total_preds, correct_preds = detail
                        
                        accuracy = f"{correct_preds}/{total_preds}" if total_preds else "0/0"
                        resolved_text.append(f"**{title}** (ID: {pred_id})\n   Answer: {correct_option} | Winners: {accuracy}")
                    
                    embed.add_field(
                        name="✅ Recent Resolved Polls",
                        value="\n\n".join(resolved_text),
                        inline=False
                    )
                    
                    if len(resolved_details) > 8:
                        embed.add_field(
                            name="📝 Note",
                            value=f"Showing 8 of {len(resolved_details)} resolved polls",
                            inline=False
                        )
                
                embed.set_footer(text="Use /reprocesspolls to fix any point distribution issues")
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error in poll status command: {e}")
                await interaction.followup.send("Error retrieving poll status.", ephemeral=True)
        
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
                
                # FIXED: Use proper PostgreSQL syntax with %s placeholders
                if self.prediction_manager.use_postgresql:
                    cursor.execute("""
                        SELECT prediction_id, title, description, prediction_type, 
                               options, closes_at, week_number, status
                        FROM predictions 
                        WHERE guild_id = %s AND status IN ('active', 'closed')
                        ORDER BY closes_at DESC
                    """, (interaction.guild.id,))
                else:
                    # SQLite syntax for fallback
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
                    if self.prediction_manager.use_postgresql:
                        # PostgreSQL returns RealDictCursor results
                        pred_id = row['prediction_id']
                        title = row['title']
                        desc = row['description']
                        pred_type = row['prediction_type']
                        options_json = row['options']
                        closes_at = row['closes_at']
                        week_num = row['week_number']
                        status = row['status']
                    else:
                        # SQLite returns tuple
                        pred_id, title, desc, pred_type, options_json, closes_at, week_num, status = row
                    
                    # Parse options JSON safely
                    try:
                        options = json.loads(options_json) if options_json else []
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Invalid options JSON for prediction {pred_id}: {options_json}")
                        options = []
                    
                    resolvable_predictions.append({
                        'id': pred_id,
                        'title': title,
                        'description': desc,
                        'type': pred_type,
                        'options': options,
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
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_shutting_down = True
        
        # Close database pool
        if hasattr(self, 'db') and hasattr(self.db, 'close_pool'):
            self.db.close_pool()
        
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
            self.check_all_content_sources.start()
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
                await self.update_batcher.cleanup_old_hourly_queue_items()  # Add this line
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
    

    @tasks.loop(minutes=10)  # Check every 10 minutes instead of every 24 hours
    async def daily_recap_task(self):
        """Daily buzz task that runs at 6:00 AM Pacific Time"""
        if self.is_shutting_down:
            return
    
        try:
            # Get current Pacific time
            pacific_tz = pytz.timezone('US/Pacific')
            now_pacific = datetime.now(pacific_tz)
            
            # Only run during 6:00-6:09 AM Pacific (10-minute window)
            if now_pacific.hour != 6 or now_pacific.minute >= 10:
                return
            
            logger.info(f"Daily buzz task running at {now_pacific.strftime('%I:%M %p Pacific')}")
            
            # Calculate the 24-hour period for buzz (previous day 6 AM to current 6 AM)
            end_time = now_pacific.replace(hour=6, minute=0, second=0, microsecond=0, tzinfo=None)
            start_time = end_time - timedelta(hours=24)
            
            # Check if we already sent a buzz for this 24-hour period
            buzz_date = start_time.date()
            if hasattr(self, '_last_buzz_date') and self._last_buzz_date == buzz_date:
                logger.info(f"Daily buzz already sent for {buzz_date} - skipping")
                return
            
            # Get all updates from the day
            daily_updates = self.db.get_daily_updates(start_time, end_time)
            
            # Calculate day number (days since season start)
            season_start = datetime(2025, 7, 8)  # Adjust this to your actual season start
            day_number = (buzz_date - season_start.date()).days + 1
            
            logger.info(f"Daily buzz for Day {day_number}: found {len(daily_updates)} updates")
            
            if not daily_updates:
                logger.info("No updates found for daily buzz - sending quiet day message")
                await self.send_quiet_day_recap(day_number)
            else:
                # Create daily buzz - FIX: Use the correct method name
                try:
                    # Try the new method first
                    buzz_embeds = await self.update_batcher._create_llm_daily_buzz(daily_updates, day_number)
                except AttributeError:
                    # Fallback to pattern-based if LLM method doesn't exist
                    logger.info("LLM daily buzz method not available, using pattern fallback")
                    buzz_embeds = self.update_batcher._create_pattern_daily_buzz(daily_updates, day_number)
                
                # Send daily buzz
                await self.send_daily_recap(buzz_embeds)
                
                logger.info(f"✅ Daily buzz sent for Day {day_number} with {len(buzz_embeds)} embeds")
            
            # Mark this date as processed
            self._last_buzz_date = buzz_date
            
        except Exception as e:
            logger.error(f"❌ Error in daily buzz task: {e}")
            logger.error(traceback.format_exc())
            
    async def send_quiet_day_recap(self, day_number: int = None):
        """Send a recap for days with no updates"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found for quiet day recap")
                return
            
            # Calculate day number if not provided
            if day_number is None:
                season_start = datetime(2025, 7, 8)
                current_date = datetime.now().date()
                day_number = (current_date - season_start.date()).days + 1
            
            quiet_messages = [
                "Even the cameras took a nap today 📷😴",
                "The houseguests were quieter than a library 📚",
                "Not even the ants were causing drama 🐜",
                "Production probably checked if the feeds were working 📺",
                "The most exciting thing was probably someone making slop 🥣"
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
                value="All houseguests accounted for and living remarkably quiet lives.",
                inline=False
            )
            
            embed.set_footer(text=f"Daily Recap • Day {day_number} • Even quiet days make history!")
            
            await channel.send(embed=embed)
            logger.info(f"✅ Sent quiet day recap for Day {day_number}")
            
        except Exception as e:
            logger.error(f"❌ Error sending quiet day recap: {e}")

    
    
    @daily_recap_task.before_loop
    async def before_daily_recap_task(self):
        """Wait for bot to be ready and sync to 8:00 AM Pacific"""
        await self.wait_until_ready()
        
        try:
            # Calculate wait time until next 8:00 AM Pacific
            pacific_tz = pytz.timezone('US/Pacific')
            now_pacific = datetime.now(pacific_tz)
            
            logger.info(f"Bot ready at {now_pacific.strftime('%I:%M %p Pacific on %A, %B %d')}")
            
            # Get next 6:00 AM Pacific
            next_recap = now_pacific.replace(hour=6, minute=0, second=0, microsecond=0)
            
            # If it's already past 6:00 AM today, schedule for tomorrow
            if now_pacific.hour >= 6:
                next_recap += timedelta(days=1)
                logger.info("Already past 6 AM Pacific today, scheduling for tomorrow")
            
            wait_seconds = (next_recap - now_pacific).total_seconds()
            
            logger.info(f"Daily buzz task will start in {wait_seconds:.0f} seconds (at {next_recap.strftime('%A, %B %d at %I:%M %p Pacific')})")
            
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
        """Send daily buzz recap to the configured channel"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            logger.warning("Update channel not configured for daily buzz")
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found for daily buzz")
                return
            
            # Send all embeds
            for embed in embeds[:5]:  # Limit to 5 embeds max
                await channel.send(embed=embed)
            
            logger.info(f"Daily buzz sent with {len(embeds)} embeds")
            
        except Exception as e:
            logger.error(f"Error sending daily buzz: {e}")

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
            
            if not embeds:
                logger.warning("No embeds created for highlights batch")
                return
            
            # Send to Discord
            for embed in embeds:
                await channel.send(embed=embed)
            
            # Clear ONLY highlights queue after successful send
            processed_count = len(self.update_batcher.highlights_queue)
            self.update_batcher.highlights_queue.clear()
            self.update_batcher.last_batch_time = datetime.now()
            await self.update_batcher.save_queue_state()
            
            logger.info(f"✅ Sent highlights batch: {len(embeds)} embeds, cleared {processed_count} updates from highlights queue")
            
        except Exception as e:
            logger.error(f"Error sending highlights batch: {e}")

    async def send_hourly_summary(self):
        """Send hourly summary - never touches the queue"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                return
            
            # Create hourly summary (always uses database with specific timeframe)
            embeds = await self.update_batcher.create_hourly_summary()
            
            if embeds:  
                for embed in embeds:
                    await channel.send(embed=embed)
                logger.info(f"Sent hourly summary with {len(embeds)} embeds")
            else:
                # Send quiet hour embed
                quiet_embed = self._create_quiet_hour_embed()
                await channel.send(embed=quiet_embed)
                logger.info("Sent quiet hour summary")
            
            # Update timestamp
            self.update_batcher.last_hourly_summary = datetime.now()
                
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
    
    async def send_quiet_day_recap(self, day_number: int = None):
        """Send a recap for days with no updates"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found for quiet day recap")
                return
            
            # Calculate day number if not provided
            if day_number is None:
                season_start = datetime(2025, 7, 8)
                current_date = datetime.now().date()
                day_number = (current_date - season_start.date()).days + 1
            
            quiet_messages = [
                "Even the cameras took a nap today 📷😴",
                "The houseguests were quieter than a library 📚",
                "Not even the ants were causing drama 🐜",
                "Production probably checked if the feeds were working 📺",
                "The most exciting thing was probably someone making slop 🥣"
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
                value="All houseguests accounted for and living remarkably quiet lives.",
                inline=False
            )
            
            embed.set_footer(text=f"Daily Recap • Day {day_number} • Even quiet days make history!")
            
            await channel.send(embed=embed)
            logger.info(f"✅ Sent quiet day recap for Day {day_number}")
            
        except Exception as e:
            logger.error(f"❌ Error sending quiet day recap: {e}")
    
    @tasks.loop(minutes=2)
    async def check_all_content_sources(self):
        """Unified content checking from RSS and Bluesky"""
        if self.is_shutting_down:
            return

        try:
            # Get updates from ALL sources
            new_updates = await self.content_monitor.check_all_sources()
            
            # Process each new update (same as before)
            # Process each new update using the proper add_update method
            # Process each new update using the proper add_update method
            for update in new_updates:
                try:
                    # Use the UpdateBatcher's add_update method (this handles queues properly)
                    await self.update_batcher.add_update(update)
                    
                    # Count processed updates in the bot class
                    self.total_updates_processed += 1
                    
                    # Process for alliance tracking
                    alliance_events = self.alliance_tracker.analyze_update_for_alliances(update)
                    # ... rest of your processing ...
                    for event in alliance_events:
                        alliance_id = self.alliance_tracker.process_alliance_event(event)
                        if alliance_id:
                            logger.info(f"Alliance event processed: {event['type'].value}")
                    
                    # Process for historical context if available
                    if hasattr(self, 'context_tracker') and self.context_tracker:
                        try:
                            await self.context_tracker.analyze_update_for_events(update)
                        except Exception as e:
                            logger.debug(f"Context processing failed: {e}")
                
                    # Process for alliance tracking
                    alliance_events = self.alliance_tracker.analyze_update_for_alliances(update)
                    for event in alliance_events:
                        alliance_id = self.alliance_tracker.process_alliance_event(event)
                        if alliance_id:
                            logger.info(f"Alliance event processed: {event['type'].value}")
                    
                    # Process for historical context if available
                    if hasattr(self, 'context_tracker') and self.context_tracker:
                        pass
                    
                    self.total_updates_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing update: {e}")
                    self.consecutive_errors += 1
        
            # Check for highlights batch (25 updates or urgent conditions)
            if self.update_batcher.should_send_highlights():
                await self.send_highlights_batch()
        
            # Update success tracking
            self.last_successful_check = datetime.now()
            self.consecutive_errors = 0
        
            if new_updates:
                sources_breakdown = {}
                for update in new_updates:
                    source = "Bluesky" if "@" in update.author else "RSS"
                    sources_breakdown[source] = sources_breakdown.get(source, 0) + 1
                
                breakdown_str = ", ".join([f"{count} {source}" for source, count in sources_breakdown.items()])
                logger.info(f"Added {len(new_updates)} updates to queues ({breakdown_str})")
                logger.info(f"Queue status - Highlights: {len(self.update_batcher.highlights_queue)}, Hourly: {len(self.update_batcher.hourly_queue)}")
            
        except Exception as e:
            logger.error(f"Error in unified content check: {e}")
            self.consecutive_errors += 1

    # Update your on_ready method to start the new task:
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
            # Start the unified content monitoring task
            self.check_all_content_sources.start()
            self.daily_recap_task.start()
            self.auto_close_predictions_task.start()
            self.hourly_summary_task.start()
            logger.info("All monitoring tasks started (RSS + Bluesky unified)")
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")

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
