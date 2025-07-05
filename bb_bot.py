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
            return []
    
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
        self.alliance_tracker = AllianceTracker(self.config.get('database_path', 'bb_updates.db'))
        
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
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
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
                
                queue_size = len(self.update_batcher.update_queue)
                embed.add_field(name="Updates in Queue", value=str(queue_size), inline=True)
                
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
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
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
                ("/momentum", "Show alliance momentum trends"),
                ("/connections", "Show a houseguest's game connections"),
                ("/suspected", "Show suspected alliances based on behavior"),
                ("/alliancehistory", "Show detailed alliance history timeline")
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

        @self.tree.command(name="alliances", description="Show current Big Brother alliances")
        async def alliances_slash(interaction: discord.Interaction):
            """Show current alliance map"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                embed = self.alliance_tracker.create_alliance_map_embed()
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error showing alliances: {e}")
                await interaction.followup.send("Error generating alliance map.", ephemeral=True)

        @self.tree.command(name="loyalty", description="Show a houseguest's alliance history")
        async def loyalty_slash(interaction: discord.Interaction, houseguest: str):
            """Show loyalty information for a houseguest"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                # Capitalize the name properly
                houseguest = houseguest.strip().title()
                
                embed = self.alliance_tracker.get_houseguest_loyalty_embed(houseguest)
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error showing loyalty: {e}")
                await interaction.followup.send("Error generating loyalty information.", ephemeral=True)

        @self.tree.command(name="betrayals", description="Show recent alliance betrayals")
        async def betrayals_slash(interaction: discord.Interaction, days: int = 7):
            """Show recent betrayals"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                if days < 1 or days > 30:
                    await interaction.followup.send("Days must be between 1 and 30", ephemeral=True)
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
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error showing betrayals: {e}")
                await interaction.followup.send("Error generating betrayal list.", ephemeral=True)
        
        @self.tree.command(name="momentum", description="Show alliance momentum trends")
        async def momentum_slash(interaction: discord.Interaction):
            """Show which alliances are rising or falling"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                embed = self.alliance_tracker.create_alliance_momentum_embed()
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error showing momentum: {e}")
                await interaction.followup.send("Error generating momentum report.", ephemeral=True)

        @self.tree.command(name="connections", description="Show a houseguest's game connections")
        async def connections_slash(interaction: discord.Interaction, houseguest: str):
            """Show connection network for a houseguest"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                # Capitalize the name properly
                houseguest = houseguest.strip().title()
                
                embed = self.alliance_tracker.create_houseguest_connections_embed(houseguest)
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error showing connections: {e}")
                await interaction.followup.send("Error generating connections.", ephemeral=True)

        @self.tree.command(name="suspected", description="Show suspected alliances based on behavior")
        async def suspected_slash(interaction: discord.Interaction):
            """Show alliances that might be forming based on subtle indicators"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                conn = sqlite3.connect(self.alliance_tracker.db_path)
                cursor = conn.cursor()
                
                # Get suspected alliances
                cursor.execute("""
                    SELECT a.alliance_id, a.name, a.confidence_level, a.momentum,
                           GROUP_CONCAT(am.houseguest_name) as members,
                           COUNT(ae.event_id) as event_count
                    FROM alliances a
                    JOIN alliance_members am ON a.alliance_id = am.alliance_id
                    LEFT JOIN alliance_events ae ON a.alliance_id = ae.alliance_id
                    WHERE a.status = ? AND am.is_active = 1
                    GROUP BY a.alliance_id
                    ORDER BY a.confidence_level DESC
                """, (AllianceStatus.SUSPECTED.value,))
                
                suspected = cursor.fetchall()
                conn.close()
                
                embed = discord.Embed(
                    title="🔍 Suspected Alliances",
                    description="Potential alliances based on subtle game behavior",
                    color=0x95a5a6,
                    timestamp=datetime.now()
                )
                
                if not suspected:
                    embed.add_field(
                        name="No Suspected Alliances",
                        value="No potential alliances detected from recent behavior",
                        inline=False
                    )
                else:
                    for alliance in suspected[:8]:
                        alliance_id, name, confidence, momentum, members, events = alliance
                        members_list = members.split(',') if members else []
                        momentum_str = f"↗️ +{momentum:.1f}" if momentum > 0 else f"↘️ {momentum:.1f}" if momentum < 0 else "→ Stable"
                        
                        embed.add_field(
                            name=f"🤔 {name}",
                            value=f"**Members**: {' + '.join(members_list)}\n"
                                  f"**Confidence**: {confidence}% • **Momentum**: {momentum_str}\n"
                                  f"**Indicators**: {events} subtle interactions",
                            inline=False
                        )
                
                embed.set_footer(text="Based on game talks, time together, and trust indicators")
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error showing suspected alliances: {e}")
                await interaction.followup.send("Error generating suspected alliances.", ephemeral=True)

        @self.tree.command(name="alliancehistory", description="Show detailed alliance history timeline")
        async def alliance_history_slash(interaction: discord.Interaction, alliance_name: str = None, days: int = 7):
            """Show detailed history of an alliance or all recent alliance events"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                if days < 1 or days > 30:
                    await interaction.followup.send("Days must be between 1 and 30", ephemeral=True)
                    return
                
                conn = sqlite3.connect(self.alliance_tracker.db_path)
                cursor = conn.cursor()
                
                cutoff_date = datetime.now() - timedelta(days=days)
                
                if alliance_name:
                    # Find alliance by name
                    cursor.execute("""
                        SELECT alliance_id, name FROM alliances 
                        WHERE LOWER(name) LIKE LOWER(?)
                        LIMIT 1
                    """, (f"%{alliance_name}%",))
                    
                    result = cursor.fetchone()
                    if not result:
                        await interaction.followup.send(f"No alliance found matching '{alliance_name}'", ephemeral=True)
                        return
                    
                    alliance_id, actual_name = result
                    
                    # Get events for specific alliance
                    cursor.execute("""
                        SELECT event_type, description, timestamp, confidence_impact, pattern_type
                        FROM alliance_events
                        WHERE alliance_id = ? AND timestamp > ?
                        ORDER BY timestamp DESC
                    """, (alliance_id, cutoff_date))
                    
                    events = cursor.fetchall()
                    
                    embed = discord.Embed(
                        title=f"📜 {actual_name} - Alliance History",
                        description=f"Events from the last {days} days",
                        color=0x9b59b6,
                        timestamp=datetime.now()
                    )
                    
                else:
                    # Get all recent alliance events
                    cursor.execute("""
                        SELECT ae.event_type, ae.description, ae.timestamp, 
                               ae.confidence_impact, a.name
                        FROM alliance_events ae
                        JOIN alliances a ON ae.alliance_id = a.alliance_id
                        WHERE ae.timestamp > ?
                        ORDER BY ae.timestamp DESC
                        LIMIT 20
                    """, (cutoff_date,))
                    
                    events = cursor.fetchall()
                    
                    embed = discord.Embed(
                        title="📜 Recent Alliance Activity",
                        description=f"All alliance events from the last {days} days",
                        color=0x9b59b6,
                        timestamp=datetime.now()
                    )
                
                conn.close()
                
                if not events:
                    embed.add_field(
                        name="No Events",
                        value="No alliance events found in this time period",
                        inline=False
                    )
                else:
                    # Group events by day
                    events_by_day = defaultdict(list)
                    
                    for event in events:
                        if alliance_name:
                            event_type, desc, timestamp, impact, pattern = event
                            alliance = actual_name
                        else:
                            event_type, desc, timestamp, impact, alliance = event
                            pattern = None
                        
                        event_date = datetime.fromisoformat(timestamp).date()
                        events_by_day[event_date].append({
                            'type': event_type,
                            'desc': desc,
                            'time': datetime.fromisoformat(timestamp),
                            'impact': impact,
                            'alliance': alliance,
                            'pattern': pattern
                        })
                    
                    # Add fields for each day
                    for date, day_events in sorted(events_by_day.items(), reverse=True)[:5]:
                        day_text = []
                        
                        for event in day_events[:4]:  # Limit events per day
                            time_str = event['time'].strftime("%I:%M %p")
                            impact_str = ""
                            
                            if event['impact']:
                                if event['impact'] > 0:
                                    impact_str = f" (+{event['impact']})"
                                else:
                                    impact_str = f" ({event['impact']})"
                            
                            # Use appropriate emoji based on event type
                            emoji = {
                                'formed': '🤝',
                                'strengthening': '💪',
                                'weakening': '⚠️',
                                'betrayal': '💔',
                                'dissolved': '☠️',
                                'suspected': '🔍'
                            }.get(event['type'], '📝')
                            
                            if alliance_name:
                                day_text.append(f"{emoji} **{time_str}**: {event['desc']}{impact_str}")
                            else:
                                day_text.append(f"{emoji} **{time_str}** [{event['alliance']}]: {event['desc'][:50]}...{impact_str}")
                        
                        embed.add_field(
                            name=f"📅 {date.strftime('%B %d')}",
                            value="\n".join(day_text),
                            inline=False
                        )
                
                embed.set_footer(text=f"Impact values show confidence changes • {len(events)} total events")
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error showing alliance history: {e}")
                await interaction.followup.send("Error generating alliance history.", ephemeral=True)
        
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
            self.daily_recap_task.start()  # Start daily recap task
            logger.info("RSS feed monitoring and daily recap tasks started")
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
        """Daily recap task that runs at 8:01 AM Pacific Time"""
        if self.is_shutting_down:
            return
        
        try:
            # Get current time in Pacific timezone
            pacific_tz = pytz.timezone('US/Pacific')
            now_pacific = datetime.now(pacific_tz)
            
            # Check if it's the right time (8:01 AM Pacific)
            if now_pacific.hour != 8 or now_pacific.minute != 1:
                return
            
            logger.info("Starting daily recap generation")
            
            # Calculate the day period (previous 8:01 AM to current 8:01 AM)
            end_time = now_pacific.replace(tzinfo=None)  # Current time
            start_time = end_time - timedelta(hours=24)  # 24 hours ago
            
            # Get all updates from the day
            daily_updates = self.db.get_daily_updates(start_time, end_time)
            
            if not daily_updates:
                logger.info("No updates found for daily recap")
                return
            
            # Calculate day number (days since season start)
            # For now, we'll use a simple calculation - you might want to adjust this
            season_start = datetime(2025, 7, 1)  # Adjust this date for actual season start
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
        """Wait for bot to be ready before starting daily recap task"""
        await self.wait_until_ready()
        
        # Calculate time until next 8:01 AM Pacific
        pacific_tz = pytz.timezone('US/Pacific')
        now_pacific = datetime.now(pacific_tz)
        
        # Find next 8:01 AM Pacific
        next_recap_time = now_pacific.replace(hour=8, minute=1, second=0, microsecond=0)
        if now_pacific >= next_recap_time:
            next_recap_time += timedelta(days=1)
        
        # Wait until that time
        wait_seconds = (next_recap_time - now_pacific).total_seconds()
        logger.info(f"Daily recap task will start in {wait_seconds/3600:.1f} hours at {next_recap_time.strftime('%I:%M %p PT')}")
        
        await asyncio.sleep(wait_seconds)
    
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
            new_updates = await self.filter_duplicates(updates)  # Now async
            
            for update in new_updates:
                try:
                    categories = self.analyzer.categorize_update(update)
                    importance = self.analyzer.analyze_strategic_importance(update)
                    
                    self.db.store_update(update, importance, categories)
                    await self.update_batcher.add_update(update)  # Now async
                    
                    # Check for alliance information
                    alliance_events = self.alliance_tracker.analyze_update_for_alliances(update)
                    for event in alliance_events:
                        alliance_id = self.alliance_tracker.process_alliance_event(event)
                        if alliance_id:
                            logger.info(f"Alliance event processed: {event['type'].value if hasattr(event['type'], 'value') else event['type']}")
                    
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
    main()import discord
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

class AllianceTracker:
    """Enhanced alliance tracking with nuanced detection and momentum analysis"""
    
    # Core alliance patterns (existing)
    ALLIANCE_FORMATION_PATTERNS = [
        (r"([\w\s]+) and ([\w\s]+) make a final (\d+)", "final_deal", 90),
        (r"([\w\s]+) forms? an? alliance with ([\w\s]+)", "alliance", 85),
        (r"([\w\s]+) and ([\w\s]+) agree to work together", "agreement", 75),
        (r"([\w\s]+) and ([\w\s]+) shake on it", "handshake", 80),
        (r"([\w\s]+), ([\w\s]+),? and ([\w\s]+) form an? alliance", "group_alliance", 85),
        (r"([\w\s]+) joins? forces with ([\w\s]+)", "joining_forces", 70),
        (r"([\w\s]+) wants? to work with ([\w\s]+)", "wants_work", 50),
        (r"([\w\s]+) trusts? ([\w\s]+) completely", "trust", 60),
    ]
    
    # NEW: Subtle alliance indicators
    SUBTLE_ALLIANCE_PATTERNS = [
        (r"([\w\s]+) and ([\w\s]+) (?:have|had) (?:a )?game talk", "game_talk", 40),
        (r"([\w\s]+) tells? ([\w\s]+) everything", "info_sharing", 55),
        (r"([\w\s]+) (?:has|have) ([\w\s]+)'s back", "loyalty_pledge", 65),
        (r"([\w\s]+) and ([\w\s]+) compare notes", "comparing_notes", 45),
        (r"([\w\s]+) checks? in with ([\w\s]+)", "check_in", 35),
        (r"([\w\s]+) (?:fills?|filled) ([\w\s]+) in", "information_loop", 40),
        (r"([\w\s]+) and ([\w\s]+) (?:are|were) whispering", "whispering", 30),
        (r"([\w\s]+) promises? ([\w\s]+) safety", "safety_promise", 50),
        (r"([\w\s]+) (?:won't|will not) put ([\w\s]+) up", "nomination_promise", 55),
        (r"([\w\s]+) and ([\w\s]+) (?:are|were) inseparable", "close_bond", 45),
        (r"([\w\s]+) spends? all (?:day|time) with ([\w\s]+)", "time_together", 35),
        (r"([\w\s]+) only trusts? ([\w\s]+)", "exclusive_trust", 70),
    ]
    
    # NEW: Alliance strengthening patterns
    ALLIANCE_STRENGTHENING_PATTERNS = [
        (r"([\w\s]+) and ([\w\s]+) reaffirm (?:their )?(?:deal|alliance)", "reaffirm", 15),
        (r"([\w\s]+) (?:proves?|proved) loyalty to ([\w\s]+)", "loyalty_proof", 20),
        (r"([\w\s]+) protects? ([\w\s]+)", "protection", 15),
        (r"([\w\s]+) (?:goes?|went) to bat for ([\w\s]+)", "defense", 20),
        (r"([\w\s]+) and ([\w\s]+) solidify", "solidify", 15),
        (r"([\w\s]+) (?:keeps?|kept) ([\w\s]+) safe", "kept_safe", 10),
        (r"([\w\s]+) follows? through for ([\w\s]+)", "follow_through", 15),
    ]
    
    # NEW: Alliance weakening patterns
    ALLIANCE_WEAKENING_PATTERNS = [
        (r"([\w\s]+) questions? ([\w\s]+)'s loyalty", "questioning", -15),
        (r"([\w\s]+) (?:doesn't|does not) trust ([\w\s]+) (?:anymore|fully)", "trust_loss", -20),
        (r"([\w\s]+) (?:is|are) sketched out by ([\w\s]+)", "sketched", -15),
        (r"([\w\s]+) avoids? ([\w\s]+)", "avoidance", -10),
        (r"([\w\s]+) and ([\w\s]+) (?:have|had) tension", "tension", -15),
        (r"([\w\s]+) (?:catches?|caught) ([\w\s]+) lying", "caught_lying", -25),
        (r"([\w\s]+) considers? targeting ([\w\s]+)", "considering_target", -20),
        (r"([\w\s]+) (?:doesn't|does not) tell ([\w\s]+) (?:everything|about)", "info_withholding", -10),
        (r"([\w\s]+) talks? about ([\w\s]+) behind (?:their|his|her) back", "talking_behind", -15),
    ]
    
    BETRAYAL_PATTERNS = [
        (r"([\w\s]+) wants? to backdoor ([\w\s]+)", "backdoor", -50),
        (r"([\w\s]+) throws? ([\w\s]+) under the bus", "bus", -40),
        (r"([\w\s]+) is now targeting ([\w\s]+)", "targeting", -35),
        (r"([\w\s]+) turns? on ([\w\s]+)", "turns", -45),
        (r"([\w\s]+) betrays? ([\w\s]+)", "betrays", -50),
        (r"([\w\s]+) flips? on ([\w\s]+)", "flips", -40),
        (r"([\w\s]+) wants? ([\w\s]+) out", "wants_out", -30),
        (r"([\w\s]+) campaigns? against ([\w\s]+)", "campaigning", -35),
        (r"([\w\s]+) exposes? ([\w\s]+)'s game", "exposes", -40),
    ]
    
    ALLIANCE_NAME_PATTERNS = [
        r"(?:The|the) ([\w\s]+) alliance",
        r"alliance (?:called|named) (?:The|the)? ?([\w\s]+)",
        r"(?:The|the) ([\w\s]+) \([\w\s,]+\)",  # The Core (Chelsie, Cam, etc)
    ]
    
    def __init__(self, db_path: str = "bb_updates.db"):
        self.db_path = db_path
        self.init_alliance_tables()
        self._alliance_cache = {}
        self._member_cache = defaultdict(set)
        self._momentum_cache = {}  # NEW: Cache for momentum calculations
    
    def init_alliance_tables(self):
        """Initialize enhanced alliance tracking tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main alliances table (enhanced)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alliances (
                alliance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                formed_date TIMESTAMP,
                dissolved_date TIMESTAMP,
                status TEXT DEFAULT 'active',
                confidence_level INTEGER DEFAULT 50,
                last_activity TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                peak_confidence INTEGER DEFAULT 50,
                momentum REAL DEFAULT 0.0,
                last_momentum_calc TIMESTAMP
            )
        """)
        
        # Alliance members table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alliance_members (
                alliance_id INTEGER,
                houseguest_name TEXT,
                joined_date TIMESTAMP,
                left_date TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                contribution_score INTEGER DEFAULT 0,
                FOREIGN KEY (alliance_id) REFERENCES alliances(alliance_id),
                UNIQUE(alliance_id, houseguest_name)
            )
        """)
        
        # Enhanced alliance events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alliance_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alliance_id INTEGER,
                event_type TEXT,
                description TEXT,
                involved_houseguests TEXT,
                timestamp TIMESTAMP,
                update_hash TEXT,
                confidence_impact INTEGER DEFAULT 0,
                pattern_type TEXT,
                FOREIGN KEY (alliance_id) REFERENCES alliances(alliance_id)
            )
        """)
        
        # NEW: Alliance interactions table for subtle tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alliance_interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                houseguest1 TEXT,
                houseguest2 TEXT,
                interaction_type TEXT,
                timestamp TIMESTAMP,
                confidence_value INTEGER,
                update_hash TEXT,
                UNIQUE(houseguest1, houseguest2, update_hash)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alliance_status ON alliances(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alliance_members ON alliance_members(houseguest_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alliance_events_type ON alliance_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions ON alliance_interactions(houseguest1, houseguest2)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON alliance_events(timestamp)")
        
        conn.commit()
        conn.close()
        
        logger.info("Enhanced alliance tracking tables initialized")
    
    def analyze_update_for_alliances(self, update: BBUpdate) -> List[Dict]:
        """Enhanced analysis including subtle patterns and momentum"""
        content = f"{update.title} {update.description}".strip()
        detected_events = []
        
        # Check all pattern types
        all_patterns = [
            (self.ALLIANCE_FORMATION_PATTERNS, AllianceEventType.FORMED),
            (self.SUBTLE_ALLIANCE_PATTERNS, AllianceEventType.SUSPECTED),
            (self.ALLIANCE_STRENGTHENING_PATTERNS, 'strengthening'),
            (self.ALLIANCE_WEAKENING_PATTERNS, 'weakening'),
            (self.BETRAYAL_PATTERNS, AllianceEventType.BETRAYAL)
        ]
        
        for patterns, event_type in all_patterns:
            for pattern, pattern_name, confidence_impact in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    houseguests = [g.strip() for g in groups if g and not g.isdigit()]
                    houseguests = [hg for hg in houseguests if hg not in EXCLUDE_WORDS]
                    
                    if len(houseguests) >= 2:
                        event = {
                            'type': event_type,
                            'houseguests': houseguests,
                            'pattern_type': pattern_name,
                            'confidence_impact': confidence_impact,
                            'update': update
                        }
                        
                        # Record subtle interactions
                        if event_type in [AllianceEventType.SUSPECTED, 'strengthening', 'weakening']:
                            self._record_interaction(
                                houseguests[0], 
                                houseguests[1], 
                                pattern_name,
                                confidence_impact,
                                update
                            )
                        
                        detected_events.append(event)
        
        # Check for named alliances (existing code)
        for pattern in self.ALLIANCE_NAME_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                alliance_name = match.group(1).strip()
                if alliance_name and len(alliance_name) > 2:
                    members = self._extract_nearby_houseguests(content, match.start(), match.end())
                    detected_events.append({
                        'type': AllianceEventType.FORMED,
                        'alliance_name': alliance_name,
                        'houseguests': members,
                        'pattern_type': 'named_alliance',
                        'confidence_impact': 80,
                        'update': update
                    })
        
        return detected_events
    
    def _record_interaction(self, hg1: str, hg2: str, interaction_type: str, 
                          confidence: int, update: BBUpdate):
        """Record subtle interactions between houseguests"""
        # Ensure consistent ordering
        if hg1 > hg2:
            hg1, hg2 = hg2, hg1
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO alliance_interactions 
                (houseguest1, houseguest2, interaction_type, timestamp, confidence_value, update_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (hg1, hg2, interaction_type, update.pub_date, confidence, update.content_hash))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
        finally:
            conn.close()
    
    def process_alliance_event(self, event: Dict) -> Optional[int]:
        """Process detected alliance events with enhanced logic"""
        try:
            event_type = event['type']
            
            if event_type == AllianceEventType.FORMED:
                return self._handle_alliance_formation(event)
            elif event_type == AllianceEventType.SUSPECTED:
                return self._handle_suspected_alliance(event)
            elif event_type == AllianceEventType.BETRAYAL:
                return self._handle_betrayal(event)
            elif event_type in ['strengthening', 'weakening']:
                return self._handle_alliance_momentum(event)
                
        except Exception as e:
            logger.error(f"Error processing alliance event: {e}")
            return None
    
    def _handle_suspected_alliance(self, event: Dict) -> Optional[int]:
        """Handle subtle alliance indicators"""
        houseguests = event.get('houseguests', [])
        if len(houseguests) < 2:
            return None
        
        # Check interaction history
        interaction_score = self._calculate_interaction_score(houseguests[0], houseguests[1])
        
        # If enough subtle interactions, consider forming alliance
        if interaction_score >= 100:  # Threshold for suspected alliance
            existing_alliance = self._find_existing_alliance(houseguests)
            
            if not existing_alliance:
                # Create suspected alliance
                alliance_name = f"{houseguests[0]}/{houseguests[1]} (Suspected)"
                return self._create_alliance(
                    name=alliance_name,
                    members=houseguests,
                    confidence=30,  # Start with lower confidence
                    formed_date=event['update'].pub_date,
                    status=AllianceStatus.SUSPECTED
                )
            else:
                # Strengthen existing alliance
                self._update_alliance_confidence(
                    existing_alliance['alliance_id'], 
                    event['confidence_impact']
                )
                return existing_alliance['alliance_id']
        
        return None
    
    def _handle_alliance_momentum(self, event: Dict) -> Optional[int]:
        """Handle alliance strengthening/weakening events"""
        houseguests = event.get('houseguests', [])
        if len(houseguests) < 2:
            return None
        
        # Find alliances containing both houseguests
        shared_alliances = self._find_shared_alliances(houseguests[0], houseguests[1])
        
        for alliance in shared_alliances:
            # Update confidence based on event type
            self._update_alliance_confidence(
                alliance['alliance_id'], 
                event['confidence_impact']
            )
            
            # Record the momentum event
            self._record_alliance_event(
                alliance_id=alliance['alliance_id'],
                event_type=event['type'],
                description=f"{event['pattern_type']}: {houseguests[0]} and {houseguests[1]}",
                involved=houseguests,
                timestamp=event['update'].pub_date,
                update_hash=event['update'].content_hash,
                confidence_impact=event['confidence_impact'],
                pattern_type=event['pattern_type']
            )
            
            # Update momentum calculation
            self._update_alliance_momentum(alliance['alliance_id'])
        
        return shared_alliances[0]['alliance_id'] if shared_alliances else None
    
    def _calculate_interaction_score(self, hg1: str, hg2: str, days: int = 7) -> int:
        """Calculate cumulative interaction score between two houseguests"""
        if hg1 > hg2:
            hg1, hg2 = hg2, hg1
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT SUM(confidence_value) as total_score, COUNT(*) as interaction_count
            FROM alliance_interactions
            WHERE houseguest1 = ? AND houseguest2 = ? AND timestamp > ?
        """, (hg1, hg2, cutoff_date))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result[0] else 0
    
    def _update_alliance_momentum(self, alliance_id: int):
        """Calculate alliance momentum based on recent events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get recent events (last 7 days)
            cutoff_date = datetime.now() - timedelta(days=7)
            
            cursor.execute("""
                SELECT confidence_impact, timestamp
                FROM alliance_events
                WHERE alliance_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (alliance_id, cutoff_date))
            
            events = cursor.fetchall()
            
            if not events:
                momentum = 0.0
            else:
                # Calculate weighted momentum (recent events matter more)
                total_weight = 0
                weighted_sum = 0
                
                for impact, timestamp in events:
                    # Calculate age in hours
                    age_hours = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds() / 3600
                    # Weight decreases over time (half-life of 48 hours)
                    weight = 0.5 ** (age_hours / 48)
                    
                    weighted_sum += impact * weight
                    total_weight += weight
                
                momentum = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Update alliance momentum
            cursor.execute("""
                UPDATE alliances 
                SET momentum = ?, last_momentum_calc = ?
                WHERE alliance_id = ?
            """, (momentum, datetime.now(), alliance_id))
            
            # Check if this alliance is rising or falling dramatically
            if momentum > 10:
                logger.info(f"Alliance {alliance_id} has strong positive momentum: {momentum:.1f}")
            elif momentum < -10:
                logger.warning(f"Alliance {alliance_id} has strong negative momentum: {momentum:.1f}")
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
        finally:
            conn.close()
    
    def get_alliance_momentum_report(self) -> List[Dict]:
        """Get alliances sorted by momentum"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.alliance_id, a.name, a.confidence_level, a.momentum,
                   GROUP_CONCAT(am.houseguest_name) as members
            FROM alliances a
            JOIN alliance_members am ON a.alliance_id = am.alliance_id
            WHERE a.status IN (?, ?) AND am.is_active = 1
            GROUP BY a.alliance_id
            ORDER BY a.momentum DESC
        """, (AllianceStatus.ACTIVE.value, AllianceStatus.SUSPECTED.value))
        
        alliances = []
        for row in cursor.fetchall():
            alliances.append({
                'alliance_id': row[0],
                'name': row[1],
                'confidence': row[2],
                'momentum': row[3] or 0,
                'members': row[4].split(',') if row[4] else []
            })
        
        conn.close()
        return alliances
    
    def get_houseguest_connections(self, houseguest: str, min_score: int = 50) -> List[Dict]:
        """Get all connections for a houseguest based on interactions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get interaction scores with all other houseguests
        cursor.execute("""
            SELECT 
                CASE WHEN houseguest1 = ? THEN houseguest2 ELSE houseguest1 END as other_hg,
                SUM(confidence_value) as total_score,
                COUNT(*) as interaction_count,
                MAX(timestamp) as last_interaction
            FROM alliance_interactions
            WHERE (houseguest1 = ? OR houseguest2 = ?)
                  AND timestamp > datetime('now', '-14 days')
            GROUP BY other_hg
            HAVING total_score >= ?
            ORDER BY total_score DESC
        """, (houseguest, houseguest, houseguest, min_score))
        
        connections = []
        for row in cursor.fetchall():
            connections.append({
                'houseguest': row[0],
                'score': row[1],
                'interaction_count': row[2],
                'last_interaction': row[3],
                'strength': 'Strong' if row[1] >= 150 else 'Moderate' if row[1] >= 100 else 'Weak'
            })
        
        conn.close()
        return connections
    
    def create_alliance_momentum_embed(self) -> discord.Embed:
        """Create embed showing alliance momentum trends"""
        alliances = self.get_alliance_momentum_report()
        
        embed = discord.Embed(
            title="📈 Alliance Momentum Report",
            description="Which alliances are rising or falling?",
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        if not alliances:
            embed.add_field(
                name="No Data",
                value="No active alliances with momentum data",
                inline=False
            )
            return embed
        
        # Split into rising, stable, and falling
        rising = [a for a in alliances if a['momentum'] > 5]
        falling = [a for a in alliances if a['momentum'] < -5]
        stable = [a for a in alliances if -5 <= a['momentum'] <= 5]
        
        if rising:
            rising_text = []
            for alliance in rising[:5]:
                members = " + ".join(alliance['members'])
                momentum_indicator = "🚀" if alliance['momentum'] > 15 else "📈"
                rising_text.append(
                    f"{momentum_indicator} **{alliance['name']}** (+{alliance['momentum']:.1f})\n"
                    f"   {members} • Confidence: {alliance['confidence']}%"
                )
            
            embed.add_field(
                name="🔥 Rising Alliances",
                value="\n\n".join(rising_text),
                inline=False
            )
        
        if falling:
            falling_text = []
            for alliance in falling[:5]:
                members = " + ".join(alliance['members'])
                momentum_indicator = "💥" if alliance['momentum'] < -15 else "📉"
                falling_text.append(
                    f"{momentum_indicator} **{alliance['name']}** ({alliance['momentum']:.1f})\n"
                    f"   {members} • Confidence: {alliance['confidence']}%"
                )
            
            embed.add_field(
                name="⚠️ Falling Alliances",
                value="\n\n".join(falling_text),
                inline=False
            )
        
        if stable and len(embed.fields) < 3:
            stable_text = []
            for alliance in stable[:3]:
                members = " + ".join(alliance['members'])
                stable_text.append(f"➡️ {alliance['name']}: {members}")
            
            embed.add_field(
                name="🔄 Stable Alliances",
                value="\n".join(stable_text),
                inline=False
            )
        
        embed.set_footer(text="Momentum based on 7-day weighted activity • Positive = strengthening")
        
        return embed
    
    def create_houseguest_connections_embed(self, houseguest: str) -> discord.Embed:
        """Create embed showing a houseguest's connection network"""
        connections = self.get_houseguest_connections(houseguest, min_score=30)
        
        embed = discord.Embed(
            title=f"🕸️ {houseguest}'s Connection Network",
            description=f"Based on game talks and interactions (last 14 days)",
            color=0x3498db,
            timestamp=datetime.now()
        )
        
        if not connections:
            embed.add_field(
                name="No Connections",
                value=f"No significant game connections detected for {houseguest}",
                inline=False
            )
            return embed
        
        # Group by strength
        strong = [c for c in connections if c['strength'] == 'Strong']
        moderate = [c for c in connections if c['strength'] == 'Moderate']
        weak = [c for c in connections if c['strength'] == 'Weak']
        
        if strong:
            strong_text = []
            for conn in strong[:5]:
                days_ago = (datetime.now() - datetime.fromisoformat(conn['last_interaction'])).days
                strong_text.append(
                    f"🔗 **{conn['houseguest']}** (Score: {conn['score']})\n"
                    f"   {conn['interaction_count']} interactions • Last: {days_ago}d ago"
                )
            
            embed.add_field(
                name="💪 Strong Connections",
                value="\n\n".join(strong_text),
                inline=False
            )
        
        if moderate:
            mod_text = []
            for conn in moderate[:4]:
                mod_text.append(f"🤝 {conn['houseguest']} ({conn['score']} pts)")
            
            embed.add_field(
                name="🤔 Moderate Connections",
                value="\n".join(mod_text),
                inline=False
            )
        
        if weak and len(embed.fields) < 3:
            weak_text = []
            for conn in weak[:3]:
                weak_text.append(f"👋 {conn['houseguest']} ({conn['score']} pts)")
            
            embed.add_field(
                name="🌱 Developing Connections",
                value="\n".join(weak_text),
                inline=False
            )
        
        # Add summary stats
        total_score = sum(c['score'] for c in connections)
        embed.add_field(
            name="📊 Network Stats",
            value=f"Total Connections: {len(connections)}\nNetwork Strength: {total_score}",
            inline=False
        )
        
        embed.set_footer(text="Based on game talks, trust indicators, and time spent together")
        
        return embed
    
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
    
    def _handle_alliance_formation(self, event: Dict) -> Optional[int]:
        """Handle alliance formation event"""
        houseguests = event.get('houseguests', [])
        if len(houseguests) < 2:
            return None
        
        # Check if these houseguests already have an alliance together
        existing_alliance = self._find_existing_alliance(houseguests)
        
        if existing_alliance:
            # Update confidence and last activity
            self._update_alliance_confidence(existing_alliance['alliance_id'], event['confidence_impact'])
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
                confidence=event['confidence_impact'],
                formed_date=event['update'].pub_date
            )
    
    def _handle_betrayal(self, event: Dict) -> None:
        """Handle betrayal event"""
        houseguests = event.get('houseguests', [])
        if len(houseguests) < 2:
            return None
        
        betrayer = houseguests[0]
        betrayed = houseguests[1]
        
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
                update_hash=event['update'].content_hash,
                confidence_impact=event['confidence_impact']
            )
            
            # Consider marking alliance as broken if confidence drops
            self._update_alliance_confidence(alliance['alliance_id'], event['confidence_impact'])
    
    def _record_alliance_event(self, alliance_id: int, event_type, description: str, 
                             involved: List[str], timestamp: datetime, update_hash: str = None,
                             confidence_impact: int = 0, pattern_type: str = None):
        """Enhanced event recording with confidence impact and pattern type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        
        cursor.execute("""
            INSERT INTO alliance_events 
            (alliance_id, event_type, description, involved_houseguests, 
             timestamp, update_hash, confidence_impact, pattern_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (alliance_id, event_type_str, description, ",".join(involved), 
              timestamp, update_hash, confidence_impact, pattern_type))
        
        conn.commit()
        conn.close()
    
    def _create_alliance(self, name: str, members: List[str], confidence: int, 
                        formed_date: datetime, status: AllianceStatus = AllianceStatus.ACTIVE) -> int:
        """Create alliance with enhanced tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO alliances 
                (name, formed_date, status, confidence_level, peak_confidence, 
                 last_activity, momentum)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, formed_date, status.value, confidence, confidence, 
                  formed_date, 0.0))
            
            alliance_id = cursor.lastrowid
            
            for member in members:
                cursor.execute("""
                    INSERT OR IGNORE INTO alliance_members 
                    (alliance_id, houseguest_name, joined_date)
                    VALUES (?, ?, ?)
                """, (alliance_id, member, formed_date))
            
            cursor.execute("""
                INSERT INTO alliance_events 
                (alliance_id, event_type, description, involved_houseguests, 
                 timestamp, confidence_impact)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (alliance_id, AllianceEventType.FORMED.value, 
                  f"Alliance '{name}' formed", ",".join(members), 
                  formed_date, confidence))
            
            conn.commit()
            logger.info(f"Created new alliance: {name} with members {members}")
            
            return alliance_id
            
        except Exception as e:
            logger.error(f"Error creating alliance: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def _update_alliance_confidence(self, alliance_id: int, change: int):
        """Enhanced confidence update with peak tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT confidence_level, status, formed_date, peak_confidence
            FROM alliances WHERE alliance_id = ?
        """, (alliance_id,))
        
        current_conf, status, formed_date, peak_conf = cursor.fetchone()
        
        was_strong = current_conf >= 70 and status == AllianceStatus.ACTIVE.value
        
        # Update confidence
        new_confidence = max(0, min(100, current_conf + change))
        
        # Update peak if necessary
        new_peak = max(peak_conf or 0, new_confidence)
        
        cursor.execute("""
            UPDATE alliances 
            SET confidence_level = ?,
                peak_confidence = ?,
                last_activity = ?
            WHERE alliance_id = ?
        """, (new_confidence, new_peak, datetime.now(), alliance_id))
        
        # Update status based on confidence
        if new_confidence < 20:
            new_status = AllianceStatus.BROKEN.value
            cursor.execute("""
                UPDATE alliances SET status = ? WHERE alliance_id = ?
            """, (new_status, alliance_id))
            
            if was_strong:
                formed = datetime.fromisoformat(formed_date)
                days_active = (datetime.now() - formed).days
                
                cursor.execute("""
                    INSERT INTO alliance_events 
                    (alliance_id, event_type, description, timestamp, confidence_impact)
                    VALUES (?, ?, ?, ?, ?)
                """, (alliance_id, AllianceEventType.DISSOLVED.value,
                      f"Alliance dissolved after {days_active} days (peak confidence: {peak_conf}%)", 
                      datetime.now(), -current_conf))
                
                logger.info(f"Major alliance break: Alliance {alliance_id} peaked at {peak_conf}%")
        elif new_confidence >= 50 and status == AllianceStatus.SUSPECTED.value:
            # Upgrade suspected alliance to active
            cursor.execute("""
                UPDATE alliances SET status = ? WHERE alliance_id = ?
            """, (AllianceStatus.ACTIVE.value, alliance_id))
            
            logger.info(f"Suspected alliance {alliance_id} confirmed as active")
        
        conn.commit()
        conn.close()
    
    def get_active_alliances(self) -> List[Dict]:
        """Get all currently active alliances"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        
        conn.close()
        return alliances
    
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Find alliances that:
        # 1. Are now marked as broken/dissolved
        # 2. Were previously at 70+ confidence for at least 7 days
        # 3. Broke within the last week
        cursor.execute("""
            SELECT DISTINCT a.alliance_id, a.name, a.formed_date, 
                   MAX(ae.timestamp) as broken_date,
                   GROUP_CONCAT(DISTINCT am.houseguest_name) as members
            FROM alliances a
            JOIN alliance_members am ON a.alliance_id = am.alliance_id
            JOIN alliance_events ae ON a.alliance_id = ae.alliance_id
            WHERE a.status IN (?, ?)
              AND a.alliance_id IN (
                  SELECT alliance_id FROM alliance_events
                  WHERE event_type = ? AND timestamp > ?
              )
              AND a.alliance_id IN (
                  SELECT alliance_id FROM alliances
                  WHERE confidence_level >= 70
                  AND julianday(last_activity) - julianday(formed_date) >= 7
              )
            GROUP BY a.alliance_id
            ORDER BY broken_date DESC
        """, (AllianceStatus.BROKEN.value, AllianceStatus.DISSOLVED.value,
              AllianceEventType.BETRAYAL.value, cutoff_date))
        
        broken_alliances = []
        for row in cursor.fetchall():
            alliance_id, name, formed_date, broken_date, members = row
            
            # Calculate how long it was strong
            formed = datetime.fromisoformat(formed_date)
            broken = datetime.fromisoformat(broken_date)
            days_strong = (broken - formed).days
            
            broken_alliances.append({
                'alliance_id': alliance_id,
                'name': name,
                'members': members.split(',') if members else [],
                'formed_date': formed_date,
                'broken_date': broken_date,
                'days_strong': days_strong
            })
        
        conn.close()
        return broken_alliances
    
    def get_houseguest_loyalty_embed(self, houseguest: str) -> discord.Embed:
        """Create an embed showing a houseguest's alliance history"""
        # Get all alliances for this houseguest
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        
        # Get betrayal count
        cursor.execute("""
            SELECT COUNT(*) FROM alliance_events
            WHERE event_type = ? AND involved_houseguests LIKE ?
        """, (AllianceEventType.BETRAYAL.value, f"%{houseguest}%"))
        
        betrayal_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Create embed
        embed = discord.Embed(
            title=f"🎭 {houseguest}'s Alliance History",
            color=0xe74c3c if betrayal_count > 2 else 0x2ecc71,
            timestamp=datetime.now()
        )
        
        if not alliances:
            embed.description = f"{houseguest} has not been detected in any alliances"
            return embed
        
        # Calculate loyalty score
        active_alliances = sum(1 for a in alliances if a[1] == AllianceStatus.ACTIVE.value)
        broken_alliances = sum(1 for a in alliances if a[1] in [AllianceStatus.BROKEN.value, AllianceStatus.DISSOLVED.value])
        
        loyalty_score = max(0, 100 - (betrayal_count * 20) - (broken_alliances * 10))
        loyalty_emoji = "🏆" if loyalty_score >= 80 else "⚠️" if loyalty_score >= 50 else "🚨"
        
        embed.description = f"**Loyalty Score**: {loyalty_emoji} {loyalty_score}/100\n"
        embed.description += f"**Betrayals**: {betrayal_count} | **Active Alliances**: {active_alliances}"
        
        # Add alliance history
        alliance_text = []
        for alliance in alliances[:6]:  # Limit display
            name, status, joined, left, confidence, members = alliance
            status_emoji = "✅" if status == AllianceStatus.ACTIVE.value else "❌"
            
            # Parse members
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
    
    def get_recent_betrayals(self, days: int = 7) -> List[Dict]:
        """Get recent betrayal events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT description, timestamp, involved_houseguests
            FROM alliance_events
            WHERE event_type = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (AllianceEventType.BETRAYAL.value, cutoff_date))
        
        betrayals = []
        for row in cursor.fetchall():
            betrayals.append({
                'description': row[0],
                'timestamp': row[1],
                'involved': row[2].split(',') if row[2] else []
            })
        
        conn.close()
        return betrayals
    
    def _find_existing_alliance(self, houseguests: List[str]) -> Optional[Dict]:
        """Find if these houseguests already have an alliance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all active alliances for first houseguest
        cursor.execute("""
            SELECT DISTINCT a.alliance_id, a.name, a.confidence_level
            FROM alliance_members am
            JOIN alliances a ON am.alliance_id = a.alliance_id
            WHERE am.houseguest_name = ? AND a.status = ? AND am.is_active = 1
        """, (houseguests[0], AllianceStatus.ACTIVE.value))
        
        for row in cursor.fetchall():
            alliance_id = row[0]
            
            # Check if all houseguests are in this alliance
            all_in = True
            for hg in houseguests[1:]:
                cursor.execute("""
                    SELECT 1 FROM alliance_members
                    WHERE alliance_id = ? AND houseguest_name = ? AND is_active = 1
                """, (alliance_id, hg))
                
                if not cursor.fetchone():
                    all_in = False
                    break
            
            if all_in:
                conn.close()
                return {
                    'alliance_id': alliance_id,
                    'name': row[1],
                    'confidence': row[2]
                }
        
        conn.close()
        return None
    
    def _find_shared_alliances(self, hg1: str, hg2: str) -> List[Dict]:
        """Find alliances containing both houseguests"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
            alliances.append({
                'alliance_id': row[0],
                'name': row[1]
            })
        
        conn.close()
        return alliances
