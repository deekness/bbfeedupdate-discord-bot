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
    main()
