#!/bin/bash

echo "Big Brother Discord Bot Setup"
echo "============================"

# Create necessary directories
mkdir -p logs

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create config file if it doesn't exist
if [ ! -f "config.json" ]; then
    echo "Creating config.json template..."
    cat > config.json << EOF
{
    "bot_token": "",
    "update_channel_id": null,
    "rss_check_interval": 2,
    "max_retries": 3,
    "retry_delay": 5,
    "database_path": "bb_updates.db",
    "enable_heartbeat": true,
    "heartbeat_interval": 300,
    "max_update_age_hours": 168,
    "enable_auto_restart": true,
    "max_consecutive_errors": 10
}
EOF
    echo "Please edit config.json and add your bot token!"
fi

echo ""
echo "Setup complete! To run the bot:"
echo "1. Edit config.json and add your bot token"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python bb_bot.py"
echo ""
echo "For Docker deployment:"
echo "1. Set BOT_TOKEN environment variable"
echo "2. Run: docker-compose up -d"
