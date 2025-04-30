from pymongo import MongoClient
from datetime import datetime
import json
import os

class DatabaseHelper:
    def __init__(self):
        # Get MongoDB connection string from environment variable
        self.mongo_uri = os.getenv('MONGO_URI', 'mongodb+srv://<username>:<password>@<cluster-url>/trading_bot?retryWrites=true&w=majority')
        self.client = None
        self.db = None
        
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client.trading_bot
            print("Successfully connected to MongoDB!")
            return True
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            return False
            
    def save_bot_data(self, data):
        """Save bot data to MongoDB"""
        try:
            # Add timestamp
            data['timestamp'] = datetime.now()
            
            # Save main bot data
            self.db.bot_data.update_one(
                {'_id': 'current_state'},
                {'$set': data},
                upsert=True
            )
            
            # Save trade history if there are new trades
            if 'recent_trades' in data:
                for trade in data['recent_trades']:
                    self.db.trade_history.update_one(
                        {
                            'timestamp': trade['timestamp'],
                            'entry': trade['entry'],
                            'side': trade['side']
                        },
                        {'$set': trade},
                        upsert=True
                    )
            
            return True
        except Exception as e:
            print(f"Error saving data to MongoDB: {e}")
            return False
            
    def get_bot_data(self):
        """Retrieve bot data from MongoDB"""
        try:
            # Get main bot data
            data = self.db.bot_data.find_one({'_id': 'current_state'})
            
            if data:
                # Remove MongoDB's _id field
                data.pop('_id', None)
                
                # Get recent trades
                recent_trades = list(self.db.trade_history
                    .find({})
                    .sort('timestamp', -1)
                    .limit(10))
                
                # Clean up trades data
                for trade in recent_trades:
                    trade.pop('_id', None)
                
                data['recent_trades'] = recent_trades
                return data
            return None
        except Exception as e:
            print(f"Error retrieving data from MongoDB: {e}")
            return None
            
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close() 