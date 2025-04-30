from pymongo import MongoClient
from datetime import datetime
import json
import os

class DatabaseHelper:
    def __init__(self):
        # Get MongoDB connection string from environment variable
        self.mongo_uri = os.getenv('MONGO_URI')
        print(f"MongoDB URI found: {'Yes' if self.mongo_uri else 'No'}")  # Debug log
        self.client = None
        self.db = None

    def connect(self):
        """Establish connection to MongoDB"""
        try:
            print("Attempting to connect to MongoDB...")  # Debug log
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.db = self.client.trading_bot
            # Test the connection
            self.client.server_info()
            print("Successfully connected to MongoDB!")
            return True
        except Exception as e:
            print(f"Error connecting to MongoDB: {str(e)}")  # More detailed error
            print(f"Connection string used: {self.mongo_uri[:20]}...")  # Show start of URI
            return False

    def save_bot_data(self, data):
        """Save bot data to MongoDB"""
        try:
            # Convert NumPy types to Python native types
            data = json.loads(json.dumps(data))
            
            # Update main bot data
            self.db.bot_data.update_one(
                {'_id': 'current_state'},
                {'$set': data},
                upsert=True
            )
            
            # Save recent trades separately
            if 'recent_trades' in data:
                for trade in data['recent_trades']:
                    self.db.trade_history.update_one(
                        {'timestamp': trade['timestamp']},
                        {'$set': trade},
                        upsert=True
                    )
            
            # Save price history separately with timestamp index
            if 'price_history' in data:
                for price_data in data['price_history']:
                    self.db.price_history.update_one(
                        {'timestamp': price_data['timestamp']},
                        {'$set': price_data},
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
                
                # Get price history
                price_history = list(self.db.price_history
                    .find({})
                    .sort('timestamp', -1)
                    .limit(100))
                    
                # Clean up price history data
                for price_data in price_history:
                    price_data.pop('_id', None)
                
                # Sort price history by timestamp
                price_history.sort(key=lambda x: x['timestamp'])
                
                data['recent_trades'] = recent_trades
                data['price_history'] = price_history
                return data
            return None
        except Exception as e:
            print(f"Error retrieving data from MongoDB: {e}")
            return None
            
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close() 