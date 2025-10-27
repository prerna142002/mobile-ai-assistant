import os
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from bson.binary import Binary
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
import io
from PIL import Image
import base64

class MongoDB:
    """Handles MongoDB connection and collection management."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDB, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize MongoDB connection and collections."""
        self.client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        self.db = self.client['mobile_shop']
        self.phones = self.db['phones']
        self.images = self.db['images']
        
        # Create indexes
        self.phones.create_index([("brand", ASCENDING)])
        self.phones.create_index([("model", ASCENDING)])
        self.images.create_index([("phone_id", ASCENDING)])

class Phone:
    """Represents a mobile phone in the database."""
    
    def __init__(self, **kwargs):
        self._id = kwargs.get('_id')
        self.brand = kwargs.get('brand', '')
        self.model = kwargs.get('model', '')
        self.price = kwargs.get('price', 0.0)
        self.ram = kwargs.get('ram')  # in GB
        self.storage = kwargs.get('storage')  # in GB
        self.screen_size = kwargs.get('screen_size')  # in inches
        self.camera = kwargs.get('camera', '')
        self.battery = kwargs.get('battery')  # in mAh
        self.os = kwargs.get('os', '')
        self.image_urls = kwargs.get('image_urls', [])  # URLs to external images
        self.images = kwargs.get('images', [])  # List of image IDs from images collection
        self.description = kwargs.get('description', '')
        self.specs = kwargs.get('specs', {})  # Additional specifications
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert phone object to dictionary."""
        result = {
            'brand': self.brand,
            'model': self.model,
            'price': self.price,
            'ram': self.ram,
            'storage': self.storage,
            'screen_size': self.screen_size,
            'camera': self.camera,
            'battery': self.battery,
            'os': self.os,
            'image_urls': self.image_urls,
            'images': self.images,
            'description': self.description,
            'specs': self.specs,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        
        if self._id:
            result['_id'] = str(self._id)
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Phone':
        """Create a Phone instance from a dictionary."""
        return cls(**data)
    
    def save(self) -> str:
        """Save the phone to the database."""
        db = MongoDB()
        phone_data = self.to_dict()
        
        # Remove _id if it's a string (from JSON)
        if '_id' in phone_data and isinstance(phone_data['_id'], str):
            del phone_data['_id']
        
        if hasattr(self, '_id') and self._id:
            # Update existing phone
            phone_data['updated_at'] = datetime.utcnow()
            db.phones.update_one(
                {'_id': self._id},
                {'$set': phone_data}
            )
        else:
            # Insert new phone
            phone_data['created_at'] = datetime.utcnow()
            phone_data['updated_at'] = datetime.utcnow()
            result = db.phones.insert_one(phone_data)
            self._id = result.inserted_id
            
        return str(self._id)
    
    def delete(self) -> bool:
        """Delete the phone from the database."""
        if not hasattr(self, '_id') or not self._id:
            return False
            
        db = MongoDB()
        result = db.phones.delete_one({'_id': self._id})
        return result.deleted_count > 0
    
    @classmethod
    def find_by_id(cls, phone_id: str) -> Optional['Phone']:
        """Find a phone by its ID."""
        try:
            db = MongoDB()
            phone_data = db.phones.find_one({'_id': ObjectId(phone_id)})
            if phone_data:
                return cls(**phone_data)
            return None
        except:
            return None
    
    @classmethod
    def find(cls, query: Dict[str, Any] = None, limit: int = 100, skip: int = 0) -> List['Phone']:
        """Find phones matching the query."""
        db = MongoDB()
        if query is None:
            query = {}
            
        phones = db.phones.find(query).skip(skip).limit(limit)
        return [cls(**phone) for phone in phones]
    
    def add_image(self, image_data: bytes, content_type: str = 'image/jpeg', thumbnail_size: tuple = (200, 200)) -> str:
        """Add an image to this phone."""
        # Create thumbnail
        img = Image.open(io.BytesIO(image_data))
        img.thumbnail(thumbnail_size)
        
        # Save thumbnail to bytes
        thumb_buffer = io.BytesIO()
        img.save(thumb_buffer, format=img.format if img.format else 'JPEG')
        thumb_data = thumb_buffer.getvalue()
        
        # Store in MongoDB
        db = MongoDB()
        image_doc = {
            'phone_id': self._id,
            'original': Binary(image_data),
            'thumbnail': Binary(thumb_data),
            'content_type': content_type,
            'created_at': datetime.utcnow()
        }
        
        result = db.images.insert_one(image_doc)
        image_id = str(result.inserted_id)
        
        # Add to phone's images list if not already present
        if image_id not in self.images:
            self.images.append(image_id)
            self.save()
            
        return image_id
    
    def get_image(self, image_id: str, thumbnail: bool = False) -> Optional[Dict[str, Any]]:
        """Get an image by ID, optionally return thumbnail."""
        try:
            db = MongoDB()
            image_doc = db.images.find_one({
                '_id': ObjectId(image_id),
                'phone_id': self._id
            })
            
            if not image_doc:
                return None
                
            return {
                'data': image_doc['thumbnail' if thumbnail else 'original'],
                'content_type': image_doc['content_type']
            }
        except:
            return None

# Initialize database connection when module is imported
db = MongoDB()
