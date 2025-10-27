from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey, Table
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a base class for declarative class definitions
Base = declarative_base()

class Phone(Base):
    """Represents a mobile phone in the database."""
    __tablename__ = 'phones'
    
    id = Column(Integer, primary_key=True)
    brand = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    price = Column(Float)
    ram = Column(Integer)  # in GB
    storage = Column(Integer)  # in GB
    screen_size = Column(Float)  # in inches
    camera = Column(String(200))
    battery = Column(Integer)  # in mAh
    os = Column(String(50))
    image_url = Column(String(500))
    description = Column(Text)
    
    def to_dict(self):
        """Convert phone object to dictionary."""
        return {
            'id': self.id,
            'brand': self.brand,
            'model': self.model,
            'price': self.price,
            'ram': self.ram,
            'storage': self.storage,
            'screen_size': self.screen_size,
            'camera': self.camera,
            'battery': self.battery,
            'os': self.os,
            'image_url': self.image_url,
            'description': self.description
        }

class Database:
    """Handles database connection and session management."""
    
    def __init__(self, db_url: str = None):
        """Initialize the database connection.
        
        Args:
            db_url: Database URL (e.g., 'sqlite:///mobiles.db')
        """
        if db_url is None:
            # Default to SQLite in the current directory
            db_url = os.getenv('DATABASE_URL', 'sqlite:///mobiles.db')
        
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)
        
    def get_session(self):
        """Get a new database session."""
        return self.Session()

# Initialize the database
db = Database()

# Import this in other modules to use the database
# from models import db, Phone
