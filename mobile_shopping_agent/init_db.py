import os
import pandas as pd
from models import db, Phone, Base
from data_loader import DataLoader

def init_database():
    """Initialize the database with required tables."""
    # Create all tables
    db.create_tables()
    print("Database tables created successfully.")

def load_sample_data(csv_path: str = 'data/sample_mobiles.csv'):
    """Load sample data from CSV into the database.
    
    Args:
        csv_path: Path to the CSV file containing sample mobile data
    """
    if not os.path.exists(csv_path):
        print(f"Sample data file not found at {csv_path}")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Clean and prepare data
    df = df.where(pd.notnull(df), None)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Add each phone to the database
    for _, row in df.iterrows():
        phone_data = {
            'brand': row.get('brand', ''),
            'model': row.get('model', ''),
            'price': float(row.get('price', 0)) if pd.notnull(row.get('price')) else None,
            'ram': int(row.get('ram', 0)) if pd.notnull(row.get('ram')) else None,
            'storage': int(row.get('storage', 0)) if pd.notnull(row.get('storage')) else None,
            'screen_size': float(row.get('screen_size', 0)) if pd.notnull(row.get('screen_size')) else None,
            'camera': str(row.get('camera', '')) if pd.notnull(row.get('camera')) else None,
            'battery': int(row.get('battery', 0)) if pd.notnull(row.get('battery')) else None,
            'os': str(row.get('os', '')) if pd.notnull(row.get('os')) else None,
            'image_url': str(row.get('image_url', '')) if pd.notnull(row.get('image_url')) else None,
            'description': str(row.get('description', '')) if pd.notnull(row.get('description')) else None
        }
        
        try:
            loader.add_phone(phone_data)
            print(f"Added {phone_data['brand']} {phone_data['model']}")
        except Exception as e:
            print(f"Error adding {row.get('brand')} {row.get('model')}: {str(e)}")
    
    print("Sample data loaded successfully.")

if __name__ == "__main__":
    # Initialize the database
    init_database()
    
    # Load sample data if available
    sample_data_path = os.path.join(os.path.dirname(__file__), 'data/sample_mobiles.csv')
    if os.path.exists(sample_data_path):
        print("Loading sample data...")
        load_sample_data(sample_data_path)
    else:
        print("No sample data found. Database initialized with empty tables.")
