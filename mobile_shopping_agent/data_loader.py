from typing import List, Dict, Any, Optional, Union, BinaryIO
import os
import io
import csv
from pathlib import Path
from PIL import Image
import pandas as pd

class DataLoader:
    """Manages mobile phone data from CSV file."""
    
    def __init__(self, csv_path: str = None):
        """Initialize the data loader with CSV file path.
        
        Args:
            csv_path: Path to the CSV file. If not provided, uses the default path.
        """
        self.csv_path = csv_path or os.path.join(os.path.dirname(__file__), 'data', 'sample_mobiles.csv')
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(self.csv_path)
            # Convert DataFrame to list of dicts
            return df.to_dict('records')
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return []
    
    def _apply_filters(self, phone: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a phone matches all the given filters."""
        for key, value in filters.items():
            if key.startswith('min_'):
                field = key[4:]
                if field in phone and float(phone[field]) < float(value):
                    return False
            elif key.startswith('max_'):
                field = key[4:]
                if field in phone and float(phone[field]) > float(value):
                    return False
            elif key in phone and str(phone[key]).lower() != str(value).lower():
                return False
        return True
    
    def get_phone_details(self, phone_id: str = None, brand: str = None, model: str = None) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific phone.
        
        Args:
            phone_id: ID of the phone (not used in CSV implementation, kept for compatibility)
            brand: Brand of the phone (case-insensitive)
            model: Model of the phone (case-insensitive, partial match)
            
        Returns:
            Dictionary with phone details if found, None otherwise
        """
        if not brand and not model and not phone_id:
            return None
            
        for phone in self.data:
            # If phone_id is provided, match by ID (if available in CSV)
            if phone_id and str(phone.get('_id', '')) == str(phone_id):
                return phone.copy()
                
            # Otherwise match by brand and model
            phone_brand = str(phone.get('brand', '')).lower()
            phone_model = str(phone.get('model', '')).lower()
            
            brand_match = not brand or (phone_brand == brand.lower())
            model_match = not model or (model.lower() in phone_model)
            
            if brand_match and model_match:
                return phone.copy()
                
        return None
        
    def get_all_phones(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """Get all mobile phones in the CSV.
        
        Args:
            limit: Maximum number of phones to return
            skip: Number of phones to skip
            
        Returns:
            List of dictionaries, each representing a mobile phone
        """
        return self.data[skip:skip+limit]
    
    def search_phones(self, filters: Dict[str, Any], limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """Search for phones matching the given filters.
        
        Args:
            filters: Dictionary of filters to apply 
                    (e.g., {'brand': 'Apple', 'min_price': 500})
            limit: Maximum number of results to return
            skip: Number of results to skip
            
        Returns:
            List of dictionaries representing matching mobile phones
        """
        # Map filter keys to the expected parameter names
        filter_mapping = {
            'brand': 'brand',
            'min_price': 'min_price',
            'max_price': 'max_price',
            'min_ram': 'min_ram',
            'min_storage': 'min_storage',
            'os': 'os_type',
            'os_type': 'os_type'
        }
        
        # Prepare filter parameters
        filter_params = {}
        for key, value in filters.items():
            if key in filter_mapping and value is not None:
                filter_params[filter_mapping[key]] = value
        
        # Use the existing filter_mobiles method
        filtered = self.filter_mobiles(**filter_params)
        
        # Apply skip and limit
        return filtered[skip:skip+limit]
        
    def filter_mobiles(self, brand: str = None, min_price: float = None, max_price: float = None,
                      min_ram: int = None, min_storage: int = None, os_type: str = None) -> List[Dict[str, Any]]:
        """Filter mobiles based on the given criteria.
        
        Args:
            brand: Filter by brand name (case-insensitive)
            min_price: Minimum price (inclusive)
            max_price: Maximum price (inclusive)
            min_ram: Minimum RAM in GB
            min_storage: Minimum storage in GB
            os_type: Filter by OS type (e.g., 'iOS', 'Android')
            
        Returns:
            List of dictionaries containing mobile phone data
        """
        filtered = []
        
        for phone in self.data:
            # Apply filters
            if brand and str(phone.get('brand', '')).lower() != brand.lower():
                continue
                
            price = float(phone.get('price', 0)) if phone.get('price') else 0
            if min_price is not None and price < min_price:
                continue
            if max_price is not None and price > max_price:
                continue
                
            # Extract RAM value (handling formats like '8 GB' or '8')
            ram_str = str(phone.get('ram', '0')).split()[0]  # Get the first part (number) from '8 GB'
            ram = int(ram_str) if ram_str.isdigit() else 0
            if min_ram is not None and ram < min_ram:
                continue
                
            # Extract storage value (handling formats like '128 GB' or '128')
            storage_str = str(phone.get('storage', '0')).split()[0]  # Get the first part (number)
            storage = int(storage_str) if storage_str.isdigit() else 0
            if min_storage is not None and storage < min_storage:
                continue
                
            # OS type check (case-insensitive, check if os_type is in the OS string)
            if os_type and os_type.lower() not in str(phone.get('os', '')).lower():
                continue
                
            filtered.append(phone)
            
        return filtered
        
    def search_phones(self, filters: Dict[str, Any], limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """Search for phones matching the given filters.
        
        Args:
            filters: Dictionary of filters to apply 
                    (e.g., {'brand': 'Apple', 'min_price': 500})
            limit: Maximum number of results to return
            skip: Number of results to skip
            
        Returns:
            List of dictionaries representing matching mobile phones
        """
        query = {}
        
        # Apply filters
        if 'brand' in filters and filters['brand']:
            query['brand'] = {'$regex': f".*{filters['brand']}.*", '$options': 'i'}
            
        if 'min_price' in filters and filters['min_price'] is not None:
            query['price'] = {'$gte': float(filters['min_price'])}
            
        if 'max_price' in filters and filters['max_price'] is not None:
            if 'price' in query:
                query['price']['$lte'] = float(filters['max_price'])
            else:
                query['price'] = {'$lte': float(filters['max_price'])}
                
        if 'min_ram' in filters and filters['min_ram'] is not None:
            query['ram'] = {'$gte': int(filters['min_ram'])}
            
        if 'min_storage' in filters and filters['min_storage'] is not None:
            query['storage'] = {'$gte': int(filters['min_storage'])}
            
        if 'os_type' in filters and filters['os_type']:
            query['os'] = {'$regex': f".*{filters['os_type']}.*", '$options': 'i'}
        
        phones = list(self.db.phones.find(query).skip(skip).limit(limit))
        return [self._phone_to_dict(phone) for phone in phones]
    
    def get_phone_details(self, phone_id: str = None, brand: str = None, model: str = None) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific phone.
        
        Args:
            phone_id: ID of the phone (takes precedence if provided)
            brand: Brand of the phone (case-insensitive, required if phone_id not provided)
            model: Model of the phone (case-insensitive, required if phone_id not provided)
            
        Returns:
            Dictionary with phone details if found, None otherwise
        """
        if phone_id:
            try:
                phone = self.db.phones.find_one({'_id': ObjectId(phone_id)})
                return self._phone_to_dict(phone) if phone else None
            except (InvalidId, TypeError):
                return None
        
        if not brand or not model:
            return None
            
        phone = self.db.phones.find_one({
            'brand': {'$regex': f'^{brand}$', '$options': 'i'},
            'model': {'$regex': f'^{model}$', '$options': 'i'}
        })
        
        return self._phone_to_dict(phone) if phone else None
    
    def compare_phones(self, phone_ids: List[str] = None, phone_specs: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Compare multiple phones by their IDs or brand/model specifications.
        
        Args:
            phone_ids: List of phone IDs to compare
            phone_specs: List of dictionaries with 'brand' and 'model' keys
            
        Returns:
            List of phone details for the specified phones
        """
        if not phone_ids and not phone_specs:
            return []
            
        query_conditions = []
        
        # Add ID-based conditions
        if phone_ids:
            try:
                object_ids = [ObjectId(pid) for pid in phone_ids if pid]
                if object_ids:
                    query_conditions.append({'_id': {'$in': object_ids}})
            except (InvalidId, TypeError):
                pass
        
        # Add brand/model conditions
        if phone_specs:
            or_conditions = []
            for spec in phone_specs:
                if 'brand' in spec and 'model' in spec and spec['brand'] and spec['model']:
                    or_conditions.append({
                        'brand': {'$regex': f'^{spec["brand"]}$', '$options': 'i'},
                        'model': {'$regex': f'^{spec["model"]}$', '$options': 'i'}
                    })
            
            if or_conditions:
                query_conditions.append({'$or': or_conditions})
        
        if not query_conditions:
            return []
            
        # Combine conditions with AND
        query = {'$and': query_conditions} if len(query_conditions) > 1 else query_conditions[0]
        
        phones = list(self.db.phones.find(query))
        return [self._phone_to_dict(phone) for phone in phones]
    
    def add_phone(self, phone_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new phone to the database.
        
        Args:
            phone_data: Dictionary containing phone details
            
        Returns:
            Dictionary of the added phone with _id
        """
        # Remove None values and empty strings
        phone_data = {k: v for k, v in phone_data.items() if v is not None and v != ''}
        
        # Ensure required fields
        if 'brand' not in phone_data or 'model' not in phone_data:
            raise ValueError("Brand and model are required fields")
        
        # Convert numeric fields
        for field in ['price', 'ram', 'storage', 'screen_size', 'battery']:
            if field in phone_data and phone_data[field] is not None:
                try:
                    phone_data[field] = float(phone_data[field])
                    if field in ['ram', 'storage', 'battery']:
                        phone_data[field] = int(phone_data[field])
                except (ValueError, TypeError):
                    del phone_data[field]
        
        # Initialize lists if not present
        if 'image_urls' not in phone_data:
            phone_data['image_urls'] = []
        if 'images' not in phone_data:
            phone_data['images'] = []
        if 'specs' not in phone_data:
            phone_data['specs'] = {}
        
        # Add timestamps
        phone_data['created_at'] = phone_data.get('created_at')
        phone_data['updated_at'] = phone_data.get('updated_at')
        
        # Insert into database
        result = self.db.phones.insert_one(phone_data)
        
        # Return the inserted document with _id
        phone_data['_id'] = str(result.inserted_id)
        return phone_data
    
    def update_phone(self, phone_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing phone's details.
        
        Args:
            phone_id: ID of the phone to update
            update_data: Dictionary of fields to update
            
        Returns:
            Updated phone dictionary if successful, None if phone not found
        """
        try:
            # Convert numeric fields
            for field in ['price', 'ram', 'storage', 'screen_size', 'battery']:
                if field in update_data and update_data[field] is not None:
                    try:
                        update_data[field] = float(update_data[field])
                        if field in ['ram', 'storage', 'battery']:
                            update_data[field] = int(update_data[field])
                    except (ValueError, TypeError):
                        del update_data[field]
            
            # Don't update _id if present
            update_data.pop('_id', None)
            
            # Add updated_at timestamp
            update_data['updated_at'] = datetime.utcnow()
            
            # Perform the update
            result = self.db.phones.update_one(
                {'_id': ObjectId(phone_id)},
                {'$set': update_data}
            )
            
            if result.matched_count == 0:
                return None
                
            # Return the updated document
            return self.get_phone_details(phone_id=phone_id)
            
        except (InvalidId, TypeError):
            return None
    
    def delete_phone(self, phone_id: str) -> bool:
        """Delete a phone from the database.
        
        Args:
            phone_id: ID of the phone to delete
            
        Returns:
            True if deletion was successful, False if phone not found or invalid ID
        """
        try:
            # First, delete all images associated with the phone
            self.db.images.delete_many({'phone_id': ObjectId(phone_id)})
            
            # Then delete the phone
            result = self.db.phones.delete_one({'_id': ObjectId(phone_id)})
            return result.deleted_count > 0
        except (InvalidId, TypeError):
            return False
            
    # Image handling methods
    
    def add_phone_image(self, phone_id: str, image_data: bytes, 
                       content_type: str = 'image/jpeg', 
                       thumbnail_size: tuple = (200, 200)) -> Optional[str]:
        """Add an image to a phone's gallery.
        
        Args:
            phone_id: ID of the phone
            image_data: Binary image data
            content_type: MIME type of the image
            thumbnail_size: Size of the thumbnail to generate
            
        Returns:
            ID of the added image, or None if failed
        """
        try:
            # Create thumbnail
            img = Image.open(io.BytesIO(image_data))
            img.thumbnail(thumbnail_size)
            
            # Save thumbnail to bytes
            thumb_buffer = io.BytesIO()
            img_format = img.format if img.format in ['JPEG', 'PNG', 'GIF'] else 'JPEG'
            img.save(thumb_buffer, format=img_format)
            thumb_data = thumb_buffer.getvalue()
            
            # Store in MongoDB
            image_doc = {
                'phone_id': ObjectId(phone_id),
                'original': Binary(image_data),
                'thumbnail': Binary(thumb_data),
                'content_type': content_type,
                'created_at': datetime.utcnow()
            }
            
            result = self.db.images.insert_one(image_doc)
            image_id = str(result.inserted_id)
            
            # Add image reference to phone
            self.db.phones.update_one(
                {'_id': ObjectId(phone_id)},
                {'$addToSet': {'images': image_id}},
                upsert=False
            )
            
            return image_id
            
        except Exception as e:
            print(f"Error adding image: {str(e)}")
            return None
    
    def get_phone_image(self, phone_id: str, image_id: str, thumbnail: bool = False) -> Optional[Dict[str, Any]]:
        """Get an image for a phone.
        
        Args:
            phone_id: ID of the phone
            image_id: ID of the image
            thumbnail: Whether to return the thumbnail version
            
        Returns:
            Dictionary with 'data' (bytes) and 'content_type', or None if not found
        """
        try:
            image = self.db.images.find_one({
                '_id': ObjectId(image_id),
                'phone_id': ObjectId(phone_id)
            })
            
            if not image:
                return None
                
            return {
                'data': image['thumbnail' if thumbnail else 'original'],
                'content_type': image['content_type']
            }
            
        except (InvalidId, TypeError):
            return None
    
    def delete_phone_image(self, phone_id: str, image_id: str) -> bool:
        """Delete an image from a phone.
        
        Args:
            phone_id: ID of the phone
            image_id: ID of the image to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Remove from images collection
            result = self.db.images.delete_one({
                '_id': ObjectId(image_id),
                'phone_id': ObjectId(phone_id)
            })
            
            if result.deleted_count == 0:
                return False
                
            # Remove reference from phone document
            self.db.phones.update_one(
                {'_id': ObjectId(phone_id)},
                {'$pull': {'images': image_id}}
            )
            
            return True
            
        except (InvalidId, TypeError):
            return False