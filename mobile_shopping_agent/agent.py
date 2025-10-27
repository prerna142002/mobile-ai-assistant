import re
import os
import random
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from data_loader import DataLoader
from llm_integration import LLMIntegration, LLMResponse
import datetime as dt

@dataclass
class Intent:
    """Represents an identified user intent."""
    name: str
    confidence: float
    entities: Dict[str, Any]

class ShoppingChatAgent:
    """A conversational agent for mobile phone shopping assistance."""
    
    def __init__(self, csv_path: str = None):
        """Initialize the shopping chat agent with CSV data source.
        
        Args:
            csv_path: Optional path to the CSV file. If not provided, uses the default path.
        """
        # Initialize data loader with CSV
        self.data_loader = DataLoader(csv_path)
        self.llm = LLMIntegration(model_name="gemini-2.5-flash")
        self.context = {
            'history': [],
            'last_intent': None,
            'current_phone': None,  # Track the current phone being discussed
            'comparison_mode': False,  # Track if we're in comparison mode
            'compared_phones': []  # Track phones being compared
        }
        self.greetings = [
            "Hello! I'm your mobile shopping assistant. How can I help you today?",
            "Hi there! Looking for a new phone? I can help you find the perfect one!",
            "Welcome! I can help you find and compare mobile phones. What are you looking for?"
        ]
        self.farewells = [
            "Thank you for using our mobile shopping assistant. Goodbye!",
            "Hope you found what you were looking for. Have a great day!",
            "Thanks for stopping by! Come back if you need more help with mobile phones."
        ]
        
        # Initialize conversation history
        self._add_to_history("system", "You are a helpful mobile shopping assistant.")
    
    def _add_to_history(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.context['history'].append({
            'role': role,
            'content': content,
            'timestamp': dt.datetime.now().isoformat()
        })
        # Keep only the last 10 messages to manage context length
        self.context['history'] = self.context['history'][-10:]

    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message and generate a response using LLM."""
        # Clean the message
        message = message.strip()
        if not message:
            return self._generate_response("I didn't catch that. Could you please rephrase?")
        
        # Add user message to history
        self._add_to_history('user', message)
        
        # Check for greetings
        if self._is_greeting(message.lower()):
            response = random.choice(self.greetings)
            self._add_to_history('assistant', response)
            return self._generate_response(response)
            
        # Check for farewells
        if self._is_farewell(message.lower()):
            response = random.choice(self.farewells)
            self._add_to_history('assistant', response)
            return self._generate_response(response)
        
        try:
            # Process the message with LLM
            llm_response = self.llm.process_message(message, self.context)
            
            # Handle the LLM response
            if llm_response.intent == 'search_phones':
                result = self._handle_search_intent(llm_response.entities)
                response_text = llm_response.response or result.get('response', 'Here are some phones that match your criteria:')
                
            elif llm_response.intent == 'compare_phones':
                result = self._handle_compare_intent({'mobiles': llm_response.entities.get('phones', [])})
                response_text = llm_response.response or result.get('response', 'Here\'s a comparison of the phones:')
                
            elif llm_response.intent == 'get_phone_details':
                result = self._handle_details_intent(llm_response.entities)
                response_text = llm_response.response or result.get('response', 'Here are the details:')
                
            elif llm_response.intent == 'chat':
                response_text = llm_response.response
                result = {}
                
            else:  # Error or unknown intent
                response_text = llm_response.response or "I'm not sure how to respond to that. Could you rephrase?"
                result = {}
            
            # Add assistant's response to history
            self._add_to_history('assistant', response_text)
            
            # Update the result with the response text
            if isinstance(result, dict):
                result['response'] = response_text
            else:
                result = {'response': response_text}
                
            return self._generate_response(**result)
            
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error: {str(e)}"
            self._add_to_history('assistant', error_msg)
            return self._generate_response(error_msg)
    
    def _is_greeting(self, message: str) -> bool:
        """Check if the message is a greeting."""
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon']
        return any(greeting in message for greeting in greetings)
    
    def _is_farewell(self, message: str) -> bool:
        """Check if the message is a farewell."""
        farewells = ['bye', 'goodbye', 'see you', 'farewell', 'take care']
        return any(farewell in message for farewell in farewells)
    
    def _detect_intent(self, message: str) -> Intent:
        """Detect the intent and extract entities from the message."""
        # This is a simplified implementation. In a real application, you would use
        # a more sophisticated NLP model for intent classification and NER.
        
        # Check for comparison intent
        if any(word in message for word in ['compare', 'vs', 'versus', 'difference between']):
            return self._extract_comparison_entities(message)
            
        # Check for details intent
        if any(word in message for word in ['details', 'specs', 'specifications', 'tell me about']):
            return self._extract_details_entities(message)
            
        # Default to search intent
        return self._extract_search_entities(message)
    
    def _extract_search_entities(self, message: str) -> Intent:
        """Extract entities for search intent."""
        entities = {}
        
        # Extract brand
        for brand in self.data_loader.get_all_brands():
            if brand.lower() in message:
                entities['brand'] = brand
                break
                
        # Extract price range
        price_match = re.search(r'(\$?\d+)\s*(?:to|\-)\s*(\$?\d+)', message)
        if price_match:
            min_price = float(price_match.group(1).replace('$', ''))
            max_price = float(price_match.group(2).replace('$', ''))
            entities['min_price'] = min(min_price, max_price)
            entities['max_price'] = max(min_price, max_price)
        else:
            price_match = re.search(r'(?:under|below|less than|lower than)\s*\$?(\d+)', message)
            if price_match:
                entities['max_price'] = float(price_match.group(1))
            
            price_match = re.search(r'(?:over|above|more than|greater than)\s*\$?(\d+)', message)
            if price_match:
                entities['min_price'] = float(price_match.group(1))
        
        # Extract storage
        storage_match = re.search(r'(\d+)\s*GB\s*(?:storage|memory|space)', message)
        if storage_match:
            entities['min_storage'] = int(storage_match.group(1))
            
        # Extract RAM
        ram_match = re.search(r'(\d+)\s*GB\s*RAM', message)
        if ram_match:
            entities['min_ram'] = int(ram_match.group(1))
            
        # Extract OS
        if 'ios' in message or 'iphone' in message:
            entities['os_type'] = 'iOS'
        elif 'android' in message:
            entities['os_type'] = 'Android'
            
        return Intent(name='search', confidence=0.9, entities=entities)
    
    def _safe_float_convert(self, value, default=0.0):
        """Safely convert a value to float, handling different formats and edge cases."""
        if value is None:
            return default
            
        if isinstance(value, dict):
            print(f"Warning: Tried to convert a dictionary to float: {value}")
            return default
            
        if isinstance(value, (int, float)):
            return float(value)
            
        try:
            # Handle string values like "$699.99" or "64 GB"
            if isinstance(value, str):
                # Remove any non-numeric characters except decimal point and minus
                cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
                return float(cleaned) if cleaned else default
            return default
        except (ValueError, TypeError) as e:
            print(f"Error converting value '{value}' to float: {str(e)}")
            return default

    def _compare_phones(self, phone1_specs: Dict[str, Any], phone2_specs: Dict[str, Any]) -> str:
        """Compare two phones and return a formatted comparison.
        
        Args:
            phone1_specs: Dictionary containing first phone's specifications
            phone2_specs: Dictionary containing second phone's specifications
            
        Returns:
            Formatted string comparing the two phones
        """
        # Debug: Print the structure of the input data
        print("\nDebug - Phone 1 specs:", phone1_specs)
        print("Debug - Phone 2 specs:", phone2_specs)
        
        comparison = "## Phone Comparison\n\n"
        
        # Define the specs to compare with their display names and formatting
        specs_to_compare = [
            ('Brand', 'brand'),
            ('Model', 'model'),
            ('Price', 'price', '${:,.2f}'),
            ('Display', 'display_size', '{} inches'),
            ('Processor', 'processor'),
            ('RAM', 'ram', '{} GB'),
            ('Storage', 'storage', '{} GB'),
            ('Camera', 'camera'),
            ('Battery', 'battery', '{} mAh'),
            ('OS', 'os'),
            ('5G', '5g', '{}')
        ]
        
        # Add comparison table header
        phone1_name = f"{phone1_specs.get('brand', '')} {phone1_specs.get('model', '')}"
        phone2_name = f"{phone2_specs.get('brand', '')} {phone2_specs.get('model', '')}"
        comparison += f"| Feature | {phone1_name} | {phone2_name} |\n"
        comparison += "|---------|----------------|----------------|\n"
        
        # Process each spec for comparison
        for spec in specs_to_compare:
            spec_name = spec[0]
            spec_key = spec[1]
            format_str = spec[2] if len(spec) > 2 else '{}'
            
            # Get values safely with better error handling
            try:
                val1 = phone1_specs.get(spec_key, 'N/A')
                val2 = phone2_specs.get(spec_key, 'N/A')
                
                # Special handling for different data types
                if isinstance(val1, dict) or isinstance(val2, dict):
                    print(f"Warning: Found dictionary value for {spec_key}")
                    val1 = str(val1) if isinstance(val1, dict) else val1
                    val2 = str(val2) if isinstance(val2, dict) else val2
                
                # Format the values if needed
                if val1 != 'N/A' and val2 != 'N/A':
                    try:
                        if spec_key in ['price', 'display_size', 'ram', 'storage', 'battery']:
                            val1 = self._safe_float_convert(val1)
                            val2 = self._safe_float_convert(val2)
                        
                        # Only apply formatting if we have a format string and the value isn't already a string
                        if format_str != '{}' and not isinstance(val1, str):
                            val1 = format_str.format(val1)
                        if format_str != '{}' and not isinstance(val2, str):
                            val2 = format_str.format(val2)
                    except Exception as e:
                        print(f"Error formatting {spec_key}: {str(e)}")
                
                comparison += f"| {spec_name} | {val1} | {val2} |\n"
                
            except Exception as e:
                print(f"Error processing {spec_key}: {str(e)}")
                comparison += f"| {spec_name} | Error | Error |\n"
        
        # Add a summary
        comparison += "\n### Summary\n"
        
        # Compare prices safely
        try:
            price1 = self._safe_float_convert(phone1_specs.get('price'))
            price2 = self._safe_float_convert(phone2_specs.get('price'))
            
            if price1 > 0 and price2 > 0:
                price_diff = abs(price1 - price2)
                if price1 > price2:
                    comparison += f"- {phone1_specs.get('brand', 'Phone 1')} {phone1_specs.get('model', '')} is ${price_diff:,.2f} more expensive than {phone2_specs.get('brand', 'Phone 2')} {phone2_specs.get('model', '')}\n"
                elif price2 > price1:
                    comparison += f"- {phone2_specs.get('brand', 'Phone 2')} {phone2_specs.get('model', '')} is ${price_diff:,.2f} more expensive than {phone1_specs.get('brand', 'Phone 1')} {phone1_specs.get('model', '')}\n"
                else:
                    comparison += "- Both phones are similarly priced\n"
        except Exception as e:
            print(f"Error comparing prices: {str(e)}")
        
        # Compare RAM safely
        try:
            ram1 = self._safe_float_convert(
                str(phone1_specs.get('ram', '0')).split()[0] 
                if isinstance(phone1_specs.get('ram'), str) 
                else phone1_specs.get('ram', 0)
            )
            ram2 = self._safe_float_convert(
                str(phone2_specs.get('ram', '0')).split()[0] 
                if isinstance(phone2_specs.get('ram'), str) 
                else phone2_specs.get('ram', 0)
            )
            
            if ram1 > 0 and ram2 > 0:
                if ram1 > ram2:
                    comparison += f"- {phone1_specs.get('brand', 'Phone 1')} {phone1_specs.get('model', '')} has {ram1-ram2:.0f}GB more RAM\n"
                elif ram2 > ram1:
                    comparison += f"- {phone2_specs.get('brand', 'Phone 2')} {phone2_specs.get('model', '')} has {ram2-ram1:.0f}GB more RAM\n"
        except Exception as e:
            print(f"Error comparing RAM: {str(e)}")
        
        return comparison
    
    def _extract_comparison_entities(self, message: str) -> Intent:
        """Extract entities for comparison intent."""
        # This is a simplified implementation
        mobiles = []
        # Look for patterns like "Pixel 8a" or "OnePlus 12R"
        brand_models = re.findall(r'([A-Za-z]+\s*\d+[A-Za-z]*)', message)
        
        for item in brand_models:
            parts = item.strip().split()
            if len(parts) >= 2:
                # Assuming first word is brand, rest is model
                mobiles.append({'brand': parts[0], 'model': ' '.join(parts[1:])})
                if len(mobiles) >= 2:  # Limit to 2 for comparison
                    break
                    
        return Intent(
            name='compare',
            confidence=0.8,
            entities={'mobiles': mobiles[:2]}  # Compare only first 2
        )
    
    def _extract_details_entities(self, message: str) -> Intent:
        """Extract entities for details intent."""
        # This is a simplified implementation
        brand_model = re.search(r'(?:about|details|specs?\s*for)?\s*((?:\b\w+\b)(?:\s+\w+)*)', message)
        if brand_model:
            parts = brand_model.group(1).strip().split()
            if len(parts) >= 2:
                return Intent(
                    name='get_details',
                    confidence=0.85,
                    entities={'brand': parts[0], 'model': ' '.join(parts[1:])}
                )
        return Intent(name='unknown', confidence=0.0, entities={})
    
    def _handle_search_intent(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search intent and return matching mobiles."""
        mobiles = self.data_loader.filter_mobiles(
            brand=entities.get('brand'),
            min_price=entities.get('min_price'),
            max_price=entities.get('max_price'),
            min_ram=entities.get('min_ram'),
            min_storage=entities.get('min_storage'),
            os_type=entities.get('os_type')
        )
        
        if not mobiles:
            return self._generate_response(
                "I couldn't find any phones matching your criteria. "
                "Would you like to try different search terms?"
            )
        
        response = "Here are some phones that match your criteria:"
        for i, mobile in enumerate(mobiles[:5], 1):  # Show top 5 results
            response += f"\n{i}. {mobile['brand']} {mobile['model']} - ${mobile['price']}"
        
        if len(mobiles) > 5:
            response += "\n\nThere are more results available. Would you like to refine your search?"
            
        return self._generate_response(response, products=mobiles[:5])

    def _search_phones(self, filters: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for phones matching the given filters.
        
        Args:
            filters: Dictionary of filters to apply
            limit: Maximum number of results to return
            
        Returns:
            List of matching phone dictionaries with enhanced data
        """
        phones = self.data_loader.search_phones(filters, limit=limit)
        
        # Enhance phone data with image URLs
        for phone in phones:
            if '_id' in phone and 'images' in phone and phone['images']:
                phone['image_urls'] = [
                    f"/api/images/{img_id}?phone_id={phone['_id']}"
                    for img_id in phone['images']
                ]
                
                # Add a thumbnail URL as well
                phone['thumbnail_url'] = f"/api/images/{phone['images'][0]}?phone_id={phone['_id']}&thumbnail=true"
        
        return phones

    def _handle_compare_intent(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compare intent and return comparison of mobiles."""
        mobiles = entities.get('mobiles', [])
        if len(mobiles) < 2:
            return self._generate_response(
                "I need at least two phones to compare. "
                "Please specify two phones like 'Compare iPhone 13 and Samsung Galaxy S21'"
            )
            
        # Get phone details for each phone
        phone1 = mobiles[0]
        phone2 = mobiles[1]
        
        # Try to find the phones in our database
        phone1_details = self.data_loader.get_phone_details(
            brand=phone1.get('brand'),
            model=phone1.get('model')
        )
        
        phone2_details = self.data_loader.get_phone_details(
            brand=phone2.get('brand'),
            model=phone2.get('model')
        )
        
        if not phone1_details or not phone2_details:
            missing_phones = []
            if not phone1_details:
                missing_phones.append(f"{phone1.get('brand', '')} {phone1.get('model', '')}")
            if not phone2_details:
                missing_phones.append(f"{phone2.get('brand', '')} {phone2.get('model', '')}")
                
            return self._generate_response(
                f"I couldn't find details for: {', '.join(missing_phones)}. "
                "Please check the names and try again."
            )
        
        # Generate the comparison
        comparison = self._compare_phones(phone1_details, phone2_details)
        
        # Add some context about the comparison
        response = (
            f"## Comparing {phone1_details.get('brand', '')} {phone1_details.get('model', '')} "
            f"vs {phone2_details.get('brand', '')} {phone2_details.get('model', '')}\n\n"
            f"{comparison}\n\n"
            "Would you like to know more about any specific aspect of these phones?"
        )
        
        return self._generate_response(response, products=[phone1_details, phone2_details])
        # Define the order of specs to display
        spec_order = ['brand', 'model', 'price', 'os', 'display_size', 'storage', 
                     'ram', 'camera', 'processor', 'battery', 'rating']
        
        # Format the comparison
        for spec in spec_order:
            if spec in all_specs:
                response += f"{spec.replace('_', ' ').title()}: "
                values = []
                for mobile in details:
                    value = mobile.get(spec, 'N/A')
                    if spec == 'price' and isinstance(value, (int, float)):
                        values.append(f"${value}")
                    else:
                        values.append(str(value))
                response += " vs ".join(values) + "\n"
                
        return self._generate_response(response, products=details)

    def _get_phone_details(self, phone_id: str = None, brand: str = None, model: str = None) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific phone.
        
        Args:
            phone_id: ID of the phone (takes precedence if provided)
            brand: Brand of the phone (required if phone_id not provided)
            model: Model of the phone (required if phone_id not provided)
            
        Returns:
            Dictionary with phone details if found, None otherwise
        """
        phone = self.data_loader.get_phone_details(phone_id=phone_id, brand=brand, model=model)
        
        # If we found a phone, update the context
        if phone and '_id' in phone:
            self.context['current_phone'] = phone
            
            # Add image URLs if available
            if 'images' in phone and phone['images']:
                phone['image_urls'] = [
                    f"/api/images/{img_id}?phone_id={phone['_id']}"
                    for img_id in phone['images']
                ]
                
                # Add a thumbnail URL as well
                phone['thumbnail_url'] = f"/api/images/{phone['images'][0]}?phone_id={phone['_id']}&thumbnail=true"
        
        return phone

    def _handle_details_intent(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get details intent and return mobile details."""
        # Try to get phone by ID if available
        if 'phone_id' in entities:
            phone = self._get_phone_details(phone_id=entities['phone_id'])
        else:
            # Otherwise use brand and model
            brand = entities.get('brand')
            model = entities.get('model')
            if not brand or not model:
                return self._generate_response(
                    "I need both the brand and model to get details. "
                    "For example, 'Tell me about iPhone 13'"
                )
            phone = self._get_phone_details(brand=brand, model=model)
        
        if not phone:
            brand = entities.get('brand', 'that brand')
            model = entities.get('model', 'that model')
            return self._generate_response(
                f"I couldn't find details for {brand} {model}. "
                "Please check the name and try again."
            )
        
        # Format the response
        response = f"Here are the details for {phone.get('brand')} {phone.get('model')}:\n"
        
        # Add basic details
        for field in ['brand', 'model', 'price', 'ram', 'storage', 'screen_size', 'camera', 'battery', 'os']:
            if field in phone and phone[field] is not None:
                response += f"- {field.replace('_', ' ').title()}: {phone[field]}\n"
        
        # Add any additional specs
        if 'specs' in phone and isinstance(phone['specs'], dict):
            for key, value in phone['specs'].items():
                if value is not None:
                    response += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        return self._generate_response(response, products=[phone])

    def _generate_response(
        self, 
        response: str,
        products: Optional[List[Dict[str, Any]]] = None,
        suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a structured response.
        
        Args:
            response: The text response to send to the user
            products: Optional list of product details to include
            suggestions: Optional list of suggested next actions
            context: Optional additional context to include in the response
            
        Returns:
            Dictionary containing the response and any additional data
        """
        # Ensure products is a list
        if products is None:
            products = []
        elif not isinstance(products, list):
            products = [products]
            
        # Enhance product data with image URLs
        for product in products:
            if '_id' in product and 'images' in product and product['images']:
                # Add full image URLs
                if 'image_urls' not in product:
                    product['image_urls'] = [
                        f"/api/images/{img_id}?phone_id={product['_id']}"
                        for img_id in product['images']
                    ]
                
                # Add thumbnail URL if not already present
                if 'thumbnail_url' not in product and product['images']:
                    product['thumbnail_url'] = f"/api/images/{product['images'][0]}?phone_id={product['_id']}&thumbnail=true"
        
        # Generate default suggestions if none provided
        if suggestions is None:
            suggestions = []
            
            # Add context-aware suggestions
            if 'current_phone' in self.context and self.context['current_phone']:
                current_brand = self.context['current_phone'].get('brand')
                current_model = self.context['current_phone'].get('model')
                
                suggestions.extend([
                    f"Compare with other {current_brand} phones",
                    f"Show me similar to {current_brand} {current_model}",
                    "Show me cheaper alternatives"
                ])
            else:
                suggestions = [
                    "Show me the latest phones",
                    "Show me phones under $500",
                    "Help me compare phones"
                ]
                
                # Try to generate more relevant suggestions based on the last user message
                try:
                    # Get the last user message from history
                    last_user_message = next(
                        (msg['content'] for msg in reversed(self.context['history']) 
                         if msg['role'] == 'user'),
                        ""
                    )
                    
                    if last_user_message:
                        prompt = f"""Based on the following user query about mobile phones, suggest 3 relevant follow-up questions or actions.
                        Keep them short (max 6 words each) and specific to mobile phone shopping.
                        
                        User query: {last_user_message}
                        
                        Suggestions:
                        1."""
                        
                        llm_response = self.llm.client.chat.completions.create(
                            model="gpt-3.5-turbo-instruct",
                            prompt=prompt,
                            max_tokens=100,
                            temperature=0.7,
                            n=1,
                            stop=["\n"]
                        )
                        
                        # Parse the response to extract suggestions
                        if llm_response and llm_response.choices:
                            text = llm_response.choices[0].text.strip()
                            # Split by numbered items and clean up
                            new_suggestions = [s.strip() for s in re.split(r'\d+\.', text) if s.strip()]
                            if new_suggestions:
                                # Replace default suggestions with generated ones
                                suggestions = new_suggestions[:3]
                
                except Exception as e:
                    print(f"Error generating suggestions: {e}")
                    # Fall back to default suggestions if there's an error
        
        # Limit number of suggestions
        suggestions = suggestions[:3]
        
        # Prepare the response
        response_data = {
            'response': response,
            'products': products,
            'suggestions': suggestions,
            'context': context if context is not None else {}
        }
        
        return response_data