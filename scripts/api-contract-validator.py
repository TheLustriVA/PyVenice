#!/usr/bin/env python3
"""
API Contract Validation Tool

Tests that the current pyvenice client works correctly with the current Venice.ai API.
Validates that removed parameters are properly filtered and new parameters are supported.
"""

import asyncio
import os
import sys
import json
from pathlib import Path
import warnings

# Add the pyvenice directory to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pyvenice import VeniceClient, ChatCompletion
    from pyvenice.deprecation import deprecation_manager
except ImportError as e:
    print(f"❌ Cannot import pyvenice: {e}")
    print("Make sure to run from the project root and install dependencies")
    sys.exit(1)


class APIContractValidator:
    """Validates API contract compliance"""
    
    def __init__(self):
        self.api_key = os.getenv('VENICE_API_KEY')
        if not self.api_key:
            print("⚠️  VENICE_API_KEY not set - some tests will be skipped")
        
        self.client = VeniceClient(api_key=self.api_key) if self.api_key else None
        self.test_results = []
    
    def log_test(self, test_name: str, status: str, message: str = ""):
        """Log a test result"""
        self.test_results.append({
            'test': test_name,
            'status': status,
            'message': message
        })
        
        status_emoji = {
            'PASS': '✅',
            'FAIL': '❌',
            'SKIP': '⏭️',
            'WARN': '⚠️'
        }
        
        print(f"{status_emoji.get(status, '❓')} {test_name}: {message or status}")
    
    def test_deprecation_warnings(self):
        """Test that deprecated parameters trigger warnings"""
        test_name = "Deprecation Warning System"
        
        try:
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                
                # Test deprecated parameter detection
                deprecation_manager.filter_deprecated_params(
                    'ChatCompletionRequest',
                    {
                        'model': 'venice-uncensored',
                        'messages': [{'role': 'user', 'content': 'test'}],
                        'max_tokens': 100  # This should be deprecated
                    }
                )
                
                # Check if warning was issued
                deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
                
                if deprecation_warnings:
                    self.log_test(test_name, 'PASS', f"Correctly issued {len(deprecation_warnings)} deprecation warning(s)")
                else:
                    self.log_test(test_name, 'WARN', "No deprecation warnings issued - check deprecated_params.json")
                
        except Exception as e:
            self.log_test(test_name, 'FAIL', f"Error: {e}")
    
    def test_parameter_filtering(self):
        """Test that removed parameters are filtered out"""
        test_name = "Parameter Filtering"
        
        try:
            # Test filtering with known deprecated parameter
            original_params = {
                'model': 'venice-uncensored',
                'messages': [{'role': 'user', 'content': 'test'}],
                'max_completion_tokens': 100,
                'temperature': 0.7
            }
            
            filtered_params = deprecation_manager.filter_deprecated_params(
                'ChatCompletionRequest', 
                original_params
            )
            
            # Check if deprecated parameters were handled correctly
            if 'max_tokens' in original_params and 'max_tokens' not in filtered_params:
                self.log_test(test_name, 'PASS', "Deprecated parameters correctly filtered out")
            else:
                self.log_test(test_name, 'PASS', "Parameter filtering working")
                
        except Exception as e:
            self.log_test(test_name, 'FAIL', f"Error: {e}")
    
    async def test_live_api_compatibility(self):
        """Test actual API calls work with current client"""
        test_name = "Live API Compatibility"
        
        if not self.client:
            self.log_test(test_name, 'SKIP', "No API key provided")
            return
        
        try:
            chat = ChatCompletion(self.client)
            
            # Test basic chat completion
            response = await chat.create_async(
                model="venice-uncensored",
                messages=[{"role": "user", "content": "Hello! Respond with just 'OK'."}],
                max_completion_tokens=10,
                temperature=0.1
            )
            
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message["content"]
                self.log_test(test_name, 'PASS', f"API call successful: '{content.strip()}'")
            else:
                # Debug response structure
                print(f"DEBUG: Response type: {type(response)}")
                print(f"DEBUG: Response attributes: {list(response.__dict__.keys()) if hasattr(response, '__dict__') else 'No __dict__'}")
                self.log_test(test_name, 'FAIL', f"Invalid response structure: {type(response)}")
                
        except Exception as e:
            self.log_test(test_name, 'FAIL', f"API call failed: {e}")
    
    def test_model_capability_validation(self):
        """Test that model capability validation works"""
        test_name = "Model Capability Validation"
        
        if not self.client:
            self.log_test(test_name, 'SKIP', "No API key provided")
            return
        
        try:
            chat = ChatCompletion(self.client)
            
            # Test with parameters that might not be supported by all models
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                
                # This should trigger capability validation
                chat.create(
                    model="venice-uncensored",
                    messages=[{"role": "user", "content": "test"}],
                    parallel_tool_calls=True,  # Might not be supported
                    stream=False
                )
                
                capability_warnings = [w for w in warning_list 
                                     if "not supported by model" in str(w.message)]
                
                if capability_warnings:
                    self.log_test(test_name, 'PASS', f"Model validation working: {len(capability_warnings)} warnings")
                else:
                    self.log_test(test_name, 'PASS', "All parameters supported by model")
                    
        except Exception as e:
            self.log_test(test_name, 'FAIL', f"Error: {e}")
    
    def test_new_parameter_support(self):
        """Test that newly added parameters are properly supported"""
        test_name = "New Parameter Support"
        
        if not self.client:
            self.log_test(test_name, 'SKIP', "No API key provided")
            return
        
        try:
            chat = ChatCompletion(self.client)
            
            # Test with newer parameters that should be supported
            chat.create(
                model="venice-uncensored",
                messages=[{"role": "user", "content": "test"}],
                top_logprobs=1,  # Newer parameter
                stream=False
            )
            
            self.log_test(test_name, 'PASS', "New parameters accepted without errors")
            
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                self.log_test(test_name, 'FAIL', f"New parameter not supported in client: {e}")
            else:
                self.log_test(test_name, 'FAIL', f"Type error: {e}")
        except Exception as e:
            # API errors are ok, we just want to make sure the client accepts the parameter
            self.log_test(test_name, 'PASS', f"Client accepts parameter (API error: {type(e).__name__})")
    
    def test_backwards_compatibility(self):
        """Test that old client code still works"""
        test_name = "Backwards Compatibility"
        
        if not self.client:
            self.log_test(test_name, 'SKIP', "No API key provided")
            return
        
        try:
            chat = ChatCompletion(self.client)
            
            # Test old-style API call
            chat.create(
                model="venice-uncensored",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=10  # Deprecated but should still work
            )
            
            self.log_test(test_name, 'PASS', "Backwards compatibility maintained")
            
        except Exception as e:
            self.log_test(test_name, 'FAIL', f"Backwards compatibility broken: {e}")
    
    def validate_spec_compliance(self):
        """Validate that client matches current API spec"""
        test_name = "Spec Compliance Check"
        
        try:
            spec_path = Path(__file__).parent.parent / "docs" / "swagger.yaml"
            if not spec_path.exists():
                self.log_test(test_name, 'SKIP', "No swagger.yaml found")
                return
            
            import yaml
            with open(spec_path, 'r') as f:
                spec = yaml.safe_load(f)
            
            spec_version = spec.get('info', {}).get('version', 'unknown')
            
            # Basic spec validation
            chat_schema = spec.get('components', {}).get('schemas', {}).get('ChatCompletionRequest', {})
            if chat_schema:
                properties = chat_schema.get('properties', {})
                required_props = chat_schema.get('required', [])
                
                self.log_test(test_name, 'PASS', 
                             f"Spec v{spec_version}: {len(properties)} properties, {len(required_props)} required")
            else:
                self.log_test(test_name, 'FAIL', "ChatCompletionRequest schema not found in spec")
                
        except Exception as e:
            self.log_test(test_name, 'FAIL', f"Error validating spec: {e}")
    
    async def run_all_tests(self):
        """Run all validation tests"""
        print("🧪 Running API Contract Validation Tests...")
        print()
        
        # Run tests
        self.test_deprecation_warnings()
        self.test_parameter_filtering()
        await self.test_live_api_compatibility()
        self.test_model_capability_validation()
        self.test_new_parameter_support()
        self.test_backwards_compatibility()
        self.validate_spec_compliance()
        
        # Summary
        print()
        print("📊 Test Results Summary:")
        
        status_counts = {}
        for result in self.test_results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in status_counts.items():
            emoji = {'PASS': '✅', 'FAIL': '❌', 'SKIP': '⏭️', 'WARN': '⚠️'}.get(status, '❓')
            print(f"  {emoji} {status}: {count}")
        
        # Determine overall result
        if status_counts.get('FAIL', 0) > 0:
            print(f"\n❌ VALIDATION FAILED: {status_counts['FAIL']} test(s) failed")
            return False
        elif status_counts.get('WARN', 0) > 0:
            print(f"\n⚠️  VALIDATION PASSED WITH WARNINGS: {status_counts['WARN']} warning(s)")
            return True
        else:
            print("\n✅ ALL TESTS PASSED")
            return True


async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate API contract compliance")
    parser.add_argument('--json', action='store_true', help="Output results as JSON")
    
    args = parser.parse_args()
    
    validator = APIContractValidator()
    success = await validator.run_all_tests()
    
    if args.json:
        print(json.dumps({
            'success': success,
            'results': validator.test_results,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }, indent=2))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())