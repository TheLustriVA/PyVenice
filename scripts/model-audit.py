#!/usr/bin/env python3
"""
Pydantic Model Audit Tool

Compares actual API responses against PyVenice Pydantic models to identify mismatches.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Set
import importlib.util

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyvenice.client import VeniceClient


class ModelAuditor:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.client = VeniceClient(api_key=os.environ.get('VENICE_API_KEY'))
        self.admin_client = VeniceClient(api_key=os.environ.get('VENICE_ADMIN_KEY'))
        
        # Endpoint test cases
        self.test_cases = {
            '/models': {
                'method': 'GET',
                'client': 'regular',
                'expected_model': 'models.ModelListResponse'
            },
            '/chat/completions': {
                'method': 'POST',
                'client': 'regular',
                'payload': {
                    'model': 'venice-uncensored',
                    'messages': [{'role': 'user', 'content': 'Hi'}],
                    'max_completion_tokens': 5
                },
                'expected_model': 'chat.ChatCompletionResponse'
            },
            '/image/generate': {
                'method': 'POST',
                'client': 'regular',
                'payload': {
                    'prompt': 'test',
                    'model': 'flux-dev',
                    'width': 256,
                    'height': 256
                },
                'expected_model': 'image.ImageGenerationResponse'
            },
            '/characters': {
                'method': 'GET',
                'client': 'regular',
                'expected_model': 'characters.CharacterListResponse'
            },
            '/models/compatibility_mapping': {
                'method': 'GET',
                'client': 'regular',
                'expected_model': None  # No model exists for this endpoint
            },
            '/billing/usage': {
                'method': 'GET',
                'client': 'admin',
                'params': {'limit': 2},
                'expected_model': 'billing.UsageResponse'
            }
        }
    
    def get_actual_response(self, endpoint: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get actual API response for comparison."""
        client = self.admin_client if config.get('client') == 'admin' else self.client
        
        try:
            if config['method'] == 'GET':
                params = config.get('params', {})
                return client.get(endpoint, params=params)
            else:
                payload = config.get('payload', {})
                return client.post(endpoint, payload)
        except Exception as e:
            print(f"❌ Failed to get response for {endpoint}: {e}")
            return {}
    
    def extract_response_structure(self, data: Dict[str, Any], prefix: str = "") -> Set[str]:
        """Extract field paths from response data."""
        fields = set()
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{prefix}.{key}" if prefix else key
                fields.add(current_path)
                
                if isinstance(value, (dict, list)):
                    fields.update(self.extract_response_structure(value, current_path))
        elif isinstance(data, list) and data:
            # Analyze first item in list
            fields.update(self.extract_response_structure(data[0], f"{prefix}[0]"))
        
        return fields
    
    def load_pydantic_model(self, model_path: str) -> Any:
        """Load Pydantic model class."""
        try:
            module_name, class_name = model_path.split('.')
            module = importlib.import_module(f'pyvenice.{module_name}')
            return getattr(module, class_name)
        except Exception as e:
            print(f"❌ Failed to load model {model_path}: {e}")
            return None
    
    def get_model_fields(self, model_class: Any) -> Set[str]:
        """Extract field names from Pydantic model."""
        if not model_class:
            return set()
        
        try:
            # Get model fields
            if hasattr(model_class, 'model_fields'):
                return set(model_class.model_fields.keys())
            elif hasattr(model_class, '__fields__'):
                return set(model_class.__fields__.keys())
            else:
                return set()
        except Exception as e:
            print(f"❌ Failed to get model fields: {e}")
            return set()
    
    def compare_model_vs_response(self, endpoint: str) -> Dict[str, Any]:
        """Compare Pydantic model against actual API response."""
        config = self.test_cases[endpoint]
        
        print(f"\n🔍 Auditing {endpoint}")
        
        # Get actual response
        actual_response = self.get_actual_response(endpoint, config)
        if not actual_response:
            return {'status': 'failed', 'error': 'No response received'}
        
        # Extract response structure
        actual_fields = self.extract_response_structure(actual_response)
        
        # Load expected model
        if not config.get('expected_model'):
            return {'status': 'no_model', 'error': 'No model defined for this endpoint', 'actual_fields': sorted(actual_fields)}
        
        expected_model = self.load_pydantic_model(config['expected_model'])
        if not expected_model:
            return {'status': 'failed', 'error': 'Could not load model'}
        
        # Get model fields
        model_fields = self.get_model_fields(expected_model)
        
        # Compare
        missing_in_model = actual_fields - model_fields
        missing_in_response = model_fields - actual_fields
        
        # Test model creation
        model_creation_success = False
        model_error = None
        try:
            expected_model(**actual_response)
            model_creation_success = True
        except Exception as e:
            model_error = str(e)
        
        return {
            'status': 'success',
            'endpoint': endpoint,
            'model_creation_success': model_creation_success,
            'model_error': model_error,
            'actual_fields': sorted(actual_fields),
            'model_fields': sorted(model_fields),
            'missing_in_model': sorted(missing_in_model),
            'missing_in_response': sorted(missing_in_response),
            'field_count_actual': len(actual_fields),
            'field_count_model': len(model_fields)
        }
    
    def run_audit(self) -> Dict[str, Any]:
        """Run complete model audit."""
        print("🔍 Running Pydantic Model Audit...")
        
        results = {}
        total_endpoints = len(self.test_cases)
        successful_models = 0
        
        for endpoint in self.test_cases:
            result = self.compare_model_vs_response(endpoint)
            results[endpoint] = result
            
            if result.get('model_creation_success'):
                successful_models += 1
                print(f"✅ {endpoint}: Model matches response")
            else:
                print(f"❌ {endpoint}: Model mismatch - {result.get('model_error', 'Unknown error')}")
                if result.get('missing_in_model'):
                    print(f"   Missing in model: {', '.join(result['missing_in_model'][:5])}")
                if result.get('missing_in_response'):
                    print(f"   Missing in response: {', '.join(result['missing_in_response'][:5])}")
        
        summary = {
            'total_endpoints': total_endpoints,
            'successful_models': successful_models,
            'failed_models': total_endpoints - successful_models,
            'success_rate': (successful_models / total_endpoints) * 100
        }
        
        print("\n📊 Audit Summary:")
        print(f"   Total endpoints: {summary['total_endpoints']}")
        print(f"   Successful models: {summary['successful_models']}")
        print(f"   Failed models: {summary['failed_models']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        
        return {
            'summary': summary,
            'results': results
        }
    
    def save_audit_report(self, audit_results: Dict[str, Any]):
        """Save audit results to file."""
        report_path = self.project_root / "docs" / "model_audit_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(audit_results, f, indent=2, default=str)
        
        print(f"\n💾 Audit report saved to: {report_path}")


if __name__ == "__main__":
    auditor = ModelAuditor()
    results = auditor.run_audit()
    auditor.save_audit_report(results)