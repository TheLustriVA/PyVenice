#!/usr/bin/env python3
"""
Response Schema Monitoring System

Monitors actual API response structures and detects changes in response schemas.
Complements the existing parameter monitoring in api-monitor.py.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyvenice.client import VeniceClient


class ResponseSchemaMonitor:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"
        self.schema_cache_path = self.docs_dir / "response_schemas.json"
        self.client = VeniceClient(api_key=os.environ.get('VENICE_API_KEY'))
        self.admin_client = VeniceClient(api_key=os.environ.get('VENICE_ADMIN_KEY'))
        
        # Define monitored endpoints
        self.endpoints = {
            '/models': {'method': 'GET', 'client': 'regular'},
            '/chat/completions': {
                'method': 'POST', 
                'client': 'regular',
                'payload': {
                    'model': 'venice-uncensored',
                    'messages': [{'role': 'user', 'content': 'test'}],
                    'max_completion_tokens': 1
                }
            },
            '/image/generate': {
                'method': 'POST',
                'client': 'regular', 
                'payload': {
                    'prompt': 'test',
                    'model': 'flux-dev',
                    'width': 256,
                    'height': 256
                }
            },
            '/characters': {'method': 'GET', 'client': 'regular'},
            '/models/compatibility_mapping': {'method': 'GET', 'client': 'regular'},
            '/billing/usage': {
                'method': 'GET',
                'client': 'admin',
                'params': {'limit': 1}
            }
        }
    
    def extract_schema_structure(self, data: Any, path: str = "") -> Dict[str, str]:
        """Extract schema structure with field types."""
        structure = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                structure[current_path] = type(value).__name__
                
                if isinstance(value, (dict, list)) and value:
                    structure.update(self.extract_schema_structure(value, current_path))
        elif isinstance(data, list) and data:
            # Analyze first item in list
            structure.update(self.extract_schema_structure(data[0], f"{path}[*]"))
        
        return structure
    
    def get_response_schema(self, endpoint: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get current response schema for endpoint."""
        client = self.admin_client if config.get('client') == 'admin' else self.client
        
        try:
            if config['method'] == 'GET':
                params = config.get('params', {})
                response = client.get(endpoint, params=params)
            else:
                payload = config.get('payload', {})
                response = client.post(endpoint, payload)
            
            schema = self.extract_schema_structure(response)
            schema_hash = hashlib.md5(json.dumps(schema, sort_keys=True).encode()).hexdigest()
            
            return {
                'endpoint': endpoint,
                'timestamp': datetime.now().isoformat(),
                'schema': schema,
                'schema_hash': schema_hash,
                'field_count': len(schema),
                'top_level_fields': list(response.keys()) if isinstance(response, dict) else []
            }
            
        except Exception as e:
            return {
                'endpoint': endpoint,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'schema': {},
                'schema_hash': '',
                'field_count': 0
            }
    
    def load_cached_schemas(self) -> Dict[str, Any]:
        """Load previously cached response schemas."""
        if not self.schema_cache_path.exists():
            return {}
        
        with open(self.schema_cache_path, 'r') as f:
            return json.load(f)
    
    def save_schemas(self, schemas: Dict[str, Any]):
        """Save response schemas to cache."""
        self.docs_dir.mkdir(exist_ok=True)
        with open(self.schema_cache_path, 'w') as f:
            json.dump(schemas, f, indent=2)
    
    def compare_schemas(self, old_schema: Dict[str, str], new_schema: Dict[str, str]) -> Dict[str, Any]:
        """Compare two schemas and identify changes."""
        old_fields = set(old_schema.keys())
        new_fields = set(new_schema.keys())
        
        added_fields = new_fields - old_fields
        removed_fields = old_fields - new_fields
        common_fields = old_fields & new_fields
        
        type_changes = []
        for field in common_fields:
            if old_schema[field] != new_schema[field]:
                type_changes.append({
                    'field': field,
                    'old_type': old_schema[field],
                    'new_type': new_schema[field]
                })
        
        return {
            'added_fields': sorted(added_fields),
            'removed_fields': sorted(removed_fields),
            'type_changes': type_changes,
            'total_changes': len(added_fields) + len(removed_fields) + len(type_changes)
        }
    
    def monitor_schemas(self) -> Dict[str, Any]:
        """Monitor all endpoint schemas for changes."""
        print("🔍 Monitoring API Response Schemas...")
        
        # Load cached schemas
        cached_data = self.load_cached_schemas()
        current_schemas = {}
        schema_changes = {}
        
        # Get current schemas
        for endpoint, config in self.endpoints.items():
            print(f"📡 Checking {endpoint}...")
            current_schema_data = self.get_response_schema(endpoint, config)
            current_schemas[endpoint] = current_schema_data
            
            # Compare with cached if available
            if endpoint in cached_data:
                cached_schema = cached_data[endpoint].get('schema', {})
                current_schema = current_schema_data.get('schema', {})
                
                if cached_schema and current_schema:
                    changes = self.compare_schemas(cached_schema, current_schema)
                    if changes['total_changes'] > 0:
                        schema_changes[endpoint] = changes
                        print(f"⚠️  Schema changes detected in {endpoint}")
                        print(f"   Added: {len(changes['added_fields'])}, Removed: {len(changes['removed_fields'])}, Type changes: {len(changes['type_changes'])}")
                    else:
                        print(f"✅ No changes in {endpoint}")
                else:
                    print(f"ℹ️  No comparison data for {endpoint}")
            else:
                print(f"🆕 New endpoint: {endpoint}")
        
        # Save current schemas
        self.save_schemas(current_schemas)
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitored_endpoints': len(self.endpoints),
            'endpoints_with_changes': len(schema_changes),
            'schema_changes': schema_changes,
            'current_schemas': {ep: data.get('schema_hash', '') for ep, data in current_schemas.items()}
        }
        
        return report
    
    def generate_model_update_tasks(self, schema_changes: Dict[str, Any]) -> List[str]:
        """Generate tasks for updating Pydantic models based on schema changes."""
        tasks = []
        
        for endpoint, changes in schema_changes.items():
            if changes['total_changes'] > 0:
                tasks.append(f"""
# Model Update Task for {endpoint}

## Changes Detected:
- Added fields: {len(changes['added_fields'])}
- Removed fields: {len(changes['removed_fields'])} 
- Type changes: {len(changes['type_changes'])}

## Recommended Actions:
1. Update Pydantic model for {endpoint} response
2. Add new fields: {changes['added_fields'][:5]}  # Show first 5
3. Remove obsolete fields: {changes['removed_fields'][:5]}  # Show first 5
4. Fix type mismatches: {[tc['field'] for tc in changes['type_changes'][:3]]}  # Show first 3

## Testing Required:
- Verify model can parse actual API response
- Run integration tests for {endpoint}
- Update mock data in tests if needed
""")
        
        return tasks


if __name__ == "__main__":
    monitor = ResponseSchemaMonitor()
    report = monitor.monitor_schemas()
    
    print("\n📊 Schema Monitoring Summary:")
    print(f"   Monitored endpoints: {report['monitored_endpoints']}")
    print(f"   Endpoints with changes: {report['endpoints_with_changes']}")
    
    if report['schema_changes']:
        print("\n⚠️  Schema changes detected! Generating update tasks...")
        tasks = monitor.generate_model_update_tasks(report['schema_changes'])
        
        # Save tasks to file
        tasks_file = monitor.docs_dir / "model_update_tasks.md"
        with open(tasks_file, 'w') as f:
            f.write("# Model Update Tasks\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            for task in tasks:
                f.write(task)
        
        print(f"   Tasks saved to: {tasks_file}")
    else:
        print("✅ No schema changes detected")
    
    # Save monitoring report
    report_file = monitor.docs_dir / "schema_monitoring_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   Report saved to: {report_file}")