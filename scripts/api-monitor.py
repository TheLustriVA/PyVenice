#!/usr/bin/env python3
"""
Venice.ai API Monitoring Tool

Downloads the latest Swagger spec, compares with local copy, and reports changes.
Designed to run daily via cron or GitHub Actions.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Set
import hashlib

import httpx
import yaml


class APIMonitor:
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"
        self.local_spec_path = self.docs_dir / "swagger.yaml"
        self.remote_spec_url = "https://api.venice.ai/api/v1/swagger.yaml"
        self.changes_log = self.docs_dir / "api_changes.json"
        
    async def fetch_remote_spec(self) -> Dict[str, Any]:
        """Download the latest Swagger spec from Venice.ai"""
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(self.remote_spec_url)
            response.raise_for_status()
            return yaml.safe_load(response.text)
    
    def load_local_spec(self) -> Dict[str, Any]:
        """Load the local Swagger spec"""
        if not self.local_spec_path.exists():
            return {}
        with open(self.local_spec_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_spec(self, spec: Dict[str, Any]):
        """Save spec to local file"""
        self.docs_dir.mkdir(exist_ok=True)
        with open(self.local_spec_path, 'w') as f:
            yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
    
    def extract_endpoints(self, spec: Dict[str, Any]) -> Set[str]:
        """Extract all endpoint paths from spec"""
        paths = spec.get('paths', {})
        endpoints = set()
        for path, methods in paths.items():
            for method in methods.keys():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    endpoints.add(f"{method.upper()} {path}")
        return endpoints
    
    def extract_schemas(self, spec: Dict[str, Any]) -> Dict[str, str]:
        """Extract schema definitions with their hashes"""
        schemas = spec.get('components', {}).get('schemas', {})
        schema_hashes = {}
        for name, schema in schemas.items():
            try:
                # Convert to string via YAML to handle datetime objects
                schema_str = yaml.dump(schema, sort_keys=True)
                schema_hashes[name] = hashlib.md5(schema_str.encode()).hexdigest()
            except Exception:
                # Fallback to string representation
                schema_hashes[name] = hashlib.md5(str(schema).encode()).hexdigest()
        return schema_hashes
    
    def extract_parameters(self, spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract all parameters from schemas for lifecycle tracking"""
        schemas = spec.get('components', {}).get('schemas', {})
        parameters = {}
        
        for schema_name, schema in schemas.items():
            if 'properties' in schema:
                parameters[schema_name] = {}
                for param_name, param_def in schema['properties'].items():
                    parameters[schema_name][param_name] = {
                        'type': param_def.get('type', 'unknown'),
                        'required': param_name in schema.get('required', []),
                        'description': param_def.get('description', ''),
                        'hash': hashlib.md5(str(param_def).encode()).hexdigest()
                    }
        
        return parameters
    
    def compare_parameters(self, old_params: Dict[str, Dict[str, Any]], new_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare parameters for lifecycle tracking"""
        param_changes = {
            'added': [],
            'removed': [],
            'modified': []
        }
        
        # Find all schemas
        all_schemas = set(old_params.keys()) | set(new_params.keys())
        
        for schema_name in all_schemas:
            old_schema_params = old_params.get(schema_name, {})
            new_schema_params = new_params.get(schema_name, {})
            
            # Find parameter changes within this schema
            old_param_names = set(old_schema_params.keys())
            new_param_names = set(new_schema_params.keys())
            
            # Added parameters
            for param_name in new_param_names - old_param_names:
                param_changes['added'].append({
                    'schema': schema_name,
                    'parameter': param_name,
                    'definition': new_schema_params[param_name]
                })
            
            # Removed parameters
            for param_name in old_param_names - new_param_names:
                param_changes['removed'].append({
                    'schema': schema_name,
                    'parameter': param_name,
                    'definition': old_schema_params[param_name]
                })
            
            # Modified parameters
            for param_name in old_param_names & new_param_names:
                if old_schema_params[param_name]['hash'] != new_schema_params[param_name]['hash']:
                    param_changes['modified'].append({
                        'schema': schema_name,
                        'parameter': param_name,
                        'old_definition': old_schema_params[param_name],
                        'new_definition': new_schema_params[param_name]
                    })
        
        return param_changes

    def compare_specs(self, old_spec: Dict[str, Any], new_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two specs and return detailed changes"""
        changes = {
            'timestamp': datetime.now().isoformat(),
            'version_change': {
                'old': old_spec.get('info', {}).get('version', 'unknown'),
                'new': new_spec.get('info', {}).get('version', 'unknown')
            },
            'endpoints': {
                'added': [],
                'removed': [],
                'modified': []
            },
            'schemas': {
                'added': [],
                'removed': [],
                'modified': []
            },
            'parameters': {
                'added': [],
                'removed': [],
                'modified': []
            },
            'summary': ''
        }
        
        # Compare endpoints
        old_endpoints = self.extract_endpoints(old_spec)
        new_endpoints = self.extract_endpoints(new_spec)
        
        changes['endpoints']['added'] = sorted(new_endpoints - old_endpoints)
        changes['endpoints']['removed'] = sorted(old_endpoints - new_endpoints)
        
        # Compare schemas
        old_schemas = self.extract_schemas(old_spec)
        new_schemas = self.extract_schemas(new_spec)
        
        changes['schemas']['added'] = sorted(set(new_schemas.keys()) - set(old_schemas.keys()))
        changes['schemas']['removed'] = sorted(set(old_schemas.keys()) - set(new_schemas.keys()))
        
        # Find modified schemas
        common_schemas = set(old_schemas.keys()) & set(new_schemas.keys())
        changes['schemas']['modified'] = [
            name for name in sorted(common_schemas)
            if old_schemas[name] != new_schemas[name]
        ]
        
        # Compare parameters for lifecycle tracking
        old_parameters = self.extract_parameters(old_spec)
        new_parameters = self.extract_parameters(new_spec)
        parameter_changes = self.compare_parameters(old_parameters, new_parameters)
        changes['parameters'] = parameter_changes
        
        # Generate summary
        total_changes = (len(changes['endpoints']['added']) + 
                        len(changes['endpoints']['removed']) +
                        len(changes['schemas']['added']) + 
                        len(changes['schemas']['removed']) +
                        len(changes['schemas']['modified']) +
                        len(changes['parameters']['added']) +
                        len(changes['parameters']['removed']) +
                        len(changes['parameters']['modified']))
        
        if total_changes == 0:
            changes['summary'] = f"No changes detected (version: {changes['version_change']['old']} → {changes['version_change']['new']})"
        else:
            parts = []
            if changes['endpoints']['added']:
                parts.append(f"{len(changes['endpoints']['added'])} new endpoints")
            if changes['endpoints']['removed']:
                parts.append(f"{len(changes['endpoints']['removed'])} removed endpoints")
            if changes['schemas']['added']:
                parts.append(f"{len(changes['schemas']['added'])} new schemas")
            if changes['schemas']['removed']:
                parts.append(f"{len(changes['schemas']['removed'])} removed schemas")
            if changes['schemas']['modified']:
                parts.append(f"{len(changes['schemas']['modified'])} modified schemas")
            if changes['parameters']['added']:
                parts.append(f"{len(changes['parameters']['added'])} new parameters")
            if changes['parameters']['removed']:
                parts.append(f"{len(changes['parameters']['removed'])} removed parameters")
            if changes['parameters']['modified']:
                parts.append(f"{len(changes['parameters']['modified'])} modified parameters")
            
            changes['summary'] = f"API changes detected: {', '.join(parts)}"
        
        return changes
    
    def log_changes(self, changes: Dict[str, Any]):
        """Append changes to the changes log"""
        log_entries = []
        if self.changes_log.exists():
            with open(self.changes_log, 'r') as f:
                log_entries = json.load(f)
        
        log_entries.append(changes)
        
        # Keep only last 100 entries
        log_entries = log_entries[-100:]
        
        with open(self.changes_log, 'w') as f:
            json.dump(log_entries, f, indent=2)
    
    def print_changes_report(self, changes: Dict[str, Any]):
        """Print a human-readable changes report"""
        print(f"🔍 Venice.ai API Monitor Report - {changes['timestamp']}")
        print(f"📋 {changes['summary']}")
        print()
        
        if changes['version_change']['old'] != changes['version_change']['new']:
            print(f"📝 Version: {changes['version_change']['old']} → {changes['version_change']['new']}")
            print()
        
        if changes['endpoints']['added']:
            print("✅ New Endpoints:")
            for endpoint in changes['endpoints']['added']:
                print(f"  + {endpoint}")
            print()
        
        if changes['endpoints']['removed']:
            print("❌ Removed Endpoints:")
            for endpoint in changes['endpoints']['removed']:
                print(f"  - {endpoint}")
            print()
        
        if changes['schemas']['added']:
            print("📄 New Schemas:")
            for schema in changes['schemas']['added']:
                print(f"  + {schema}")
            print()
        
        if changes['schemas']['removed']:
            print("🗑️  Removed Schemas:")
            for schema in changes['schemas']['removed']:
                print(f"  - {schema}")
            print()
        
        if changes['schemas']['modified']:
            print("🔄 Modified Schemas:")
            for schema in changes['schemas']['modified']:
                print(f"  ~ {schema}")
            print()
        
        # Parameter-level changes
        if changes.get('parameters', {}).get('added'):
            print("➕ New Parameters:")
            for param in changes['parameters']['added']:
                req_marker = " (required)" if param['definition']['required'] else ""
                print(f"  + {param['schema']}.{param['parameter']}{req_marker}")
            print()
        
        if changes.get('parameters', {}).get('removed'):
            print("❌ Removed Parameters:")
            for param in changes['parameters']['removed']:
                req_marker = " (required)" if param['definition']['required'] else ""
                print(f"  - {param['schema']}.{param['parameter']}{req_marker}")
            print()
        
        if changes.get('parameters', {}).get('modified'):
            print("🔄 Modified Parameters:")
            for param in changes['parameters']['modified']:
                print(f"  ~ {param['schema']}.{param['parameter']}")
            print()
    
    async def run(self, save_changes: bool = True) -> Dict[str, Any]:
        """Main monitoring logic"""
        print("🚀 Fetching latest Venice.ai API specification...")
        
        try:
            remote_spec = await self.fetch_remote_spec()
            local_spec = self.load_local_spec()
            
            changes = self.compare_specs(local_spec, remote_spec)
            
            self.print_changes_report(changes)
            
            if save_changes:
                self.save_spec(remote_spec)
                self.log_changes(changes)
                
                if any(changes['endpoints'].values()) or any(changes['schemas'].values()):
                    print("💾 Updated local spec and logged changes")
                    print(f"📊 Changes log: {self.changes_log}")
                    return changes
                else:
                    print("💤 No changes detected")
            
            return changes
            
        except Exception as e:
            error_msg = f"❌ Error monitoring API: {e}"
            print(error_msg)
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}


async def run_changelog_monitoring():
    """Run Venice.ai changelog monitoring alongside API monitoring."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import importlib.util
        spec = importlib.util.spec_from_file_location("changelog_monitor", Path(__file__).parent / "changelog-monitor.py")
        changelog_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(changelog_module)
        VeniceChangelogMonitor = changelog_module.VeniceChangelogMonitor
        
        print("\n🔍 Checking Venice.ai Changelog for API Changes...")
        changelog_monitor = VeniceChangelogMonitor()
        changelog_report = await changelog_monitor.monitor_changelog()
        
        if changelog_report.get('status') == 'success':
            print(f"   Changelog entries found: {changelog_report['total_entries']}")
            if changelog_report['api_relevant_new'] > 0:
                print(f"   ⚠️  API-relevant changes: {changelog_report['api_relevant_new']}")
                return changelog_report['api_relevant_entries']
            else:
                print("   ✅ No new API-relevant changes")
        else:
            print(f"   ❌ Changelog monitoring failed: {changelog_report.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ⚠️  Changelog monitoring error: {e}")
    
    return []


async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Venice.ai API for changes")
    parser.add_argument('--dry-run', action='store_true', 
                       help="Don't save changes, just report them")
    parser.add_argument('--json', action='store_true',
                       help="Output changes as JSON instead of human-readable format")
    parser.add_argument('--skip-changelog', action='store_true',
                       help="Skip changelog monitoring and only check API parameters")
    
    args = parser.parse_args()
    
    # Run changelog monitoring first (unless skipped)
    changelog_changes = []
    if not args.skip_changelog:
        changelog_changes = await run_changelog_monitoring()
    
    monitor = APIMonitor()
    changes = await monitor.run(save_changes=not args.dry_run)
    
    # Merge changelog changes into the main changes report
    if changelog_changes:
        changes['changelog_changes'] = changelog_changes
    
    if args.json:
        print(json.dumps(changes, indent=2))
    
    # Exit with non-zero if changes detected (useful for CI/CD)
    if 'error' in changes:
        sys.exit(1)
    
    total_changes = 0
    if 'endpoints' in changes and 'schemas' in changes:
        total_changes = (len(changes['endpoints'].get('added', [])) + 
                        len(changes['endpoints'].get('removed', [])) +
                        len(changes['schemas'].get('added', [])) + 
                        len(changes['schemas'].get('removed', [])) +
                        len(changes['schemas'].get('modified', [])))
    
    sys.exit(0 if total_changes == 0 else 2)  # 2 = changes detected


if __name__ == "__main__":
    asyncio.run(main())