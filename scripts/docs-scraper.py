#!/usr/bin/env python3
"""
Venice.ai Documentation Scraper

Scrapes the Venice.ai documentation pages to supplement swagger spec
with examples, validation rules, and detailed parameter descriptions.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import httpx
from bs4 import BeautifulSoup
import yaml


class VeniceDocsScraper:
    """Scraper for Venice.ai API documentation"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"
        self.base_url = "https://docs.venice.ai"
        self.scraped_data = {}
        
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a documentation page"""
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            return None
    
    def extract_api_info(self, html: str, endpoint_path: str) -> Dict[str, Any]:
        """Extract API information from a documentation page"""
        soup = BeautifulSoup(html, 'html.parser')
        
        info = {
            'endpoint': endpoint_path,
            'title': None,
            'description': None,
            'parameters': {},
            'examples': {
                'request': None,
                'response': None
            },
            'validation_rules': [],
            'models': []
        }
        
        # Extract title
        title_elem = soup.find('h1')
        if title_elem:
            info['title'] = title_elem.get_text().strip()
        
        # Extract description
        # Look for description paragraph after the title
        desc_elem = soup.find('p')
        if desc_elem:
            info['description'] = desc_elem.get_text().strip()
        
        # Extract parameter information from tables
        tables = soup.find_all('table')
        for table in tables:
            headers = [th.get_text().strip() for th in table.find_all('th')]
            
            # Look for parameter tables
            if any(header in ['Parameter', 'Name', 'Field'] for header in headers):
                rows = table.find_all('tr')[1:]  # Skip header row
                
                for row in rows:
                    cells = [td.get_text().strip() for td in row.find_all('td')]
                    if len(cells) >= 2:
                        param_name = cells[0]
                        param_info = {
                            'description': cells[1] if len(cells) > 1 else '',
                            'type': cells[2] if len(cells) > 2 else '',
                            'required': 'required' in ' '.join(cells).lower(),
                            'default': None
                        }
                        
                        # Extract default value if present
                        for cell in cells:
                            if 'default' in cell.lower():
                                match = re.search(r'default[:\s]+([^\s,]+)', cell.lower())
                                if match:
                                    param_info['default'] = match.group(1)
                        
                        info['parameters'][param_name] = param_info
        
        # Extract code examples
        code_blocks = soup.find_all('code')
        for code_block in code_blocks:
            code_text = code_block.get_text()
            
            # Try to identify request vs response examples
            if '{' in code_text and '"' in code_text:
                try:
                    parsed = json.loads(code_text)
                    
                    # Heuristic: if it has common request fields, it's a request
                    if any(key in parsed for key in ['model', 'messages', 'prompt']):
                        info['examples']['request'] = parsed
                    
                    # Heuristic: if it has response fields, it's a response
                    elif any(key in parsed for key in ['choices', 'data', 'usage']):
                        info['examples']['response'] = parsed
                        
                except json.JSONDecodeError:
                    continue
        
        # Extract validation rules from text
        text_content = soup.get_text()
        
        # Look for validation patterns
        validation_patterns = [
            r'must be between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)',
            r'maximum (?:of )?(\d+)',
            r'minimum (?:of )?(\d+)',
            r'required|optional',
            r'must be one of: ([^.]+)',
            r'cannot exceed (\d+)',
        ]
        
        for pattern in validation_patterns:
            matches = re.finditer(pattern, text_content, re.IGNORECASE)
            for match in matches:
                info['validation_rules'].append({
                    'rule': match.group(0),
                    'context': text_content[max(0, match.start()-50):match.end()+50].strip()
                })
        
        # Extract model information
        model_mentions = re.finditer(r'([a-z-]+(?:-[a-z0-9]+)*)', text_content)
        potential_models = set()
        
        for match in model_mentions:
            model_name = match.group(1)
            
            # Filter for likely model names
            if any(keyword in model_name for keyword in ['venice', 'flux', 'gpt', 'claude', 'llama', 'gemini']):
                potential_models.add(model_name)
        
        info['models'] = list(potential_models)
        
        return info
    
    async def scrape_endpoint_docs(self, endpoint_paths: List[str]) -> Dict[str, Any]:
        """Scrape documentation for multiple endpoints"""
        scraped_docs = {}
        
        for endpoint_path in endpoint_paths:
            # Convert endpoint path to docs URL
            # e.g., "/chat/completions" -> "https://docs.venice.ai/api-reference/endpoint/chat/completions"
            clean_path = endpoint_path.strip('/')
            docs_url = f"{self.base_url}/api-reference/endpoint/{clean_path.replace('/', '/')}"
            
            print(f"🔍 Scraping: {docs_url}")
            
            html = await self.fetch_page(docs_url)
            if html:
                info = self.extract_api_info(html, endpoint_path)
                scraped_docs[endpoint_path] = info
                
                # Brief delay to be respectful
                await asyncio.sleep(1)
        
        return scraped_docs
    
    def merge_with_swagger(self, swagger_spec: Dict[str, Any], scraped_docs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge scraped documentation with swagger specification"""
        enhanced_spec = swagger_spec.copy()
        
        # Add scraped info to swagger paths
        for endpoint_path, docs_info in scraped_docs.items():
            # Find matching path in swagger
            swagger_path = None
            for path in enhanced_spec.get('paths', {}):
                if path == endpoint_path or path.endswith(endpoint_path):
                    swagger_path = path
                    break
            
            if swagger_path:
                path_info = enhanced_spec['paths'][swagger_path]
                
                # Enhance each HTTP method
                for method, method_info in path_info.items():
                    if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                        # Add scraped description if better than swagger
                        if docs_info['description'] and len(docs_info['description']) > len(method_info.get('description', '')):
                            method_info['description'] = docs_info['description']
                        
                        # Add examples
                        if docs_info['examples']['request']:
                            if 'requestBody' in method_info:
                                method_info['requestBody']['content'] = method_info['requestBody'].get('content', {})
                                for content_type in method_info['requestBody']['content']:
                                    method_info['requestBody']['content'][content_type]['example'] = docs_info['examples']['request']
                        
                        if docs_info['examples']['response']:
                            method_info['responses'] = method_info.get('responses', {})
                            for status_code, response_info in method_info['responses'].items():
                                if 'content' in response_info:
                                    for content_type in response_info['content']:
                                        response_info['content'][content_type]['example'] = docs_info['examples']['response']
                        
                        # Add validation rules as custom extension
                        if docs_info['validation_rules']:
                            method_info['x-validation-rules'] = docs_info['validation_rules']
                        
                        # Add model list
                        if docs_info['models']:
                            method_info['x-supported-models'] = docs_info['models']
        
        return enhanced_spec
    
    async def scrape_all_endpoints(self) -> Dict[str, Any]:
        """Scrape documentation for all known endpoints"""
        # Common Venice.ai endpoints
        endpoints = [
            "/chat/completions",
            "/image/generate",
            "/images/generations",
            "/image/upscale",
            "/audio/speech",
            "/embeddings",
            "/models",
            "/api_keys",
            "/characters",
            "/billing/usage"
        ]
        
        print("🕷️ Scraping Venice.ai documentation...")
        scraped_docs = await self.scrape_endpoint_docs(endpoints)
        
        return scraped_docs
    
    def save_enhanced_spec(self, enhanced_spec: Dict[str, Any], filename: str = "swagger_enhanced.yaml"):
        """Save enhanced specification with scraped data"""
        output_path = self.docs_dir / filename
        
        with open(output_path, 'w') as f:
            yaml.dump(enhanced_spec, f, default_flow_style=False, sort_keys=False)
        
        print(f"💾 Enhanced specification saved: {output_path}")
        return output_path
    
    async def run(self) -> Dict[str, Any]:
        """Main scraping workflow"""
        # Load existing swagger spec
        swagger_path = self.docs_dir / "swagger.yaml"
        if swagger_path.exists():
            with open(swagger_path, 'r') as f:
                swagger_spec = yaml.safe_load(f)
        else:
            print("⚠️ No swagger.yaml found - creating from scratch")
            swagger_spec = {'paths': {}}
        
        # Scrape documentation
        scraped_docs = await self.scrape_all_endpoints()
        
        # Merge data
        enhanced_spec = self.merge_with_swagger(swagger_spec, scraped_docs)
        
        # Save results
        enhanced_path = self.save_enhanced_spec(enhanced_spec)
        
        # Save raw scraped data for reference
        scraped_path = self.docs_dir / "scraped_docs.json"
        with open(scraped_path, 'w') as f:
            json.dump(scraped_docs, f, indent=2)
        
        print(f"📄 Raw scraped data saved: {scraped_path}")
        
        # Generate summary
        summary = {
            'endpoints_scraped': len(scraped_docs),
            'total_parameters': sum(len(info['parameters']) for info in scraped_docs.values()),
            'total_examples': sum(1 for info in scraped_docs.values() 
                                if info['examples']['request'] or info['examples']['response']),
            'total_validation_rules': sum(len(info['validation_rules']) for info in scraped_docs.values()),
            'enhanced_spec_path': str(enhanced_path),
            'scraped_data_path': str(scraped_path)
        }
        
        print("\n📊 Scraping Summary:")
        print(f"  Endpoints: {summary['endpoints_scraped']}")
        print(f"  Parameters: {summary['total_parameters']}")
        print(f"  Examples: {summary['total_examples']}")
        print(f"  Validation rules: {summary['total_validation_rules']}")
        
        return summary


async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape Venice.ai documentation")
    parser.add_argument('--endpoint', action='append', help="Specific endpoint to scrape")
    parser.add_argument('--output', default="swagger_enhanced.yaml", help="Output filename")
    
    args = parser.parse_args()
    
    scraper = VeniceDocsScraper()
    
    if args.endpoint:
        scraped_docs = await scraper.scrape_endpoint_docs(args.endpoint)
    else:
        scraped_docs = await scraper.scrape_all_endpoints()
    
    # Load and enhance swagger spec
    swagger_path = scraper.docs_dir / "swagger.yaml"
    if swagger_path.exists():
        with open(swagger_path, 'r') as f:
            swagger_spec = yaml.safe_load(f)
        
        enhanced_spec = scraper.merge_with_swagger(swagger_spec, scraped_docs)
        scraper.save_enhanced_spec(enhanced_spec, args.output)
    
    print("✅ Documentation scraping complete")


if __name__ == "__main__":
    asyncio.run(main())