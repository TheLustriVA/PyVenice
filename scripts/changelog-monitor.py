#!/usr/bin/env python3
"""
Venice.ai Changelog Monitor

Fetches and monitors Venice.ai changelog for API-related changes.
Integrates with existing automation system to provide proactive change detection.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from bs4 import BeautifulSoup
import hashlib
import asyncio
from playwright.async_api import async_playwright


class VeniceChangelogMonitor:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"
        self.changelog_cache_path = self.docs_dir / "venice_changelog_cache.json"
        self.changelog_url = "https://featurebase.venice.ai/changelog"
        
        # Credentials
        self.username = os.environ.get('VENICE_USERNAME')
        self.password = os.environ.get('VENICE_PASS')
        
        if not self.username or not self.password:
            raise ValueError("VENICE_USERNAME and VENICE_PASS environment variables required")
        
        # Keywords that indicate API-relevant changes
        self.api_keywords = [
            'api', 'endpoint', 'parameter', 'response', 'schema',
            'model', 'deprecat', 'remov', 'add', 'chang', 'updat',
            'chat', 'completion', 'image', 'billing', 'character',
            'authentication', 'rate limit', 'token', 'pricing'
        ]
    
    async def authenticate_and_fetch_changelog(self) -> Optional[str]:
        """Authenticate with Venice.ai featurebase and fetch changelog using Playwright."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                
                # Navigate to the changelog URL
                print(f"🔄 Navigating to {self.changelog_url}")
                response = await page.goto(self.changelog_url, wait_until='networkidle')
                
                # Check for Cloudflare challenge
                page_content = await page.content()
                if "Just a moment..." in page_content or "Enable JavaScript" in page_content:
                    print("🔄 Cloudflare challenge detected, waiting for completion...")
                    try:
                        # Wait for the challenge to complete - look for actual content instead of challenge page
                        await page.wait_for_function(
                            "document.body.innerText.includes('changelog') || document.body.innerText.includes('change') || document.body.innerText.includes('update') || document.body.innerText.includes('feature')",
                            timeout=30000
                        )
                        print("✅ Cloudflare challenge completed")
                    except Exception:
                        print("⚠️  Cloudflare challenge timeout, trying anyway...")
                
                # Check final status
                final_content = await page.content()
                if "Just a moment..." not in final_content and "Enable JavaScript" not in final_content:
                    print("✅ Changelog accessible after challenge")
                    return final_content
                
                print(f"   Status: {response.status}, trying authentication...")
                
                # Save screenshot for debugging
                screenshot_path = self.docs_dir / "venice_changelog_debug.png"
                await page.screenshot(path=screenshot_path)
                print(f"   Screenshot saved to: {screenshot_path}")
                
                # Get page content for analysis
                page_content = await page.content()
                debug_html_path = self.docs_dir / "venice_changelog_debug.html" 
                with open(debug_html_path, 'w', encoding='utf-8') as f:
                    f.write(page_content)
                print(f"   HTML content saved to: {debug_html_path}")
                
                # Look for login link or form
                login_selectors = [
                    'a[href*="login"]',
                    'a[href*="signin"]', 
                    'a[href*="auth"]',
                    'button:has-text("Login")',
                    'button:has-text("Sign In")',
                    'input[type="email"]',
                    'input[type="password"]'
                ]
                
                login_element = None
                for selector in login_selectors:
                    try:
                        login_element = await page.wait_for_selector(selector, timeout=2000)
                        if login_element:
                            print(f"   Found login element: {selector}")
                            break
                    except Exception:
                        continue
                
                if not login_element:
                    # Try navigating to common login URLs
                    login_urls = [
                        'https://featurebase.venice.ai/login',
                        'https://featurebase.venice.ai/signin',
                        'https://featurebase.venice.ai/auth/login'
                    ]
                    
                    for login_url in login_urls:
                        try:
                            print(f"   Trying login URL: {login_url}")
                            await page.goto(login_url)
                            
                            # Wait for email/username input
                            email_input = await page.wait_for_selector('input[type="email"], input[name="email"], input[name="username"]', timeout=5000)
                            password_input = await page.wait_for_selector('input[type="password"], input[name="password"]', timeout=2000)
                            
                            if email_input and password_input:
                                print("   Found login form, filling credentials...")
                                
                                # Fill in credentials
                                await email_input.fill(self.username)
                                await password_input.fill(self.password)
                                
                                # Submit form
                                submit_button = await page.query_selector('button[type="submit"], input[type="submit"], button:has-text("Login"), button:has-text("Sign In")')
                                if submit_button:
                                    await submit_button.click()
                                else:
                                    # Try pressing Enter on password field
                                    await password_input.press('Enter')
                                
                                # Wait for navigation or success indicator
                                try:
                                    await page.wait_for_load_state('networkidle', timeout=10000)
                                    print("   Login submitted, checking result...")
                                    
                                    # Try accessing changelog again
                                    response = await page.goto(self.changelog_url)
                                    if response.status == 200:
                                        print("✅ Successfully authenticated and accessed changelog")
                                        return await page.content()
                                    
                                except Exception as e:
                                    print(f"   Login wait failed: {e}")
                                    
                        except Exception as e:
                            print(f"   Login attempt failed: {e}")
                            continue
                
                print("❌ Failed to authenticate with Venice.ai")
                return None
                
            except Exception as e:
                print(f"❌ Browser automation failed: {e}")
                return None
            finally:
                await browser.close()
    
    def parse_changelog_entries(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse changelog HTML and extract entries."""
        entries = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for common changelog entry patterns
            entry_selectors = [
                '.changelog-entry',
                '.changelog-item', 
                '.update-entry',
                '.release-note',
                'article',
                '.post'
            ]
            
            changelog_entries = []
            for selector in entry_selectors:
                found = soup.select(selector)
                if found:
                    changelog_entries = found
                    break
            
            # Fallback: look for date patterns
            if not changelog_entries:
                # Find elements with date-like text
                date_pattern = re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\w+ \d{1,2},? \d{4}\b')
                all_elements = soup.find_all(text=date_pattern)
                for element in all_elements[:10]:  # Limit to prevent too many
                    parent = element.parent
                    if parent:
                        changelog_entries.append(parent)
            
            for i, entry_elem in enumerate(changelog_entries[:20]):  # Limit entries
                try:
                    # Extract date
                    date_text = entry_elem.get_text()
                    date_match = re.search(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\w+ \d{1,2},? \d{4}\b', date_text)
                    entry_date = date_match.group(0) if date_match else f"entry_{i}"
                    
                    # Extract content
                    content = entry_elem.get_text(strip=True)
                    
                    # Check for API relevance
                    is_api_relevant = any(keyword.lower() in content.lower() for keyword in self.api_keywords)
                    
                    # Extract title (usually first line or header)
                    title_elem = entry_elem.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    title = title_elem.get_text(strip=True) if title_elem else content[:100]
                    
                    entry = {
                        'id': hashlib.md5(content.encode()).hexdigest()[:12],
                        'date': entry_date,
                        'title': title,
                        'content': content,
                        'is_api_relevant': is_api_relevant,
                        'raw_html': str(entry_elem),
                        'parsed_timestamp': datetime.now().isoformat()
                    }
                    
                    entries.append(entry)
                    
                except Exception as e:
                    print(f"⚠️  Failed to parse entry {i}: {e}")
                    continue
            
        except Exception as e:
            print(f"❌ Failed to parse changelog HTML: {e}")
        
        return entries
    
    def load_cached_changelog(self) -> Dict[str, Any]:
        """Load previously cached changelog data."""
        if not self.changelog_cache_path.exists():
            return {'entries': [], 'last_fetch': None, 'entry_ids': set()}
        
        try:
            with open(self.changelog_cache_path, 'r') as f:
                data = json.load(f)
                # Convert entry_ids back to set
                data['entry_ids'] = set(data.get('entry_ids', []))
                return data
        except Exception as e:
            print(f"⚠️  Failed to load changelog cache: {e}")
            return {'entries': [], 'last_fetch': None, 'entry_ids': set()}
    
    def save_changelog_cache(self, data: Dict[str, Any]):
        """Save changelog data to cache."""
        self.docs_dir.mkdir(exist_ok=True)
        
        # Convert set to list for JSON serialization
        data_to_save = data.copy()
        data_to_save['entry_ids'] = list(data['entry_ids'])
        
        try:
            with open(self.changelog_cache_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save changelog cache: {e}")
    
    def detect_new_entries(self, current_entries: List[Dict[str, Any]], cached_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect new changelog entries since last fetch."""
        cached_ids = cached_data.get('entry_ids', set())
        new_entries = []
        
        for entry in current_entries:
            if entry['id'] not in cached_ids:
                new_entries.append(entry)
        
        return new_entries
    
    async def monitor_changelog(self) -> Dict[str, Any]:
        """Monitor changelog for changes and return report."""
        print("🔍 Monitoring Venice.ai Changelog...")
        
        # Fetch current changelog
        html_content = await self.authenticate_and_fetch_changelog()
        if not html_content:
            return {'status': 'failed', 'error': 'Failed to fetch changelog'}
        
        # Parse entries
        current_entries = self.parse_changelog_entries(html_content)
        if not current_entries:
            print("⚠️  No changelog entries found - may need to adjust parsing")
        
        # Load cached data
        cached_data = self.load_cached_changelog()
        
        # Detect new entries
        new_entries = self.detect_new_entries(current_entries, cached_data)
        api_relevant_new = [e for e in new_entries if e['is_api_relevant']]
        
        # Update cache
        all_entry_ids = {entry['id'] for entry in current_entries}
        updated_cache = {
            'entries': current_entries,
            'last_fetch': datetime.now().isoformat(),
            'entry_ids': all_entry_ids
        }
        self.save_changelog_cache(updated_cache)
        
        # Generate report
        report = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_entries': len(current_entries),
            'new_entries': len(new_entries),
            'api_relevant_new': len(api_relevant_new),
            'new_entries_data': new_entries,
            'api_relevant_entries': api_relevant_new
        }
        
        return report
    
    def generate_api_change_alerts(self, report: Dict[str, Any]) -> List[str]:
        """Generate alerts for API-relevant changes."""
        alerts = []
        
        api_entries = report.get('api_relevant_entries', [])
        
        for entry in api_entries:
            alert = f"""
# API Change Alert: {entry['title']}

**Date**: {entry['date']}
**Relevance**: API-related change detected

## Content:
{entry['content'][:500]}{'...' if len(entry['content']) > 500 else ''}

## Recommended Actions:
1. Review change for impact on PyVenice client
2. Check if any endpoints/parameters are affected
3. Update Pydantic models if response schemas changed
4. Run API contract validation
5. Update documentation if needed

**Change ID**: {entry['id']}
"""
            alerts.append(alert)
        
        return alerts


async def main():
    monitor = VeniceChangelogMonitor()
    report = await monitor.monitor_changelog()
    
    print("\n📊 Changelog Monitoring Summary:")
    print(f"   Status: {report.get('status', 'unknown')}")
    
    if report.get('status') == 'success':
        print(f"   Total entries: {report['total_entries']}")
        print(f"   New entries: {report['new_entries']}")
        print(f"   API-relevant new: {report['api_relevant_new']}")
        
        if report['api_relevant_new'] > 0:
            print("\n⚠️  API-relevant changes detected!")
            alerts = monitor.generate_api_change_alerts(report)
            
            # Save alerts
            alerts_file = monitor.docs_dir / "changelog_api_alerts.md"
            with open(alerts_file, 'w') as f:
                f.write("# Venice.ai Changelog API Alerts\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                for alert in alerts:
                    f.write(alert)
            
            print(f"   Alerts saved to: {alerts_file}")
        else:
            print("✅ No API-relevant changes detected")
    else:
        print(f"   Error: {report.get('error', 'Unknown error')}")
    
    # Save full report
    report_file = monitor.docs_dir / "changelog_monitoring_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   Report saved to: {report_file}")


if __name__ == "__main__":
    asyncio.run(main())