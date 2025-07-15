#!/bin/bash
# Daily Venice.ai API monitoring script
# Add to crontab: 0 9 * * * /path/to/this/script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "🕘 $(date): Starting Venice.ai API monitoring..."

# Run the API monitor
if python scripts/api-monitor.py; then
    echo "✅ API monitoring completed successfully"
echo "📊 Generating monitoring report..."
python scripts/monitoring-report.py
    
    # Run the changelog monitor
    echo "📰 Running changelog monitoring..."
    python scripts/changelog-monitor.py
    CHANGELOG_EXIT_CODE=$?
    
    if [ $CHANGELOG_EXIT_CODE -eq 0 ]; then
        echo "✅ Changelog monitoring completed - no changes"
    elif [ $CHANGELOG_EXIT_CODE -eq 2 ]; then
        echo "📰 Changelog monitoring completed - changes detected"
    elif [ $CHANGELOG_EXIT_CODE -eq 3 ]; then
        echo "⚠️  Changelog monitoring - access restricted"
    else
        echo "❌ Changelog monitoring failed"
    fi
    
    # Check if changes were detected
    HAS_API_CHANGES=false
    HAS_CHANGELOG_CHANGES=false
    
    if [ -f "docs/api_changes.json" ]; then
        LATEST_CHANGE=$(python -c "
import json
with open('docs/api_changes.json', 'r') as f:
    changes = json.load(f)
if changes:
    latest = changes[-1]
    if any(latest.get('endpoints', {}).values()) or any(latest.get('schemas', {}).values()):
        print('CHANGES_DETECTED')
        print(f'Summary: {latest[\"summary\"]}')
        print(f'Version: {latest[\"version_change\"][\"old\"]} → {latest[\"version_change\"][\"new\"]}')
    else:
        print('NO_CHANGES')
else:
    print('NO_CHANGES')
")
        if [[ "$LATEST_CHANGE" == *"CHANGES_DETECTED"* ]]; then
            HAS_API_CHANGES=true
        fi
    fi
    
    # Check for changelog changes
    if [ -f "docs/changelog_changes.json" ]; then
        CHANGELOG_CHANGE=$(python -c "
import json
with open('docs/changelog_changes.json', 'r') as f:
    changes = json.load(f)
if changes:
    latest = changes[-1]
    api_entries = latest.get('api_related_entries', [])
    high_severity = latest.get('high_severity_entries', [])
    if api_entries or high_severity:
        print('CHANGELOG_CHANGES_DETECTED')
        print(f'Summary: {latest[\"summary\"]}')
        print(f'API-related entries: {len(api_entries)}')
        print(f'High-severity entries: {len(high_severity)}')
    else:
        print('NO_CHANGELOG_CHANGES')
else:
    print('NO_CHANGELOG_CHANGES')
")
        if [[ "$CHANGELOG_CHANGE" == *"CHANGELOG_CHANGES_DETECTED"* ]]; then
            HAS_CHANGELOG_CHANGES=true
        fi
    fi
    
    # Handle detected changes
    if [[ "$HAS_API_CHANGES" == "true" ]] || [[ "$HAS_CHANGELOG_CHANGES" == "true" ]]; then
        echo "🚨 Changes detected!"
        
        if [[ "$HAS_API_CHANGES" == "true" ]]; then
            echo "📋 API Spec Changes:"
            echo "$LATEST_CHANGE"
        fi
        
        if [[ "$HAS_CHANGELOG_CHANGES" == "true" ]]; then
            echo "📰 Changelog Changes:"
            echo "$CHANGELOG_CHANGE"
        fi
        
        # Generate code recommendations if API changes detected
        if [[ "$HAS_API_CHANGES" == "true" ]]; then
            echo "🤖 Generating code update recommendations..."
            python scripts/generate-endpoint.py
        fi
        
        # Send notification (customize as needed)
        echo "📧 Changes detected - check docs/ for details"
        
        # Optionally commit changes
        if [[ "${AUTO_COMMIT:-false}" == "true" ]]; then
            FILES_TO_ADD="docs/"
            
            if [[ "$HAS_API_CHANGES" == "true" ]]; then
                FILES_TO_ADD="$FILES_TO_ADD docs/swagger.yaml docs/api_changes.json docs/api_update_report.md"
            fi
            
            if [[ "$HAS_CHANGELOG_CHANGES" == "true" ]]; then
                FILES_TO_ADD="$FILES_TO_ADD docs/changelog_history.json docs/changelog_changes.json"
            fi
            
            git add $FILES_TO_ADD
            
            COMMIT_MSG="🤖 Auto-update Venice.ai monitoring data

"
            if [[ "$HAS_API_CHANGES" == "true" ]]; then
                COMMIT_MSG="$COMMIT_MSG$(echo "$LATEST_CHANGE" | tail -n +2)

"
            fi
            
            if [[ "$HAS_CHANGELOG_CHANGES" == "true" ]]; then
                COMMIT_MSG="$COMMIT_MSG$(echo "$CHANGELOG_CHANGE" | tail -n +2)

"
            fi
            
            git commit -m "$COMMIT_MSG🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
            echo "✅ Changes committed to git"
        fi
    else
        echo "💤 No significant changes detected"
    fi
else
    echo "❌ API monitoring failed"
    exit 1
fi

echo "🎯 Monitoring complete: $(date)"