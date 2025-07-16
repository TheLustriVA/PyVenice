# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Setup Checklist

**ALWAYS verify when starting work on a new project:**

- [ ] GRIMOIRE.md symlinked: `ls -la GRIMOIRE.md` (should show symlink to ~/code/central_projections/GRIMOIRE.md)
- [ ] If missing: `ln -s ~/code/central_projections/GRIMOIRE.md ./GRIMOIRE.md`
- [ ] Remind user if symlink is missing or broken

## Project Overview

"pyvenice" - A comprehensive Python client library for the Venice.ai API, providing a wrapper for all API endpoints with type safety, automatic validation, and convenience features.

## User vs Maintainer Separation

**IMPORTANT**: This library has two distinct user groups with different needs:

### **End Users** (Library Consumers)

- Install via `pip install pyvenice`
- Use the library for Venice.ai API integration
- Get clear error messages if version incompatible with API
- Upgrade library when needed: `pip install --upgrade pyvenice`
- **Do NOT need**: API monitoring, model updating, or maintainer automation

### **Maintainer** (Library Developer)

- Monitor Venice.ai API changes via automation scripts
- Update PyVenice when API evolves
- Release new versions to PyPI
- Handle all API compatibility complexity internally
- **Uses**: All the automation scripts and monitoring tools

## Development Roadmap

**Note**: Comprehensive roadmap was moved to central project repository.

- Documentation and wiki development plans
- Feature enhancement priorities  
- Community growth strategies
- Long-term vision for agentic frameworks
- Realistic timeframes for all major milestones

## Key Architecture

The library implements a **decorator-based validation pattern** that requires understanding across multiple files:

1. **validators.py** defines `@validate_model_params` decorator that intercepts API calls
2. **models.py** provides model capability checking via `get_model_traits()`
3. **Endpoint modules** (chat.py, image.py, etc.) use the decorator to automatically filter unsupported parameters before API calls
4. This prevents API errors when users pass parameters like `parallel_tool_calls` to models that don't support them

## Distribution Infrastructure (v0.2.0+)

**Multi-Platform Support**: Professional CI/CD pipeline with ARM64 wheel building solves Termux/Android installation issues:

- GitHub Actions builds wheels for Linux/macOS/Windows on x86_64 and ARM64
- Docker images published to GitHub Container Registry with multi-arch support
- Conda-forge recipe ready for scientific Python ecosystem submission

**Security Framework**: Automated vulnerability scanning and structured threat modeling:

- Daily security scans with Safety, Bandit, Semgrep, pip-audit
- Tiered security assessment (individual/corporate/nation-state threats)
- Professional security documentation and incident response procedures

## Essential Commands

### **For End Users** (Library Development)

```bash
# Development setup
source .venv/bin/activate      # Activate uv virtual environment
pip install -e .[dev]         # Include test dependencies

# Code quality
ruff check .                   # Run linting checks
ruff check --fix .            # Fix auto-fixable linting issues
pytest tests/ -m "not integration" --cov=pyvenice --cov-report=term-missing  # Unit tests with coverage

# Testing
pytest tests/test_chat.py -v  # Run specific test module
pytest --cov=pyvenice --cov-report=html  # Generate HTML coverage report

# Manual testing
export VENICE_API_KEY='your-api-key'
python src/simple_example.py  # Basic functionality test
python src/example_usage.py   # Comprehensive API demonstration
```

### **For Maintainer Only** (API Maintenance)

```bash
# Security scanning
scripts/security-scan.sh      # Local security audit (Safety, Bandit, Semgrep)

# API monitoring and automated maintenance (v0.3.0)
scripts/api-monitor.py         # Check for Venice.ai API changes
scripts/api-monitor.py --dry-run    # Check without saving changes
scripts/schema-diff.py --old docs/swagger_old.yaml --new docs/swagger.yaml  # Detailed schema comparison
scripts/generate-endpoint.py  # Generate discrete AI code update tasks
scripts/daily-monitor.sh       # Full monitoring workflow for cron

# Safety and validation systems
scripts/safety-validator.py   # Multi-layered safety validation with backup/restore
scripts/ci-feedback.py --check-safety  # Check CI/CD status for deployment safety
scripts/api-contract-validator.py      # Test API compatibility and backwards compat
scripts/dead-code-detector.py          # Find unused code for cleanup
scripts/docs-scraper.py               # Scrape Venice.ai docs for enhanced validation

# Automated deployment (ADHD/PTSD-optimized)
scripts/safe-auto-deploy.py   # Fully automated safe deployment pipeline
scripts/safe-auto-deploy.py --dry-run  # Simulate deployment without changes

# Distribution testing
cibuildwheel --platform linux --archs x86_64  # Test wheel building locally
docker build -t pyvenice:test .               # Test Docker image build
```

## Development Workflow

1. **Before modifying endpoints**: Check if the parameter needs validation in `validators.py`
2. **When adding new endpoints**: Follow the pattern in existing modules (e.g., chat.py) - create a class with sync/async methods
3. **For streaming endpoints**: See `chat.py` and `audio.py` for the streaming response handling pattern
4. **Testing changes**: Always run the test suite - the project maintains 82% coverage minimum

## Critical Implementation Details

### Known Issues & Workarounds

1. **Brotli Compression**: Venice.ai supports `br` encoding but httpx has decoding issues. The client only requests `gzip` compression in `client.py:59`

2. **Parameter Defaults**: Avoid default values for optional parameters (e.g., `parallel_tool_calls`) as they'll always be sent. Use `None` defaults in `chat.py:176`

3. **Compatibility Mapping**: The `/models/compatibility_mapping` endpoint returns a dict, not a list. See `models.py:170` for correct handling

### Cross-Module Dependencies

- **All endpoint modules** depend on the `@validate_model_params` decorator from `validators.py`
- **validators.py** depends on `models.py` for capability checking
- **models.py** caches are shared across all endpoint instances via the client
- **Streaming responses** in chat/audio require special handling of SSE format and chunk processing

### Project Status & Notes

- **v0.3.0 Release**: COMPLETE AUTOMATED API MAINTENANCE SYSTEM (Maintainer-Only)
- **User-Maintainer Separation**: Clear distinction between end-user library and maintainer automation
- **ADHD/PTSD-Optimized**: Maintainer automation designed for solo dev with review limitations
- **Zero-Manual-Review Pipeline**: Comprehensive safety validation without human review requirement
- **Multi-Layered Safety**: 6+ independent validation systems with automatic rollback
- **AI-Powered Code Generation**: Discrete additive tasks optimized for AI strengths
- **Complete Observability**: CI/CD feedback loops, failure classification, audit trails
- **Production-Ready Automation**: Can run unattended with `AUTO_COMMIT=true`

**Architecture Philosophy**:

- **End Users**: Install library → Use library → Get clear errors if incompatible → Upgrade when needed
- **Maintainer**: Monitor API → Update library → Release new version → Users benefit automatically

**Previous Infrastructure (v0.2.0)**:

- **ARM64 Support**: Wheel building solves Android/Termux installation issues  
- **Security Framework**: Automated scanning with tiered threat assessment
- **Distribution Ready**: Conda-forge recipe prepared, Docker images published
- **Banner added**: README.md (src/pyvenice_oldschool_banner.png)

## File Modification Best Practices

**Follow these 3 rules to avoid edit loops and file corruption:**

### Rule 1: Read First, Edit Second

- **Always use `Read` tool** to understand context before any `Edit`
- **Check ±10 lines** around target location for conflicts
- **Understand the structure** before making changes

### Rule 2: Verify Target Uniqueness  

- **Use `grep -n "target_string" file`** to check uniqueness before Edit
- **If multiple matches**: Use more context or different strategy
- **Tool selection**:
  - `Edit` - Only for unique targeting strings
  - `Bash` append (`>>`) - For end-of-file additions
  - `Write` - For complete file rewrites (rare)

### Rule 3: Verify Immediately After Changes

- **Syntax check**: `python -c "import module"` right after edit
- **Placement check**: `grep -n "new_code"` to confirm location  
- **Duplicate check**: `grep -c "function_name"` should equal 1
- **If ANY check fails**: `git checkout -- file` and restart

### Emergency Recovery

- **Before risky edits**: `git stash push -m "backup"`
- **If stuck in loop**: `git checkout -- file` to reset
- **Nuclear option**: `git stash pop` to restore backup

**Core insight**: 2x ceremony prevents 4x rework = 2x faster overall

**For process improvements**: Document in PROCEDURES.md immediately while solution is fresh

## Automated API Maintenance System (v0.3.0) - MAINTAINER ONLY

### Overview

Complete zero-manual-review pipeline for handling Venice.ai API changes. **This is maintainer-only infrastructure** - end users of the library don't need to interact with this system. Designed specifically for developers with ADHD/PTSD who cannot reliably review generated code. System prioritizes safety through automation rather than human oversight.

### System Architecture

**1. Detection Layer**

- `api-monitor.py`: Parameter-level change tracking with version history
- `docs-scraper.py`: Enhanced data from Venice.ai documentation pages
- `schema-diff.py`: Detailed analysis of specific changes between versions

**2. Safety Validation Layer**

- `safety-validator.py`: Multi-layered validation (syntax, imports, tests, linting, API compatibility)
- `api-contract-validator.py`: Live API testing with backwards compatibility verification
- `dead-code-detector.py`: Identifies safe-to-remove unused code
- `ci-feedback.py`: Real-time CI/CD monitoring with failure classification

**3. Code Generation Layer**

- `generate-endpoint.py`: AI-optimized discrete tasks (additive vs modification detection)
- Deprecation system with graceful parameter removal
- Automatic backup creation for rollback safety

**4. Deployment Layer**

- `safe-auto-deploy.py`: Complete pipeline with branch isolation and automatic rollback
- CI/CD integration with workflow completion monitoring
- Comprehensive audit trails for debugging

### Safety Philosophy

**No Manual Review Required**: System designed around the constraint that generated code cannot be reliably reviewed by the developer. Safety comes from:

1. **Conservative Automation**: Only auto-adds optional parameters, never removes functionality
2. **Multiple Independent Validation**: 6+ separate safety checks that must all pass
3. **Automatic Rollback**: System restores from backup on any validation failure
4. **Branch Isolation**: All changes tested in isolation before merging to main
5. **CI/CD Validation**: Waits for all workflows to pass before proceeding

**Risk Mitigation**:

- Syntax errors caught by AST parsing
- Import errors caught by test imports
- API compatibility validated against live endpoints
- Backwards compatibility tested with existing client code
- Dead code removal only after comprehensive usage analysis

### Usage Patterns

**Daily Automated Run** (Recommended):

```bash
# Set up daily cron job
0 9 * * * cd /path/to/project && python scripts/safe-auto-deploy.py

# Or run manually
python scripts/safe-auto-deploy.py
```

**Manual Validation** (When needed):

```bash
# Check if it's safe to deploy
python scripts/ci-feedback.py --check-safety

# Validate current state
python scripts/safety-validator.py

# Test API compatibility
python scripts/api-contract-validator.py
```

**Emergency Procedures**:

```bash
# If deployment fails, backup location is logged
python scripts/safety-validator.py --restore /tmp/backup_path

# Reset to last known good state
git reset --hard HEAD~1
python scripts/safe-auto-deploy.py
```

### Failure Handling

**Automatic Rollback Triggers**:

- Any syntax error in generated code
- Import failures after code changes  
- Test suite failures
- Linting errors (not warnings)
- API compatibility test failures
- CI/CD workflow failures

**Failure Classification**:

- `test_failure`: Code broke existing functionality
- `syntax_error`: Generated code has syntax issues
- `import_error`: Dependency or module issues
- `api_incompatible`: Changes break API contract
- `timeout`: Validation took too long

### Key Files and Data

**Generated Artifacts**:

- `docs/api_changes.json`: Historical change log with parameter tracking
- `docs/api_update_report.md`: Human-readable change analysis
- `docs/deprecated_params.json`: Deprecation configuration
- `docs/last_deployment_report.md`: Latest deployment audit trail

**Backup Locations**:

- `/tmp/pyvenice_backup_*`: Safety backups (auto-cleaned after success)
- `/tmp/claude_prompt_*.md`: Generated AI tasks for manual review if needed

### Integration Points

**GitHub Actions**: `.github/workflows/api-monitor.yml`

- Daily automated monitoring
- Issue creation for detected changes
- Automatic PR creation with safety validations

**Local Development**: `scripts/daily-monitor.sh`

- Cron-compatible monitoring script
- Email/notification integration points
- Manual override capabilities

**CI/CD Feedback**: Real-time workflow monitoring

- Success rate tracking
- Failure pattern analysis
- Go/no-go deployment decisions

### Troubleshooting

**System Not Running**:

1. Check prerequisites: `python scripts/safe-auto-deploy.py` (will validate tools)
2. Verify API key: `echo $VENICE_API_KEY`
3. Check CI/CD status: `python scripts/ci-feedback.py --check-safety`

**Validation Failures**:

1. Check latest report: `cat docs/last_deployment_report.md`
2. Review specific failure: `python scripts/safety-validator.py`
3. Test manual API compatibility: `python scripts/api-contract-validator.py`

**Code Generation Issues**:

1. Check generated tasks: `ls /tmp/claude_prompt_*.md`
2. Manual task execution: `claude-code "$(cat /tmp/claude_prompt_1.md)"`
3. Validate result: `python scripts/safety-validator.py`

### Performance Characteristics

**Runtime**: Full pipeline takes 5-15 minutes depending on:

- Number of API changes detected
- Test suite execution time  
- CI/CD workflow completion time

**Resource Usage**: Minimal - primarily I/O bound

- Backup storage: ~50MB per deployment
- Network: API calls for monitoring and validation
- CPU: Test suite and validation processes

**Reliability**: Designed for 99%+ success rate on valid API changes

- False positives (blocks valid changes): <1%
- False negatives (allows bad changes): <0.1% target

This system represents a complete solution for the "ADHD developer cannot review code" constraint while maintaining production-quality safety standards.
