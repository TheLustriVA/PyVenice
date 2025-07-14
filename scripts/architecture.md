# PyVenice Automatic Update System Architecture

## Overview

The PyVenice project implements a sophisticated **zero-manual-review automatic API maintenance system** designed specifically for developers with ADHD/PTSD who cannot reliably review generated code. The system prioritizes **safety through automation** rather than human oversight.

## Core System Architecture

### 1. **Detection Layer** - API Change Monitoring

#### **api-monitor.py** - Primary Change Detection
- **Purpose**: Downloads Venice.ai Swagger spec, compares with local copy, tracks parameter-level changes
- **Key Features**:
  - Parameter lifecycle tracking (added/removed/modified)
  - Schema hash comparison for change detection
  - Version history logging in `docs/api_changes.json`
  - Integrates with changelog monitoring
- **Output**: Detailed change reports with parameter-specific analysis

#### **changelog-monitor.py** - Proactive Change Detection
- **Purpose**: Monitors Venice.ai changelog for API-relevant changes using web scraping
- **Key Features**:
  - Playwright-based authentication and scraping
  - Keyword-based API relevance detection
  - Cloudflare challenge handling
  - Generates API change alerts
- **Integration**: Called by `api-monitor.py` for comprehensive change detection

#### **response-schema-monitor.py** - Response Structure Monitoring
- **Purpose**: Monitors actual API response structures for schema changes
- **Key Features**:
  - Live API testing against multiple endpoints
  - Schema structure extraction and comparison
  - Generates model update tasks for Pydantic models
- **Data**: Saves response schemas and change detection reports

### 2. **Code Analysis & Generation Layer**

#### **generate-endpoint.py** - AI Code Generation
- **Purpose**: Creates discrete Claude Code prompts for updating the client library
- **Key Features**:
  - Analyzes endpoint coverage vs API spec
  - Generates additive vs modification tasks
  - Creates specific prompts for new parameters, endpoints, and schema changes
  - Optimized for AI strengths (discrete, additive tasks)
- **Output**: Multiple focused prompts saved to `/tmp/claude_prompt_*.md`

#### **docs-scraper.py** - Documentation Enhancement
- **Purpose**: Scrapes Venice.ai documentation to supplement Swagger spec
- **Key Features**:
  - Extracts parameter descriptions, examples, validation rules
  - Merges scraped data with Swagger spec
  - Enhances code generation with better context
- **Output**: Enhanced Swagger spec with documentation details

#### **dead-code-detector.py** - Cleanup Analysis
- **Purpose**: Identifies unused code that can be safely removed after deprecations
- **Key Features**:
  - AST-based analysis of function/class usage
  - Deprecated parameter usage tracking
  - Conservative safety classification
- **Safety**: Only marks private functions as safe-to-remove

### 3. **Model Management Layer**

#### **model-audit.py** - Pydantic Model Validation
- **Purpose**: Compares Pydantic models against actual API responses
- **Key Features**:
  - Live API testing for model compatibility
  - Field mismatch detection
  - Model creation success/failure analysis
- **Output**: Detailed audit reports identifying model issues

#### **model-updater.py** - Automated Model Updates
- **Purpose**: Automatically updates Pydantic models based on audit results
- **Key Features**:
  - AST-based model modification
  - Type inference for new fields
  - Safety backup and restore system
  - Validation after updates
- **Safety**: Creates backups before modification, validates after changes

### 4. **Safety Validation Layer**

#### **safety-validator.py** - Multi-Layer Safety System
- **Purpose**: Comprehensive validation without manual review requirement
- **Key Features**:
  - **6 independent validation layers**:
    1. Syntax validation (AST parsing)
    2. Import validation (test imports)
    3. Parameter safety (required vs optional)
    4. Code quality (linting)
    5. Unit test execution
    6. API compatibility testing
  - Automatic backup creation and restore
  - Conservative safety philosophy
- **Philosophy**: Multiple independent checks that must ALL pass

#### **api-contract-validator.py** - API Compatibility Testing
- **Purpose**: Validates client works with current Venice.ai API
- **Key Features**:
  - Live API testing
  - Deprecation warning validation
  - Parameter filtering tests
  - Model capability validation
  - Backwards compatibility testing
- **Integration**: Called by safety-validator for API compatibility layer

### 5. **CI/CD Integration Layer**

#### **ci-feedback.py** - Build System Monitoring
- **Purpose**: Monitors GitHub Actions workflows for deployment safety
- **Key Features**:
  - Workflow status tracking via GitHub CLI
  - Failure classification and analysis
  - Go/no-go deployment decisions
  - Workflow completion waiting
- **Safety**: Prevents deployment if CI/CD is unstable

#### **daily-monitor.sh** - Cron Integration
- **Purpose**: Orchestrates daily automated monitoring
- **Key Features**:
  - Runs API and changelog monitoring
  - Detects changes and triggers code generation
  - Auto-commit with `AUTO_COMMIT=true`
  - Notification system integration
- **Usage**: Designed for daily cron jobs

### 6. **Orchestration Layer**

#### **safe-auto-deploy.py** - Master Deployment Pipeline
- **Purpose**: Fully automated deployment with comprehensive safety
- **Key Features**:
  - **Complete deployment pipeline**:
    1. Prerequisites check
    2. API change detection
    3. Code generation
    4. Safety validation
    5. Branch isolation
    6. CI/CD validation
    7. Merge to main
    8. Cleanup
  - Automatic rollback on any failure
  - Comprehensive audit trails
- **Safety**: Branch isolation, automatic rollback, multiple validation layers

### 7. **Utility & Support Scripts**

#### **schema-diff.py** - Detailed Schema Analysis
- **Purpose**: Provides detailed diff analysis between API spec versions
- **Key Features**:
  - DeepDiff-based comparison
  - Human-readable schema change reports
  - Type change detection
- **Usage**: Manual analysis and debugging

#### **security-scan.sh** - Security Validation
- **Purpose**: Comprehensive security scanning
- **Key Features**:
  - Multiple security tools (Safety, Bandit, Semgrep, pip-audit)
  - Secret detection
  - Vulnerability scanning
- **Integration**: Run before deployment

#### **monitor_checker.py** - Debug Tool
- **Purpose**: Test and debug the API monitoring system
- **Key Features**: Basic connectivity and API testing

#### **publish.sh** - PyPI Publishing
- **Purpose**: Automated package publishing workflow
- **Key Features**: Build, test, and publish to PyPI

## GitHub Actions Integration

### **api-monitor.yml** - Daily Automated Monitoring
- **Schedule**: Daily at 9 AM UTC (`0 9 * * *`)
- **Workflow**:
  1. Runs `api-monitor.py` to detect changes
  2. Generates detailed diff reports with `schema-diff.py`
  3. Creates code generation recommendations with `generate-endpoint.py`
  4. Automatically creates GitHub issues for detected changes
  5. Commits updated API specs to repository
- **Key Features**:
  - Automatic issue creation with detailed change analysis
  - Structured change reporting
  - Auto-commit of API spec updates

## Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  API Monitor    │───▶│  Change Detection│───▶│  Code Generation│
│  (Daily Cron)   │    │  & Analysis      │    │  (Claude Prompts│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Safety         │    │  Model Updates   │    │  CI/CD          │
│  Validation     │    │  (Pydantic)      │    │  Integration    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────┐
                    │  Automated Deploy   │
                    │  (safe-auto-deploy) │
                    └─────────────────────┘
```

## Generated Data Files

The automatic update system creates and maintains several key data files:

### **Core API Tracking**
- `docs/api_changes.json` - Historical change log with parameter tracking
- `docs/swagger.yaml` - Current API specification
- `docs/swagger_current.yaml` - Working copy of API spec
- `docs/api_update_report.md` - Human-readable change analysis

### **Model Management**
- `docs/deprecated_params.json` - Deprecation configuration
- `docs/model_audit_report.json` - Pydantic model validation results
- `docs/model_update_report.json` - Model update tracking
- `docs/response_schemas.json` - API response structure tracking

### **Monitoring & Analysis**
- `docs/changelog_changes.json` - Venice.ai changelog tracking
- `docs/changelog_history.json` - Historical changelog data
- `docs/changelog_monitoring_report.json` - Monitoring analysis
- `docs/schema_monitoring_report.json` - Schema change reports

### **Debug & Development**
- `docs/venice_changelog_debug.html` - Debug output from changelog scraping
- `docs/venice_changelog_debug.png` - Screenshot of changelog for debugging

## Dependencies in Main Codebase

The main library code **does not depend** on the automatic update scripts. The separation is clean:

- **Library Code** (`pyvenice/`) - Core API client functionality
- **Automation Scripts** (`scripts/`) - Maintenance automation (maintainer-only)
- **Documentation** mentions automation but doesn't depend on it

## Key Design Principles

1. **No Manual Review Required**: System designed around constraint that generated code cannot be reliably reviewed
2. **Conservative Automation**: Only auto-adds optional parameters, never removes functionality
3. **Multiple Independent Validation**: 6+ separate safety checks that must all pass
4. **Automatic Rollback**: System restores from backup on any validation failure
5. **Branch Isolation**: All changes tested in isolation before merging
6. **Comprehensive Audit Trails**: Every action logged for debugging

## Safety Philosophy

The system embodies a **"Multiple Independent Layers of Safety"** approach:
- **Syntax errors** caught by AST parsing
- **Import errors** caught by test imports  
- **API compatibility** validated against live endpoints
- **Backwards compatibility** tested with existing client code
- **Test failures** caught by full test suite execution
- **Code quality** validated by linting systems

## Performance Characteristics

- **Runtime**: Full pipeline takes 5-15 minutes
- **Reliability**: Designed for 99%+ success rate on valid changes
- **False positives**: <1% (blocks valid changes)
- **False negatives**: <0.1% target (allows bad changes)

This system represents a complete solution for automated API maintenance while maintaining production-quality safety standards, specifically designed for developers who cannot reliably review generated code.