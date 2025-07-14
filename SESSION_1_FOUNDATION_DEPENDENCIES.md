# Session 1: Foundation & Dependencies

## Context from Previous Audit

A comprehensive audit of the PyVenice API monitoring system revealed several critical issues that must be addressed before proceeding with feature development:

### Key Findings:
- **Local API Version**: v20250612.151220 (June 12, 2025)
- **Live API Version**: v20250713.224148 (July 13, 2025)
- **Version Gap**: ~1 month behind live API
- **Missing Dependencies**: `beautifulsoup4` (bs4), `ruff`
- **Monitoring System Effectiveness**: 65% (partially working)

### Critical Missing Dependencies:
1. **beautifulsoup4 (bs4)**: Required for changelog monitoring
2. **ruff**: Required for code quality checks during safety validation

### Current Repository Status:
- **Branch**: main
- **Working Directory**: `/home/websinthe/code/pyvenice`
- **Virtual Environment**: `.venv` (activated with `source .venv/bin/activate`)
- **API Key**: Available as `$VENICE_API_KEY` environment variable

## Prerequisites

This is Session 1 - no previous sessions required. However, ensure:
- You are in the correct directory: `/home/websinthe/code/pyvenice`
- Virtual environment is activated: `source .venv/bin/activate`
- API key is available: `echo $VENICE_API_KEY | head -c 10` should show first 10 chars

## Goals

1. **Install Missing Dependencies**: Resolve bs4 and ruff installation issues
2. **Update API Specification**: Sync local swagger.yaml to latest version (v20250713.224148)
3. **Verify Monitoring Scripts**: Ensure all monitoring scripts work with new dependencies
4. **Run API Synchronization**: Update local API specification to match live API

## Detailed Tasks

### Task 1: Install Missing Dependencies
```bash
# Activate virtual environment
source .venv/bin/activate

# Install missing dependencies
pip install beautifulsoup4 ruff

# Verify installations
python -c "import bs4; print('bs4 installed successfully')"
python -c "import ruff; print('ruff available')" || which ruff
```

### Task 2: Verify Current API Monitoring Status
```bash
# Check current API version in local swagger
grep -n "version:" docs/swagger.yaml | head -5

# Run API monitoring in dry-run mode to see what changes are detected
python scripts/api-monitor.py --dry-run
```

### Task 3: Update API Specification
```bash
# Run API monitoring to update local swagger.yaml
python scripts/api-monitor.py

# Verify the update worked
grep -n "version:" docs/swagger.yaml | head -5
# Should now show v20250713.224148
```

### Task 4: Verify Monitoring Scripts Functionality
```bash
# Test security scan (should no longer timeout on missing dependencies)
bash scripts/security-scan.sh

# Test safety validation (should no longer skip ruff checks)
python scripts/safety-validator.py

# Test API contract validation
python scripts/api-contract-validator.py
```

### Task 5: Run Code Quality Checks
```bash
# Run ruff linting (should work now)
ruff check .

# If there are any issues, fix them
ruff check --fix .
```

## Success Criteria

Before proceeding to Session 2, verify:

1. **Dependencies Installed**: 
   - `python -c "import bs4"` works without error
   - `which ruff` returns a path

2. **API Version Updated**:
   - `grep "version:" docs/swagger.yaml` shows `20250713.224148`

3. **Monitoring Scripts Working**:
   - `python scripts/api-monitor.py --dry-run` runs without dependency errors
   - `python scripts/safety-validator.py` doesn't skip ruff checks

4. **Code Quality Passes**:
   - `ruff check .` returns "All checks passed!" or shows only minor issues

5. **API Changes Detected**:
   - API monitoring should detect the new `/image/edit` endpoint
   - Should show schema changes in billing, embeddings, etc.

## Expected Outputs

After successful completion, you should see:
- Updated `docs/swagger.yaml` with version `20250713.224148`
- Updated `docs/api_changes.json` with latest changes
- Updated `docs/api_update_report.md` with current gaps
- All monitoring scripts running without dependency errors

## Handoff to Session 2

**Status**: Foundation dependencies resolved, API specification updated

**Next Session Focus**: Schema fixes and integration test resolution

**Key Information for Session 2**:
- API specification is now current (v20250713.224148)
- All monitoring tools are functional
- Integration test failures are in: billing usage, embeddings dimensions
- Schema mismatches to fix: BillingUsageRequest, BillingUsageResponse, ChatCompletionRequest, GenerateImageRequest, ModelResponse

**Files to Focus On**:
- `pyvenice/billing.py` - billing schema fixes
- `pyvenice/embeddings.py` - embeddings schema fixes  
- `tests/test_billing.py` - billing integration tests
- `tests/test_embeddings.py` - embeddings integration tests

## Notes

- If API monitoring fails, check network connectivity and API key validity
- If dependency installation fails, ensure you're in the correct virtual environment
- The API specification update may take a few minutes to complete
- Save any error messages for troubleshooting in next session if needed

## Validation Commands

```bash
# Final validation before ending session
source .venv/bin/activate
python -c "import bs4; print('✅ bs4 installed')"
which ruff && echo "✅ ruff installed"
grep "version:" docs/swagger.yaml | grep "20250713.224148" && echo "✅ API version updated"
python scripts/api-monitor.py --dry-run | grep -E "image/edit|New Endpoints" && echo "✅ New endpoints detected"
ruff check . && echo "✅ Code quality checks passed"
```

If all validation commands pass, Session 1 is complete and ready for Session 2.