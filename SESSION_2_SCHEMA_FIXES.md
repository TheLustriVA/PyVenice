# Session 2: Schema Fixes & Integration Tests

## Context from Previous Sessions

**Session 1 Results**: Foundation dependencies resolved, API specification updated to v20250713.224148

## Prerequisites

Before starting this session, verify Session 1 was completed successfully:
```bash
# Verify you're in the correct directory
cd /home/websinthe/code/pyvenice

# Verify virtual environment and dependencies
source .venv/bin/activate
python -c "import bs4; print('✅ bs4 available')"
which ruff && echo "✅ ruff available"

# Verify API version was updated
grep "version:" docs/swagger.yaml | grep "20250713.224148" && echo "✅ API version current"
```

## Goals

1. **Fix Integration Test Failures**: Resolve failing billing and embeddings tests
2. **Update Schema Models**: Align request/response models with API v20250713.224148
3. **Verify Parameter Validation**: Ensure parameter filtering works with new schemas
4. **Run Complete Test Suite**: Achieve 100% integration test pass rate

## Context from API Audit

### Integration Test Failures Identified:
1. **Billing Usage Test**: `tests/test_billing.py::TestBillingIntegration::test_real_usage_retrieval`
2. **Embeddings Dimensions Test**: `tests/test_embeddings.py::TestEmbeddingsIntegration::test_real_embeddings_with_dimensions`

### Schema Changes Detected:
- `BillingUsageRequest` - Modified parameters
- `BillingUsageResponse` - Modified response structure  
- `ChatCompletionRequest` - Parameter updates
- `GenerateImageRequest` - New parameters
- `ModelResponse` - Structure changes

## Detailed Tasks

### Task 1: Analyze Current Integration Test Failures

```bash
# Run integration tests to see current failures
source .venv/bin/activate
pytest tests/test_billing.py::TestBillingIntegration::test_real_usage_retrieval -v
pytest tests/test_embeddings.py::TestEmbeddingsIntegration::test_real_embeddings_with_dimensions -v
```

### Task 2: Compare Current vs Updated API Schemas

```bash
# Check what parameters are available in current API for billing
python -c "
import json
import yaml
with open('docs/swagger.yaml') as f:
    spec = yaml.safe_load(f)
    
# Look for BillingUsageRequest and BillingUsageResponse schemas
schemas = spec.get('components', {}).get('schemas', {})
for schema_name in ['BillingUsageRequest', 'BillingUsageResponse']:
    if schema_name in schemas:
        print(f'{schema_name}:')
        print(json.dumps(schemas[schema_name], indent=2))
"
```

### Task 3: Update Billing Schema Models

**File**: `pyvenice/billing.py`

**Current Issues**: Integration test suggests parameter mismatch

**Actions**:
1. Read current billing.py implementation
2. Compare with updated swagger.yaml schema
3. Update BillingUsageRequest parameters to match API spec
4. Update BillingUsageResponse parsing to match API response
5. Update parameter validation and filtering

**Pattern to Follow**:
```python
# Check existing billing.py structure
head -50 pyvenice/billing.py

# Look for the get_usage method signature
grep -n "def get_usage" pyvenice/billing.py -A 10
```

### Task 4: Update Embeddings Schema Models

**File**: `pyvenice/embeddings.py`

**Current Issues**: Dimension parameter test failure

**Actions**:
1. Read current embeddings.py implementation  
2. Compare with updated swagger.yaml schema
3. Update CreateEmbeddingRequest parameters
4. Fix dimension parameter handling
5. Update response parsing

**Pattern to Follow**:
```python
# Check existing embeddings.py structure
head -50 pyvenice/embeddings.py

# Look for dimension-related parameters
grep -n "dimension" pyvenice/embeddings.py
```

### Task 5: Update Chat Completion Schema

**File**: `pyvenice/chat.py`

**Potential Issues**: Parameter updates in ChatCompletionRequest

**Actions**:
1. Check ChatCompletionRequest model in chat.py
2. Compare with swagger.yaml schema
3. Update parameter definitions and validation
4. Test with existing chat functionality

### Task 6: Update Image Generation Schema

**File**: `pyvenice/image.py`

**Potential Issues**: New parameters in GenerateImageRequest

**Actions**:
1. Check GenerateImageRequest model in image.py
2. Compare with swagger.yaml schema  
3. Add any new parameters
4. Update validation logic

### Task 7: Update Model Response Schema

**File**: `pyvenice/models.py`

**Potential Issues**: ModelResponse structure changes

**Actions**:
1. Check ModelResponse parsing in models.py
2. Compare with swagger.yaml schema
3. Update response handling
4. Test model listing functionality

### Task 8: Run Comprehensive Testing

```bash
# Run all integration tests
pytest tests/ -m integration -v

# Run specific failed tests
pytest tests/test_billing.py::TestBillingIntegration::test_real_usage_retrieval -v
pytest tests/test_embeddings.py::TestEmbeddingsIntegration::test_real_embeddings_with_dimensions -v

# Run unit tests to ensure no regression
pytest tests/ -m "not integration" --cov=pyvenice --cov-report=term-missing

# Run API contract validation
python scripts/api-contract-validator.py
```

## Success Criteria

Before proceeding to Session 3, verify:

1. **Integration Tests Pass**:
   - `pytest tests/test_billing.py -m integration` passes
   - `pytest tests/test_embeddings.py -m integration` passes
   - All other integration tests continue to pass

2. **Schema Validation Works**:
   - `python scripts/api-contract-validator.py` shows all tests passing
   - No schema mismatch errors in API calls

3. **Unit Tests Stable**:
   - Unit test coverage remains at 80%+ 
   - No regressions in existing functionality

4. **Code Quality Maintained**:
   - `ruff check .` passes without errors
   - No new linting issues introduced

## Expected File Changes

After successful completion:
- `pyvenice/billing.py` - Updated parameter handling and response parsing
- `pyvenice/embeddings.py` - Fixed dimension parameter support  
- `pyvenice/chat.py` - Updated ChatCompletionRequest parameters
- `pyvenice/image.py` - Added new GenerateImageRequest parameters
- `pyvenice/models.py` - Updated ModelResponse handling

## Handoff to Session 3

**Status**: Schema fixes completed, integration tests passing

**Next Session Focus**: Implement new `/image/edit` endpoint

**Key Information for Session 3**:
- All existing schemas are now current with API v20250713.224148
- Integration tests are passing (stable foundation)
- New endpoint `/image/edit` needs implementation
- Follow patterns from existing image.py endpoints

**Files to Focus On**:
- `pyvenice/image.py` - Add edit() method and EditImageRequest model
- `tests/test_image.py` - Add comprehensive tests for edit functionality
- `docs/swagger.yaml` - Reference for EditImageRequest schema

## Troubleshooting

**If billing tests still fail**:
- Check parameter names match exactly between billing.py and swagger.yaml
- Verify response parsing handles new response structure
- Test with a simple API call to see actual vs expected structure

**If embeddings tests still fail**:
- Check if dimension parameter is properly supported by the model
- Verify parameter is correctly passed through validation
- Test with different models to isolate the issue

**If schema validation fails**:
- Compare parameter names in code vs swagger.yaml (case sensitivity)
- Check for missing required parameters
- Verify optional parameters are handled correctly

## Validation Commands

```bash
# Final validation before ending session
source .venv/bin/activate

# Test specific integration issues
pytest tests/test_billing.py::TestBillingIntegration::test_real_usage_retrieval -v
pytest tests/test_embeddings.py::TestEmbeddingsIntegration::test_real_embeddings_with_dimensions -v

# Test all integration tests
pytest tests/ -m integration | grep -E "(PASSED|FAILED|ERROR)" | tail -10

# Test API contract validation
python scripts/api-contract-validator.py | grep -E "(PASS|FAIL)" | tail -10

# Test code quality
ruff check . && echo "✅ Code quality maintained"
```

If all validation commands pass, Session 2 is complete and ready for Session 3.