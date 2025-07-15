# PyPI Trusted Publisher Configuration

## Problem
The GitHub Actions workflow is failing to publish to PyPI because the trusted publisher is not configured.

## Error Message
```
* `invalid-publisher`: valid token, but no corresponding publisher (Publisher with matching claims was not found)
```

## Solution: Configure PyPI Trusted Publisher

### Step 1: Go to PyPI Project Settings
1. Go to https://pypi.org/manage/project/pyvenice/settings/
2. Login with your PyPI account that has maintainer access

### Step 2: Configure Trusted Publisher
1. Scroll down to "Trusted publishers"
2. Click "Add a new trusted publisher"
3. Fill in these exact details:
   - **Repository name**: `TheLustriVA/PyVenice`
   - **Workflow filename**: `.github/workflows/build-wheels.yml`
   - **Environment name**: `pypi`
   - **Repository owner**: `TheLustriVA`

### Step 3: Save Configuration
Click "Add trusted publisher" to save the configuration.

### Step 4: Test the Configuration
1. Create a new tag and push it:
   ```bash
   git tag v0.3.1
   git push origin v0.3.1
   ```
2. Check the GitHub Actions run to verify PyPI upload works

## Alternative: Manual Upload

If trusted publishing setup is not possible, you can manually upload releases:

```bash
# Build the package
python -m build

# Upload to PyPI
twine upload dist/*
```

## Security Note
Trusted publishing is the recommended approach as it eliminates the need for PyPI tokens in GitHub secrets.

## Claims from Failed Attempt
For reference, the claims from the failed attempt were:
- `sub`: `repo:TheLustriVA/PyVenice:environment:pypi`
- `repository`: `TheLustriVA/PyVenice`
- `repository_owner`: `TheLustriVA`
- `workflow_ref`: `TheLustriVA/PyVenice/.github/workflows/build-wheels.yml@refs/tags/v0.3.0`
- `ref`: `refs/tags/v0.3.0`

These match the configuration requirements above.