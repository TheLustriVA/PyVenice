# Next Session Task List

## Priority 1: Separate User vs Maintainer Concerns

### **Remove User-Facing Automation** 
- [ ] Remove automation scripts from CI/CD workflows that require user API keys
- [ ] Move maintainer scripts to `maintainer/` or `scripts/maintainer/` directory
- [ ] Update README.md to remove references to user needing to run monitoring scripts
- [ ] Remove `VENICE_API_KEY` requirements from user-facing documentation

### **Add Proper Version/Compatibility Checking**
- [ ] Implement API version detection in client initialization
- [ ] Add version compatibility matrix to library
- [ ] Create clear error messages for version mismatches
- [ ] Add graceful degradation for minor API changes
- [ ] Test version checking with simulated old/new API scenarios

### **Clean Up User Experience**
- [ ] Remove user-facing references to model auditing/updating
- [ ] Update installation instructions to remove maintainer dependencies
- [ ] Simplify README to focus on library usage, not maintenance
- [ ] Create separate MAINTAINER.md for automation documentation

## Priority 2: Fix Immediate User Blockers

### **Dependencies & Installation**
- [ ] Add missing dependencies (`playwright`, `beautifulsoup4`) to optional `[maintainer]` extra
- [ ] Test clean installation on fresh environment
- [ ] Fix executable permissions on user-facing scripts (if any remain)
- [ ] Create proper async usage examples in README

### **API Client Improvements**
- [ ] Implement remaining 6 missing endpoints for completeness
- [ ] Add retry logic with exponential backoff
- [ ] Improve error handling consistency across all endpoints
- [ ] Add model compatibility checking before API calls

## Priority 3: Polish User Experience

### **Documentation**
- [ ] Create comprehensive async examples
- [ ] Add streaming examples for chat/audio
- [ ] Document model compatibility matrix
- [ ] Add troubleshooting section for common issues

### **Developer Experience**
- [ ] Add proper progress indicators for long operations
- [ ] Implement caching for expensive calls (models list)
- [ ] Create configuration file support
- [ ] Add comprehensive type hints and validation

## Maintainer-Only Features (Keep Separate)

### **Automation System** (Your Tools Only)
- [ ] Move all monitoring scripts to maintainer-only directory
- [ ] Document maintainer workflow separately
- [ ] Keep zero-review automation for your productivity
- [ ] Maintain CI/CD automation for releases

### **API Maintenance**
- [ ] Continue monitoring Venice.ai API changes
- [ ] Automated model updates when API evolves
- [ ] Release automation to PyPI when changes detected
- [ ] Maintain backwards compatibility where possible

## Architecture Principle

**Users**: Install library → Use library → Get clear errors if version mismatch → Upgrade when needed
**Maintainer**: Monitor API → Update library → Release new version → Users benefit automatically

The automation system is **your** productivity tool, not something users interact with.