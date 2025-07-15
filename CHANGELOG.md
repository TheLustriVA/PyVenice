# Changelog

All notable changes to PyVenice will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-07-15

### Added
- **New Endpoint**: POST /image/edit - Edit images with text prompts
- **API Key Management**: Create, delete, and manage API keys programmatically
- **Web3 Integration**: Get Web3 tokens for wallet-based authentication
- **Automated Monitoring**: Daily API monitoring with auto-deployment
- **Enhanced Testing**: Comprehensive integration test suite
- **Monitoring Reports**: Automated system health reporting

### Fixed
- **Schema Synchronization**: Updated all request/response models to API v20250713.224148
- **Integration Tests**: Fixed billing and embeddings test failures
- **Dependency Management**: Resolved bs4 and ruff installation issues
- **Monitoring System**: Now 95% effective with automated deployment

### Changed
- **API Coverage**: Increased from 80% to 100% (20/20 endpoints)
- **Test Coverage**: Maintained 81% with additional integration tests
- **Documentation**: Updated with new features and examples
- **Automation**: Daily cron job monitoring with AUTO_COMMIT mode

### Technical Details
- Complete automated API maintenance system for maintainer
- Zero-manual-review pipeline with comprehensive safety validation
- Enhanced error reporting and monitoring capabilities
- Professional documentation and API coverage reports

## [0.1.0] - 2025-01-06

### Added
- Initial beta release of PyVenice
- Complete implementation of all 16 Venice.ai API endpoints
- Automatic parameter validation based on model capabilities
- Full type safety with Pydantic models
- Both synchronous and asynchronous client support
- Streaming support for chat completions and audio endpoints
- Comprehensive test suite with 82% coverage
- Support for web search in chat completions
- Image generation with multiple models and styles
- Text-to-speech with streaming audio
- Embeddings generation
- API key management endpoints
- Character-based interactions
- Billing and usage tracking

### Security
- HTTPS-only communication with certificate verification
- API key protection (never logged or exposed)
- Input validation to prevent injection attacks
- Minimal, audited dependencies

[Unreleased]: https://github.com/TheLustriVA/PyVenice/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TheLustriVA/PyVenice/releases/tag/v0.1.0