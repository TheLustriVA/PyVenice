#!/usr/bin/env python3
"""
Safety Validation System

Multi-layered validation for automated code changes to ensure safety
without requiring manual review.
"""

import ast
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import asyncio


class SafetyValidator:
    """Multi-layered safety validation for automated code changes"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.pyvenice_dir = self.project_root / "pyvenice"
        self.tests_dir = self.project_root / "tests"
        self.validation_results = []
        
    def log_validation(self, check_name: str, status: str, message: str = ""):
        """Log a validation result"""
        self.validation_results.append({
            'check': check_name,
            'status': status,
            'message': message
        })
        
        status_emoji = {
            'PASS': '✅',
            'FAIL': '❌',
            'WARN': '⚠️',
            'SKIP': '⏭️'
        }
        
        print(f"{status_emoji.get(status, '❓')} {check_name}: {message or status}")
    
    def validate_syntax(self, file_path: Path) -> bool:
        """Validate Python syntax of a file"""
        try:
            with open(file_path, 'r') as f:
                source = f.read()
            
            ast.parse(source)
            return True
        except SyntaxError as e:
            self.log_validation(f"Syntax Check: {file_path.name}", "FAIL", 
                              f"Line {e.lineno}: {e.msg}")
            return False
        except Exception as e:
            self.log_validation(f"Syntax Check: {file_path.name}", "FAIL", str(e))
            return False
    
    def validate_imports(self, file_path: Path) -> bool:
        """Test that all imports in a file work"""
        try:
            # Create a temporary test file that just imports the module
            rel_path = file_path.relative_to(self.project_root)
            module_path = str(rel_path).replace('/', '.').replace('.py', '')
            
            test_script = f"""
import sys
sys.path.insert(0, '{self.project_root}')
try:
    import {module_path}
    print("IMPORT_SUCCESS")
except Exception as e:
    print(f"IMPORT_FAILED: {{e}}")
    sys.exit(1)
"""
            
            result = subprocess.run(
                [sys.executable, '-c', test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if "IMPORT_SUCCESS" in result.stdout:
                return True
            else:
                self.log_validation(f"Import Check: {file_path.name}", "FAIL",
                                  result.stderr or result.stdout)
                return False
                
        except subprocess.TimeoutExpired:
            self.log_validation(f"Import Check: {file_path.name}", "FAIL", "Timeout")
            return False
        except Exception as e:
            self.log_validation(f"Import Check: {file_path.name}", "FAIL", str(e))
            return False
    
    def validate_function_signatures(self, file_path: Path, expected_functions: List[str] = None) -> bool:
        """Validate that expected functions exist with reasonable signatures"""
        try:
            with open(file_path, 'r') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            found_functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    found_functions.append(node.name)
            
            if expected_functions:
                missing = set(expected_functions) - set(found_functions)
                if missing:
                    self.log_validation(f"Function Check: {file_path.name}", "FAIL",
                                      f"Missing functions: {missing}")
                    return False
            
            self.log_validation(f"Function Check: {file_path.name}", "PASS",
                              f"Found {len(found_functions)} functions")
            return True
            
        except Exception as e:
            self.log_validation(f"Function Check: {file_path.name}", "FAIL", str(e))
            return False
    
    def validate_parameter_safety(self, changes: Dict[str, Any]) -> bool:
        """Validate that parameter changes are safe"""
        safe = True
        
        # Check added parameters - these should be safe if optional
        for param in changes.get('parameters', {}).get('added', []):
            if param['definition']['required']:
                self.log_validation("Parameter Safety", "FAIL",
                                  f"New required parameter: {param['schema']}.{param['parameter']}")
                safe = False
            else:
                self.log_validation("Parameter Safety", "PASS",
                                  f"Optional parameter: {param['schema']}.{param['parameter']}")
        
        # Check removed parameters - always dangerous
        removed_params = changes.get('parameters', {}).get('removed', [])
        if removed_params:
            self.log_validation("Parameter Safety", "WARN",
                              f"{len(removed_params)} parameters removed - handled by deprecation system")
        
        # Check modified parameters - potentially breaking
        modified_params = changes.get('parameters', {}).get('modified', [])
        for param in modified_params:
            if param['old_definition']['type'] != param['new_definition']['type']:
                self.log_validation("Parameter Safety", "WARN",
                                  f"Type change: {param['schema']}.{param['parameter']}")
        
        return safe
    
    def run_unit_tests(self) -> bool:
        """Run the test suite to ensure changes don't break functionality"""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            if result.returncode == 0:
                self.log_validation("Unit Tests", "PASS", "All tests passed")
                return True
            else:
                self.log_validation("Unit Tests", "FAIL",
                                  f"Tests failed:\n{result.stdout}\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_validation("Unit Tests", "FAIL", "Tests timed out")
            return False
        except Exception as e:
            self.log_validation("Unit Tests", "FAIL", str(e))
            return False
    
    def run_linting(self) -> bool:
        """Run code quality checks"""
        try:
            # Check if ruff is available
            result = subprocess.run(
                ['ruff', 'check', '.', '--format', 'json'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.log_validation("Code Quality", "PASS", "No linting issues")
                return True
            else:
                try:
                    issues = json.loads(result.stdout)
                    error_count = len([i for i in issues if i.get('level') == 'error'])
                    warning_count = len([i for i in issues if i.get('level') == 'warning'])
                    
                    if error_count > 0:
                        self.log_validation("Code Quality", "FAIL",
                                          f"{error_count} errors, {warning_count} warnings")
                        return False
                    else:
                        self.log_validation("Code Quality", "WARN",
                                          f"{warning_count} warnings")
                        return True
                except json.JSONDecodeError:
                    self.log_validation("Code Quality", "FAIL", result.stdout)
                    return False
                    
        except subprocess.TimeoutExpired:
            self.log_validation("Code Quality", "FAIL", "Linting timed out")
            return False
        except FileNotFoundError:
            self.log_validation("Code Quality", "SKIP", "Ruff not installed")
            return True  # Don't fail if linter not available
        except Exception as e:
            self.log_validation("Code Quality", "FAIL", str(e))
            return False
    
    async def validate_api_compatibility(self) -> bool:
        """Test that changes don't break API compatibility"""
        try:
            # Run the API contract validator
            result = subprocess.run(
                [sys.executable, 'scripts/api-contract-validator.py', '--json'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                # Parse results
                try:
                    results = json.loads(result.stdout)
                    if results.get('success'):
                        self.log_validation("API Compatibility", "PASS", "All compatibility tests passed")
                        return True
                    else:
                        failed_tests = [r for r in results.get('results', []) if r['status'] == 'FAIL']
                        self.log_validation("API Compatibility", "FAIL",
                                          f"{len(failed_tests)} compatibility tests failed")
                        return False
                except json.JSONDecodeError:
                    self.log_validation("API Compatibility", "PASS", "Tests completed")
                    return True
            else:
                self.log_validation("API Compatibility", "FAIL",
                                  f"Compatibility tests failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_validation("API Compatibility", "FAIL", "API tests timed out")
            return False
        except Exception as e:
            self.log_validation("API Compatibility", "FAIL", str(e))
            return False
    
    def create_safety_backup(self) -> Path:
        """Create a backup of current state for rollback"""
        try:
            backup_dir = Path(tempfile.mkdtemp(prefix="pyvenice_backup_"))
            
            # Backup pyvenice directory
            shutil.copytree(self.pyvenice_dir, backup_dir / "pyvenice")
            
            # Backup key config files
            for file_name in ["pyproject.toml", "CLAUDE.md"]:
                file_path = self.project_root / file_name
                if file_path.exists():
                    shutil.copy2(file_path, backup_dir / file_name)
            
            self.log_validation("Safety Backup", "PASS", f"Backup created: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            self.log_validation("Safety Backup", "FAIL", str(e))
            return None
    
    def restore_from_backup(self, backup_dir: Path) -> bool:
        """Restore from safety backup"""
        try:
            if not backup_dir.exists():
                self.log_validation("Backup Restore", "FAIL", "Backup directory not found")
                return False
            
            # Restore pyvenice directory
            if (backup_dir / "pyvenice").exists():
                shutil.rmtree(self.pyvenice_dir)
                shutil.copytree(backup_dir / "pyvenice", self.pyvenice_dir)
            
            # Restore config files
            for file_name in ["pyproject.toml", "CLAUDE.md"]:
                backup_file = backup_dir / file_name
                if backup_file.exists():
                    shutil.copy2(backup_file, self.project_root / file_name)
            
            self.log_validation("Backup Restore", "PASS", "Successfully restored from backup")
            return True
            
        except Exception as e:
            self.log_validation("Backup Restore", "FAIL", str(e))
            return False
    
    async def comprehensive_validation(self, changes: Dict[str, Any] = None) -> Tuple[bool, Optional[Path]]:
        """Run comprehensive safety validation with backup/restore"""
        print("🛡️ Running Comprehensive Safety Validation...")
        print()
        
        # Create safety backup
        backup_dir = self.create_safety_backup()
        
        # Run all validation checks
        all_passed = True
        
        # 1. Syntax validation for all Python files
        for py_file in self.pyvenice_dir.glob("**/*.py"):
            if not self.validate_syntax(py_file):
                all_passed = False
        
        # 2. Import validation
        for py_file in self.pyvenice_dir.glob("*.py"):
            if py_file.name != "__init__.py":
                if not self.validate_imports(py_file):
                    all_passed = False
        
        # 3. Parameter safety validation
        if changes:
            if not self.validate_parameter_safety(changes):
                all_passed = False
        
        # 4. Code quality checks
        if not self.run_linting():
            all_passed = False
        
        # 5. Unit tests
        if not self.run_unit_tests():
            all_passed = False
        
        # 6. API compatibility
        if not await self.validate_api_compatibility():
            all_passed = False
        
        # Summary
        print()
        print("📊 Validation Summary:")
        
        status_counts = {}
        for result in self.validation_results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in status_counts.items():
            emoji = {'PASS': '✅', 'FAIL': '❌', 'WARN': '⚠️', 'SKIP': '⏭️'}.get(status, '❓')
            print(f"  {emoji} {status}: {count}")
        
        if all_passed and status_counts.get('FAIL', 0) == 0:
            print("\n✅ ALL VALIDATIONS PASSED - Changes are safe to deploy")
            return True, backup_dir
        else:
            print("\n❌ VALIDATION FAILED - Changes are NOT safe")
            if backup_dir:
                print(f"💾 Backup available for restore: {backup_dir}")
            return False, backup_dir


async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive safety validation")
    parser.add_argument('--changes-file', type=Path, help="JSON file with API changes")
    parser.add_argument('--restore', type=Path, help="Restore from backup directory")
    
    args = parser.parse_args()
    
    validator = SafetyValidator()
    
    if args.restore:
        success = validator.restore_from_backup(args.restore)
        sys.exit(0 if success else 1)
    
    changes = None
    if args.changes_file and args.changes_file.exists():
        with open(args.changes_file, 'r') as f:
            changes = json.load(f)
    
    success, backup_dir = await validator.comprehensive_validation(changes)
    
    if success:
        print("\n🎉 Validation complete - system is stable")
    else:
        print("\n🚨 Validation failed - manual intervention required")
        if backup_dir:
            print(f"   Restore command: python scripts/safety-validator.py --restore {backup_dir}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())