#!/usr/bin/env python3
"""
Dead Code Detection Tool

Identifies unused functions, classes, and parameters in the pyvenice codebase
that can be safely removed after API deprecations.
"""

import ast
from pathlib import Path
from typing import Dict, Set, List, Any
import json


class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze code usage"""

    def __init__(self):
        self.definitions: Dict[str, Set[str]] = {
            "functions": set(),
            "classes": set(),
            "methods": set(),
            "variables": set(),
        }
        self.usages: Dict[str, Set[str]] = {
            "functions": set(),
            "classes": set(),
            "methods": set(),
            "variables": set(),
        }
        self.current_class = None

    def visit_FunctionDef(self, node):
        if self.current_class:
            # This is a method
            method_name = f"{self.current_class}.{node.name}"
            self.definitions["methods"].add(method_name)
        else:
            # This is a function
            self.definitions["functions"].add(node.name)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        # Treat async functions same as regular functions
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.definitions["classes"].add(node.name)
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            # This is a usage
            if hasattr(node, "id"):
                self.usages["variables"].add(node.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        # Function calls
        if isinstance(node.func, ast.Name):
            self.usages["functions"].add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                # Method call like obj.method()
                method_name = f"{node.func.value.id}.{node.func.attr}"
                self.usages["methods"].add(method_name)

        self.generic_visit(node)


class DeadCodeDetector:
    """Main dead code detection logic"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.pyvenice_dir = self.project_root / "pyvenice"
        self.tests_dir = self.project_root / "tests"
        self.docs_dir = self.project_root / "docs"

    def analyze_file(self, file_path: Path) -> CodeAnalyzer:
        """Analyze a single Python file"""
        try:
            with open(file_path, "r") as f:
                source = f.read()

            tree = ast.parse(source)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            return analyzer
        except Exception as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")
            return CodeAnalyzer()

    def get_python_files(self, directory: Path) -> List[Path]:
        """Get all Python files in a directory"""
        if not directory.exists():
            return []
        return list(directory.glob("**/*.py"))

    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the entire codebase for dead code"""
        all_definitions = {
            "functions": set(),
            "classes": set(),
            "methods": set(),
            "variables": set(),
        }
        all_usages = {
            "functions": set(),
            "classes": set(),
            "methods": set(),
            "variables": set(),
        }

        # Analyze main codebase
        pyvenice_files = self.get_python_files(self.pyvenice_dir)
        test_files = self.get_python_files(self.tests_dir)

        all_files = pyvenice_files + test_files

        file_analyses = {}
        for file_path in all_files:
            analyzer = self.analyze_file(file_path)
            file_analyses[str(file_path)] = analyzer

            # Merge into global sets
            for category in all_definitions:
                all_definitions[category].update(analyzer.definitions[category])
                all_usages[category].update(analyzer.usages[category])

        # Find potentially dead code
        dead_code = {}
        for category in all_definitions:
            dead_items = all_definitions[category] - all_usages[category]
            # Filter out special methods and common patterns
            dead_items = {
                item for item in dead_items if not self._is_special_item(item)
            }
            dead_code[category] = dead_items

        return {
            "file_analyses": file_analyses,
            "definitions": all_definitions,
            "usages": all_usages,
            "dead_code": dead_code,
            "stats": {
                "total_files": len(all_files),
                "pyvenice_files": len(pyvenice_files),
                "test_files": len(test_files),
                "total_definitions": sum(
                    len(defs) for defs in all_definitions.values()
                ),
                "total_usages": sum(len(usages) for usages in all_usages.values()),
                "dead_items": sum(len(dead) for dead in dead_code.values()),
            },
        }

    def _is_special_item(self, item: str) -> bool:
        """Check if an item should be excluded from dead code detection"""
        special_patterns = [
            "__init__",
            "__str__",
            "__repr__",
            "__enter__",
            "__exit__",
            "setUp",
            "tearDown",
            "test_",
            "_test_",
            "main",
            "cli",
            "app",  # Entry points
            "_.*",  # Private items (might be used dynamically)
        ]

        for pattern in special_patterns:
            if pattern.endswith("_") and item.startswith(pattern[:-1]):
                return True
            elif pattern.startswith("_") and pattern.endswith(".*"):
                if item.startswith(pattern[:-2]):
                    return True
            elif item == pattern:
                return True

        return False

    def check_deprecated_parameters(self) -> Dict[str, Any]:
        """Check which deprecated parameters are still used in code"""
        deprecated_file = self.docs_dir / "deprecated_params.json"
        if not deprecated_file.exists():
            return {"error": "No deprecated_params.json found"}

        with open(deprecated_file, "r") as f:
            deprecated_params = json.load(f)

        # Find usages of deprecated parameters in codebase
        param_usages = {}

        for file_path in self.get_python_files(self.pyvenice_dir):
            try:
                with open(file_path, "r") as f:
                    content = f.read()

                for schema, params in deprecated_params.items():
                    for param_name, param_info in params.items():
                        if param_name in content:
                            if schema not in param_usages:
                                param_usages[schema] = {}
                            if param_name not in param_usages[schema]:
                                param_usages[schema][param_name] = []
                            param_usages[schema][param_name].append(str(file_path))

            except Exception as e:
                print(f"⚠️  Error checking deprecated params in {file_path}: {e}")

        return {
            "deprecated_params": deprecated_params,
            "param_usages": param_usages,
            "removable": {
                schema: {
                    param: info
                    for param, info in params.items()
                    if schema not in param_usages or param not in param_usages[schema]
                }
                for schema, params in deprecated_params.items()
            },
        }

    def generate_removal_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a plan for removing dead code"""
        plan = {
            "safe_to_remove": {"functions": [], "classes": [], "methods": []},
            "needs_review": {"functions": [], "classes": [], "methods": []},
            "deprecated_params_removable": [],
            "actions": [],
        }

        # Categorize dead code by safety
        for category in ["functions", "classes", "methods"]:
            for item in analysis["dead_code"].get(category, set()):
                # Very conservative - only mark as safe if it's clearly internal
                if item.startswith("_") and not item.startswith("__"):
                    plan["safe_to_remove"][category].append(item)
                else:
                    plan["needs_review"][category].append(item)

        # Check deprecated parameters
        deprecated_analysis = self.check_deprecated_parameters()
        if "removable" in deprecated_analysis:
            for schema, params in deprecated_analysis["removable"].items():
                for param, info in params.items():
                    if info.get("behavior") == "remove":
                        plan["deprecated_params_removable"].append(f"{schema}.{param}")

        # Generate action items
        if plan["safe_to_remove"]["functions"]:
            plan["actions"].append(
                f"Remove {len(plan['safe_to_remove']['functions'])} unused private functions"
            )

        if plan["needs_review"]["functions"]:
            plan["actions"].append(
                f"Review {len(plan['needs_review']['functions'])} potentially unused public functions"
            )

        if plan["deprecated_params_removable"]:
            plan["actions"].append(
                f"Remove {len(plan['deprecated_params_removable'])} deprecated parameters"
            )

        return plan

    def run(self) -> Dict[str, Any]:
        """Main execution"""
        print("🔍 Analyzing codebase for dead code...")

        analysis = self.analyze_codebase()

        print("📊 Analysis complete:")
        print(f"  Files analyzed: {analysis['stats']['total_files']}")
        print(f"  Definitions found: {analysis['stats']['total_definitions']}")
        print(f"  Dead items detected: {analysis['stats']['dead_items']}")

        # Generate removal plan
        plan = self.generate_removal_plan(analysis)

        # Print summary
        print("\n🎯 Dead Code Summary:")
        for category in ["functions", "classes", "methods"]:
            safe_count = len(plan["safe_to_remove"][category])
            review_count = len(plan["needs_review"][category])
            if safe_count > 0:
                print(f"  Safe to remove: {safe_count} {category}")
            if review_count > 0:
                print(f"  Needs review: {review_count} {category}")

        deprecated_count = len(plan["deprecated_params_removable"])
        if deprecated_count > 0:
            print(f"  Deprecated params: {deprecated_count} removable")

        if plan["actions"]:
            print("\n📋 Recommended Actions:")
            for action in plan["actions"]:
                print(f"  • {action}")
        else:
            print("\n✨ No dead code detected - codebase is clean!")

        return {
            "analysis": analysis,
            "plan": plan,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Detect dead code in pyvenice")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--save", type=Path, help="Save analysis to file")

    args = parser.parse_args()

    detector = DeadCodeDetector()
    result = detector.run()

    if args.json:
        print(json.dumps(result, indent=2, default=str))

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"💾 Analysis saved to {args.save}")


if __name__ == "__main__":
    main()
