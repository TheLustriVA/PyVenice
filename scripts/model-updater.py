#!/usr/bin/env python3
"""
Pydantic Model Updater

Automatically updates PyVenice Pydantic models based on detected API schema changes.
Integrates with model audit and response schema monitoring systems.
"""

import sys
import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))


class ModelUpdater:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"
        self.pyvenice_dir = self.project_root / "pyvenice"

        # Load current audit and schema data
        self.model_audit_path = self.docs_dir / "model_audit_report.json"
        self.schema_monitoring_path = self.docs_dir / "schema_monitoring_report.json"

        # Safety validation settings
        self.backup_dir = Path(tempfile.mkdtemp(prefix="pyvenice_model_backup_"))

    def load_audit_report(self) -> Dict[str, Any]:
        """Load latest model audit report."""
        if not self.model_audit_path.exists():
            print("⚠️  No model audit report found. Run scripts/model-audit.py first.")
            return {}

        with open(self.model_audit_path, "r") as f:
            return json.load(f)

    def load_schema_monitoring_report(self) -> Dict[str, Any]:
        """Load latest schema monitoring report."""
        if not self.schema_monitoring_path.exists():
            print(
                "⚠️  No schema monitoring report found. Run scripts/response-schema-monitor.py first."
            )
            return {}

        with open(self.schema_monitoring_path, "r") as f:
            return json.load(f)

    def create_backup(self):
        """Create backup of all PyVenice model files."""
        print(f"📦 Creating backup in {self.backup_dir}")

        for py_file in self.pyvenice_dir.glob("*.py"):
            if py_file.name != "__init__.py":
                shutil.copy2(py_file, self.backup_dir / py_file.name)

        # Create restore script
        restore_script = self.backup_dir / "restore.sh"
        with open(restore_script, "w") as f:
            f.write(
                f"""#!/bin/bash
echo "Restoring PyVenice models from backup..."
for file in {self.backup_dir}/*.py; do
    filename=$(basename "$file")
    cp "$file" "{self.pyvenice_dir}/$filename"
    echo "Restored $filename"
done
echo "Restore completed."
"""
            )
        restore_script.chmod(0o755)

        print(f"✅ Backup created. Restore with: {restore_script}")

    def analyze_missing_fields(
        self, audit_results: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze which fields are missing from each model."""
        missing_fields_by_endpoint = {}

        for endpoint, result in audit_results.get("results", {}).items():
            if result.get("status") == "success":
                missing_fields = result.get("missing_in_model", [])
                if missing_fields:
                    # Filter out nested array fields and focus on top-level additions
                    top_level_missing = []
                    for field in missing_fields:
                        # Skip deeply nested array fields for now
                        if not re.search(r"\[\d+\]\..*\.", field):
                            top_level_missing.append(field)

                    if top_level_missing:
                        missing_fields_by_endpoint[endpoint] = top_level_missing[
                            :10
                        ]  # Limit to avoid overwhelming

        return missing_fields_by_endpoint

    def find_model_file_and_class(self, endpoint: str) -> Optional[tuple]:
        """Find the Python file and class name for an endpoint's model."""
        endpoint_to_module = {
            "/models": ("models.py", "ModelListResponse"),
            "/chat/completions": ("chat.py", "ChatCompletionResponse"),
            "/image/generate": ("image.py", "ImageGenerationResponse"),
            "/characters": ("characters.py", "CharacterListResponse"),
            "/billing/usage": ("billing.py", "UsageResponse"),
            "/models/compatibility_mapping": (
                "models.py",
                "CompatibilityMappingResponse",
            ),
        }

        if endpoint in endpoint_to_module:
            filename, class_name = endpoint_to_module[endpoint]
            file_path = self.pyvenice_dir / filename
            if file_path.exists():
                return file_path, class_name

        return None

    def parse_python_file(self, file_path: Path) -> ast.AST:
        """Parse Python file into AST."""
        with open(file_path, "r") as f:
            content = f.read()
        return ast.parse(content)

    def find_class_in_ast(
        self, tree: ast.AST, class_name: str
    ) -> Optional[ast.ClassDef]:
        """Find a specific class in the AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def field_exists_in_class(self, class_node: ast.ClassDef, field_name: str) -> bool:
        """Check if a field already exists in the class."""
        for node in ast.walk(class_node):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == field_name:
                    return True
        return False

    def infer_field_type(self, field_name: str, actual_fields: List[str]) -> str:
        """Infer the appropriate Pydantic field type based on field name and context."""
        field_name_lower = field_name.lower()

        # Common type patterns
        if any(
            keyword in field_name_lower
            for keyword in ["id", "key", "token", "name", "title"]
        ):
            return "str"
        elif any(
            keyword in field_name_lower
            for keyword in ["count", "total", "limit", "page"]
        ):
            return "int"
        elif any(
            keyword in field_name_lower for keyword in ["price", "amount", "rate"]
        ):
            return "float"
        elif any(
            keyword in field_name_lower for keyword in ["timestamp", "date", "time"]
        ):
            return "str"  # Usually ISO format strings
        elif field_name_lower.endswith("s") and not field_name_lower.endswith("ss"):
            return "List[Dict[str, Any]]"  # Likely array
        elif any(
            keyword in field_name_lower for keyword in ["enabled", "active", "success"]
        ):
            return "bool"
        else:
            # Default to optional Any for unknown types
            return "Any"

    def generate_field_addition(self, field_name: str, field_type: str) -> str:
        """Generate the field addition code."""
        # Handle nested field names (convert dots to underscores for Python)
        python_field_name = (
            field_name.replace(".", "_").replace("[0]", "").replace("[]", "")
        )

        # Make field optional by default for safety
        if field_type == "Any":
            return f'    {python_field_name}: Optional[{field_type}] = Field(None, description="{field_name}")'
        else:
            return f'    {python_field_name}: Optional[{field_type}] = Field(None, description="{field_name}")'

    def update_model_file(
        self, file_path: Path, class_name: str, missing_fields: List[str]
    ) -> bool:
        """Update a model file to add missing fields."""
        print(f"🔧 Updating {file_path.name}::{class_name}")

        # Read current file
        with open(file_path, "r") as f:
            content = f.read()

        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"❌ Syntax error in {file_path}: {e}")
            return False

        # Find the target class
        class_node = self.find_class_in_ast(tree, class_name)
        if not class_node:
            print(f"❌ Class {class_name} not found in {file_path}")
            return False

        # Determine which fields to add
        fields_to_add = []
        for field in missing_fields:
            # Skip complex nested fields for now
            if "[0]." in field or field.count(".") > 2:
                continue

            # Simplify field name for Python
            simple_field = field.split(".")[-1] if "." in field else field

            if not self.field_exists_in_class(class_node, simple_field):
                field_type = self.infer_field_type(simple_field, missing_fields)
                fields_to_add.append((simple_field, field_type, field))

        if not fields_to_add:
            print(f"   ℹ️  No fields to add to {class_name}")
            return True

        # Generate new field definitions
        new_fields = []
        for field_name, field_type, original_field in fields_to_add:
            field_def = self.generate_field_addition(field_name, field_type)
            new_fields.append(field_def)
            print(f"   + Adding field: {field_name} ({field_type})")

        # Find insertion point (end of class)
        lines = content.split("\n")
        class_start_line = None
        class_end_line = None

        for i, line in enumerate(lines):
            if f"class {class_name}" in line:
                class_start_line = i
                # Find end of class (next class or end of file)
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("class ") or (j == len(lines) - 1):
                        class_end_line = j if lines[j].startswith("class ") else j + 1
                        break
                break

        if class_start_line is None:
            print(f"❌ Could not locate class {class_name} in file")
            return False

        # Find last field definition in the class
        last_field_line = class_start_line
        for i in range(class_start_line + 1, class_end_line):
            line = lines[i].strip()
            if ":" in line and ("=" in line or "Field(" in line):
                last_field_line = i

        # Insert new fields after the last existing field
        insert_pos = last_field_line + 1
        for field_def in reversed(new_fields):  # Insert in reverse to maintain order
            lines.insert(insert_pos, field_def)

        # Add imports if needed
        if any("Optional[" in field for field in new_fields):
            if "from typing import" in content and "Optional" not in content:
                # Add Optional to existing typing import
                for i, line in enumerate(lines):
                    if line.startswith("from typing import"):
                        if "Optional" not in line:
                            lines[i] = line.rstrip() + ", Optional"
                        break
            elif "from typing import" not in content:
                # Add new typing import
                lines.insert(0, "from typing import Optional, Any, Dict, List")

        if any("Field(" in field for field in new_fields):
            if "from pydantic import" in content and "Field" not in content:
                # Add Field to existing pydantic import
                for i, line in enumerate(lines):
                    if line.startswith("from pydantic import"):
                        if "Field" not in line:
                            lines[i] = line.rstrip() + ", Field"
                        break

        # Write updated content
        new_content = "\n".join(lines)
        with open(file_path, "w") as f:
            f.write(new_content)

        print(f"✅ Updated {class_name} with {len(fields_to_add)} new fields")
        return True

    def validate_updated_models(self) -> bool:
        """Validate that updated models can still be imported and parsed."""
        print("🔍 Validating updated models...")

        try:
            # Try importing each module
            modules_to_test = ["models", "chat", "image", "characters", "billing"]

            for module_name in modules_to_test:
                try:
                    exec(f"from pyvenice import {module_name}")
                    print(f"   ✅ {module_name}.py imports successfully")
                except Exception as e:
                    print(f"   ❌ {module_name}.py import failed: {e}")
                    return False

            print("✅ All model validations passed")
            return True

        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False

    def run_model_updates(self) -> Dict[str, Any]:
        """Run the complete model update process."""
        print("🚀 Starting Pydantic Model Updates...")

        # Create backup first
        self.create_backup()

        # Load audit data
        audit_results = self.load_audit_report()
        if not audit_results:
            return {"status": "failed", "error": "No audit data available"}

        # Analyze missing fields
        missing_fields = self.analyze_missing_fields(audit_results)

        if not missing_fields:
            print("✅ No missing fields detected in models")
            return {
                "status": "success",
                "updates": 0,
                "backup_path": str(self.backup_dir),
            }

        print(f"📊 Found missing fields in {len(missing_fields)} endpoints")

        # Update each model
        updates_made = 0
        failed_updates = []

        for endpoint, fields in missing_fields.items():
            model_info = self.find_model_file_and_class(endpoint)
            if not model_info:
                print(f"⚠️  Could not locate model file for {endpoint}")
                continue

            file_path, class_name = model_info
            try:
                if self.update_model_file(file_path, class_name, fields):
                    updates_made += 1
                else:
                    failed_updates.append(endpoint)
            except Exception as e:
                print(f"❌ Failed to update {endpoint}: {e}")
                failed_updates.append(endpoint)

        # Validate updates
        if updates_made > 0:
            if not self.validate_updated_models():
                print("❌ Model validation failed, restoring backup...")
                self.restore_backup()
                return {
                    "status": "failed",
                    "error": "Model validation failed",
                    "backup_path": str(self.backup_dir),
                }

        result = {
            "status": "success",
            "updates_made": updates_made,
            "failed_updates": failed_updates,
            "backup_path": str(self.backup_dir),
            "timestamp": datetime.now().isoformat(),
        }

        print(
            f"✅ Model updates completed: {updates_made} successful, {len(failed_updates)} failed"
        )
        return result

    def restore_backup(self):
        """Restore models from backup."""
        print(f"🔄 Restoring models from {self.backup_dir}")

        for backup_file in self.backup_dir.glob("*.py"):
            target_file = self.pyvenice_dir / backup_file.name
            shutil.copy2(backup_file, target_file)
            print(f"   Restored {backup_file.name}")

        print("✅ Backup restored")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Update PyVenice Pydantic models based on schema changes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument("--restore", type=str, help="Restore from backup directory")

    args = parser.parse_args()

    if args.restore:
        # Restore from specified backup
        backup_dir = Path(args.restore)
        if not backup_dir.exists():
            print(f"❌ Backup directory not found: {backup_dir}")
            sys.exit(1)

        updater = ModelUpdater()
        updater.backup_dir = backup_dir
        updater.restore_backup()
        sys.exit(0)

    if args.dry_run:
        print("🔍 Dry-run mode: analyzing what would be updated...")
        updater = ModelUpdater()
        audit_results = updater.load_audit_report()
        missing_fields = updater.analyze_missing_fields(audit_results)

        if not missing_fields:
            print("✅ No missing fields detected")
        else:
            print(f"📊 Would update {len(missing_fields)} models:")
            for endpoint, fields in missing_fields.items():
                print(f"   {endpoint}: {len(fields)} missing fields")
                for field in fields[:5]:  # Show first 5
                    print(f"     + {field}")
                if len(fields) > 5:
                    print(f"     ... and {len(fields) - 5} more")
    else:
        updater = ModelUpdater()
        result = updater.run_model_updates()

        # Save result
        result_file = updater.docs_dir / "model_update_report.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"📝 Update report saved to: {result_file}")

        if result["status"] == "failed":
            sys.exit(1)
