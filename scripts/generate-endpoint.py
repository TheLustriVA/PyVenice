#!/usr/bin/env python3
"""
Endpoint Code Generator

Uses Claude Code to generate Python client code for new/modified Venice.ai API endpoints.
This script identifies changes and uses AI to generate appropriate client code following
the existing patterns in the pyvenice library.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import yaml


class EndpointGenerator:
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"
        self.pyvenice_dir = self.project_root / "pyvenice"
        self.changes_log = self.docs_dir / "api_changes.json"

    def load_latest_changes(self) -> Dict[str, Any]:
        """Load the most recent API changes"""
        if not self.changes_log.exists():
            return {}

        with open(self.changes_log, "r") as f:
            changes_list = json.load(f)

        return changes_list[-1] if changes_list else {}

    def load_current_spec(self) -> Dict[str, Any]:
        """Load the current Swagger spec"""
        spec_path = self.docs_dir / "swagger.yaml"
        with open(spec_path, "r") as f:
            return yaml.safe_load(f)

    def analyze_endpoint_coverage(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which endpoints are implemented vs available"""
        spec_endpoints = set()
        paths = spec.get("paths", {})

        for path, methods in paths.items():
            for method in methods.keys():
                if method.lower() in ["get", "post", "put", "delete", "patch"]:
                    spec_endpoints.add((method.upper(), path))

        # Map spec endpoints to likely implementation files
        endpoint_mapping = {
            "/chat/completions": "chat.py",
            "/image/generate": "image.py",
            "/images/generations": "image.py",
            "/image/upscale": "image.py",
            "/image/styles": "image.py",
            "/audio/speech": "audio.py",
            "/embeddings": "embeddings.py",
            "/models": "models.py",
            "/api_keys": "api_keys.py",
            "/characters": "characters.py",
            "/billing/usage": "billing.py",
        }

        # Check which endpoints are likely implemented
        implemented = []
        missing = []

        for method, path in spec_endpoints:
            expected_file = endpoint_mapping.get(path)
            if expected_file and (self.pyvenice_dir / expected_file).exists():
                implemented.append((method, path, expected_file))
            else:
                missing.append((method, path, expected_file or "unknown"))

        return {
            "total_endpoints": len(spec_endpoints),
            "implemented": implemented,
            "missing": missing,
            "coverage_percent": (
                (len(implemented) / len(spec_endpoints)) * 100 if spec_endpoints else 0
            ),
        }

    def analyze_change_types(
        self, changes: Dict[str, Any], spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze whether changes are additive or require modifications"""
        change_analysis = {
            "additive_changes": {
                "new_parameters": [],
                "new_schemas": [],
                "new_endpoints": [],
            },
            "modification_changes": {
                "modified_parameters": [],
                "modified_schemas": [],
                "constraint_changes": [],
            },
            "removal_changes": {
                "removed_parameters": [],
                "removed_schemas": [],
                "removed_endpoints": [],
            },
        }

        # Categorize parameter changes
        for param in changes.get("parameters", {}).get("added", []):
            change_analysis["additive_changes"]["new_parameters"].append(
                {
                    "schema": param["schema"],
                    "parameter": param["parameter"],
                    "type": param["definition"]["type"],
                    "required": param["definition"]["required"],
                    "description": param["definition"]["description"],
                }
            )

        for param in changes.get("parameters", {}).get("removed", []):
            change_analysis["removal_changes"]["removed_parameters"].append(
                {
                    "schema": param["schema"],
                    "parameter": param["parameter"],
                    "was_required": param["definition"]["required"],
                }
            )

        for param in changes.get("parameters", {}).get("modified", []):
            change_analysis["modification_changes"]["modified_parameters"].append(
                {
                    "schema": param["schema"],
                    "parameter": param["parameter"],
                    "old_type": param["old_definition"]["type"],
                    "new_type": param["new_definition"]["type"],
                }
            )

        # Categorize schema changes
        change_analysis["additive_changes"]["new_schemas"] = changes.get(
            "schemas", {}
        ).get("added", [])
        change_analysis["removal_changes"]["removed_schemas"] = changes.get(
            "schemas", {}
        ).get("removed", [])
        change_analysis["modification_changes"]["modified_schemas"] = changes.get(
            "schemas", {}
        ).get("modified", [])

        # Categorize endpoint changes
        change_analysis["additive_changes"]["new_endpoints"] = changes.get(
            "endpoints", {}
        ).get("added", [])
        change_analysis["removal_changes"]["removed_endpoints"] = changes.get(
            "endpoints", {}
        ).get("removed", [])

        return change_analysis

    def generate_additive_code_prompts(
        self, additive_changes: Dict[str, Any], spec: Dict[str, Any]
    ) -> List[str]:
        """Generate discrete prompts for additive changes"""
        prompts = []
        schemas = spec.get("components", {}).get("schemas", {})

        # Generate prompts for new parameters
        for param in additive_changes["new_parameters"]:
            schema_def = schemas.get(param["schema"], {})
            param_def = schema_def.get("properties", {}).get(param["parameter"], {})

            prompt = f"""# Add New Parameter: {param['schema']}.{param['parameter']}

## Task: Add support for new parameter `{param['parameter']}` in {param['schema']}

### Parameter Details:
- **Name**: {param['parameter']}
- **Type**: {param['type']}
- **Required**: {param['required']}
- **Description**: {param['description']}

### Full Parameter Definition:
```yaml
{yaml.dump({param['parameter']: param_def}, default_flow_style=False)}
```

### Instructions:
1. **Add parameter to appropriate endpoint method signature**
   - Location: Likely in `pyvenice/chat.py` or `pyvenice/image.py`
   - Add `{param['parameter']}: {self._get_python_type(param['type'])} = None` to method signature
   
2. **Update validators.py if needed**
   - Add capability check if this parameter requires specific model support
   - Follow existing pattern in `capability_required_params` dict

3. **Add type hints and documentation**
   - Use proper Python type annotation
   - Add docstring parameter documentation

### Code Pattern to Follow:
```python
def create(self, *, model: str, messages: List[Message], 
           {param['parameter']}: {self._get_python_type(param['type'])} = None, **kwargs):
    \"\"\"
    Create a chat completion.
    
    Args:
        {param['parameter']}: {param['description']}
    \"\"\"
```

Generate the exact code additions needed."""

            prompts.append(prompt)

        # Generate prompts for new endpoints
        for endpoint in additive_changes["new_endpoints"]:
            method, path = endpoint.split(" ", 1)
            prompt = f"""# Implement New Endpoint: {endpoint}

## Task: Create complete implementation for new endpoint `{method} {path}`

### Instructions:
1. **Create new endpoint class** (if needed)
2. **Implement sync and async methods**
3. **Add proper request/response models**
4. **Follow existing patterns from similar endpoints**
5. **Add comprehensive error handling**
6. **Include proper documentation**

### Pattern to Follow:
Look at existing endpoint implementations in `pyvenice/` directory and create similar structure.

Generate the complete implementation for this new endpoint."""

            prompts.append(prompt)

        return prompts

    def generate_modification_prompts(
        self, modification_changes: Dict[str, Any], spec: Dict[str, Any]
    ) -> List[str]:
        """Generate prompts for modification changes (more complex)"""
        prompts = []

        if modification_changes["modified_parameters"]:
            prompt_parts = [
                "# Parameter Modifications Task",
                "",
                "Several existing parameters have been modified in the Venice.ai API. These require careful updates to maintain compatibility.",
                "",
                "## Modified Parameters:",
            ]

            for param in modification_changes["modified_parameters"]:
                prompt_parts.extend(
                    [
                        f"### {param['schema']}.{param['parameter']}",
                        f"- **Old Type**: {param['old_type']}",
                        f"- **New Type**: {param['new_type']}",
                        "",
                    ]
                )

            prompt_parts.extend(
                [
                    "## Instructions:",
                    "1. **Review each parameter modification carefully**",
                    "2. **Update type hints and validation**",
                    "3. **Ensure backward compatibility where possible**",
                    "4. **Add migration warnings if needed**",
                    "5. **Update tests to cover new parameter types**",
                    "",
                    "Generate the necessary code changes to handle these parameter modifications.",
                ]
            )

            prompts.append("\\n".join(prompt_parts))

        return prompts

    def _get_python_type(self, json_type: str) -> str:
        """Convert JSON schema type to Python type hint"""
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]",
        }
        return type_mapping.get(json_type, "Any")

    def generate_claude_prompts(
        self, changes: Dict[str, Any], spec: Dict[str, Any]
    ) -> List[str]:
        """Generate multiple discrete prompts for different types of changes"""
        change_analysis = self.analyze_change_types(changes, spec)

        prompts = []

        # Generate additive prompts (easier for AI)
        additive_prompts = self.generate_additive_code_prompts(
            change_analysis["additive_changes"], spec
        )
        prompts.extend(additive_prompts)

        # Generate modification prompts (more complex)
        modification_prompts = self.generate_modification_prompts(
            change_analysis["modification_changes"], spec
        )
        prompts.extend(modification_prompts)

        # Generate removal guidance
        if any(change_analysis["removal_changes"].values()):
            removal_prompt = self._generate_removal_prompt(
                change_analysis["removal_changes"]
            )
            prompts.append(removal_prompt)

        return prompts

    def _generate_removal_prompt(self, removal_changes: Dict[str, Any]) -> str:
        """Generate prompt for handling removals"""
        prompt_parts = [
            "# Parameter/Feature Removal Task",
            "",
            "The following parameters/features have been removed from the Venice.ai API:",
            "",
        ]

        for param in removal_changes["removed_parameters"]:
            prompt_parts.append(f"- {param['schema']}.{param['parameter']}")

        prompt_parts.extend(
            [
                "",
                "## Instructions:",
                "1. **Update deprecated_params.json** to mark these as removed",
                "2. **Add deprecation warnings** in the client code",
                "3. **Filter out removed parameters** in validators.py",
                "4. **Update documentation** to reflect removals",
                "",
                "These should be handled through the deprecation system, not direct code removal.",
            ]
        )

        return "\\n".join(prompt_parts)

    def invoke_claude_code(self, prompt: str) -> bool:
        """Invoke Claude Code via shell to update the codebase"""
        try:
            # Write prompt to temporary file
            prompt_file = Path("/tmp/claude_prompt.md")
            with open(prompt_file, "w") as f:
                f.write(prompt)

            print("🤖 Invoking Claude Code to update client library...")
            print(f"📝 Prompt written to: {prompt_file}")

            # Use claude-code CLI (if available) or provide instructions
            cmd = f"claude-code '{prompt}'"
            print("💡 Run this command to update the code:")
            print(f"   {cmd}")
            print()
            print(
                "Alternatively, copy the prompt from the file above and paste it into Claude Code."
            )

            return True

        except Exception as e:
            print(f"❌ Error invoking Claude Code: {e}")
            return False

    def generate_summary_report(
        self, changes: Dict[str, Any], coverage: Dict[str, Any]
    ) -> str:
        """Generate a summary report of changes and recommendations"""
        report = [
            "# Venice.ai API Update Report",
            f"Generated: {changes.get('timestamp', 'unknown')}",
            "",
            f"## Version Change: {changes.get('version_change', {}).get('old')} → {changes.get('version_change', {}).get('new')}",
            "",
            f"## Endpoint Coverage: {coverage['coverage_percent']:.1f}% ({len(coverage['implemented'])}/{coverage['total_endpoints']})",
            "",
        ]

        if changes.get("schemas", {}).get("modified"):
            report.extend(["## Schema Changes Requiring Client Updates:", ""])
            for schema in changes["schemas"]["modified"]:
                report.append(
                    f"- **{schema}**: Likely affects client parameter validation"
                )
            report.append("")

        if changes.get("endpoints", {}).get("added"):
            report.extend(["## New Endpoints to Implement:", ""])
            for endpoint in changes["endpoints"]["added"]:
                report.append(
                    f"- `{endpoint}`: New endpoint requiring client implementation"
                )
            report.append("")

        if coverage["missing"]:
            report.extend(["## Missing Endpoint Implementations:", ""])
            for method, path, file in coverage["missing"]:
                report.append(f"- `{method} {path}`: Should be in {file}")
            report.append("")

        report.extend(
            [
                "## Recommended Actions:",
                "1. Review schema changes and update parameter validation",
                "2. Test existing functionality with new API version",
                "3. Implement any missing endpoints",
                "4. Update documentation and examples",
                "5. Run full test suite to ensure compatibility",
                "",
            ]
        )

        return "\\n".join(report)

    async def run(self) -> bool:
        """Main generation logic"""
        print("🔍 Analyzing API changes for code generation...")

        changes = self.load_latest_changes()
        if not changes:
            print("📭 No recent changes found")
            return False

        spec = self.load_current_spec()
        coverage = self.analyze_endpoint_coverage(spec)

        print(f"📊 API Coverage: {coverage['coverage_percent']:.1f}%")
        print(
            f"🔄 Schema changes: {len(changes.get('schemas', {}).get('modified', []))}"
        )
        print(f"➕ New endpoints: {len(changes.get('endpoints', {}).get('added', []))}")

        # Generate summary report
        report = self.generate_summary_report(changes, coverage)
        report_file = self.docs_dir / "api_update_report.md"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"📋 Report saved: {report_file}")

        # Generate discrete Claude prompts for different types of changes
        prompts = self.generate_claude_prompts(changes, spec)

        if prompts:
            print(f"🤖 Generated {len(prompts)} discrete code update tasks:")

            for i, prompt in enumerate(prompts, 1):
                prompt_file = Path(f"/tmp/claude_prompt_{i}.md")
                with open(prompt_file, "w") as f:
                    f.write(prompt)

                # Extract task title from prompt
                title = prompt.split("\n")[0].replace("#", "").strip()
                print(f"  {i}. {title}")
                print(f"     📝 Prompt: {prompt_file}")

            print("\n💡 Execute these tasks in order:")
            for i in range(len(prompts)):
                print(f'   claude-code "$(cat /tmp/claude_prompt_{i+1}.md)"')

            print(
                "\n🎯 Or run all at once by feeding each prompt file to Claude Code sequentially."
            )
            return True
        else:
            print("✨ No changes requiring code updates")
            return True


async def main():
    """CLI entry point"""
    generator = EndpointGenerator()
    success = await generator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
