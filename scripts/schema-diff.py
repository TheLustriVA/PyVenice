#!/usr/bin/env python3
"""
Schema Diff Tool

Compares two versions of the Venice.ai API spec and shows detailed schema differences.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

import yaml
from deepdiff import DeepDiff


def load_spec(path: Path) -> Dict[str, Any]:
    """Load a YAML spec file"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_schema_diff(
    old_spec: Dict[str, Any], new_spec: Dict[str, Any], schema_name: str
) -> Dict[str, Any]:
    """Get detailed differences for a specific schema"""
    old_schemas = old_spec.get("components", {}).get("schemas", {})
    new_schemas = new_spec.get("components", {}).get("schemas", {})

    old_schema = old_schemas.get(schema_name, {})
    new_schema = new_schemas.get(schema_name, {})

    if not old_schema and not new_schema:
        return {"error": f"Schema {schema_name} not found in either spec"}

    if not old_schema:
        return {"type": "added", "schema": new_schema}

    if not new_schema:
        return {"type": "removed", "schema": old_schema}

    diff = DeepDiff(old_schema, new_schema, ignore_order=True)
    return {
        "type": "modified",
        "diff": diff,
        "old_schema": old_schema,
        "new_schema": new_schema,
    }


def print_schema_diff(schema_name: str, diff_result: Dict[str, Any]):
    """Print a human-readable schema diff"""
    print(f"\n🔍 Schema: {schema_name}")
    print("=" * (len(schema_name) + 10))

    if "error" in diff_result:
        print(f"❌ {diff_result['error']}")
        return

    if diff_result["type"] == "added":
        print("✅ New schema added")
        print(json.dumps(diff_result["schema"], indent=2))
        return

    if diff_result["type"] == "removed":
        print("❌ Schema removed")
        print(json.dumps(diff_result["schema"], indent=2))
        return

    diff = diff_result["diff"]

    if "dictionary_item_added" in diff:
        print("➕ Added properties:")
        for key in diff["dictionary_item_added"]:
            clean_key = key.replace("root['properties']['", "").replace("']", "")
            print(f"  + {clean_key}")

    if "dictionary_item_removed" in diff:
        print("➖ Removed properties:")
        for key in diff["dictionary_item_removed"]:
            clean_key = key.replace("root['properties']['", "").replace("']", "")
            print(f"  - {clean_key}")

    if "values_changed" in diff:
        print("🔄 Modified properties:")
        for key, change in diff["values_changed"].items():
            clean_key = (
                key.replace("root['properties']['", "")
                .replace("']", "")
                .replace("root['", "")
                .replace("']", "")
            )
            print(f"  ~ {clean_key}")
            print(f"    Old: {change['old_value']}")
            print(f"    New: {change['new_value']}")

    if "type_changes" in diff:
        print("🔀 Type changes:")
        for key, change in diff["type_changes"].items():
            clean_key = key.replace("root['properties']['", "").replace("']", "")
            print(f"  ~ {clean_key}")
            print(f"    Old type: {change['old_type']}")
            print(f"    New type: {change['new_type']}")


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Compare Venice.ai API schemas")
    parser.add_argument("--old", type=Path, required=True, help="Old spec file")
    parser.add_argument("--new", type=Path, required=True, help="New spec file")
    parser.add_argument(
        "--schema", help="Specific schema to compare (otherwise compare all modified)"
    )

    args = parser.parse_args()

    if not args.old.exists():
        print(f"❌ Old spec file not found: {args.old}")
        sys.exit(1)

    if not args.new.exists():
        print(f"❌ New spec file not found: {args.new}")
        sys.exit(1)

    print(f"📊 Comparing schemas: {args.old} → {args.new}")

    old_spec = load_spec(args.old)
    new_spec = load_spec(args.new)

    old_version = old_spec.get("info", {}).get("version", "unknown")
    new_version = new_spec.get("info", {}).get("version", "unknown")

    print(f"🏷️  Version: {old_version} → {new_version}")

    if args.schema:
        # Compare specific schema
        diff_result = get_schema_diff(old_spec, new_spec, args.schema)
        print_schema_diff(args.schema, diff_result)
    else:
        # Find all modified schemas
        old_schemas = set(old_spec.get("components", {}).get("schemas", {}).keys())
        new_schemas = set(new_spec.get("components", {}).get("schemas", {}).keys())

        added_schemas = new_schemas - old_schemas
        removed_schemas = old_schemas - new_schemas
        common_schemas = old_schemas & new_schemas

        # Check for modifications in common schemas
        modified_schemas = []
        for schema_name in common_schemas:
            diff_result = get_schema_diff(old_spec, new_spec, schema_name)
            if diff_result["type"] == "modified" and diff_result["diff"]:
                modified_schemas.append(schema_name)

        if added_schemas:
            print(f"\n✅ Added schemas ({len(added_schemas)}):")
            for schema in sorted(added_schemas):
                print(f"  + {schema}")

        if removed_schemas:
            print(f"\n❌ Removed schemas ({len(removed_schemas)}):")
            for schema in sorted(removed_schemas):
                print(f"  - {schema}")

        if modified_schemas:
            print(f"\n🔄 Modified schemas ({len(modified_schemas)}):")
            for schema in sorted(modified_schemas):
                print(f"  ~ {schema}")

            print("\n" + "=" * 50)
            print("DETAILED SCHEMA DIFFERENCES")
            print("=" * 50)

            for schema_name in sorted(modified_schemas):
                diff_result = get_schema_diff(old_spec, new_spec, schema_name)
                print_schema_diff(schema_name, diff_result)

        if not added_schemas and not removed_schemas and not modified_schemas:
            print("\n✨ No schema changes detected")


if __name__ == "__main__":
    main()
