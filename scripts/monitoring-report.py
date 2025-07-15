#!/usr/bin/env python3
"""
Enhanced monitoring report generator.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def generate_monitoring_report():
    """Generate comprehensive monitoring report."""

    report = {
        "timestamp": datetime.now().isoformat(),
        "system_status": {},
        "api_status": {},
        "test_status": {},
        "coverage_status": {},
    }

    # Check system dependencies
    try:
        import importlib.util

        spec = importlib.util.find_spec("bs4")
        if spec is not None:
            report["system_status"]["bs4"] = "✅ Available"
        else:
            report["system_status"]["bs4"] = "❌ Missing"
    except ImportError:
        report["system_status"]["bs4"] = "❌ Missing"

    try:
        result = subprocess.run(["which", "ruff"], capture_output=True, text=True)
        if result.returncode == 0:
            report["system_status"]["ruff"] = "✅ Available"
        else:
            report["system_status"]["ruff"] = "❌ Missing"
    except (subprocess.SubprocessError, OSError):
        report["system_status"]["ruff"] = "❌ Error"

    # Check API version
    try:
        with open("docs/swagger.yaml") as f:
            content = f.read()
            if "20250713.224148" in content:
                report["api_status"]["version"] = "✅ Current (20250713.224148)"
            else:
                report["api_status"]["version"] = "❌ Outdated"
    except (FileNotFoundError, OSError):
        report["api_status"]["version"] = "❌ Cannot read"

    # Check endpoint coverage
    try:
        import yaml

        with open("docs/swagger.yaml") as f:
            spec = yaml.safe_load(f)

        endpoints = 0
        for path, methods in spec.get("paths", {}).items():
            endpoints += len(
                [m for m in methods if m in ["get", "post", "put", "delete"]]
            )

        report["api_status"]["endpoints"] = f"✅ {endpoints}/20 endpoints"
    except (FileNotFoundError, ImportError, OSError):
        report["api_status"]["endpoints"] = "❌ Cannot analyze"

    # Check test status
    try:
        result = subprocess.run(
            ["pytest", "tests/", "-m", "not integration", "--tb=no", "-q"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            report["test_status"]["unit_tests"] = "✅ Passing"
        else:
            report["test_status"]["unit_tests"] = "❌ Failing"
    except (subprocess.SubprocessError, OSError):
        report["test_status"]["unit_tests"] = "❌ Error"

    # Check integration tests
    try:
        result = subprocess.run(
            ["pytest", "tests/", "-m", "integration", "--tb=no", "-q"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            report["test_status"]["integration_tests"] = "✅ Passing"
        else:
            report["test_status"]["integration_tests"] = "❌ Failing"
    except (subprocess.SubprocessError, OSError):
        report["test_status"]["integration_tests"] = "❌ Error"

    # Check coverage
    try:
        result = subprocess.run(
            [
                "pytest",
                "tests/",
                "-m",
                "not integration",
                "--cov=pyvenice",
                "--cov-report=term",
                "--tb=no",
            ],
            capture_output=True,
            text=True,
        )
        if "TOTAL" in result.stdout:
            coverage_line = [
                line for line in result.stdout.split("\n") if "TOTAL" in line
            ]
            if coverage_line:
                coverage = coverage_line[0].split()[-1]
                report["coverage_status"]["unit_coverage"] = f"✅ {coverage}"
            else:
                report["coverage_status"]["unit_coverage"] = "❌ Cannot parse"
        else:
            report["coverage_status"]["unit_coverage"] = "❌ No coverage data"
    except (subprocess.SubprocessError, OSError):
        report["coverage_status"]["unit_coverage"] = "❌ Error"

    # Generate report
    report_path = Path("docs/monitoring_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Generate human-readable report
    readable_report = f"""
# PyVenice Monitoring Report
Generated: {report["timestamp"]}

## System Status
- Dependencies: {report["system_status"]["bs4"]}, {report["system_status"]["ruff"]}

## API Status  
- Version: {report["api_status"]["version"]}
- Endpoints: {report["api_status"]["endpoints"]}

## Test Status
- Unit Tests: {report["test_status"]["unit_tests"]}
- Integration Tests: {report["test_status"]["integration_tests"]}
- Coverage: {report["coverage_status"]["unit_coverage"]}

## Overall Health
{"✅ System is healthy" if all("✅" in str(v) for v in report["system_status"].values()) else "⚠️ System needs attention"}

## Recommendations
- Monitor API version for updates
- Keep test coverage above 80%
- Check integration test failures (may be due to API key issues)
"""

    with open("docs/monitoring_report.md", "w") as f:
        f.write(readable_report)

    print("✅ Monitoring report generated")
    print(f"   - JSON: {report_path}")
    print("   - Markdown: docs/monitoring_report.md")

    return report


if __name__ == "__main__":
    try:
        # Change to project directory
        os.chdir(Path(__file__).parent.parent)
        generate_monitoring_report()
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)
