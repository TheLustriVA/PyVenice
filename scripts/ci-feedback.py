#!/usr/bin/env python3
"""
CI/CD Feedback System

Monitors GitHub Actions workflows and pulls failure information back
for debugging and go/no-go decisions on automated code changes.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timedelta


class CIFeedbackMonitor:
    """Monitor CI/CD status and provide feedback for automated changes"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.repo_name = self._get_repo_name()

    def _get_repo_name(self) -> str:
        """Get the GitHub repository name"""
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                url = result.stdout.strip()
                # Extract owner/repo from various URL formats
                if "github.com" in url:
                    parts = url.split("/")
                    if len(parts) >= 2:
                        repo = parts[-1].replace(".git", "")
                        owner = parts[-2]
                        return f"{owner}/{repo}"

            return "unknown/unknown"

        except Exception:
            return "unknown/unknown"

    def get_workflow_runs(
        self, workflow_name: str = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent workflow runs using gh CLI"""
        try:
            cmd = [
                "gh",
                "run",
                "list",
                "--json",
                "status,conclusion,workflowName,createdAt,headBranch,url",
            ]

            if workflow_name:
                cmd.extend(["--workflow", workflow_name])

            cmd.extend(["--limit", str(limit)])

            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                print(f"❌ Failed to get workflow runs: {result.stderr}")
                return []

        except FileNotFoundError:
            print(
                "❌ GitHub CLI (gh) not installed - install from https://cli.github.com/"
            )
            return []
        except Exception as e:
            print(f"❌ Error getting workflow runs: {e}")
            return []

    def get_workflow_logs(self, run_id: str) -> Optional[str]:
        """Get logs for a specific workflow run"""
        try:
            result = subprocess.run(
                ["gh", "run", "view", run_id, "--log-failed"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return result.stdout
            else:
                return None

        except Exception:
            return None

    def analyze_workflow_failure(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a failed workflow run and extract actionable information"""
        analysis = {
            "run_id": run_data.get("url", "").split("/")[-1],
            "workflow": run_data.get("workflowName", "unknown"),
            "status": run_data.get("status"),
            "conclusion": run_data.get("conclusion"),
            "branch": run_data.get("headBranch"),
            "created_at": run_data.get("createdAt"),
            "failure_type": "unknown",
            "actionable_items": [],
            "logs": None,
        }

        # Get logs if failed
        if analysis["conclusion"] == "failure":
            logs = self.get_workflow_logs(analysis["run_id"])
            if logs:
                analysis["logs"] = logs
                analysis["failure_type"] = self._classify_failure(logs)
                analysis["actionable_items"] = self._extract_actionable_items(logs)

        return analysis

    def _classify_failure(self, logs: str) -> str:
        """Classify the type of failure based on logs"""
        logs_lower = logs.lower()

        if "test" in logs_lower and ("failed" in logs_lower or "error" in logs_lower):
            return "test_failure"
        elif "lint" in logs_lower or "ruff" in logs_lower:
            return "linting_failure"
        elif "syntax" in logs_lower and "error" in logs_lower:
            return "syntax_error"
        elif "import" in logs_lower and (
            "error" in logs_lower or "failed" in logs_lower
        ):
            return "import_error"
        elif "timeout" in logs_lower:
            return "timeout"
        elif "permission" in logs_lower or "denied" in logs_lower:
            return "permission_error"
        elif "network" in logs_lower or "connection" in logs_lower:
            return "network_error"
        else:
            return "unknown"

    def _extract_actionable_items(self, logs: str) -> List[str]:
        """Extract actionable items from failure logs"""
        items = []
        lines = logs.split("\n")

        for line in lines:
            line = line.strip()

            # Test failures
            if "FAILED" in line and "::" in line:
                items.append(f"Fix failing test: {line}")

            # Syntax errors
            elif "SyntaxError:" in line:
                items.append(f"Fix syntax error: {line}")

            # Import errors
            elif "ImportError:" in line or "ModuleNotFoundError:" in line:
                items.append(f"Fix import issue: {line}")

            # Linting issues
            elif any(word in line for word in ["error:", "E999", "F401", "F811"]):
                items.append(f"Fix linting issue: {line}")

        return items[:10]  # Limit to top 10 issues

    def check_deployment_safety(
        self, branch: str = "main", lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """Check if it's safe to deploy based on recent CI/CD history"""
        runs = self.get_workflow_runs(limit=50)

        # Filter runs for the target branch within lookback period
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_runs = []

        for run in runs:
            if run.get("headBranch") == branch:
                try:
                    run_time = datetime.fromisoformat(
                        run["createdAt"].replace("Z", "+00:00")
                    )
                    if run_time >= cutoff_time:
                        recent_runs.append(run)
                except Exception:
                    continue

        # Analyze recent runs
        success_count = sum(
            1 for run in recent_runs if run.get("conclusion") == "success"
        )
        failure_count = sum(
            1 for run in recent_runs if run.get("conclusion") == "failure"
        )
        total_runs = len(recent_runs)

        success_rate = (success_count / total_runs) if total_runs > 0 else 0

        # Determine safety status
        if total_runs == 0:
            safety_status = "unknown"
            safety_message = "No recent CI/CD runs found"
        elif success_rate >= 0.8:
            safety_status = "safe"
            safety_message = (
                f"High success rate: {success_rate:.1%} ({success_count}/{total_runs})"
            )
        elif success_rate >= 0.5:
            safety_status = "caution"
            safety_message = f"Moderate success rate: {success_rate:.1%} ({success_count}/{total_runs})"
        else:
            safety_status = "unsafe"
            safety_message = (
                f"Low success rate: {success_rate:.1%} ({success_count}/{total_runs})"
            )

        # Get details of recent failures
        recent_failures = [
            run for run in recent_runs if run.get("conclusion") == "failure"
        ]
        failure_analyses = [
            self.analyze_workflow_failure(run) for run in recent_failures[:3]
        ]

        return {
            "safety_status": safety_status,
            "safety_message": safety_message,
            "success_rate": success_rate,
            "total_runs": total_runs,
            "success_count": success_count,
            "failure_count": failure_count,
            "recent_failures": failure_analyses,
            "recommendation": self._get_deployment_recommendation(
                safety_status, failure_analyses
            ),
        }

    def _get_deployment_recommendation(
        self, safety_status: str, failures: List[Dict[str, Any]]
    ) -> str:
        """Get deployment recommendation based on CI/CD status"""
        if safety_status == "safe":
            return "✅ Safe to deploy - CI/CD is stable"
        elif safety_status == "caution":
            return "⚠️ Deploy with caution - monitor closely"
        elif safety_status == "unsafe":
            failure_types = [f["failure_type"] for f in failures]
            common_failures = list(set(failure_types))
            return f"❌ Do not deploy - fix {', '.join(common_failures)} issues first"
        else:
            return "❓ Unknown safety status - proceed manually"

    def wait_for_workflow_completion(self, timeout_minutes: int = 30) -> Dict[str, Any]:
        """Wait for currently running workflows to complete"""
        print(f"⏳ Waiting for workflows to complete (timeout: {timeout_minutes}m)...")

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            runs = self.get_workflow_runs(limit=5)
            running_runs = [run for run in runs if run.get("status") == "in_progress"]

            if not running_runs:
                print("✅ All workflows completed")
                return self.check_deployment_safety()

            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(
                    f"⏰ Timeout reached - {len(running_runs)} workflows still running"
                )
                return {
                    "safety_status": "timeout",
                    "safety_message": f"{len(running_runs)} workflows still running after {timeout_minutes}m",
                    "recommendation": "❌ Do not deploy - workflows did not complete in time",
                }

            print(
                f"⏳ {len(running_runs)} workflows running... (elapsed: {elapsed/60:.1f}m)"
            )
            time.sleep(30)  # Check every 30 seconds

    def generate_failure_report(self, failures: List[Dict[str, Any]]) -> str:
        """Generate a detailed failure report"""
        if not failures:
            return "✅ No recent failures found"

        report_lines = ["🚨 CI/CD Failure Report", "=" * 50, ""]

        for i, failure in enumerate(failures, 1):
            report_lines.extend(
                [
                    f"## Failure #{i}: {failure['workflow']}",
                    f"**Run ID**: {failure['run_id']}",
                    f"**Branch**: {failure['branch']}",
                    f"**Time**: {failure['created_at']}",
                    f"**Type**: {failure['failure_type']}",
                    "",
                ]
            )

            if failure["actionable_items"]:
                report_lines.extend(["**Action Items**:", ""])
                for item in failure["actionable_items"]:
                    report_lines.append(f"- {item}")
                report_lines.append("")

        return "\n".join(report_lines)


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor CI/CD status for deployment safety"
    )
    parser.add_argument(
        "--check-safety", action="store_true", help="Check deployment safety"
    )
    parser.add_argument(
        "--wait-completion",
        type=int,
        metavar="MINUTES",
        help="Wait for workflows to complete (timeout in minutes)",
    )
    parser.add_argument(
        "--failure-report", action="store_true", help="Generate failure report"
    )
    parser.add_argument(
        "--branch", default="main", help="Branch to check (default: main)"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    monitor = CIFeedbackMonitor()

    if args.wait_completion:
        result = monitor.wait_for_workflow_completion(args.wait_completion)
    elif args.check_safety:
        result = monitor.check_deployment_safety(args.branch)
    else:
        # Default: check safety
        result = monitor.check_deployment_safety(args.branch)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"🔍 CI/CD Status Check for '{args.branch}' branch")
        print(f"📊 {result['safety_message']}")
        print(f"💡 {result['recommendation']}")

        if args.failure_report and result.get("recent_failures"):
            print("\n" + monitor.generate_failure_report(result["recent_failures"]))

    # Exit with appropriate code for automation
    if result["safety_status"] in ["safe", "caution"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
