"""Apply native human-approved GitHub PR decision receipts from trusted main only."""

from __future__ import annotations
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from harness.core import agent_flow as flow


class ReviewError(ValueError):
    pass


def api(path):
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    req = Request(
        "https://api.github.com/repos/" + repo + path,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "openresearch-review",
        },
    )
    with urlopen(req, timeout=15) as result:
        return json.loads(result.read())


def checked_event(event):
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request_review":
        raise ReviewError("Authentic GitHub review event required")
    review = event.get("review") or {}
    pr = event.get("pull_request") or {}
    actor = event.get("sender") or {}
    if review.get("state") != "approved" or actor.get("type") != "User":
        raise ReviewError("Human GitHub Approved review required")
    login = actor.get("login")
    if not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,39}", login):
        raise ReviewError("Invalid human reviewer")
    user = review.get("user") or actor
    if user.get("login") != login or user.get("id") not in (None, actor.get("id")):
        raise ReviewError("Review author differs from event actor")
    target_repo = (pr.get("base", {}).get("repo") or {}).get("full_name")
    if target_repo and target_repo != os.environ.get("GITHUB_REPOSITORY"):
        raise ReviewError("GitHub review targets a different repository")
    if pr.get("base", {}).get("ref") != "main":
        raise ReviewError("Review must target main")
    if review.get("commit_id") != pr.get("head", {}).get("sha"):
        raise ReviewError("Review stale against PR head")
    number = pr.get("number")
    if type(number) is not int:
        raise ReviewError("PR number missing")
    permission = api("/collaborators/" + login + "/permission").get("permission")
    if permission not in ("write", "maintain", "admin"):
        raise ReviewError("Reviewed actor requires write-level repository permission")
    return number, login, pr["head"]["sha"]


def execute():
    event_file = os.environ.get("GITHUB_EVENT_PATH")
    if not event_file:
        raise ReviewError("Missing trusted GitHub event file")
    event = json.loads(Path(event_file).read_text(encoding="utf-8"))
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request_review":
        raise ReviewError("Authenticated GitHub review event required")
    pull_request = event.get("pull_request") or {}
    number = pull_request.get("number")
    if type(number) is not int:
        raise ReviewError("Reviewed PR number missing")
    files = api("/pulls/" + str(number) + "/files?per_page=100")
    if not isinstance(files, list):
        raise ReviewError("Invalid GitHub PR file response")
    # A normal source/Knowledge PR is reviewed by GitHub's native process.
    # This narrowly privileged handler is ONLY for review-decision receipts.
    if not any(
        str(entry.get("filename", "")).endswith(".review-decisions.json") for entry in files
    ):
        print("Ordinary PR approval; Agent Flow receipt handler skipped.")
        return
    number, reviewer, commit_sha = checked_event(event)
    meta = api("/pulls/" + str(number))
    if meta.get("head", {}).get("sha") != commit_sha or meta.get("changed_files") != 1:
        raise ReviewError("Approval must cover an unchanged one-file receipt PR")
    if len(files) != 1:
        raise ReviewError("Exactly one receipt is permitted")
    name = files[0].get("filename", "")
    match = re.fullmatch(
        r"harness/github_work_plan/agent_flow/([A-Za-z][A-Za-z0-9_-]{2,63})\.review-decisions\.json",
        name,
    )
    if match is None or files[0].get("status") not in ("added", "modified"):
        raise ReviewError("Only .review-decisions.json under agent_flow/ is allowed")
    plan = flow.load(match.group(1))
    content = api("/contents/" + name + "?ref=" + commit_sha)
    if content.get("encoding") != "base64":
        raise ReviewError("Unexpected receipt encoding")
    if int(content.get("size", 0)) > 65536:
        raise ReviewError("Receipt exceeds the maximum review size")
    decoded = base64.b64decode(content["content"], validate=True)
    if len(decoded) > 65536:
        raise ReviewError("Receipt exceeds the maximum decoded size")
    receipt = json.loads(decoded.decode("utf-8"))
    if (
        receipt.get("schema") != "openresearch.agent_flow.decisions.v1"
        or receipt.get("id") != plan["id"]
        or receipt.get("preview_sha256") != plan["preview_sha256"]
    ):
        raise ReviewError("Receipt does not match audited plan digest")
    decisions = receipt.get("decisions", [])
    if receipt.get("authorize") is True:
        if decisions:
            raise ReviewError("Preflight authorization requires its own review")
        flow.authorize(plan, reviewer, human=True, channel="github_pr_review")
    else:
        if not isinstance(decisions, list) or not decisions or len(decisions) > len(plan["items"]):
            raise ReviewError("Bounded decisions required")
        ids = [x.get("key") for x in decisions]
        if len(ids) != len(set(ids)):
            raise ReviewError("Duplicate receipt nodes")
        for row in decisions:
            flow.decide(
                plan,
                row.get("key"),
                row.get("decision"),
                reviewer,
                marks=row.get("marks", []),
                reason=row.get("reason", ""),
                human=True,
                channel="github_pr_review",
            )
        if receipt.get("finalize") is True:
            flow.finalize(plan)
    files_to_commit = [f"harness/github_work_plan/agent_flow/{plan['id']}.json"]
    files_to_commit.extend(plan.get("adopted_authority_paths", []))
    if plan["phase"] == "finalized":
        files_to_commit.append(f"harness/github_work_plan/agent_flow/{plan['id']}.final.json")
    subprocess.run(["git", "config", "user.name", "openresearch-review[bot]"], check=True)
    subprocess.run(
        ["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
        check=True,
    )
    subprocess.run(["git", "add", "--", *files_to_commit], check=True)
    subprocess.run(["git", "commit", "-m", f"review(flow): {plan['id']} by {reviewer}"], check=True)
    subprocess.run(["git", "push", "origin", "HEAD:main"], check=True)
    print(
        json.dumps(
            {"plan": plan["id"], "phase": plan["phase"], "reviewer": reviewer, "receipt_pr": number}
        )
    )


if __name__ == "__main__":
    try:
        execute()
    except (ReviewError, flow.FlowError, ValueError, KeyError, OSError) as exc:
        print("GitHub approval rejected:", exc, file=sys.stderr)
        raise SystemExit(1)
