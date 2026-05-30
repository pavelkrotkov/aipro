"""Real GitHub API client using httpx."""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import quote

import httpx

from ai_pr_orchestrator.github import graphql, models

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.github.com"
_GRAPHQL_URL = "https://api.github.com/graphql"
_MAX_RETRIES = 4
_INITIAL_BACKOFF = 1.0
_TOKEN_REDACT_RE = re.compile(r"((?:gh[pousr]|github_pat)_\w{4})\w+")


def _redact_token(text: str) -> str:
    return _TOKEN_REDACT_RE.sub(r"\1****", text)


class GitHubClientError(Exception):
    """Raised on unrecoverable GitHub API errors."""

    def __init__(self, message: str) -> None:
        super().__init__(_redact_token(message))


class GitHubClient:
    """GitHub REST and GraphQL client with rate-limit handling and dry-run support."""

    def __init__(
        self,
        token: str,
        owner: str,
        repo: str,
        *,
        dry_run: bool = False,
        base_url: str = _DEFAULT_BASE_URL,
        graphql_url: str = _GRAPHQL_URL,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._owner = owner
        self._repo = repo
        self._dry_run = dry_run
        self._base_url = base_url.rstrip("/")
        self._graphql_url = graphql_url
        self._client = http_client or httpx.Client(
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )
        self._owns_client = http_client is None
        # Per-(pr_number, head_sha) cache of changed files; see get_pr.
        self._pr_files_cache: dict[tuple[int, str], list[str]] = {}
        # Short-lived cache of review threads, keyed by pr_number. Within one
        # reviewer-poll tick the runner calls get_review_threads twice (once via
        # the reviewer's collect_findings, once via has_responded); caching
        # collapses that to a single GraphQL request. The runner calls
        # reset_request_cache() at the start of every tick so cross-tick
        # freshness is preserved — this is a within-tick memo, not a TTL cache.
        self._review_threads_cache: dict[int, list[models.ReviewThread]] = {}

    def reset_request_cache(self) -> None:
        """Drop within-tick memoized responses (currently review threads).

        Called by the runner between reviewer-poll iterations so each tick sees
        fresh data while still de-duplicating the repeated reads inside a single
        tick. The ``_pr_files_cache`` is intentionally NOT cleared: it is keyed
        by head SHA and only changes when the PR head moves.
        """
        self._review_threads_cache.clear()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- REST API ---

    def get_pr(self, number: int) -> models.PullRequest:
        data = self._get(f"/repos/{self._owner}/{self._repo}/pulls/{number}")
        # Guard every nested access: a missing or null ``head``/``base`` (from an
        # unexpected payload shape or a mock) would otherwise raise KeyError/
        # TypeError mid-construction. Use the same ``.get(...) or {}`` idiom
        # consistently for both branches.
        head_data = data.get("head") or {}
        base_data = data.get("base") or {}
        head_repo = head_data.get("repo") or {}
        is_fork = bool(head_repo.get("fork", False))
        head_sha = head_data.get("sha") or ""
        # Fetch changed files for safety checks (e.g. disallow_workflow_file_changes).
        # The runner refetches the PR on every transition/poll iteration, but the
        # changed-files list only moves when the head SHA does. Cache by
        # (number, head_sha) so repeated get_pr calls within a run don't issue a
        # redundant (paginated) /files request on every tick.
        cache_key = (number, head_sha)
        changed_files = self._pr_files_cache.get(cache_key)
        if changed_files is None:
            changed_files = self.get_pr_files(number)
            self._pr_files_cache[cache_key] = changed_files
        return models.PullRequest(
            number=data["number"],
            title=data["title"],
            body=data.get("body") or "",
            state=data["state"],
            head_sha=head_sha,
            head_ref=head_data.get("ref") or "",
            base_ref=base_data.get("ref") or "",
            author=(data.get("user") or {}).get("login", "ghost"),
            draft=data.get("draft", False),
            mergeable=data.get("mergeable"),
            labels=[label["name"] for label in data.get("labels", [])],
            is_fork=is_fork,
            changed_files=changed_files,
            author_association=data.get("author_association") or "",
        )

    def get_pr_files(self, pr_number: int) -> list[str]:
        data = self._get_paginated(
            f"/repos/{self._owner}/{self._repo}/pulls/{pr_number}/files",
            items_key=None,
        )
        return [item["filename"] for item in data if isinstance(item, dict) and "filename" in item]

    def post_comment(self, issue_number: int, body: str) -> models.Comment:
        data = self._post(
            f"/repos/{self._owner}/{self._repo}/issues/{issue_number}/comments",
            json={"body": body},
        )
        if data is None:
            return models.Comment(id=0, body=body, user="dry-run", created_at="", updated_at="")
        return _parse_comment(data)

    def edit_comment(self, comment_id: int, body: str) -> models.Comment:
        data = self._patch(
            f"/repos/{self._owner}/{self._repo}/issues/comments/{comment_id}",
            json={"body": body},
        )
        if data is None:
            return models.Comment(
                id=comment_id, body=body, user="dry-run", created_at="", updated_at=""
            )
        return _parse_comment(data)

    def add_label(self, issue_number: int, label: str) -> list[dict[str, Any]]:
        data = self._post(
            f"/repos/{self._owner}/{self._repo}/issues/{issue_number}/labels",
            json={"labels": [label]},
        )
        if data is None:
            return [{"name": label}]
        return data

    def remove_label(self, issue_number: int, label: str) -> None:
        self._delete(
            f"/repos/{self._owner}/{self._repo}/issues/{issue_number}/labels/{quote(label, safe='')}",
        )

    def get_comment(self, comment_id: int) -> models.Comment | None:
        """Fetch a single issue comment by id, or None if it no longer exists.

        Lets callers that already know the comment id (e.g. the runner's state
        comment) avoid paging the entire comment list. A deleted comment yields
        a 404, which we translate to None rather than propagating.
        """
        try:
            data = self._get(f"/repos/{self._owner}/{self._repo}/issues/comments/{comment_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        return _parse_comment(data)

    def get_pr_comments(self, issue_number: int) -> list[models.Comment]:
        data = self._get_paginated(
            f"/repos/{self._owner}/{self._repo}/issues/{issue_number}/comments",
            items_key=None,
        )
        return [_parse_comment(c) for c in data]

    def get_check_runs(self, ref: str) -> list[models.CheckRun]:
        data = self._get_paginated(
            f"/repos/{self._owner}/{self._repo}/commits/{quote(ref, safe='')}/check-runs",
            items_key="check_runs",
        )
        return [
            models.CheckRun(
                id=cr["id"],
                name=cr["name"],
                status=cr["status"],
                conclusion=cr.get("conclusion"),
                html_url=cr.get("html_url", ""),
            )
            for cr in data
        ]

    def get_commit_statuses(self, ref: str) -> list[models.CheckRun]:
        """Fetch legacy commit-status contexts and adapt them to CheckRun shape.

        Required checks that report via the Statuses API (rather than the
        Checks API) never appear in ``get_check_runs``. The orchestrator's CI
        gate consumes ``CheckRun``s, so each status context is mapped to one:
        ``state`` becomes the equivalent check ``status``/``conclusion``
        (``pending`` -> still-running, ``success`` -> passing, ``failure``/
        ``error`` -> failing). Status contexts have no numeric id, so ``id`` is
        a process-stable hash of the context name (see
        ``models.stable_check_run_id``).
        """
        data = self._get_paginated(
            f"/repos/{self._owner}/{self._repo}/commits/{quote(ref, safe='')}/statuses",
            items_key=None,
        )
        # The statuses endpoint returns newest-first and may list the same
        # context multiple times (one per update). Keep only the first (latest)
        # entry per context so the CI gate sees each context once.
        latest_by_context: dict[str, dict[str, Any]] = {}
        for status in data:
            if not isinstance(status, dict):
                continue
            context = status.get("context")
            if isinstance(context, str) and context not in latest_by_context:
                latest_by_context[context] = status
        runs: list[models.CheckRun] = []
        for context, status in latest_by_context.items():
            state = status.get("state")
            if state in ("success", "failure", "error"):
                check_status = "completed"
                conclusion = "success" if state == "success" else "failure"
            else:
                # "pending" (or any unknown state) => not yet conclusive.
                check_status = "in_progress"
                conclusion = None
            runs.append(
                models.CheckRun(
                    id=models.stable_check_run_id(context),
                    name=context,
                    status=check_status,
                    conclusion=conclusion,
                    html_url=status.get("target_url") or "",
                )
            )
        return runs

    def get_pull_request_reviews(self, pr_number: int) -> list[models.Review]:
        data = self._get_paginated(
            f"/repos/{self._owner}/{self._repo}/pulls/{pr_number}/reviews",
            items_key=None,
        )
        return [
            models.Review(
                id=review["id"],
                author=(review.get("user") or {}).get("login", "ghost"),
                body=review.get("body") or "",
                state=review.get("state") or "",
                submitted_at=review.get("submitted_at") or "",
            )
            for review in data
            if isinstance(review, dict) and "id" in review
        ]

    # --- GraphQL API ---

    def get_review_threads(self, pr_number: int) -> list[models.ReviewThread]:
        cached = self._review_threads_cache.get(pr_number)
        if cached is not None:
            return cached
        threads = self._fetch_review_threads(pr_number)
        self._review_threads_cache[pr_number] = threads
        return threads

    def _fetch_review_threads(self, pr_number: int) -> list[models.ReviewThread]:
        threads: list[models.ReviewThread] = []
        cursor: str | None = None

        while True:
            variables: dict[str, Any] = {
                "owner": self._owner,
                "repo": self._repo,
                "number": pr_number,
            }
            if cursor:
                variables["after"] = cursor

            data = self._graphql(graphql.REVIEW_THREADS_QUERY, variables)
            data_dict = data.get("data") or {}
            if "repository" in data_dict and data_dict["repository"] is None:
                raise GitHubClientError(
                    f"Repository {self._owner}/{self._repo} not found or inaccessible"
                )
            repo_dict = data_dict.get("repository") or {}
            if "pullRequest" in repo_dict and repo_dict["pullRequest"] is None:
                raise GitHubClientError(
                    f"Pull request #{pr_number} not found in {self._owner}/{self._repo}"
                )
            pr_dict = repo_dict.get("pullRequest") or {}
            thread_connection = pr_dict.get("reviewThreads") or {}

            for node in thread_connection.get("nodes") or []:
                if not node:
                    continue
                comments = [
                    models.ReviewComment(
                        id=c["id"],
                        body=c["body"],
                        author=(c.get("author") or {}).get("login", ""),
                        path=c.get("path", ""),
                        created_at=c.get("createdAt", ""),
                    )
                    for c in (node.get("comments") or {}).get("nodes") or []
                    if c
                ]
                threads.append(
                    models.ReviewThread(
                        id=node["id"],
                        is_resolved=node["isResolved"],
                        is_outdated=node["isOutdated"],
                        path=node.get("path", ""),
                        comments=comments,
                    )
                )

            page_info = thread_connection.get("pageInfo") or {}
            if page_info.get("hasNextPage"):
                cursor = page_info.get("endCursor")
            else:
                break

        return threads

    def reply_to_review_thread(self, thread_id: str, body: str) -> dict[str, Any] | None:
        # Mutating a thread invalidates the within-tick threads cache.
        self._review_threads_cache.clear()
        if self._dry_run:
            logger.info("DRY-RUN: would reply to review thread %s", thread_id)
            return {"comment": {"id": "dry-run"}}
        return self._graphql(
            graphql.REPLY_TO_REVIEW_THREAD_MUTATION,
            {"threadId": thread_id, "body": body},
        )

    def resolve_review_thread(self, thread_id: str) -> dict[str, Any] | None:
        # Mutating a thread invalidates the within-tick threads cache.
        self._review_threads_cache.clear()
        if self._dry_run:
            logger.info("DRY-RUN: would resolve review thread %s", thread_id)
            return {"thread": {"id": thread_id, "isResolved": True}}
        return self._graphql(
            graphql.RESOLVE_REVIEW_THREAD_MUTATION,
            {"threadId": thread_id},
        )

    # --- HTTP helpers ---

    def _get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path).json()

    def _post(self, path: str, *, json: Any) -> Any:
        if self._dry_run:
            logger.info("DRY-RUN: would POST %s", path)
            return None
        return self._request("POST", path, json=json).json()

    def _patch(self, path: str, *, json: Any) -> Any:
        if self._dry_run:
            logger.info("DRY-RUN: would PATCH %s", path)
            return None
        return self._request("PATCH", path, json=json).json()

    def _delete(self, path: str) -> None:
        if self._dry_run:
            logger.info("DRY-RUN: would DELETE %s", path)
            return
        self._request("DELETE", path)

    def _get_paginated(self, path: str, *, items_key: str | None = None) -> list[dict[str, Any]]:
        all_items: list[dict[str, Any]] = []
        url: str | None = f"{self._base_url}{path}"

        while url:
            response = self._request("GET", url, absolute_url=True)
            body = response.json()
            if items_key is not None:
                if not isinstance(body, dict):
                    raise GitHubClientError(
                        f"Expected a dict response with key {items_key!r},"
                        f" got {type(body).__name__}"
                    )
                all_items.extend(body.get(items_key, []))
            elif isinstance(body, list):
                all_items.extend(body)
            else:
                raise GitHubClientError(f"Expected a list response, got {type(body).__name__}")
            url = _parse_next_link(response.headers.get("link", ""))

        return all_items

    def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        response = self._request(
            "POST",
            self._graphql_url,
            json={"query": query, "variables": variables},
            absolute_url=True,
        )
        body: dict[str, Any] = response.json()
        if "errors" in body:
            raise GitHubClientError(f"GraphQL error: {body['errors']}")
        return body

    def _request(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        absolute_url: bool = False,
    ) -> httpx.Response:
        full_url = url if absolute_url else f"{self._base_url}{url}"

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.request(method, full_url, json=json)
            except httpx.RequestError as exc:
                if attempt < _MAX_RETRIES - 1:
                    wait = _INITIAL_BACKOFF * (2**attempt)
                    logger.warning(
                        "Network error (%s), retrying in %.1fs (attempt %d/%d)",
                        exc,
                        wait,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise GitHubClientError(
                    f"Network error after {_MAX_RETRIES} retries: {exc}"
                ) from exc

            logger.debug(
                "%s %s -> %d",
                method,
                full_url,
                response.status_code,
            )

            if response.status_code in (429, 500, 502, 503, 504) or (
                response.status_code == 403 and _is_rate_limited(response)
            ):
                wait = _retry_wait(response, attempt)
                if attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "Transient error (%d), retrying in %.1fs (attempt %d/%d)",
                        response.status_code,
                        wait,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise GitHubClientError(
                    f"Request failed after {_MAX_RETRIES} retries: "
                    f"{method} {full_url} -> {response.status_code}"
                )

            response.raise_for_status()
            return response

        raise GitHubClientError(f"Request failed after {_MAX_RETRIES} retries: {method} {full_url}")


def _parse_comment(data: dict[str, Any]) -> models.Comment:
    return models.Comment(
        id=data["id"],
        body=data["body"],
        user=(data.get("user") or {}).get("login", "ghost"),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


def _is_rate_limited(response: httpx.Response) -> bool:
    remaining = response.headers.get("x-ratelimit-remaining")
    if remaining == "0":
        return True
    return "retry-after" in response.headers


def _retry_wait(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    reset_at = response.headers.get("x-ratelimit-reset")
    if reset_at:
        try:
            wait = float(reset_at) - time.time()
            if wait > 0:
                return min(wait, 60.0)
        except ValueError:
            pass

    return _INITIAL_BACKOFF * (2**attempt)


def _parse_next_link(link_header: str) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' in part:
            url_match = re.search(r"<([^>]+)>", part)
            if url_match:
                return url_match.group(1)
    return None
