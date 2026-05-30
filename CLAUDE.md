# Project notes for Claude

## Resolving PR review threads from a Claude Code sandbox

**This now works from inside the sandbox** (updated 2026-05-29). Earlier the
GitHub MCP server couldn't resolve threads, but the current server does:

- `pull_request_read` with `method: get_review_comments` **does** return each
  thread's GraphQL node ID in the `ID` field (e.g. `PRRT_kwDO...`), along with
  `is_resolved` so you can tell which are still open.
- `pull_request_review_write` with `method: resolve_thread` and that `threadId`
  resolves the conversation (returns "review thread resolved successfully").
  `unresolve_thread` reverses it.

**Workflow:** fetch threads with `get_review_comments`, filter to
`is_resolved == false`, post your reply via `add_reply_to_pull_request_comment`,
then call `resolve_thread` with the thread's `ID`. Note `get_review_comments`
is paginated — check `pageInfo.hasNextPage` and follow `endCursor` via the
`after` param, since open threads may live on a later page than resolved ones.

If `resolve_thread` ever regresses, the local fallback is:
```bash
gh api graphql -f query='
  { repository(owner:"OWNER", name:"REPO") { pullRequest(number:N) {
      reviewThreads(first:50) { nodes { id isResolved } } } } }' \
  --jq '.data.repository.pullRequest.reviewThreads.nodes[]
        | select(.isResolved==false) | .id' \
| xargs -I{} gh api graphql -f query='
    mutation { resolveReviewThread(input:{threadId:"{}"})
               { thread { id isResolved } } }'
```

## RuntimeState concurrency: serialize runner jobs per PR

The orchestrator persists `RuntimeState` in a single hidden PR comment and
guards concurrent writers with an **optimistic lock** on `updated_at`
(`_save_state` checks the expected `updated_at` before editing, and re-reads
after editing to confirm our write landed — raising `StateConflictError` if a
concurrent runner clobbered it). This is best-effort: GitHub's issue-comment
`PATCH` has **no conditional/If-Match support**, so the read-then-PATCH window
cannot be closed at the API level — two runners racing within that window can
still interleave, and the re-read only *detects* it after the fact.

**The real fix is deployment-level:** run the orchestrator workflow under a
GitHub Actions [`concurrency:`](https://docs.github.com/actions/using-jobs/using-concurrency)
group keyed by PR number so only one runner job is in flight per PR at a time,
e.g.

```yaml
concurrency:
  group: ai-pr-orchestrator-${{ github.event.pull_request.number || github.event.issue.number }}
  cancel-in-progress: false
```

With that group in place the optimistic lock + re-read is a backstop, not the
primary serialization mechanism.
