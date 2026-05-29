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
