# Project notes for Claude

## Resolving PR review threads from a Claude Code sandbox

**Known limitation:** the GitHub MCP server in the Claude Code remote sandbox
cannot resolve PR review threads. The `resolve_thread` endpoint requires the
thread's GraphQL node ID (`PRRT_*`), but `pull_request_read get_review_comments`
omits that field from its response (despite the tool description claiming
otherwise). No other MCP method exposes it, and the sandbox has no `gh` CLI,
no `GITHUB_TOKEN`, and no GraphQL passthrough.

**Do not try to resolve threads from inside the sandbox.** Don't guess node IDs,
don't brute-force, don't waste turns on it.

**What to do instead:** post the reply (which works fine via
`add_reply_to_pull_request_comment`), then tell the user one of:
- Click "Resolve conversation" in the GitHub UI.
- Or run locally:
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
