"""GraphQL query strings and helpers for the GitHub API."""

REVIEW_THREADS_QUERY = """
query($owner: String!, $repo: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $after) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          isResolved
          isOutdated
          path
          comments(first: 100) {
            nodes {
              id
              body
              author { login }
              path
              createdAt
            }
          }
        }
      }
    }
  }
}
"""

REPLY_TO_REVIEW_THREAD_MUTATION = """
mutation($threadId: ID!, $body: String!) {
  addPullRequestReviewComment(input: {
    pullRequestReviewThreadId: $threadId
    body: $body
  }) {
    comment {
      id
    }
  }
}
"""

RESOLVE_REVIEW_THREAD_MUTATION = """
mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread {
      id
      isResolved
    }
  }
}
"""
