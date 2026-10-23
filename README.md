# Invoice Action Agent (Render)

## Important limitations
This is a deployable starter, not a guarantee of full grader compliance. Review and test the decision logic against the actual synthetic packages. Do not connect it to a real payment rail. The agent proposes actions only; accepted results produce receipts, not actual bank transfers.

## Deploy
1. Create a GitHub repository and upload these files.
2. In Render, choose **New + → Blueprint**, connect the repository, and deploy `render.yaml`.
3. Set `PUBLIC_BASE_URL` to `https://YOUR-SERVICE.onrender.com` (without `/a2a`; the card appends `/a2a`).
4. Set `OPENAI_API_KEY` in the Render environment if you want model-backed document analysis. `OPENAI_BASE_URL` is optional for compatible providers. Never commit keys.
5. Wait for deploy. The A2A base URL to submit is:
   `https://YOUR-SERVICE.onrender.com/a2a/`
6. Copy the generated `AGENT_BEARER_TOKEN` from Render environment variables and use it as the exact Bearer token in grader requests.

## Routes
- `GET /.well-known/agent-card.json` (public)
- `POST /a2a/message:send`
- `GET /a2a/tasks/{id}`
- `GET /a2a/tasks`
- `POST /a2a/tasks/{id}:cancel`

Authenticated requests require `Authorization: Bearer <AGENT_BEARER_TOKEN>` and `A2A-Version: 1.0`. POST requests require `Content-Type: application/a2a+json`.

## Health check
The app's Render health check is `/health`. Add this endpoint to `main.py` before deployment if needed:
```python
@app.get("/health")
def health():
    return {"ok": True}
```

## Known items to harden before grading
- Use a transaction-level advisory lock or database row locks for cross-worker receipt/cancel atomicity; the included process lock is not sufficient across multiple workers.
- Confirm the grader's exact A2A envelope expectations and task state serialization.
- Add stricter continuation completeness checks if the grader requires one result for every proposal.
- Validate model output facts and evidence refs against source documents; never trust generated evidence without checking it appears in the decisive paragraph.
- Configure persistent Postgres; SQLite is not suitable for ephemeral production deployments.
