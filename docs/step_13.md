# Step 13 — Material Source Status

1. Problem being solved in this step: Users have no visibility into whether content was generated fresh or loaded from cache/database.
2. Proposed technical solution: Show a brief status message when a lesson loads indicating the material source.
3. Implementation plan: Map the source (`llm`, `cache`, `db`) to user-facing strings and send a short message before rendering the lesson; apply both for new lessons and resume.
4. Example code or pseudocode (if relevant): `status_map = {"cache": LOADED_CACHE, "db": LOADED_DB, "llm": LOADED_NEW}`.
5. Risks and edge cases: Too many messages if users spam requests; messages are short and informational.
6. Suggested next improvements: Provide a toggle in settings to hide status messages.
