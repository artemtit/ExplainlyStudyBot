# Step 1 — Foundation and Onboarding Snapshot

1. Problem being solved in this step: Establish a clear, runnable foundation so a new contributor or deploy pipeline can bring the Telegram tutor bot up quickly without guessing environment variables, boot commands, or required services.
2. Proposed technical solution: Document the current architecture and startup requirements, provide a complete `.env.example`, and expand README with a short quickstart and environment variable list aligned to the existing code.
3. Implementation plan: Add `.env.example` with required and optional variables; update README with setup, run, and test commands; add a short Step 1 summary file to capture the current stage and the rationale for these changes.
4. Example code or pseudocode (if relevant): Example environment variables are provided in `.env.example` and referenced in README.
5. Risks and edge cases: Missing or mismatched env vars can fail startup; Groq API key not configured disables LLM responses; Redis and metrics ports can conflict on shared hosts; local runs without Supabase credentials will fail user/material persistence.
6. Suggested next improvements: Add a minimal local dev script to validate env vars before boot; add a lightweight smoke test for `/start`; add a migration or schema doc for Supabase tables.
