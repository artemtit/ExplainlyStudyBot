# Step 8 — Profile Activity Context

1. Problem being solved in this step: The profile screen shows only basic identity info and lacks progress context.
2. Proposed technical solution: Include the user’s streak and last topic in the profile view.
3. Implementation plan: Fetch stats in the profile handler and extend the profile template with `daily_streak` and `last_topic`.
4. Example code or pseudocode (if relevant): `stats = get_user_stats(); render streak + last_topic`.
5. Risks and edge cases: Stats may be missing for new users; default values should display safely.
6. Suggested next improvements: Show last active date and tests passed.
