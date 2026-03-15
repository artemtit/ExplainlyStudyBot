# Step 47 — Support in Profile

1. Problem being solved in this step: Users in the profile screen do not see a quick support link.
2. Proposed technical solution: Add a support button to the profile keyboard when `SUPPORT_URL` is configured.
3. Implementation plan: Extend `create_profile_keyboard` with an optional support URL; pass `support_url` into the profile router.
4. Example code or pseudocode (if relevant): `create_profile_keyboard(support_url=...)`.
5. Risks and edge cases: None; button is optional.
6. Suggested next improvements: Add a feedback shortcut from profile.
