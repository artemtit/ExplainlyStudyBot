# Step 23 — Profile Progress Stats

1. Problem being solved in this step: The profile view lacks summary progress stats.
2. Proposed technical solution: Show topics learned, tests passed, and flashcards reviewed in the profile.
3. Implementation plan: Extend the profile template to include stats fields and fill them from `get_user_stats`.
4. Example code or pseudocode (if relevant): `topics = stats.get('topics_learned', 0)`.
5. Risks and edge cases: Stats may be missing; default to zero.
6. Suggested next improvements: Display last active date or badge levels.
