# Step 28 — Exit Hint in Topic Prompt

1. Problem being solved in this step: Users may not know how to exit the topic entry flow.
2. Proposed technical solution: Add a short hint about `/cancel` and `/menu` in the topic prompt.
3. Implementation plan: Extend `format_topic_prompt` with an exit hint line.
4. Example code or pseudocode (if relevant): `"Для выхода используйте /cancel или /menu."`.
5. Risks and edge cases: Keep the prompt concise to avoid overwhelming users.
6. Suggested next improvements: Show exit hints in other long flows as well.
