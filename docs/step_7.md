# Step 7 — Clear Topic Validation Feedback

1. Problem being solved in this step: Users receive a generic prompt when their topic is too short or too long, which is not helpful.
2. Proposed technical solution: Provide explicit feedback for short/long topics before re-prompting.
3. Implementation plan: Add formatting helpers for short/long topic messages; enforce min/max lengths in the topic handler and show the specific feedback.
4. Example code or pseudocode (if relevant): `if len < min: send too-short + prompt; if len > max: send too-long + prompt`.
5. Risks and edge cases: Multiple messages could feel noisy; keep messages concise.
6. Suggested next improvements: Add examples of good topics and a validation for repeated spam content.
