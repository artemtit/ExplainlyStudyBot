# Step 11 — Unknown Command Guidance

1. Problem being solved in this step: Users can enter unknown slash commands and receive no feedback.
2. Proposed technical solution: Add a fallback handler for unknown commands that points to available commands and returns to the main menu.
3. Implementation plan: Add a final message handler in the start router that matches any `/` command and responds with a short message plus main menu.
4. Example code or pseudocode (if relevant): `if text.startswith('/') -> reply unknown -> show_main_menu()`.
5. Risks and edge cases: Handler order must not override valid commands; keep it after specific command handlers.
6. Suggested next improvements: Add command suggestions in the help text based on common typos.
