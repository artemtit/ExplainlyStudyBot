# Step 4 — Cancel Current Flow

1. Problem being solved in this step: Users can get stuck mid-flow (topic entry, test, etc.) without a clear way to reset and return to the main menu.
2. Proposed technical solution: Add a global `/cancel` command that clears the FSM state and shows the main menu.
3. Implementation plan: Add a `/cancel` handler in the start router; reuse `show_main_menu` with a short confirmation message.
4. Example code or pseudocode (if relevant): `/cancel` -> `state.clear()` -> `show_main_menu(text=CANCEL_TEXT)`.
5. Risks and edge cases: Users might accidentally cancel a flow; the handler should be safe even when no state is set.
6. Suggested next improvements: Add a “Cancel” inline button during long flows; log cancellations for UX tuning.
