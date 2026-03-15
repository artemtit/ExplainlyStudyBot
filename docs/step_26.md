# Step 26 — Practice From Lesson

1. Problem being solved in this step: Practice is only accessible after finishing tests, making it harder to reach directly.
2. Proposed technical solution: Add a “Practice” button to the lesson action keyboard.
3. Implementation plan: Extend the lesson keyboard with a practice button and handle `lesson:practice` by opening the practice view.
4. Example code or pseudocode (if relevant): `lesson:practice -> open_practice(show_solution=False)`.
5. Risks and edge cases: Ensure material exists; existing loader handles this.
6. Suggested next improvements: Add a “Next step” recommendation after practice.
