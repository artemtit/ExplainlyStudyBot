PROJECT: Explainly AI Tutor Bot

STACK:
Python
aiogram
Supabase/PostgreSQL
LLM API

RULES:
- clean architecture
- business logic only in services
- handlers must stay thin
- use type hints
- avoid overengineering
- project must stay runnable


=== PROMPT 001 ===
DONE
Create base project architecture.

Structure:

bot/
handlers/
services/
ai/
database/
utils/

Add __init__.py files.


=== PROMPT 002 ===
DONE
Create main.py.

Responsibilities:
- initialize bot
- configure aiogram
- register handlers
- start polling.


=== PROMPT 003 ===
DONE
Create bot/handlers/start.py.

Responsibilities:
- handle /start command
- explain how bot works
- show example topics.


=== PROMPT 004 ===
DONE
Create LessonService.

File:
services/lesson_service.py

Responsibilities:
- receive topic
- call LLM
- return explanation.


=== PROMPT 005 ===
DONE
Create QuestionService.

File:
services/question_service.py

Responsibilities:
- generate practice questions
- validate answers.


=== PROMPT 006 ===
DONE
Create ai/llm_client.py.

Responsibilities:
- send requests to LLM API
- retry failed calls
- parse responses.


=== PROMPT 007 ===
Create ai/prompts folder.

Add prompt templates:

explain_topic.txt
generate_questions.txt
evaluate_answer.txt


=== PROMPT 008 ===
Implement prompt loader.

File:
ai/prompts/loader.py

Responsibilities:
- load prompt templates
- format with variables.


=== PROMPT 009 ===
Create bot/handlers/lesson.py.

Responsibilities:
- receive topic from user
- call LessonService
- send explanation.


=== PROMPT 010 ===
Create bot/handlers/practice.py.

Responsibilities:
- ask practice questions
- receive answers.


=== PROMPT 011 ===
Implement question evaluation.

Flow:

user answer
↓
QuestionService
↓
LLM evaluation
↓
feedback


=== PROMPT 012 ===
Create utils/formatting.py.

Functions:
- format lesson text
- format questions.


=== PROMPT 013 ===
Create utils/validation.py.

Functions:
- validate topic input
- sanitize text.


=== PROMPT 014 ===
Create database/models.py.

Tables:

users
lessons
answers


=== PROMPT 015 ===
Create database/repository.py.

Responsibilities:
- store user
- store lesson request
- store answers.


=== PROMPT 016 ===
Connect LessonService to database.

Save:

topic
lesson
timestamp.


=== PROMPT 017 ===
Add logging system.

Use Python logging.


=== PROMPT 018 ===
Add environment config.

Create config.py.

Variables:

BOT_TOKEN
LLM_API_KEY
DATABASE_URL.


=== PROMPT 019 ===
Implement error handling middleware.

Handle:

API errors
invalid input
timeouts.


=== PROMPT 020 ===
Add basic caching for LLM responses.


=== PROMPT 021 ===
Add topic normalization.

Example:

"quadratic equations"
→ canonical topic.


=== PROMPT 022 ===
Add example generation to lessons.


=== PROMPT 023 ===
Add "3 practice questions" after lesson.


=== PROMPT 024 ===
Implement feedback if answer incorrect.


=== PROMPT 025 ===
Add hint generation.


=== PROMPT 026 ===
Improve Telegram message formatting.


=== PROMPT 027 ===
Add lesson retry command.


=== PROMPT 028 ===
Add topic suggestion system.


=== PROMPT 029 ===
Add user session tracking.


=== PROMPT 030 ===
Store learning history.


=== PROMPT 031 ===
Create analytics module.

Track:

lessons requested
topics.


=== PROMPT 032 ===
Add rate limiting.


=== PROMPT 033 ===
Add LLM cost logging.


=== PROMPT 034 ===
Optimize prompt usage.


=== PROMPT 035 ===
Improve prompt instructions for teaching.


=== PROMPT 036 ===
Add exam-style questions.


=== PROMPT 037 ===
Add difficulty levels.


=== PROMPT 038 ===
Add lesson summary generation.


=== PROMPT 039 ===
Add learning streak tracking.


=== PROMPT 040 ===
Prepare project for deployment on Render.
