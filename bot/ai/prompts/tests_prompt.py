from __future__ import annotations

from bot.ai.prompts.base import PromptTemplate


class TestsPrompt(PromptTemplate):
    name = "tests_generation"
    version = "1.0"

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            version=self.version,
            system_template=(
                "\u0422\u044b \u2014 \u0441\u0442\u0440\u043e\u0433\u0438\u0439 \u0442\u0435\u0441\u0442\u043e\u0432\u044b\u0439 \u0433\u0435\u043d\u0435\u0440\u0430\u0442\u043e\u0440. "
                "\u0412\u0435\u0440\u043d\u0438 \u0442\u043e\u043b\u044c\u043a\u043e JSON \u0438 \u043d\u0438\u0447\u0435\u0433\u043e \u0431\u043e\u043b\u044c\u0448\u0435. "
                "\u0421\u0442\u0440\u043e\u0433\u0430\u044f \u0441\u0442\u0440\u0443\u043a\u0442\u0443\u0440\u0430: {tests:[{question:string,options:[string,string,string,string],correct:string,explanation:string}]}. "
                "\u0420\u043e\u0432\u043d\u043e 5 \u0442\u0435\u0441\u0442\u043e\u0432, \u0432 \u043a\u0430\u0436\u0434\u043e\u043c 4 \u0432\u0430\u0440\u0438\u0430\u043d\u0442\u0430 \u043e\u0442\u0432\u0435\u0442\u0430, correct \u0442\u043e\u043b\u044c\u043a\u043e A/B/C/D, \u043a\u0430\u0436\u0434\u044b\u0439 \u0442\u0435\u0441\u0442 \u0441 explanation.\n"
                "Return ONLY valid JSON.\n"
                "Rules:\n"
                "- Do not wrap JSON in markdown\n"
                "- Do not use ```json\n"
                "- Do not add explanations\n"
                "- Output must start with { and end with }\n"
                "Options rules:\n"
                "- options must contain ONLY answer text\n"
                "- NEVER include A), B), C), D) or A., B., C., D.\n"
                "- NEVER include numbering prefixes\n"
                "- exactly 4 options\n"
                "- correct must be one of: A, B, C, D"
            ),
            user_template=(
                "\u0422\u0435\u043c\u0430: {topic}\n"
                "\u0421\u043b\u043e\u0436\u043d\u043e\u0441\u0442\u044c: {difficulty}\n"
                "\u0421\u0433\u0435\u043d\u0435\u0440\u0438\u0440\u0443\u0439 \u0442\u043e\u043b\u044c\u043a\u043e \u0442\u0435\u0441\u0442\u044b \u043f\u043e \u0441\u0445\u0435\u043c\u0435 tests."
            ),
            metadata={
                "type": "tests",
                "language": "ru",
                "temperature": 0.7,
                "max_tokens": 1500,
            },
        )


__all__ = ["TestsPrompt"]
