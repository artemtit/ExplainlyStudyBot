from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PromptInstance:
    name: str
    version: str
    variant: str
    system: str
    user: str
    metadata: dict[str, Any]
    variables: dict[str, Any]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    version: str
    system_template: str
    user_template: str
    metadata: dict[str, Any] = field(default_factory=dict)
    variant: str = "default"

    def render(self, **variables: Any) -> PromptInstance:
        system = self.system_template
        user = self.user_template.format_map(variables)
        model, temperature, max_tokens = _extract_llm_params(self.metadata)
        return PromptInstance(
            name=self.name,
            version=self.version,
            variant=self.variant,
            system=system,
            user=user,
            metadata=dict(self.metadata),
            variables=dict(variables),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


@dataclass(frozen=True, slots=True)
class _PromptVariant:
    template: PromptTemplate
    weight: float


class PromptRegistry:
    def __init__(self) -> None:
        self._templates: dict[str, list[_PromptVariant]] = {}

    def register(self, template: PromptTemplate, *, weight: float = 1.0) -> None:
        if weight <= 0:
            raise ValueError("weight must be positive")
        self._templates.setdefault(template.name, []).append(_PromptVariant(template=template, weight=weight))

    def register_version(self, template: PromptTemplate, *, weight: float = 1.0) -> None:
        self.register(template, weight=weight)

    def get(self, name: str, *, version: str | None = None, variant: str | None = None) -> PromptTemplate:
        candidates = list(self._templates.get(name, []))
        if not candidates:
            raise KeyError(f"Prompt not found: {name}")

        if version is not None:
            candidates = [item for item in candidates if item.template.version == version]
        if variant is not None:
            candidates = [item for item in candidates if item.template.variant == variant]

        if not candidates:
            raise KeyError(f"Prompt not found for name={name}, version={version}, variant={variant}")

        return candidates[0].template

    def choose(self, name: str, *, key: str | int | None = None) -> PromptTemplate:
        candidates = list(self._templates.get(name, []))
        if not candidates:
            raise KeyError(f"Prompt not found: {name}")
        if len(candidates) == 1:
            return candidates[0].template

        total = sum(item.weight for item in candidates)
        if total <= 0:
            return candidates[0].template

        if key is None:
            pick = random.random() * total
        else:
            digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
            pick = (int(digest[:16], 16) / 2**64) * total

        acc = 0.0
        for item in candidates:
            acc += item.weight
            if pick <= acc:
                return item.template

        return candidates[-1].template


class PromptBuilder:
    def __init__(
        self,
        registry: PromptRegistry,
        name: str,
        *,
        version: str | None = None,
        variant: str | None = None,
        ab_key: str | int | None = None,
    ) -> None:
        self._registry = registry
        self._name = name
        self._version = version
        self._variant = variant
        self._ab_key = ab_key

    def build(self, **variables: Any) -> PromptInstance:
        if self._version or self._variant:
            template = self._registry.get(self._name, version=self._version, variant=self._variant)
        else:
            template = self._registry.choose(self._name, key=self._ab_key)
        return template.render(**variables)


def _extract_llm_params(metadata: dict[str, Any]) -> tuple[str | None, float | None, int | None]:
    llm = metadata.get("llm")
    if isinstance(llm, dict):
        model = llm.get("model")
        temperature = llm.get("temperature")
        max_tokens = llm.get("max_tokens")
    else:
        model = metadata.get("model")
        temperature = metadata.get("temperature")
        max_tokens = metadata.get("max_tokens")

    return (
        str(model) if model is not None else None,
        float(temperature) if temperature is not None else None,
        int(max_tokens) if max_tokens is not None else None,
    )


__all__ = ["PromptTemplate", "PromptRegistry", "PromptBuilder", "PromptInstance"]
