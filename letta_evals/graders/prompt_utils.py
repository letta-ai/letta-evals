"""Prompt construction for rubric-based graders.

A rubric is a Jinja template. It is rendered in Jinja's sandbox and sent
verbatim to the judge: the framework adds no wrapping text, headers or system
prompt of its own.

Variables available to every rubric:
    {{ input }}         the sample's input (a string, or a list of turns)
    {{ ground_truth }}  the expected answer (empty string if not provided)
    {{ submission }}    the agent's extracted output

Every key in ``sample.rubric_vars`` is also available, with its value as given,
so lists and dicts can be looped over (``{% for c in criteria %}``).

Only ``{{ ... }}`` is a placeholder, so literal braces such as JSON examples need
no escaping. Referencing a variable the sample doesn't provide is an error.
"""

import re
from functools import lru_cache
from typing import Any, Dict, Iterable

from jinja2 import StrictUndefined, Template, meta
from jinja2.sandbox import SandboxedEnvironment

from letta_evals.models import Sample

RESERVED_VARS = ("input", "ground_truth", "submission")

# The sandbox matters: a rubric can come from a dataset row, including a
# downloaded one, and unsandboxed Jinja can reach Python internals.
_ENV = SandboxedEnvironment(undefined=StrictUndefined, keep_trailing_newline=True)

_SINGLE_BRACE = re.compile(r"(?<!\{)\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}(?!\})")


@lru_cache(maxsize=256)
def _compile(rubric: str) -> Template:
    return _ENV.from_string(rubric)


@lru_cache(maxsize=256)
def template_variables(rubric: str) -> frozenset:
    """Return the top-level variables a rubric references.

    Raises:
        jinja2.TemplateSyntaxError: If the rubric is not a valid template.
    """
    return frozenset(meta.find_undeclared_variables(_ENV.parse(rubric)))


def check_rubric(rubric: str, provided: Iterable[str]) -> None:
    """Check a rubric against the variable names a sample provides.

    Raises:
        ValueError: If the rubric uses ``{name}`` for a provided variable (the
            pre-Jinja placeholder syntax), or references a variable that isn't provided.
        jinja2.TemplateSyntaxError: If the rubric is not a valid template.
    """
    provided = set(provided)
    legacy = sorted({name for name in _SINGLE_BRACE.findall(rubric) if name in provided})
    if legacy:
        examples = ", ".join(f"{{{name}}} -> {{{{ {name} }}}}" for name in legacy)
        raise ValueError(f"Rubric uses single-brace placeholders, which are now literal text: {examples}")

    missing = template_variables(rubric) - provided
    if missing:
        raise ValueError(
            f"Rubric references variables the sample doesn't provide: {sorted(missing)}. "
            f"Available: {sorted(provided)}; add others via sample.rubric_vars."
        )


def judge_variables(sample: Sample, submission: str) -> Dict[str, Any]:
    """The variables a rubric is rendered with for one sample.

    Raises:
        ValueError: If ``sample.rubric_vars`` contains a reserved variable name.
    """
    variables: Dict[str, Any] = {
        "input": sample.input,
        "ground_truth": sample.ground_truth if sample.ground_truth is not None else "",
        "submission": submission,
    }
    for key, value in (sample.rubric_vars or {}).items():
        if key in RESERVED_VARS:
            raise ValueError(
                f"rubric_vars key {key!r} collides with a reserved variable. Reserved names: {RESERVED_VARS}"
            )
        variables[key] = value
    return variables


def build_judge_prompt(rubric: str, sample: Sample, submission: str) -> str:
    """Render a rubric template for one sample.

    Raises:
        ValueError: If ``sample.rubric_vars`` contains a reserved variable name,
            or the rubric fails ``check_rubric``.
        jinja2.TemplateError: If the rubric doesn't parse or fails while rendering.
    """
    variables = judge_variables(sample, submission)
    check_rubric(rubric, variables)
    return _compile(rubric).render(variables)
