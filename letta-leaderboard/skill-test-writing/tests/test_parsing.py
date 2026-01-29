"""Tests for pure parsing functions."""

import pytest
from pathlib import Path

# Import functions under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_extractor import extract_json_from_message, load_test_context, format_for_judge
from parse_results import slugify, extract_skill_name, extract_model_short_name
from generate_dataset import format_additional_files, read_skill_files, get_skill_file_tree


class TestExtractJsonFromMessage:
    """Tests for extract_json_from_message()."""

    def test_simple_json(self):
        msg = '{"sandbox_path": "/tmp/test"}'
        assert extract_json_from_message(msg) == {"sandbox_path": "/tmp/test"}

    def test_json_in_code_block(self):
        msg = """Here's the result:
```json
{"sandbox_path": "/tmp/my-test"}
```
"""
        assert extract_json_from_message(msg) == {"sandbox_path": "/tmp/my-test"}

    def test_json_in_unmarked_code_block(self):
        msg = """Done:
```
{"sandbox_path": "/tmp/foo"}
```
"""
        assert extract_json_from_message(msg) == {"sandbox_path": "/tmp/foo"}

    def test_json_with_extra_fields(self):
        msg = '{"sandbox_path": "/tmp/x", "note": "extra"}'
        result = extract_json_from_message(msg)
        assert result["sandbox_path"] == "/tmp/x"

    def test_multiple_json_prefers_sandbox_path_first_key(self):
        msg = """
{"other": "value"}
{"sandbox_path": "/tmp/correct", "foo": "bar"}
{"foo": "bar", "sandbox_path": "/tmp/wrong-order"}
"""
        result = extract_json_from_message(msg)
        assert result["sandbox_path"] == "/tmp/correct"

    def test_json_embedded_in_text(self):
        msg = 'I created the test at {"sandbox_path": "/tmp/test"} as requested.'
        assert extract_json_from_message(msg) == {"sandbox_path": "/tmp/test"}

    def test_no_sandbox_path_raises(self):
        msg = '{"foo": "bar"}'
        with pytest.raises(ValueError, match="No JSON with 'sandbox_path'"):
            extract_json_from_message(msg)

    def test_invalid_json_raises(self):
        msg = "no json here"
        with pytest.raises(ValueError, match="No JSON with 'sandbox_path'"):
            extract_json_from_message(msg)

    def test_nested_json(self):
        msg = '{"sandbox_path": "/tmp/test", "config": {"key": "value"}}'
        result = extract_json_from_message(msg)
        assert result["sandbox_path"] == "/tmp/test"


class TestSlugify:
    """Tests for slugify()."""

    def test_simple_text(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_characters(self):
        assert slugify("Test: With (special) chars!") == "test-with-special-chars"

    def test_max_length(self):
        long_text = "a" * 100
        assert len(slugify(long_text, max_len=50)) == 50

    def test_underscores_to_dashes(self):
        assert slugify("foo_bar_baz") == "foo-bar-baz"

    def test_strips_trailing_dashes(self):
        assert slugify("test---") == "test"

    def test_multiple_spaces(self):
        assert slugify("hello    world") == "hello-world"


class TestExtractSkillName:
    """Tests for extract_skill_name()."""

    def test_extracts_skill_name(self):
        text = "Some text\n**Skill Name:** my-skill\nMore text"
        assert extract_skill_name(text) == "my-skill"

    def test_no_skill_name_returns_none(self):
        text = "No skill name here"
        assert extract_skill_name(text) is None

    def test_skill_name_with_spaces(self):
        text = "**Skill Name:** hugging-face-datasets"
        assert extract_skill_name(text) == "hugging-face-datasets"

    def test_multiline_input(self):
        text = """## The Skill

**Skill Name:** webapp-testing

**SKILL.md Content:**
```markdown
content here
```"""
        assert extract_skill_name(text) == "webapp-testing"


class TestExtractModelShortName:
    """Tests for extract_model_short_name()."""

    def test_anthropic_model(self):
        assert extract_model_short_name("anthropic/claude-haiku-4-5-20251001") == "claude-haiku-4-5"

    def test_openai_model(self):
        assert extract_model_short_name("openai/gpt-4o-20240513") == "gpt-4o"

    def test_no_provider(self):
        assert extract_model_short_name("claude-sonnet-4-5-20250929") == "claude-sonnet-4-5"

    def test_no_date_suffix(self):
        assert extract_model_short_name("gpt-4-turbo") == "gpt-4-turbo"


class TestFormatAdditionalFiles:
    """Tests for format_additional_files()."""

    def test_empty_list(self):
        assert format_additional_files([]) == ""

    def test_single_file(self):
        files = [("README.md", "# Title\nContent")]
        result = format_additional_files(files)
        assert "## Additional Skill Files" in result
        assert "**README.md:**" in result
        assert "# Title" in result

    def test_limits_to_two_files(self):
        files = [
            ("a.md", "content a"),
            ("b.md", "content b"),
            ("c.md", "content c"),
        ]
        result = format_additional_files(files)
        assert "**a.md:**" in result
        assert "**b.md:**" in result
        assert "**c.md:**" not in result


class TestLoadTestContext:
    """Tests for load_test_context()."""

    def test_loads_test_yaml(self):
        files = [(Path("test.yaml"), "name: my-test\nprompt: do something")]
        result = load_test_context(files)
        assert result["test_config"]["name"] == "my-test"
        assert result["grader_code"] is None

    def test_loads_grader_py(self):
        files = [
            (Path("test.yaml"), "name: test"),
            (Path("grader.py"), "def grade(): pass"),
        ]
        result = load_test_context(files)
        assert result["grader_code"] == "def grade(): pass"

    def test_invalid_yaml(self):
        files = [(Path("test.yaml"), "invalid: yaml: content:")]
        result = load_test_context(files)
        assert "error" in result["test_config"]


class TestFormatForJudge:
    """Tests for format_for_judge()."""

    def test_formats_prompt(self):
        context = {
            "test_config": {"prompt": "Do this task"},
            "grader_code": None,
        }
        result = format_for_judge(context)
        assert result["prompt"] == "Do this task"

    def test_formats_grader_config(self):
        context = {
            "test_config": {
                "prompt": "task",
                "grader": {"kind": "letta_judge", "prompt": "score it"},
            },
            "grader_code": None,
        }
        result = format_for_judge(context)
        assert "letta_judge" in result["grader_config"]

    def test_formats_grader_code(self):
        context = {
            "test_config": {"prompt": "task"},
            "grader_code": "def grade(): return 1.0",
        }
        result = format_for_judge(context)
        assert "def grade()" in result["grader_code_section"]

    def test_truncates_long_grader_code(self):
        context = {
            "test_config": {"prompt": "task"},
            "grader_code": "x" * 4000,
        }
        result = format_for_judge(context)
        assert "(truncated)" in result["grader_code_section"]


class TestReadSkillFiles:
    """Tests for read_skill_files() - uses tmp_path fixture."""

    def test_reads_skill_md(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill\nContent here")

        name, content, additional = read_skill_files(skill_dir)
        assert name == "my-skill"
        assert "# My Skill" in content
        assert additional == []

    def test_extracts_name_from_frontmatter(self, tmp_path):
        skill_dir = tmp_path / "dir-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: actual-name\n---\n# Skill")

        name, content, _ = read_skill_files(skill_dir)
        assert name == "actual-name"

    def test_reads_additional_md_files(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Main")
        (skill_dir / "GOTCHAS.md").write_text("# Gotchas")

        _, _, additional = read_skill_files(skill_dir)
        assert len(additional) == 1
        assert additional[0][0] == "GOTCHAS.md"

    def test_no_skill_md_returns_none(self, tmp_path):
        skill_dir = tmp_path / "empty"
        skill_dir.mkdir()

        name, content, additional = read_skill_files(skill_dir)
        assert name is None
        assert content is None


class TestGetSkillFileTree:
    """Tests for get_skill_file_tree()."""

    def test_lists_files(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("content")
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "helper.py").write_text("code")

        tree = get_skill_file_tree(skill_dir)
        assert "SKILL.md" in tree
        assert "scripts/helper.py" in tree

    def test_excludes_hidden_files(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("content")
        (skill_dir / ".hidden").write_text("secret")

        tree = get_skill_file_tree(skill_dir)
        assert ".hidden" not in tree

    def test_limits_to_20_files(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        for i in range(25):
            (skill_dir / f"file{i:02d}.txt").write_text("content")

        tree = get_skill_file_tree(skill_dir)
        lines = tree.strip().split("\n")
        assert len(lines) == 20
