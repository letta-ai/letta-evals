"""Tests for pure parsing functions."""

import pytest
from pathlib import Path

# Import functions under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_dataset import format_additional_files, read_skill_files, get_skill_file_tree


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
