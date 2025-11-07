from pathlib import Path

import yaml


def build_tree_structure(files: list[str]) -> str:
    """
    Build a tree structure from a list of file paths.

    Args:
        files: List of file paths relative to a base directory

    Returns:
        A string representation of the tree structure
    """
    tree = {}

    for file_path in files:
        parts = Path(file_path).parts
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]

    def _build_tree_lines(node: dict, prefix: str = "", is_root: bool = True) -> list[str]:
        """Recursively build tree lines with proper formatting."""
        lines = []
        items = sorted(node.items())

        for i, (name, children) in enumerate(items):
            is_last_item = i == len(items) - 1

            if is_root:
                # Root level - no connector, but children get proper indentation
                connector = ""
                new_prefix = ""
                lines.append(name)
            else:
                connector = "└── " if is_last_item else "├── "
                new_prefix = prefix + ("    " if is_last_item else "│   ")
                lines.append(prefix + connector + name)

            if children:
                lines.extend(_build_tree_lines(children, new_prefix, is_root=False))

        return lines

    tree_lines = _build_tree_lines(tree)
    return "\n".join(tree_lines)


def extract_skill_metadata(skill_md_path: Path) -> str:
    """
    Extract name and description from YAML frontmatter in SKILL.md file.

    Args:
        skill_md_path: Path to the SKILL.md file

    Returns:
        Dictionary with 'name' and 'description' keys
    """
    with open(skill_md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if the file has YAML frontmatter
    if content.startswith("---"):
        # Find the closing ---
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter_text = parts[1]
            try:
                frontmatter = yaml.safe_load(frontmatter_text)
                metadata = {"name": frontmatter.get("name", ""), "description": frontmatter.get("description", "")}
            except yaml.YAMLError as e:
                print(f"Error parsing YAML frontmatter in {skill_md_path}: {e}")
                metadata = {"name": "", "description": ""}

    if not metadata["name"] or not metadata["description"]:
        return ""

    return f"- Skill: {metadata['name']}\n  - Description: {metadata['description']}\n"
