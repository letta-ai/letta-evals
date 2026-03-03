"""Unit tests for letta_evals.datasets.loader module."""

import json
import tempfile
from pathlib import Path

import pytest

from letta_evals.datasets.loader import (
    _parse_json_dict_field,
    _parse_string_or_list,
    load_csv,
    load_dataset,
    load_jsonl,
)

# ── _parse_string_or_list ──


class TestParseStringOrList:
    def test_plain_string(self):
        assert _parse_string_or_list("hello", "input", 0) == "hello"

    def test_json_array(self):
        assert _parse_string_or_list('["a", "b"]', "input", 0) == ["a", "b"]

    def test_whitespace_stripped(self):
        assert _parse_string_or_list('  ["a"]  ', "input", 0) == ["a"]

    def test_invalid_json_array(self):
        """Starts with [ and ends with ] but invalid JSON inside."""
        with pytest.raises(ValueError, match="invalid"):
            _parse_string_or_list("[not, json]", "input", 0)

    def test_non_bracket_json_treated_as_string(self):
        """JSON that doesn't start with [ is treated as plain string."""
        assert _parse_string_or_list('{"a": 1}', "input", 0) == '{"a": 1}'

    def test_partial_bracket_treated_as_string(self):
        """Starts with [ but doesn't end with ] — treated as plain string."""
        assert _parse_string_or_list("[not closed", "input", 0) == "[not closed"


# ── _parse_json_dict_field ──


class TestParseJsonDictField:
    def _make_df_and_row(self, field_name, value):
        import pandas as pd

        df = pd.DataFrame([{field_name: value}])
        return df, df.iloc[0]

    def test_valid_json_dict(self):
        df, row = self._make_df_and_row("agent_args", '{"key": "value"}')
        result = _parse_json_dict_field(df, row, "agent_args", 0)
        assert result == {"key": "value"}

    def test_missing_column(self):
        import pandas as pd

        df = pd.DataFrame([{"other": "x"}])
        result = _parse_json_dict_field(df, df.iloc[0], "agent_args", 0)
        assert result is None

    def test_null_value(self):
        import pandas as pd

        df = pd.DataFrame([{"agent_args": None}])
        result = _parse_json_dict_field(df, df.iloc[0], "agent_args", 0)
        assert result is None

    def test_non_dict_json(self):
        df, row = self._make_df_and_row("agent_args", "[1, 2]")
        with pytest.raises(ValueError, match="must be a JSON object/dict"):
            _parse_json_dict_field(df, row, "agent_args", 0)

    def test_invalid_json(self):
        df, row = self._make_df_and_row("agent_args", "{bad json")
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_json_dict_field(df, row, "agent_args", 0)


# ── load_jsonl ──


class TestLoadJsonl:
    def _write_jsonl(self, samples: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for s in samples:
            tmp.write(json.dumps(s) + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_basic_load(self):
        path = self._write_jsonl(
            [
                {"input": "hello", "ground_truth": "world"},
                {"input": "foo", "ground_truth": "bar"},
            ]
        )
        samples = list(load_jsonl(path))
        assert len(samples) == 2
        assert samples[0].input == "hello"
        assert samples[0].ground_truth == "world"
        assert samples[1].id == 1

    def test_max_samples(self):
        path = self._write_jsonl([{"input": f"q{i}"} for i in range(10)])
        samples = list(load_jsonl(path, max_samples=3))
        assert len(samples) == 3

    def test_optional_fields(self):
        path = self._write_jsonl(
            [
                {
                    "input": "test",
                    "agent_args": {"key": "val"},
                    "rubric_vars": {"v": 1},
                    "extra_vars": {"e": "x"},
                }
            ]
        )
        samples = list(load_jsonl(path))
        assert samples[0].agent_args == {"key": "val"}
        assert samples[0].rubric_vars == {"v": 1}
        assert samples[0].extra_vars == {"e": "x"}

    def test_no_ground_truth(self):
        path = self._write_jsonl([{"input": "test"}])
        samples = list(load_jsonl(path))
        assert samples[0].ground_truth is None

    def test_malformed_json_line(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write("{bad json}\n")
        tmp.close()
        with pytest.raises(json.JSONDecodeError):
            list(load_jsonl(Path(tmp.name)))

    def test_missing_input_key(self):
        path = self._write_jsonl([{"ground_truth": "answer"}])
        with pytest.raises(KeyError):
            list(load_jsonl(path))


# ── load_csv ──


class TestLoadCsv:
    def _write_csv(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_basic_load(self):
        path = self._write_csv("input,ground_truth\nhello,world\nfoo,bar\n")
        samples = list(load_csv(path))
        assert len(samples) == 2
        assert samples[0].input == "hello"
        assert samples[0].ground_truth == "world"

    def test_max_samples(self):
        rows = "input\n" + "\n".join(f"q{i}" for i in range(10)) + "\n"
        path = self._write_csv(rows)
        samples = list(load_csv(path, max_samples=3))
        assert len(samples) == 3

    def test_json_dict_fields(self):
        path = self._write_csv(
            'input,agent_args,rubric_vars,extra_vars\nhello,"{""k"":""v""}","{""r"":1}","{""e"":""x""}"\n'
        )
        samples = list(load_csv(path))
        assert samples[0].agent_args == {"k": "v"}
        assert samples[0].rubric_vars == {"r": 1}
        assert samples[0].extra_vars == {"e": "x"}

    def test_missing_input_column(self):
        path = self._write_csv("other\nvalue\n")
        with pytest.raises(ValueError, match="missing required column 'input'"):
            list(load_csv(path))

    def test_empty_csv(self):
        path = self._write_csv("input\n")
        with pytest.raises(ValueError, match="empty"):
            list(load_csv(path))

    def test_null_input(self):
        path = self._write_csv("input,ground_truth\n,world\n")
        with pytest.raises(ValueError, match="cannot be null"):
            list(load_csv(path))

    def test_list_input(self):
        path = self._write_csv('input,ground_truth\n"[""q1"",""q2""]","[""a1"",""a2""]"\n')
        samples = list(load_csv(path))
        assert samples[0].input == ["q1", "q2"]
        assert samples[0].ground_truth == ["a1", "a2"]

    def test_list_length_mismatch(self):
        """Input and ground_truth lists with different lengths should raise."""
        path = self._write_csv('input,ground_truth\n"[""q1"",""q2""]","[""a1""]"\n')
        with pytest.raises(ValueError, match="Failed to create Sample"):
            list(load_csv(path))


# ── load_dataset ──


class TestLoadDataset:
    def test_jsonl_dispatch(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write(json.dumps({"input": "test"}) + "\n")
        tmp.close()
        samples = list(load_dataset(tmp.name))
        assert len(samples) == 1

    def test_csv_dispatch(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp.write("input\nhello\n")
        tmp.close()
        samples = list(load_dataset(tmp.name))
        assert len(samples) == 1

    def test_unsupported_format(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write("hello")
        tmp.close()
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            list(load_dataset(tmp.name))

    def test_nonexistent_file(self):
        with pytest.raises(ValueError, match="does not exist"):
            list(load_dataset("/tmp/nonexistent_file_12345.jsonl"))
