from datetime import date

import pytest

from sales_agent.features.calls.import_workflow import CallImportParseError, parse_call_file


def test_parse_call_file_uses_directory_date_and_first_speaker(tmp_path) -> None:
    root = tmp_path / "week1"
    date_directory = root / "2026-03-02"
    date_directory.mkdir(parents=True)
    source = date_directory / "90328338.txt"
    source.write_text(
        "【通话ID：ignored】\n"
        "日期：ignored\n"
        "通话内容如下：\n"
        "用户0：喂， 0:00:01\n"
        "用户1：你好， 0:00:02\n"
        "-------\n",
        encoding="utf-8",
    )

    parsed = parse_call_file(source, root=root, fallback_date=date(2026, 9, 1))

    assert parsed.external_call_id == "90328338"
    assert parsed.call_date == date(2026, 3, 2)
    assert parsed.source_filename == "2026-03-02/90328338.txt"
    assert parsed.transcript_text == "用户0：喂， 0:00:01\n用户1：你好， 0:00:02"
    assert len(parsed.source_hash) == 64


def test_parse_call_file_accepts_conversation_starting_with_user_one(tmp_path) -> None:
    root = tmp_path / "week1"
    root.mkdir()
    source = root / "call.txt"
    source.write_text("头部\n用户1：你好， 0:00:01", encoding="utf-8")

    parsed = parse_call_file(source, root=root, fallback_date=date(2026, 9, 1))

    assert parsed.transcript_text == "用户1：你好， 0:00:01"


def test_parse_call_file_rejects_non_call_text(tmp_path) -> None:
    root = tmp_path / "week1"
    root.mkdir()
    source = root / "summary.txt"
    source.write_text("本周实际电话接通量：138", encoding="utf-8")

    with pytest.raises(CallImportParseError, match="no speaker line found"):
        parse_call_file(source, root=root, fallback_date=date(2026, 9, 1))
