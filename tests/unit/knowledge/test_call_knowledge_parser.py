import pytest

from sales_agent.features.knowledge.parser import TranscriptParseError, parse_transcript_turns


def test_parse_transcript_turns_preserves_roles_timestamps_and_source_lines() -> None:
    turns = parse_transcript_turns(
        "用户0：您好， 0:00:01\n用户1：需要二十台电脑， 0:01:02.500"
    )

    assert [turn.turn_no for turn in turns] == [1, 2]
    assert [turn.speaker.value for turn in turns] == ["unknown", "unknown"]
    assert [turn.timestamp_ms for turn in turns] == [1000, 62500]
    assert turns[1].text == "需要二十台电脑"
    assert turns[1].raw_line == "用户1：需要二十台电脑， 0:01:02.500"


def test_parse_transcript_turns_rejects_unrecognized_nonempty_lines() -> None:
    with pytest.raises(TranscriptParseError, match="unrecognized transcript line"):
        parse_transcript_turns("用户0：您好\n这不是一个轮次")
