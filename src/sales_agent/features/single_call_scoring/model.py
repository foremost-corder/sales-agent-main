"""Strict-JSON model agents for layered single-call scoring."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from sales_agent.core.config import Settings
from sales_agent.features.knowledge.contracts import TranscriptTurn
from sales_agent.features.single_call_scoring.contracts import (
    CallLevel,
    CallLevelClassification,
    DimensionJudgment,
    EvidenceReviewOutput,
    ModuleJudgmentOutput,
    WhiteboardRule,
    WhiteboardScoringModule,
    WhiteboardSpeakerAssignment,
)
from sales_agent.integrations.strict_json_completion import complete_strict_json


class WhiteboardModelOutputError(ValueError):
    pass


class _AgentBase:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _complete(
        self,
        *,
        instruction: str,
        payload: dict[str, object],
        operation: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        raw = complete_strict_json(
            self._settings,
            instruction=instruction,
            payload=payload,
            operation=operation,
            schema=schema,
        )
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            raise WhiteboardModelOutputError(
                f"{operation} violated its strict response schema"
            ) from exc


class OpenAICallLevelClassifier(_AgentBase):
    version = "single-call-level-classifier-v1"

    def classify(
        self, *, call_id: str, turns: list[TranscriptTurn]
    ) -> CallLevelClassification:
        output = self._complete(
            instruction=_LEVEL_INSTRUCTION,
            payload={"call_id": call_id, "turns": _turn_payload(turns)},
            operation="single call level classification",
            schema=CallLevelClassification,
        )
        assert isinstance(output, CallLevelClassification)
        return output


class OpenAIModuleJudge(_AgentBase):
    version = "single-call-module-judge-v1"

    def judge_module(
        self,
        *,
        call_id: str,
        level: CallLevel,
        turns: list[TranscriptTurn],
        speaker_assignments: list[WhiteboardSpeakerAssignment],
        module: WhiteboardScoringModule,
        rules: tuple[WhiteboardRule, ...],
    ) -> ModuleJudgmentOutput:
        output = self._complete(
            instruction=_MODULE_INSTRUCTION,
            payload={
                "call_id": call_id,
                "call_level": level,
                "module": module.model_dump(mode="json"),
                "rules": [rule.model_dump(mode="json") for rule in rules],
                "speaker_assignments": [
                    item.model_dump(mode="json") for item in speaker_assignments
                ],
                "input_modality": "transcript_text",
                "turns": _turn_payload(turns),
            },
            operation=f"single call module {module.module_id}",
            schema=ModuleJudgmentOutput,
        )
        assert isinstance(output, ModuleJudgmentOutput)
        return output


class OpenAIEvidenceReviewer(_AgentBase):
    version = "single-call-evidence-reviewer-v1"

    def review(
        self,
        *,
        call_id: str,
        level: CallLevel,
        turns: list[TranscriptTurn],
        rules: tuple[WhiteboardRule, ...],
        judgments: list[DimensionJudgment],
    ) -> EvidenceReviewOutput:
        output = self._complete(
            instruction=_REVIEW_INSTRUCTION,
            payload={
                "call_id": call_id,
                "call_level": level,
                "rules": [rule.model_dump(mode="json") for rule in rules],
                "candidate_judgments": [
                    item.model_dump(mode="json") for item in judgments
                ],
                "turns": _turn_payload(turns),
            },
            operation="single call evidence review",
            schema=EvidenceReviewOutput,
        )
        assert isinstance(output, EvidenceReviewOutput)
        return output


def _turn_payload(turns: list[TranscriptTurn]) -> list[dict[str, object]]:
    return [
        {
            "turn_no": turn.turn_no,
            "source_speaker_label": turn.source_speaker_label,
            "timestamp_ms": turn.timestamp_ms,
            "text": turn.text,
        }
        for turn in turns
    ]


_LEVEL_INSTRUCTION = """你是销售电话业务层级分类器，只依据当前完整转写判断本通电话的主要业务目标，并统一识别说话人角色。
层级只能选择一个：initial_contact=初次接触或业务介绍；needs_discovery=需求和信息挖掘；solution_negotiation=方案推荐、报价、优惠或异议洽谈；contract_delivery=签约、风控、发货或交付；after_sales=故障、换机、账单、发票、付款、增租等售后服务。
优先选择实际对话占比最高且最能代表本通电话目的的层级，不根据外部 sales_stage 标签猜测。
speaker_assignments 必须覆盖 turns 中全部且仅有的 source_speaker_label，每个标签一次。声纹编号不代表角色，无法可靠区分时标 unknown。
层级 evidence 必须选择1至5个决定层级的原文轮次，turn_no、标签和角色必须准确；quote 只需简短描述该轮证据，本地程序会按 turn_no 回填完整原文，不要求逐字复制。"""


_MODULE_INSTRUCTION = """你是单通销售电话的一个无状态评分模块，只评估输入 rules 中列出的维度，不计算分数。
必须按 rules 顺序返回全部且仅有这些 dimension_id，module_id 必须与输入一致。
present=true 仅表示详细规则确实命中；matched_count 是彼此独立的命中数。binary 只能为0或1，count_capped 不得超过规则分数上限对应次数。
不要默认把 criteria 列表理解为必须全部同时满足。对于“介绍业务、承上启下、挖到需求或信息、介绍卖点/优势”，列表中的每一项都是可独立成立的命中路径，满足任意一项即可判定；只有规则解释明确要求推荐依据、客户反馈或处理结果时，才要求这些必要条件同时出现。
present=true 必须提供证据轮次；turn_no、标签和 speaker 必须与统一 speaker_assignments 一致。quote 只需简短描述证据，本地程序会按 turn_no 回填完整原文，不要求逐字复制。未命中必须 count=0 且 evidence=[]。
采用贴近真实电话口语的语义判断，不要求销售使用规则中的标准术语：
- 介绍业务：只要销售让客户知道自己提供 IT、电脑、办公/硬件设备、供应、租赁或相关企业服务即可；不要求说出公司全名、完整业务介绍或“我们公司是做”这类固定句式。
- 承上启下：提到“之前、上次、此前、听某某说、您之前提过、后面会新增”等历史沟通或旧信息，并借此询问、回应或继续推进当前对话即可；不强制位于最初两轮，也不要求同时给建议和询问进展。
- 挖到需求或信息：客户明确给出的肯定或否定信息都算，例如“暂时不要、设备都有、不负责这块、要找老板、项目未定”；信息可以是客户在销售铺垫后主动说出，也可以由销售问题和相邻客户回答共同表达。单纯只有销售问题而没有答案不算，同一类别不得重复拆分。
- 介绍卖点/优势：六大卖点及公司实力都允许使用口语等价表达并分别独立计数：随用随还包括“随时退、不用可退、可换可升级”；一台起租包括“少量也能租、一台也可以”；免押金包括“零押金、不交保证金、符合条件不用押金”，结合“先寄设备且一分钱不收”的上下文也可成立；全程保修包括“免费上门、工程师全包、故障换机”；灵活支付包括“先用后付、按月付、一次付有折扣”；高性价比包括明确低月租或节省成本；公司网点、规模、供货和服务覆盖属于实力优势。无需使用标准名称、一次讲完六项或获得客户回应。
凡要求客户非负向、同意或有后续态度的规则，必须有客户回应证据；角色为 unknown 时不得满足这类规则。
互动性按可见轮次、时间戳和客户字数占比判断，缺时间戳不得假设时长。"""


_REVIEW_INSTRUCTION = """你是独立证据复核器，不重新评分，只检查每个已命中的候选判定是否被其证据和完整上下文充分支持。
必须按 candidate_judgments 顺序为全部且仅有这些 dimension_id 返回 review。
accepted=true 需要同时满足：引用原文确实表达该事实；说话人角色正确；独立命中数没有重复拆分；需要客户态度、同意或异议结果时上下文足以证明；结论符合对应规则。
证据轮次是定位点，不是对完整事实的穷举。必须结合该轮前后相邻内容以及输入中的完整 turns 审核；只要完整原文能支持结论，不得仅因评分模块少选了一条相邻回答、quote 不是逐字引文或理由不够完整而拒绝。
对介绍业务、承上启下、挖到需求或信息、介绍卖点/优势采用召回优先口径：接受真实电话中的简称、省略句、口头语和语义等价表达；肯定信息与否定信息都可以构成挖掘结果；这些规则的 criteria 是可选命中路径而非必须全部满足。
只有完整原文不存在支持、说话人明显错误、命中次数重复/超过独立事实数，或规则明确要求的客户态度确实缺失时，才返回 accepted=false。不得创造原文中没有的事实。"""
