"""Self-contained 14-dimension policy for single-call whiteboard scoring."""

from sales_agent.features.single_call_scoring.contracts import (
    CallLevel,
    WhiteboardRule,
    WhiteboardScoringModule,
)


def _rule(
    dimension_id: str,
    name: str,
    explanation: str,
    criteria: list[str],
    points: int,
    *,
    maximum: int | None = None,
    mode: str = "binary",
) -> WhiteboardRule:
    return WhiteboardRule(
        dimension_id=dimension_id,
        name=name,
        explanation=explanation,
        criteria=criteria,
        points_per_match=points,
        maximum_points=points if maximum is None else maximum,
        scoring_mode=mode,  # type: ignore[arg-type]
    )


WHITEBOARD_RULES: tuple[WhiteboardRule, ...] = (
    _rule(
        "business_introduction", "介绍业务", "销售介绍公司的业务性质。",
        ["介绍公司从事 IT 或办公设备供应", "介绍公司从事电脑或设备租赁"], 10,
    ),
    _rule(
        "prior_context_bridge", "承上启下了", "开场衔接历史沟通并继续推进。",
        ["开场提及客户此前关注的问题、业务现状或历史沟通", "针对旧关注点给出建议、方案或询问进展"], 10,
    ),
    _rule(
        "discovery_information", "挖到了需求或信息", "销售通过提问获得明确且彼此独立的需求或业务信息。",
        [
            "联系人岗位或需求决策链", "是否有需求及新项目、新场地、新职场、新中标、换新、入职或扩张等原因",
            "需求时间", "首次或未来需求数量", "设备类型、品牌、岗位、CPU、内存、硬盘、尺寸或显卡配置",
            "现有采购或租赁渠道、IT方案、价格、售后、供货、发票、使用周期、闲置或服务现状",
            "合作意向抬头", "公司主营、组织架构或IT岗位人数", "现金流、灵活性、售后或资产管理痛点",
            "固定资产、成本、版权、数据安全或决策流程等推动阻力", "预算或价格接受区间",
            "通过其他联系人了解公司或现有联系人想法", "旧设备回收需求",
        ], 10, maximum=50, mode="count_capped",
    ),
    _rule(
        "value_proposition", "介绍卖点/优势", "销售介绍租赁卖点或公司实力优势。",
        [
            "随用随还", "一台起租", "免押金", "全程保修，包括在线服务、快速上门或整机快换",
            "灵活支付，包括按月或一次支付折扣", "高性价比或明确低月租", "公司核心实力优势",
        ], 5, maximum=20, mode="count_capped",
    ),
    _rule(
        "product_recommendation", "产品与配置推荐能力好", "结合需求推荐适配设备且客户无负面反馈。",
        ["推荐基于客户岗位、用途或参数需求", "推荐具体设备型号或 CPU、内存等配置", "客户后续反馈非负向"], 10,
    ),
    _rule(
        "price_negotiation", "价格洽谈与优惠引导能力好", "结合需求报价、介绍优惠或妥善处理价格异议。",
        ["针对需求给出租赁报价", "说明限时折扣、租满赠礼或利益价值", "处理价格异议后客户态度正向或中立"], 10,
    ),
    _rule(
        "objection_resolved", "解决了客户异议", "针对客户异议给出对应解释或方案且客户态度非负向。",
        [
            "针对IT服务、现金流等真实痛点给方案", "回应不考虑租赁、价格贵或流程复杂",
            "处理风控、缺货、合同、注册授信等卡点", "解决产品、租期、售后、故障或卡顿问题",
            "处理价格或租赁模式质疑", "处理租卖对比或租租对比",
        ], 20,
    ),
    _rule(
        "empathy_relationship", "同理心好或拉近了客户关系", "站在客户立场沟通或通过额外价值和非工作互动拉近关系。",
        [
            "结合岗位或行业调整价值表达", "说明提问是为了提供适配方案", "先共情认同再解决",
            "结合客户过往困难推荐方案", "不急于反驳并从客户角度分析", "结合需求推荐配置并解释原因",
            "家乡、爱好、节日、近况等非工作话题双向互动", "提供办公资源、巡检、维修、礼物或特产等额外价值",
        ], 20,
    ),
    _rule(
        "conversation_interaction", "沟通互动性好", "依据阶段、来回轮次、时长和客户发言占比判断。",
        [
            "线索阶段至少3个来回，或通话至少40秒且客户字数占比至少20%",
            "非线索阶段至少5个来回，或通话至少40秒且客户字数占比至少35%",
        ], 20,
    ),
    _rule(
        "contract_delivery_process", "签约流程和发货流程熟悉", "清晰讲解签约、风控和发货流程并能解答疑问。",
        ["讲解注册授信资料、签字、盖章、风控或上门拍视频", "准确解决签约流程疑问", "讲解发货时效、流程或签收注意事项"], 5,
    ),
    _rule(
        "after_sales_solution", "售后问题解决好", "熟悉售后流程并提供可行方案且客户态度非负向。",
        ["说明设备故障换机完整流程", "解决增租、故障、账单、发票或付款问题", "客户态度正向或中立"], 5,
    ),
    _rule(
        "wechat_agreement", "客户愿意加微信", "客户主动提出加微信或明确同意销售的请求。",
        ["客户主动提出添加微信", "销售提出加微信且客户明确同意"], 10,
    ),
    _rule(
        "abnormal_speech", "语速语调异常", "语速过快或过慢、声音过小或普通话明显不标准。",
        ["必须有音频或可信声学分析；纯TXT不可判定"], -10, mode="audio_required",
    ),
    _rule(
        "objection_unresolved", "不能解决客户异议", "无法解释或解决异议，或回应后客户仍明确负向。",
        ["没有给出原因解释或解决方案", "表达结巴或逻辑混乱导致客户无法理解", "回应后客户持续不满或明确拒绝合作"], -30,
    ),
)


RULE_BY_ID = {rule.dimension_id: rule for rule in WHITEBOARD_RULES}


ALL_LEVELS: list[CallLevel] = [
    "initial_contact",
    "needs_discovery",
    "solution_negotiation",
    "contract_delivery",
    "after_sales",
]


WHITEBOARD_MODULES: tuple[WhiteboardScoringModule, ...] = (
    WhiteboardScoringModule(
        module_id="opening_discovery",
        name="开场与需求挖掘",
        purpose="判断业务介绍、历史衔接和需求信息挖掘。",
        rule_ids=["business_introduction", "prior_context_bridge", "discovery_information"],
        applicable_levels=["initial_contact", "needs_discovery", "solution_negotiation"],
    ),
    WhiteboardScoringModule(
        module_id="solution_offer",
        name="方案、产品与价格",
        purpose="判断卖点介绍、配置推荐以及报价优惠能力。",
        rule_ids=["value_proposition", "product_recommendation", "price_negotiation"],
        # Cold-call introductions often include a compact value pitch, rough
        # price, or configuration suggestion before a formal opportunity exists.
        applicable_levels=["initial_contact", "needs_discovery", "solution_negotiation", "contract_delivery"],
    ),
    WhiteboardScoringModule(
        module_id="objection_handling",
        name="异议处理",
        purpose="分别判断异议是否被解决以及是否存在未解决异议。",
        rule_ids=["objection_resolved", "objection_unresolved"],
        applicable_levels=ALL_LEVELS,
    ),
    WhiteboardScoringModule(
        module_id="relationship_interaction",
        name="关系与互动",
        purpose="判断同理心、沟通互动和添加微信意愿。",
        rule_ids=["empathy_relationship", "conversation_interaction", "wechat_agreement"],
        applicable_levels=ALL_LEVELS,
    ),
    WhiteboardScoringModule(
        module_id="fulfillment_support",
        name="签约交付与售后",
        purpose="判断签约发货流程及售后问题解决能力。",
        rule_ids=["contract_delivery_process", "after_sales_solution"],
        applicable_levels=["contract_delivery", "after_sales"],
    ),
)


def modules_for_level(level: CallLevel) -> tuple[WhiteboardScoringModule, ...]:
    return tuple(
        module for module in WHITEBOARD_MODULES if level in module.applicable_levels
    )
