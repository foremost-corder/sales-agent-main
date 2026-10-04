"""Versioned key-behavior policy supplied by the business."""

from sales_agent.features.scoring.contracts import ScoreRule, ScoreSection, ScoringPolicy


def _rule(
    rule_id: str,
    name: str,
    explanation: str,
    criteria: list[str],
    queries: list[str],
    points: int,
    *,
    maximum: int | None = None,
    mode: str = "binary",
    per_query_k: int = 8,
    final_k: int = 20,
    per_call_k: int = 2,
    low_confidence_k: int = 6,
    min_similarity: float = 0.35,
) -> ScoreRule:
    return ScoreRule(
        rule_id=rule_id,
        name=name,
        explanation=explanation,
        criteria=criteria,
        retrieval_queries=queries,
        points_per_match=points,
        maximum_points=points if maximum is None else maximum,
        scoring_mode=mode,  # type: ignore[arg-type]
        retrieval_per_query_k=per_query_k,
        retrieval_final_k=final_k,
        retrieval_per_call_k=per_call_k,
        retrieval_low_confidence_k=low_confidence_k,
        retrieval_min_similarity=min_similarity,
    )


SALES_KEY_BEHAVIOR_POLICY_V1 = ScoringPolicy(
    policy_id="computer-rental-key-behaviors",
    version="v1",
    rules=[
        _rule(
            "business_introduction", "介绍业务", "销售介绍公司从事 IT、办公设备供应或电脑设备租赁。",
            ["介绍公司是做 IT 或办公设备供应", "介绍电脑或设备租赁业务"],
            [
                "销售：我们公司主要给企业提供电脑、打印机等 IT 办公设备供应服务。",
                "销售：我们是做企业电脑租赁和办公设备租赁的。",
            ], 10, per_call_k=1,
        ),
        _rule(
            "prior_context_bridge", "承上启下", "开场衔接历史沟通或客户旧关注点并继续推进。",
            ["主动提及此前问题、现状或历史沟通", "针对旧关注点给建议、方案或询问进展"],
            [
                "销售：上次您提到新办公室近期要采购电脑，现在项目进展怎么样了？",
                "销售：之前您比较关注售后响应，我根据这个问题整理了一个方案。",
            ], 10, per_call_k=1,
        ),
        _rule(
            "discovery_information", "挖到需求或信息", "销售通过提问获取一项明确且彼此独立的客户需求或业务信息。",
            [
                "联系人岗位或决策链", "需求及缺口原因、新项目、新场地、扩员或换新", "需求时间",
                "首次或未来需求数量", "设备类型、品牌、岗位或核心配置", "现有 IT 获取渠道、采购、租赁、售后、供货、发票、周期或闲置现状",
                "合作意向抬头", "公司主营、组织架构或 IT 岗位人数", "现金流、灵活性、售后或资产管理痛点",
                "固定资产、成本核算、版权、数据安全或决策流程阻力", "预算或价格接受区间", "通过其他联系人打探情况", "回收需求",
            ],
            [
                "销售：这批设备预计什么时候需要，大概需要多少台？客户回答了明确时间或数量。",
                "销售：电脑主要给什么岗位使用，对 CPU、内存、硬盘或品牌有什么要求？客户回答了用途或配置。",
                "销售：您现在的设备是采购、租赁还是员工自带，目前售后和供货怎么样？客户介绍了 IT 现状。",
                "销售：这件事由哪些部门或负责人决策，您主要负责哪个环节？客户说明了岗位或决策链。",
                "销售：最近是有新项目、新场地、人员扩张还是旧设备换新？客户说明了需求原因。",
                "销售询问预算或可接受价格，客户给出预算范围、价格区间或成本要求。",
                "客户描述现金流、设备闲置、售后响应、资产管理、数据安全或内部决策阻力。",
                "销售询问公司业务、组织规模、IT 人员、开票抬头或旧设备回收，客户提供了明确信息。",
            ], 10, maximum=50, mode="count_capped", final_k=40,
            per_call_k=5, low_confidence_k=12,
        ),
        _rule(
            "value_proposition", "介绍卖点或优势", "销售介绍一项明确的租赁卖点或公司实力优势。",
            ["随用随还", "一台起租", "免押金", "全程保修", "灵活支付", "高性价比", "公司核心实力优势"],
            [
                "销售：我们的设备可以随用随还，也可以一台起租，数量和租期都比较灵活。",
                "销售：符合条件可以免押金，租期内全程保修并提供快速上门或整机更换。",
                "销售：可以按月付款节省现金，也可以一次支付享受折扣，月租价格性价比较高。",
                "销售向客户介绍公司的服务网点、供货能力、客户规模或企业实力。",
            ],
            5, maximum=20, mode="count_capped", final_k=30,
            per_call_k=4, low_confidence_k=9,
        ),
        _rule(
            "product_recommendation", "产品与配置推荐能力好", "销售基于客户用途或参数需求推荐具体设备且客户无负面反馈。",
            ["推荐建立在客户岗位、用途或参数需求上", "推荐具体设备型号或 CPU、内存等配置", "客户后续反馈不是负向"],
            [
                "客户说明办公岗位和软件用途后，销售推荐了具体电脑型号、CPU、内存或硬盘配置，并解释推荐原因。",
                "销售根据客户的使用场景推荐打印机或电脑配置，客户表示可以、合适或没有反对。",
            ], 10,
        ),
        _rule(
            "price_negotiation", "价格洽谈与优惠引导能力好", "销售针对需求报价、介绍优惠，或妥善处理价格异议且客户反馈非负向。",
            ["针对需求给出设备租赁报价", "说明限时折扣、赠礼或对应价值", "解释价格异议并获得正向或中立反馈"],
            [
                "销售根据客户需要的设备和租期给出明确的月租或整体报价。",
                "销售介绍近期折扣、租满赠礼或一次性支付优惠，并说明客户能节省的成本。",
                "客户认为价格高，销售解释价格构成或提出调整方案，客户接受或继续沟通。",
            ], 10,
        ),
        _rule(
            "objection_resolved", "解决了客户异议", "销售针对明确异议或卡点给出对应解释或方案，客户后续态度非负向。",
            ["解决痛点", "回应不租、价格贵或流程复杂", "处理风控、缺货、合同、授信卡点", "解答产品、租期、售后或故障疑问", "处理租卖或租租对比"],
            [
                "客户担心租赁价格、流程或模式，销售给出针对性解释和解决方案，客户随后接受或保持中立。",
                "客户提出风控、合同、授信、缺货或发货卡点，销售说明了可执行的处理办法。",
                "客户反馈设备故障、售后慢、租期过长或型号不合适，销售提供换机、维修或调整方案。",
                "客户比较购买与租赁或不同租赁公司，销售完成对比解释后客户愿意继续推进。",
            ], 20,
        ),
        _rule(
            "empathy_relationship", "同理心好或拉近客户关系", "销售站在客户立场沟通，或通过双向非工作互动、额外价值服务拉近关系。",
            ["根据岗位或行业调整价值表达", "解释提问是为了适配方案", "先共情再解决", "结合客户困难给方案", "不急于反驳", "推荐配置并解释原因", "双向非工作互动", "提供资源、巡检、维修、礼物等额外价值"],
            [
                "销售：我理解您担心预算或流程的问题，我们先站在您这个岗位的角度算一下。",
                "销售根据客户行业、岗位、人数或使用周期调整推荐重点，并解释为什么更适合客户。",
                "销售和客户双向交流家乡、爱好、节日或近况等非工作话题。",
                "销售主动提供巡检、维修、办公资源渠道、礼物或其他额外帮助。",
            ], 20,
        ),
        _rule(
            "conversation_interaction", "沟通互动性好", "根据销售阶段、互动轮次、时长和客户发言占比确定。",
            ["线索阶段至少3个来回，或时长至少40秒且客户字数占比至少20%", "非线索阶段至少5个来回，或时长至少40秒且客户字数占比至少35%"],
            ["销售与客户沟通互动轮次 客户发言占比 通话时长"], 20, mode="interaction_metric",
        ),
        _rule(
            "contract_delivery_process", "签约流程和发货流程熟悉", "销售清楚讲解签约、授信、签字盖章、风控、发货或签收流程。",
            ["讲解注册授信资料、签字或上门拍视频", "准确解答签约流程疑问", "讲解发货时效、流程或签收注意事项"],
            [
                "销售说明注册授信需要提交哪些资料，以及签字、盖章、风控审核或上门拍摄的流程。",
                "销售向客户讲解发货时效、物流步骤和设备签收注意事项。",
            ], 5, final_k=15, low_confidence_k=4,
        ),
        _rule(
            "after_sales_solution", "售后问题解决好", "销售熟悉售后流程并给出可行方案，客户态度正向或中立。",
            ["讲解设备故障换机流程", "解决增租、故障、账单、发票或付款问题", "客户反馈正向或中立"],
            [
                "客户反馈设备故障，销售清楚说明报修、上门、换机或整机替换的操作步骤。",
                "客户询问增租、账单、发票或付款问题，销售给出明确可执行的售后方案。",
            ], 5, final_k=15, low_confidence_k=4,
        ),
        _rule(
            "wechat_agreement", "客户愿意加微信", "客户主动提出加微信，或明确同意销售的加微信请求。",
            ["客户主动提出添加微信", "销售提出加微信且客户明确同意"],
            [
                "销售：我加一下您的微信，把方案发给您。客户：可以，你加我吧。",
                "客户主动询问销售微信号或提出通过微信继续联系。",
            ], 10, final_k=15, per_call_k=1, low_confidence_k=4,
        ),
        _rule(
            "abnormal_speech", "语速语调异常", "语速过快或过慢、声音过小或普通话明显不标准。",
            ["需要音频或可信声学分析结果，不能只根据转写文本判断"],
            ["语速过快 语速过慢 声音小 普通话不标准"], -10, mode="audio_required",
        ),
        _rule(
            "objection_unresolved", "不能解决客户异议", "销售未能解释或解决客户异议，或回应后客户仍明确负向。",
            ["没有给出解释或方案", "表达结巴或逻辑混乱导致无法理解", "回应后客户持续不满或明确拒绝"],
            [
                "客户提出价格、流程、合同或售后异议，销售表示不知道、无法处理或没有给出方案。",
                "销售回应异议时逻辑混乱或反复解释，客户仍表示不接受、不满意或拒绝合作。",
                "客户明确拒绝后销售没有处理异议，沟通停滞或客户状态进一步变差。",
            ], -30, final_k=15, low_confidence_k=4, min_similarity=0.45,
        ),
    ],
    sections=[
        ScoreSection(
            section_id="business_introduction", name="介绍业务",
            purpose="召回并判断销售是否介绍公司业务性质。", rule_ids=["business_introduction"],
        ),
        ScoreSection(
            section_id="prior_context_bridge", name="承上启下",
            purpose="召回并判断开场是否衔接历史沟通。", rule_ids=["prior_context_bridge"],
        ),
        ScoreSection(
            section_id="discovery_information", name="挖到需求或信息",
            purpose="召回并判断销售获得的独立客户需求与业务信息。", rule_ids=["discovery_information"],
        ),
        ScoreSection(
            section_id="value_proposition", name="介绍卖点或优势",
            purpose="召回并判断租赁卖点与公司实力表达。", rule_ids=["value_proposition"],
        ),
        ScoreSection(
            section_id="product_recommendation", name="产品与配置推荐能力好",
            purpose="召回并判断基于需求的具体设备推荐。", rule_ids=["product_recommendation"],
        ),
        ScoreSection(
            section_id="price_negotiation", name="价格洽谈与优惠引导能力好",
            purpose="召回并判断报价、优惠与价格异议处理。", rule_ids=["price_negotiation"],
        ),
        ScoreSection(
            section_id="objection_resolved", name="解决了客户异议",
            purpose="召回并判断异议解决及客户后续反馈。", rule_ids=["objection_resolved"],
        ),
        ScoreSection(
            section_id="empathy_relationship", name="同理心好或拉近客户关系",
            purpose="召回并判断共情、关系建立与额外价值。", rule_ids=["empathy_relationship"],
        ),
        ScoreSection(
            section_id="conversation_interaction", name="沟通互动性好",
            purpose="根据通话轮次、时长和客户发言比例本地计算。", rule_ids=["conversation_interaction"],
        ),
        ScoreSection(
            section_id="contract_delivery_process", name="签约流程和发货流程熟悉",
            purpose="召回并判断签约、风控、发货与签收流程说明。", rule_ids=["contract_delivery_process"],
        ),
        ScoreSection(
            section_id="after_sales_solution", name="售后问题解决好",
            purpose="召回并判断售后问题处理方案。", rule_ids=["after_sales_solution"],
        ),
        ScoreSection(
            section_id="wechat_agreement", name="客户愿意加微信",
            purpose="召回并判断客户主动提出或明确同意添加微信。", rule_ids=["wechat_agreement"],
        ),
        ScoreSection(
            section_id="abnormal_speech", name="语速语调异常",
            purpose="仅依据声学分析判断语速、音量和普通话问题。", rule_ids=["abnormal_speech"],
        ),
        ScoreSection(
            section_id="objection_unresolved", name="不能解决客户异议",
            purpose="召回并判断异议未解决及客户负向变化。", rule_ids=["objection_unresolved"],
        ),
    ],
)
