"""基于 z3 的小样本格值暴露风险审查与公开汇总核对。

审查模型（不依赖真实检出数作为约束）：
- 每个格一个未知整数 x_i，仅受 0 <= x_i <= 人数_i 界限约束；
- 每条历次公开记录：所选交集格之和 == 公开总数；
- 候选发布：所选交集格之和 == 候选总数。

判定：对每个"原检出数为 1~4"的格，在上述约束下必须
- 仍存在 x_i == 0 的可行整数解，且
- 仍存在 x_i >= 5 的可行整数解。
任一不满足则该格为受影响格，发布须被拒绝，并给出该格的可行范围 [min, max]。

核对模型（reconcile）：
- 修订量 r_i 为各格修订后的检出数，仅受 0 <= r_i <= 人数_i 与
  数据员复核得到的"分区合计、年龄组合计"两组约束；
  历史公开总数仅作对照，不作为修订约束，也不改动任何已存数据。
- 在满足两组合计的可行修订中，先求"改动格数最少"（r_i != 原检出数 的格数），
  再在该最优修订集合上：
  * 标出每次都要改的格（所有最优修订都改）与仅有部分修订会改的格；
  * 求所选公开记录交集格之和的可行上下界；
  * 单独以可满足性检查核实原公开总数是否确实可取（不只看区间）。
- 结论：无可行修订 → 无法对账；原总数不可取 → 确定失准；
  上下界收敛于原总数 → 仍准确；其余（原总数可取但界不收敛）→ 可能失准。

注意：z3 的全局默认 Context 非线程安全。每次评估使用独立 Context，
以支持多个请求并发求解（预览/发布/核对在事件循环外的工作线程中执行）。
"""

from z3 import Context, If, Int, Optimize, Solver, Sum, sat

SMALL_MIN, SMALL_MAX = 1, 4   # 受保护的小样本检出数区间
SAFE_LOW, SAFE_HIGH = 0, 5    # 须同时保留 "取0" 与 "至少5" 的可行解


def _indices(cells: list[dict], selection: dict) -> list[int]:
    ds = set(selection["districts"])
    ags = set(selection["age_groups"])
    return [
        i for i, c in enumerate(cells)
        if c["district"] in ds and c["age_group"] in ags
    ]


def _feasible_range(ctx: Context, constraints, x) -> tuple[int | None, int | None]:
    opt = Optimize(ctx=ctx)
    opt.add(constraints)
    h = opt.minimize(x)
    if opt.check() != sat:
        return None, None
    lo = opt.lower(h).as_long()
    opt2 = Optimize(ctx=ctx)
    opt2.add(constraints)
    h2 = opt2.maximize(x)
    if opt2.check() != sat:
        return lo, None
    hi = opt2.upper(h2).as_long()
    return lo, hi


def evaluate(cells: list[dict], publications: list[dict], candidate: dict) -> dict:
    """评估候选发布是否安全。

    cells:        全部格 {district, age_group, population, detected}
    publications: 历次公开记录 {districts, age_groups, total}
    candidate:    候选发布 {districts, age_groups, total}
    返回 {"safe": bool, "affected": [...]}
    """
    ctx = Context()  # 每次评估独立上下文，保证并发求解安全
    xs = [Int(f"x_{i}", ctx=ctx) for i in range(len(cells))]

    constraints = []
    for i, c in enumerate(cells):
        constraints.append(xs[i] >= 0)
        constraints.append(xs[i] <= int(c["population"]))
    for pub in list(publications) + [candidate]:
        idx = _indices(cells, pub)
        constraints.append(Sum([xs[i] for i in idx]) == int(pub["total"]))

    base = Solver(ctx=ctx)
    base.add(constraints)
    if base.check() != sat:
        # 理论上不会发生（真实格值恒满足全部约束），防御性处理。
        affected = [
            {
                "district": c["district"],
                "age_group": c["age_group"],
                "detected": c["detected"],
                "min": None,
                "max": None,
                "zero_feasible": False,
                "ge5_feasible": False,
            }
            for c in cells
            if SMALL_MIN <= c["detected"] <= SMALL_MAX
        ]
        return {"safe": False, "reason": "约束不可满足", "affected": affected}

    affected = []
    for i, c in enumerate(cells):
        if not (SMALL_MIN <= c["detected"] <= SMALL_MAX):
            continue
        s = Solver(ctx=ctx)
        s.add(constraints)
        s.add(xs[i] == SAFE_LOW)
        zero_ok = s.check() == sat
        s = Solver(ctx=ctx)
        s.add(constraints)
        s.add(xs[i] >= SAFE_HIGH)
        ge5_ok = s.check() == sat
        if not (zero_ok and ge5_ok):
            lo, hi = _feasible_range(ctx, constraints, xs[i])
            affected.append(
                {
                    "district": c["district"],
                    "age_group": c["age_group"],
                    "detected": c["detected"],
                    "min": lo,
                    "max": hi,
                    "zero_feasible": zero_ok,
                    "ge5_feasible": ge5_ok,
                }
            )
    return {"safe": not affected, "affected": affected}


# ---------------- 公开汇总核对 ----------------

# 核对结论取值
VERDICT_INFEASIBLE = "无法对账"    # 两组合计与人数界限下不存在任何可行修订
VERDICT_WRONG = "确定失准"        # 所有最少改动修订下，公开交集和都取不到原公开总数
VERDICT_MAYBE = "可能失准"        # 部分最少改动修订保持原总数，部分不保持
VERDICT_OK = "仍准确"            # 所有最少改动修订下，公开交集和恒等于原总数


def reconcile(
    cells: list[dict],
    district_totals: dict[str, int],
    age_totals: dict[str, int],
    publication: dict,
) -> dict:
    """按复核合计核对一条公开汇总。

    cells:           全部格 {district, age_group, population, detected}
    district_totals: 每个现有分区的复核检出合计（须恰好覆盖现有分区）
    age_totals:      每个现有年龄组的复核检出合计（须恰好覆盖现有年龄组）
    publication:     所选公开记录 {districts, age_groups, total}（仅作对照，不约束修订）

    返回 {"feasible": bool, ...}；可行时含 min_changes、always_changed、
    sometimes_changed、bound_min/bound_max、original_attainable、verdict。
    不改动任何输入数据。
    """
    ctx = Context()  # 独立上下文，保证并发求解安全
    n = len(cells)
    xs = [Int(f"r_{i}", ctx=ctx) for i in range(n)]

    constraints = []
    for i, c in enumerate(cells):
        constraints.append(xs[i] >= 0)
        constraints.append(xs[i] <= int(c["population"]))
    for d in sorted({c["district"] for c in cells}):
        idx = [i for i, c in enumerate(cells) if c["district"] == d]
        constraints.append(Sum([xs[i] for i in idx]) == int(district_totals[d]))
    for a in sorted({c["age_group"] for c in cells}):
        idx = [i for i, c in enumerate(cells) if c["age_group"] == a]
        constraints.append(Sum([xs[i] for i in idx]) == int(age_totals[a]))

    base = Solver(ctx=ctx)
    base.add(constraints)
    if base.check() != sat:
        # 两组合计与人数界限无可行修订（如行列合计总和不一致、合计超容量）
        return {"feasible": False, "verdict": VERDICT_INFEASIBLE}

    # 改动格数：r_i 与原检出数不同的格计 1
    n_changes = Sum(
        [If(xs[i] != int(c["detected"]), 1, 0) for i, c in enumerate(cells)]
    )
    opt = Optimize(ctx=ctx)
    opt.add(constraints)
    h = opt.minimize(n_changes)
    if opt.check() != sat:
        # 理论上不会发生（上面已判可行），防御性处理
        return {"feasible": False, "verdict": VERDICT_INFEASIBLE}
    min_changes = opt.lower(h).as_long()

    # 在最优（改动最少）修订集合上分析
    opt_cons = constraints + [n_changes == min_changes]

    always_changed, sometimes_changed = [], []
    for i, c in enumerate(cells):
        d0 = int(c["detected"])
        s = Solver(ctx=ctx)
        s.add(opt_cons)
        s.add(xs[i] != d0)
        can_change = s.check() == sat
        s = Solver(ctx=ctx)
        s.add(opt_cons)
        s.add(xs[i] == d0)
        can_keep = s.check() == sat
        cell = {
            "district": c["district"],
            "age_group": c["age_group"],
            "population": c["population"],
            "detected": c["detected"],
        }
        if can_change and not can_keep:
            always_changed.append(cell)      # 每次都要改
        elif can_change and can_keep:
            sometimes_changed.append(cell)   # 仅部分修订会改
        # can_keep and not can_change：任何最优修订都不动，不列出

    # 所选公开记录交集格之和的可行上下界（仍在最优修订集合上）
    idx = _indices(cells, publication)
    pub_sum = Sum([xs[i] for i in idx])
    lo, hi = _feasible_range(ctx, opt_cons, pub_sum)

    # 原公开总数是否确实可取（区间含端点不等于每个中间值都可取，单独判定）
    original_total = int(publication["total"])
    s = Solver(ctx=ctx)
    s.add(opt_cons)
    s.add(pub_sum == original_total)
    attainable = s.check() == sat

    if not attainable:
        verdict = VERDICT_WRONG
    elif lo == hi:
        verdict = VERDICT_OK
    else:
        verdict = VERDICT_MAYBE

    return {
        "feasible": True,
        "min_changes": min_changes,
        "always_changed": always_changed,
        "sometimes_changed": sometimes_changed,
        "bound_min": lo,
        "bound_max": hi,
        "original_total": original_total,
        "original_attainable": attainable,
        "verdict": verdict,
    }
