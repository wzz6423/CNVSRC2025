import math
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeedbackQueryDecision:
    queried: bool
    policy_requested: bool
    manifest_requested: bool
    source: str
    reason: str
    block_index: int | None
    position_in_block: int | None
    reliability_threshold: float | None

    def to_dict(self):
        return asdict(self)


def _quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("分位数至少需要一个数值")
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class FeedbackQueryPolicy:
    SUPPORTED_STRATEGIES = {"periodic", "random", "uncertainty"}

    def __init__(
        self,
        strategy,
        every,
        total_samples,
        *,
        random_seed=0,
        uncertainty_history_window=100,
    ):
        self.strategy = str(strategy)
        if self.strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(f"未知反馈查询策略：{self.strategy}")
        self.every = int(every)
        self.total_samples = int(total_samples)
        self.random_seed = int(random_seed)
        self.uncertainty_history_window = int(uncertainty_history_window)
        if self.every < 0:
            raise ValueError("feedback.every 不能为负数")
        if self.total_samples < 0:
            raise ValueError("流式样本数不能为负数")
        if self.strategy != "periodic" and self.every < 1:
            raise ValueError("random/uncertainty 查询要求 feedback.every 大于 0")
        if (
            self.strategy == "uncertainty"
            and self.uncertainty_history_window < self.every
        ):
            raise ValueError(
                "uncertainty_history_window 不能小于一个反馈窗口"
            )

        self.planned_budget = (
            self.total_samples // self.every if self.every > 0 else 0
        )
        rng = random.Random(self.random_seed)
        self._random_positions = tuple(
            rng.randrange(self.every) for _ in range(self.planned_budget)
        )
        self._reliability_history = []
        self._uncertainty_block_threshold = None
        self._uncertainty_block_queried = False
        self.policy_queries = 0
        self.manifest_queries = 0
        self.total_queries = 0
        self.reason_counts = {}

    def _policy_decision(self, index, reliability_score):
        if self.every == 0:
            return False, "querying_disabled", None, None, None

        block_index, position = divmod(index, self.every)
        if block_index >= self.planned_budget:
            return False, "partial_window", block_index, position, None

        threshold = None
        if self.strategy == "periodic":
            requested = position == self.every - 1
            reason = "periodic_slot" if requested else "not_periodic_slot"
        elif self.strategy == "random":
            requested = position == self._random_positions[block_index]
            reason = "random_slot" if requested else "not_random_slot"
        else:
            if position == 0:
                history = self._reliability_history[
                    -self.uncertainty_history_window :
                ]
                self._uncertainty_block_threshold = (
                    _quantile(history, 1.0 / self.every)
                    if len(history) >= self.every
                    else None
                )
                self._uncertainty_block_queried = False
            threshold = self._uncertainty_block_threshold
            requested = (
                not self._uncertainty_block_queried
                and threshold is not None
                and reliability_score <= threshold
            )
            if requested:
                reason = "uncertainty_below_history_quantile"
            elif (
                position == self.every - 1
                and not self._uncertainty_block_queried
            ):
                requested = True
                reason = "uncertainty_window_fallback"
            elif self._uncertainty_block_queried:
                reason = "uncertainty_window_budget_used"
            else:
                reason = "uncertainty_above_history_quantile"
            if requested:
                self._uncertainty_block_queried = True

        return requested, reason, block_index, position, threshold

    def decide(self, index, reliability_score, *, manifest_requested=False):
        index = int(index)
        reliability_score = float(reliability_score)
        if index < 0 or index >= self.total_samples:
            raise ValueError(f"反馈查询索引越界：{index}")
        if not math.isfinite(reliability_score):
            raise ValueError("反馈查询可靠度必须是有限数值")

        policy_requested, reason, block_index, position, threshold = (
            self._policy_decision(index, reliability_score)
        )
        if self.strategy == "uncertainty":
            self._reliability_history.append(reliability_score)

        manifest_requested = bool(manifest_requested)
        queried = manifest_requested or policy_requested
        if manifest_requested and policy_requested:
            source = "manifest_and_policy"
        elif manifest_requested:
            source = "manifest"
            reason = "manifest_requested"
        elif policy_requested:
            source = "policy"
        else:
            source = "none"

        if policy_requested:
            self.policy_queries += 1
        if manifest_requested:
            self.manifest_queries += 1
        if queried:
            self.total_queries += 1
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

        return FeedbackQueryDecision(
            queried=queried,
            policy_requested=policy_requested,
            manifest_requested=manifest_requested,
            source=source,
            reason=reason,
            block_index=block_index,
            position_in_block=position,
            reliability_threshold=threshold,
        )

    def summary(self):
        return {
            "strategy": self.strategy,
            "every": self.every,
            "total_samples": self.total_samples,
            "planned_budget": self.planned_budget,
            "policy_queries": self.policy_queries,
            "manifest_queries": self.manifest_queries,
            "total_queries": self.total_queries,
            "query_rate": (
                self.total_queries / self.total_samples
                if self.total_samples
                else 0.0
            ),
            "random_seed": self.random_seed,
            "uncertainty_history_window": self.uncertainty_history_window,
            "reason_counts": dict(self.reason_counts),
        }
