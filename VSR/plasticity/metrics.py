from collections import defaultdict

from .reliability import edit_distance


class StreamMetrics:
    def __init__(self):
        self.total_edits = 0
        self.total_characters = 0
        self.samples = 0
        self.update_counts = defaultdict(int)
        self.domain_totals = defaultdict(lambda: [0, 0, 0])
        self.reliability_sum = 0.0

    def state_dict(self):
        return {
            "total_edits": self.total_edits,
            "total_characters": self.total_characters,
            "samples": self.samples,
            "update_counts": dict(self.update_counts),
            "domain_totals": dict(self.domain_totals),
            "reliability_sum": self.reliability_sum,
        }

    def load_state_dict(self, state):
        self.total_edits = int(state["total_edits"])
        self.total_characters = int(state["total_characters"])
        self.samples = int(state["samples"])
        self.update_counts = defaultdict(int, state["update_counts"])
        self.domain_totals = defaultdict(
            lambda: [0, 0, 0],
            {
                domain: list(values)
                for domain, values in state["domain_totals"].items()
            },
        )
        self.reliability_sum = float(state["reliability_sum"])

    def update(self, prediction, target, domain, reliability, update_status):
        self.samples += 1
        self.reliability_sum += float(reliability)
        self.update_counts[update_status] += 1
        if target is None:
            return
        edits = edit_distance(list(prediction), list(target))
        length = len(target)
        self.total_edits += edits
        self.total_characters += length
        domain_total = self.domain_totals[str(domain)]
        domain_total[0] += edits
        domain_total[1] += length
        domain_total[2] += 1

    def summary(self):
        domains = {}
        for domain, (edits, characters, samples) in self.domain_totals.items():
            domains[domain] = {
                "cer": edits / characters if characters else None,
                "edits": edits,
                "characters": characters,
                "samples": samples,
            }
        attempted_updates = sum(
            self.update_counts.get(status, 0)
            for status in ("accepted", "rolled_back", "failed")
        )
        accepted_updates = self.update_counts.get("accepted", 0)
        return {
            "samples": self.samples,
            "cer": (
                self.total_edits / self.total_characters
                if self.total_characters
                else None
            ),
            "edits": self.total_edits,
            "characters": self.total_characters,
            "mean_reliability": (
                self.reliability_sum / self.samples if self.samples else 0.0
            ),
            "updates": dict(self.update_counts),
            "attempted_updates": attempted_updates,
            "accepted_update_rate": (
                accepted_updates / attempted_updates if attempted_updates else 0.0
            ),
            "domains": domains,
        }
