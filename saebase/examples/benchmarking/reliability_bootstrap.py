"""Offline grouped-bootstrap reliability demonstration, no downloads or model calls."""

from saebase.reliability import Observation, AuditConfig, bootstrap_metric, paired_bootstrap


def main():
    a = []
    b = []
    for group in range(8):
        for role in ("clean", "wrapper"):
            value = group * 0.2 + (role == "wrapper") * 0.05
            for dest, sae, shift in ((a, "a", 0.0), (b, "b", 0.08)):
                dest.append(
                    Observation(
                        "toy",
                        sae,
                        "intervention.logit_delta_l2",
                        "offline",
                        "test",
                        f"group-{group}",
                        f"{group}-{role}",
                        value + shift,
                    )
                )
    config = AuditConfig(
        evaluation_seeds=(0, 1, 2, 3, 4), bootstrap_replicates=128, sample_sizes=(4, 8)
    )
    print("Fixed SAE:", bootstrap_metric(a, config))
    print("Paired SAE A-B:", paired_bootstrap(a, b, config))


if __name__ == "__main__":
    main()
