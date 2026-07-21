import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasticity.signature_calibration import (
    calibrate_signature_artifacts,
    write_calibration_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="使用无泄漏 validation 路由签名校准 motion order 和路由阈值",
        epilog="输入应来自 frozen base checkpoint；禁止使用 test 数据调参。",
    )
    parser.add_argument("--input", required=True, help="route_signatures.npz 路径")
    parser.add_argument(
        "--metadata",
        help="签名 metadata JSON；默认使用 <input>.meta.json",
    )
    parser.add_argument("--output", required=True, help="校准结果 JSON 路径")
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="预先锁定的 validation manifest SHA-256",
    )
    parser.add_argument(
        "--expected-base-checkpoint-sha256",
        required=True,
        help="预先锁定的 frozen base checkpoint SHA-256",
    )
    parser.add_argument("--seed", type=int, default=42, help="按域拆分的固定随机种子")
    parser.add_argument(
        "--reference-fraction",
        type=float,
        default=0.5,
        help="每个域用于 reference prototype 的样本比例",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        help="backbone feature 维度；缺省时从 metadata/签名维度推断",
    )
    args = parser.parse_args()

    try:
        output_path = Path(args.output)
        if output_path.exists():
            raise FileExistsError(f"拒绝覆盖已有校准结果：{output_path}")
        result = calibrate_signature_artifacts(
            args.input,
            metadata_path=args.metadata,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_base_checkpoint_sha256=args.expected_base_checkpoint_sha256,
            seed=args.seed,
            reference_fraction=args.reference_fraction,
            feature_dim=args.feature_dim,
        )
        write_calibration_json(output_path, result)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    best = result["orders"][str(result["best_motion_order"])]
    json.dump(
        {
            "output": str(output_path),
            "best_motion_order": result["best_motion_order"],
            "route_threshold": best["unknown_detection"]["route_threshold"],
            "unknown_f1": best["unknown_detection"]["best_f1"],
            "unknown_auroc": best["unknown_detection"]["auroc"],
            "clustering_purity": best["clustering_purity"],
        },
        sys.stdout,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
