#!/usr/bin/env python3
"""Audit completion and frozen decisions for Stage 4D paper assets."""

import argparse
import csv
import hashlib
import os
import os.path as osp


EXPECTED_ROLES = {
    "full_cache": ("quality_resource_reference", "REFERENCE_ONLY"),
    "stage3_2_k4": ("primary_bounded_deployment", "FROZEN_PRIMARY"),
    "old_dino_k6": ("robust_bounded_alternative", "FROZEN_ROBUST"),
    "temporal_binned_dino_k8": (
        "long_sequence_pose_specialist",
        "FROZEN_SPECIALIST",
    ),
}
REQUIRED_ASSETS = (
    "source_manifest.json",
    "tables/table_video_depth.csv",
    "tables/table_video_depth.tex",
    "tables/table_method_roles.csv",
    "tables/table_method_roles.tex",
    "tables/table_long_sequence.csv",
    "tables/table_long_sequence.tex",
    "tables/table_cross_task.csv",
    "figures/fig_video_depth_pareto.png",
    "figures/fig_video_depth_pareto.pdf",
    "figures/fig_cross_task_regret.png",
    "figures/fig_cross_task_regret.pdf",
    "figures/fig_stage4c_scaling.png",
    "figures/fig_stage4c_scaling.pdf",
    "figures/fig_stage4c_pose_scaling.png",
    "figures/fig_stage4c_pose_scaling.pdf",
    "figures/fig_stage4c_trajectories.png",
    "figures/fig_stage4c_trajectories.pdf",
    "figures/fig_stage4c_cache_timeline.png",
    "figures/fig_stage4c_cache_timeline.pdf",
    "figures/fig_stage4e_fusion_failure.png",
    "figures/fig_stage4e_fusion_failure.pdf",
)
FIELDS = (
    "source_coverage_ok",
    "frozen_roles_ok",
    "stage4c_evidence_ok",
    "stage4e_stop_decision_ok",
    "case_assets_ok",
    "required_assets_ok",
    "manifest_hashes_ok",
    "num_manifest_assets",
    "decision",
)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key)
    return None if value in (None, "") else float(value)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yes(value):
    return "yes" if value else "no"


def main():
    parser = argparse.ArgumentParser("Check Stage 4D paper-asset gate")
    parser.add_argument(
        "--cross-task-summary", default="stage4b_cross_task_summary.csv"
    )
    parser.add_argument("--method-roles", default="stage4b_method_roles.csv")
    parser.add_argument("--stage4c-results", default="stage4c_results.csv")
    parser.add_argument(
        "--stage4e-results", default="stage4e_a_sequence_results.csv"
    )
    parser.add_argument("--case-audit", default="stage4d_case_audit.csv")
    parser.add_argument(
        "--asset-manifest", default="stage4d_asset_manifest.csv"
    )
    parser.add_argument(
        "--asset-root", default="paper_assets/stage4d"
    )
    parser.add_argument("--output", default="stage4d_gate.csv")
    args = parser.parse_args()

    cross_task = read_csv(args.cross_task_summary)
    roles = read_csv(args.method_roles)
    stage4c = read_csv(args.stage4c_results)
    stage4e = read_csv(args.stage4e_results)
    cases = read_csv(args.case_audit)
    manifest = read_csv(args.asset_manifest)

    source_coverage_ok = (
        len(cross_task) == 41
        and len(roles) == 4
        and len(stage4c) == 42
        and len(stage4e) == 18
        and all(row.get("status") == "ok" for row in stage4e)
    )
    by_method = {row["method"]: row for row in roles}
    frozen_roles_ok = set(by_method) == set(EXPECTED_ROLES) and all(
        (
            by_method[method]["final_role"],
            by_method[method]["status"],
        )
        == expected
        for method, expected in EXPECTED_ROLES.items()
    )
    bounded = [
        row
        for row in stage4c
        if row["method"] != "full_cache"
    ]
    full = [row for row in stage4c if row["method"] == "full_cache"]
    stage4c_evidence_ok = (
        len(bounded) == 36
        and all(row["status"] == "ok" for row in bounded)
        and len(full) == 6
        and sum(row["status"] == "ok" for row in full) == 3
        and sum(row["status"] != "ok" for row in full) == 3
    )
    direct = [
        row
        for row in stage4e
        if row["variant"] == "direct_k4_geometry_k8_pose"
    ]
    component = [
        row
        for row in stage4e
        if row["variant"]
        == "component_k4_translation_k8_rotation"
    ]
    stage4e_stop_ok = (
        len(direct) == 9
        and len(component) == 9
        and max(number(row, "ate_ratio_to_k4") for row in direct) > 1.10
        and max(
            number(row, "rpe_trans_ratio_to_k8") for row in component
        )
        > 1.10
    )
    case_assets_ok = (
        len(cases) == 4
        and len({row["case_id"] for row in cases}) == 4
        and all(row["source_available"] == "yes" for row in cases)
    )
    manifest_by_path = {row["relative_path"]: row for row in manifest}
    required_assets_ok = all(
        relative in manifest_by_path
        and int(manifest_by_path[relative]["size_bytes"]) > 0
        for relative in REQUIRED_ASSETS
    )
    manifest_hashes_ok = bool(manifest) and all(
        osp.isfile(osp.join(args.asset_root, row["relative_path"]))
        and sha256(osp.join(args.asset_root, row["relative_path"]))
        == row["sha256"]
        for row in manifest
    )
    checks = (
        source_coverage_ok,
        frozen_roles_ok,
        stage4c_evidence_ok,
        stage4e_stop_ok,
        case_assets_ok,
        required_assets_ok,
        manifest_hashes_ok,
    )
    decision = (
        "PASS_PAPER_ASSETS_FROZEN"
        if all(checks)
        else "FAIL_INCOMPLETE_PAPER_ASSETS"
    )
    output = {
        "source_coverage_ok": yes(source_coverage_ok),
        "frozen_roles_ok": yes(frozen_roles_ok),
        "stage4c_evidence_ok": yes(stage4c_evidence_ok),
        "stage4e_stop_decision_ok": yes(stage4e_stop_ok),
        "case_assets_ok": yes(case_assets_ok),
        "required_assets_ok": yes(required_assets_ok),
        "manifest_hashes_ok": yes(manifest_hashes_ok),
        "num_manifest_assets": len(manifest),
        "decision": decision,
    }
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(output)
    print(f"Stage 4D decision: {decision}")
    print(f"Wrote Stage 4D gate to {args.output}")
    if not all(checks):
        raise RuntimeError(decision)


if __name__ == "__main__":
    main()
