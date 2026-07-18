from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "publish_private_agent1_v5_metadata.py"
SPEC = importlib.util.spec_from_file_location(
    "publish_private_agent1_v5_metadata", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(
    size: int, *, sha: str | None = None, blob: str | None = None
) -> SimpleNamespace:
    lfs = SimpleNamespace(sha256=sha) if sha else None
    return SimpleNamespace(size=size, lfs=lfs, blob_id=blob)


def file_row(
    path: str, size: int, *, sha: str | None = None, blob: str | None = None
) -> SimpleNamespace:
    result = row(size, sha=sha, blob=blob)
    result.path = path
    return result


class FakeCommitOperationAdd:
    """Small stand-in whose payload has the same bytes contract as HF Hub."""

    def __init__(self, *, path_in_repo: str, path_or_fileobj: bytes) -> None:
        if not isinstance(path_or_fileobj, bytes):
            raise TypeError("test publisher must pass an immutable bytes snapshot")
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class FakeHub:
    def __init__(
        self,
        *,
        base_revision: str,
        commit_revision: str,
        pending_path: Path,
    ) -> None:
        self.base_revision = base_revision
        self.commit_revision = commit_revision
        self.pending_path = pending_path
        self.head = base_revision
        self.private = True
        self.create_behavior = "success"
        self.download_corrupt = False
        self.create_calls = 0
        self.create_parents: list[str] = []
        self.pending_seen_during_remote_reads: list[bool] = []
        self.base_payloads = {
            ".gitattributes": b"*.parquet filter=lfs\n",
            "README.md": b"old card\n",
            "data/part-00000.parquet": b"fake immutable parquet payload",
        }
        self.trees: dict[str, dict[str, SimpleNamespace]] = {
            base_revision: {
                ".gitattributes": file_row(
                    ".gitattributes",
                    len(self.base_payloads[".gitattributes"]),
                    blob="attributes-blob",
                ),
                "README.md": file_row(
                    "README.md", len(self.base_payloads["README.md"]), blob="old-readme"
                ),
                "data/part-00000.parquet": file_row(
                    "data/part-00000.parquet",
                    len(self.base_payloads["data/part-00000.parquet"]),
                    sha="d" * 64,
                ),
            }
        }
        self.payloads: dict[str, dict[str, bytes]] = {
            base_revision: dict(self.base_payloads)
        }

    def repo_info(self, **_: Any) -> SimpleNamespace:
        if self.head != self.base_revision:
            self.pending_seen_during_remote_reads.append(self.pending_path.exists())
        return SimpleNamespace(sha=self.head, private=self.private)

    def list_repo_tree(self, *, revision: str, **_: Any) -> list[SimpleNamespace]:
        if revision != self.base_revision:
            self.pending_seen_during_remote_reads.append(self.pending_path.exists())
        return list(self.trees[revision].values())

    def install_metadata_commit(
        self,
        operations: list[FakeCommitOperationAdd],
        *,
        parent_revision: str | None = None,
        commit_revision: str | None = None,
    ) -> None:
        parent_revision = parent_revision or self.head
        commit_revision = commit_revision or self.commit_revision
        updated = dict(self.trees[parent_revision])
        payloads = dict(self.payloads[parent_revision])
        for operation in operations:
            payload = operation.path_or_fileobj
            updated[operation.path_in_repo] = file_row(
                operation.path_in_repo,
                len(payload),
                blob=hashlib.sha256(payload).hexdigest(),
            )
            payloads[operation.path_in_repo] = payload
        self.trees[commit_revision] = updated
        self.payloads[commit_revision] = payloads
        self.head = commit_revision

    def create_commit(
        self,
        *,
        operations: list[FakeCommitOperationAdd],
        parent_commit: str,
        **_: Any,
    ) -> SimpleNamespace:
        self.create_calls += 1
        self.create_parents.append(parent_commit)
        self.install_metadata_commit(operations, parent_revision=parent_commit)
        if self.create_behavior == "timeout_committed":
            raise TimeoutError("simulated response timeout after commit")
        return SimpleNamespace(oid=self.commit_revision)

    def hf_hub_download(
        self,
        *,
        filename: str,
        revision: str,
        local_dir: str,
        **_: Any,
    ) -> str:
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.payloads[revision][filename]
        target.write_bytes(b"corrupt" if self.download_corrupt else payload)
        return str(target)


def write_request(tmp_path: Path) -> dict[str, Any]:
    repo_id = "owner/private-agent1-v5"
    publication = tmp_path / "publication.json"
    publication.write_text(
        json.dumps(
            {
                "schema_version": "agent1_v5_private_hf_publication_receipt_v1",
                "status": "passed",
                "private": True,
                "repo_id": repo_id,
                "commit_sha": "base-sha",
            }
        ),
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_bytes(b"new readiness card\n")
    provenance = tmp_path / "provenance.json"
    provenance.write_bytes(b'{"status":"intermediate"}\n')
    output = tmp_path / "metadata-publication.json"
    return {
        "repo_id": repo_id,
        "publication": publication,
        "readme": readme,
        "provenance": provenance,
        "output": output,
        "argv": [
            "--repo-id",
            repo_id,
            "--publication-receipt",
            str(publication),
            "--add",
            f"{readme}=README.md",
            "--add",
            f"{provenance}=manifests/provenance.json",
            "--output",
            str(output),
            "--execute",
            "--token",
            "test-token",
        ],
    }


def install_previous_metadata(
    request: dict[str, Any],
    hub: FakeHub,
    *,
    commit_sha: str = "metadata-v1-sha",
) -> Path:
    old_readme = b"first readiness card\n"
    old_provenance = b'{"status":"first-metadata-commit"}\n'
    operations = [
        FakeCommitOperationAdd(path_in_repo="README.md", path_or_fileobj=old_readme),
        FakeCommitOperationAdd(
            path_in_repo="manifests/provenance.json",
            path_or_fileobj=old_provenance,
        ),
    ]
    hub.install_metadata_commit(
        operations,
        parent_revision="base-sha",
        commit_revision=commit_sha,
    )
    receipt = request["output"].with_name("previous-metadata-publication.json")
    receipt.write_text(
        MODULE.canonical_json(
            {
                "schema_version": MODULE.METADATA_PUBLICATION_SCHEMA,
                "status": "passed",
                "repo_id": request["repo_id"],
                "private": True,
                "base_revision": "base-sha",
                "base_publication_receipt": {
                    "sha256": MODULE.sha256_file(request["publication"])
                },
                "additions": {
                    operation.path_in_repo: {
                        "bytes": len(operation.path_or_fileobj),
                        "sha256": hashlib.sha256(operation.path_or_fileobj).hexdigest(),
                    }
                    for operation in operations
                },
                "commit_sha": commit_sha,
                "unchanged_files": 2,
                "unchanged_bytes": 42,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    request["argv"][4:4] = ["--previous-metadata-receipt", str(receipt)]
    return receipt


def install_fake_hub(monkeypatch: pytest.MonkeyPatch, hub: FakeHub) -> None:
    fake_module = ModuleType("huggingface_hub")
    fake_module.CommitOperationAdd = FakeCommitOperationAdd
    fake_module.HfApi = lambda token: hub
    fake_module.hf_hub_download = hub.hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)


def pending_payload(request: dict[str, Any], commit_sha: str) -> dict[str, Any]:
    additions = MODULE.local_additions(
        [
            f"{request['readme']}=README.md",
            f"{request['provenance']}=manifests/provenance.json",
        ]
    )
    result = {
        "schema_version": MODULE.METADATA_PUBLICATION_SCHEMA,
        "status": "verifying",
        "repo_id": request["repo_id"],
        "base_revision": "base-sha",
        "additions": {
            path: {"bytes": value["bytes"], "sha256": value["sha256"]}
            for path, value in additions.items()
        },
        "commit_sha": commit_sha,
    }
    if "--previous-metadata-receipt" in request["argv"]:
        previous_path = Path(
            request["argv"][request["argv"].index("--previous-metadata-receipt") + 1]
        )
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        result.update(
            {
                "parent_revision": previous["commit_sha"],
                "previous_metadata_receipt": {
                    "sha256": MODULE.sha256_file(previous_path)
                },
            }
        )
    return result


def test_same_remote_object_prefers_lfs_sha() -> None:
    assert MODULE.same_remote_object(row(10, sha="a" * 64), row(10, sha="a" * 64))
    assert not MODULE.same_remote_object(row(10, sha="a" * 64), row(10, sha="b" * 64))


def test_same_remote_object_checks_small_blob_id() -> None:
    assert MODULE.same_remote_object(row(10, blob="abc"), row(10, blob="abc"))
    assert not MODULE.same_remote_object(row(10, blob="abc"), row(11, blob="abc"))


def test_parse_addition_rejects_parent_path(tmp_path: Path) -> None:
    local = tmp_path / "README.md"
    local.write_text("x", encoding="utf-8")
    try:
        MODULE.parse_addition(f"{local}=../README.md")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("parent traversal should be rejected")


def test_parse_addition_rejects_dataset_payload_path(tmp_path: Path) -> None:
    local = tmp_path / "shard.parquet"
    local.write_bytes(b"not parquet")
    with pytest.raises(ValueError):
        MODULE.parse_addition(f"{local}=data/000000.parquet")


def test_parse_addition_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "README.md"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        MODULE.parse_addition(f"{link}=README.md")


def test_installed_commit_operation_add_accepts_immutable_bytes_snapshot(
    tmp_path: Path,
) -> None:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    source = tmp_path / "README.md"
    source.write_bytes(b"snapshot before mutation")
    additions = MODULE.local_additions([f"{source}=README.md"])

    source.write_bytes(b"mutated after snapshot")
    operation = huggingface_hub.CommitOperationAdd(
        path_in_repo="README.md",
        path_or_fileobj=additions["README.md"]["payload"],
    )

    assert isinstance(operation.path_or_fileobj, bytes)
    assert operation.path_or_fileobj == b"snapshot before mutation"
    assert (
        additions["README.md"]["sha256"]
        == hashlib.sha256(b"snapshot before mutation").hexdigest()
    )


def test_fresh_commit_writes_pending_then_verifies_and_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="metadata-sha",
        pending_path=pending,
    )
    install_fake_hub(monkeypatch, hub)

    assert MODULE.main(request["argv"]) == 0

    receipt = json.loads(request["output"].read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["commit_sha"] == "metadata-sha"
    assert receipt["recovered_existing_head"] is False
    assert receipt["unchanged_files"] == 2
    assert receipt["unchanged_bytes"] == sum(
        len(hub.base_payloads[path])
        for path in (".gitattributes", "data/part-00000.parquet")
    )
    assert hub.create_calls == 1
    assert any(hub.pending_seen_during_remote_reads)
    assert not pending.exists()


def test_second_metadata_commit_uses_passed_metadata_head_as_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="metadata-v2-sha",
        pending_path=pending,
    )
    previous_path = install_previous_metadata(request, hub)
    previous_sha256 = MODULE.sha256_file(previous_path)
    install_fake_hub(monkeypatch, hub)

    assert MODULE.main(request["argv"]) == 0

    receipt = json.loads(request["output"].read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["base_revision"] == "base-sha"
    assert receipt["parent_revision"] == "metadata-v1-sha"
    assert receipt["previous_metadata_receipt"] == {"sha256": previous_sha256}
    assert receipt["commit_sha"] == "metadata-v2-sha"
    assert hub.create_calls == 1
    assert hub.create_parents == ["metadata-v1-sha"]
    assert not pending.exists()


def test_second_metadata_commit_preserves_inherited_metadata_and_all_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    provenance_argument = f"{request['provenance']}=manifests/provenance.json"
    provenance_index = request["argv"].index(provenance_argument)
    del request["argv"][provenance_index - 1 : provenance_index + 1]
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="metadata-v2-sha",
        pending_path=pending,
    )
    install_previous_metadata(request, hub)
    prior_tree = dict(hub.trees["metadata-v1-sha"])
    prior_provenance = hub.payloads["metadata-v1-sha"]["manifests/provenance.json"]
    install_fake_hub(monkeypatch, hub)

    assert MODULE.main(request["argv"]) == 0

    receipt = json.loads(request["output"].read_text(encoding="utf-8"))
    assert receipt["unchanged_files"] == 3
    assert receipt["unchanged_bytes"] == sum(
        row.size for path, row in prior_tree.items() if path != "README.md"
    )
    assert MODULE.same_remote_object(
        prior_tree["manifests/provenance.json"],
        hub.trees["metadata-v2-sha"]["manifests/provenance.json"],
    )
    assert (
        hub.payloads["metadata-v2-sha"]["manifests/provenance.json"] == prior_provenance
    )
    assert MODULE.same_remote_object(
        prior_tree["data/part-00000.parquet"],
        hub.trees["metadata-v2-sha"]["data/part-00000.parquet"],
    )


def test_second_metadata_commit_rejects_tampered_previous_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="metadata-v2-sha",
        pending_path=pending,
    )
    previous_path = install_previous_metadata(request, hub)
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous["additions"]["README.md"]["sha256"] = "0" * 64
    previous_path.write_text(MODULE.canonical_json(previous) + "\n", encoding="utf-8")
    install_fake_hub(monkeypatch, hub)

    with pytest.raises(RuntimeError, match="remote metadata checksum mismatch"):
        MODULE.main(request["argv"])

    assert hub.create_calls == 0
    assert not pending.exists()
    assert not request["output"].exists()


def test_second_metadata_commit_rejects_prior_data_object_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="metadata-v2-sha",
        pending_path=pending,
    )
    install_previous_metadata(request, hub)
    hub.trees["metadata-v1-sha"]["data/part-00000.parquet"] = file_row(
        "data/part-00000.parquet",
        len(hub.base_payloads["data/part-00000.parquet"]),
        sha="e" * 64,
    )
    install_fake_hub(monkeypatch, hub)

    with pytest.raises(RuntimeError, match="changed data objects"):
        MODULE.main(request["argv"])

    assert hub.create_calls == 0
    assert not pending.exists()
    assert not request["output"].exists()


def test_second_metadata_commit_rejects_unrelated_live_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="metadata-v2-sha",
        pending_path=pending,
    )
    install_previous_metadata(request, hub)
    hub.install_metadata_commit(
        [
            FakeCommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=b"x" * len(request["readme"].read_bytes()),
            ),
            FakeCommitOperationAdd(
                path_in_repo="manifests/provenance.json",
                path_or_fileobj=b"y" * len(request["provenance"].read_bytes()),
            ),
        ],
        parent_revision="metadata-v1-sha",
        commit_revision="unrelated-sha",
    )
    install_fake_hub(monkeypatch, hub)

    with pytest.raises(RuntimeError, match="remote metadata checksum mismatch"):
        MODULE.main(request["argv"])

    journal = json.loads(pending.read_text(encoding="utf-8"))
    assert journal["parent_revision"] == "metadata-v1-sha"
    assert journal["commit_sha"] == "unrelated-sha"
    assert hub.create_calls == 0
    assert not request["output"].exists()


def test_timed_out_but_committed_head_is_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="metadata-sha",
        pending_path=pending,
    )
    hub.create_behavior = "timeout_committed"
    install_fake_hub(monkeypatch, hub)

    assert MODULE.main(request["argv"]) == 0

    receipt = json.loads(request["output"].read_text(encoding="utf-8"))
    assert receipt["commit_sha"] == "metadata-sha"
    assert receipt["recovered_existing_head"] is True
    assert receipt["status"] == "passed"
    assert hub.create_calls == 1
    assert not pending.exists()


def test_second_metadata_commit_timeout_is_recovered_from_exact_new_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="metadata-v2-sha",
        pending_path=pending,
    )
    install_previous_metadata(request, hub)
    hub.create_behavior = "timeout_committed"
    install_fake_hub(monkeypatch, hub)

    assert MODULE.main(request["argv"]) == 0

    receipt = json.loads(request["output"].read_text(encoding="utf-8"))
    assert receipt["commit_sha"] == "metadata-v2-sha"
    assert receipt["parent_revision"] == "metadata-v1-sha"
    assert receipt["recovered_existing_head"] is True
    assert hub.create_calls == 1
    assert hub.create_parents == ["metadata-v1-sha"]
    assert not pending.exists()


def test_second_metadata_commit_resumes_matching_pending_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="metadata-v2-sha",
        pending_path=pending,
    )
    install_previous_metadata(request, hub)
    hub.install_metadata_commit(
        [
            FakeCommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=request["readme"].read_bytes(),
            ),
            FakeCommitOperationAdd(
                path_in_repo="manifests/provenance.json",
                path_or_fileobj=request["provenance"].read_bytes(),
            ),
        ],
        parent_revision="metadata-v1-sha",
        commit_revision="metadata-v2-sha",
    )
    pending.write_text(
        MODULE.canonical_json(pending_payload(request, "metadata-v2-sha")) + "\n",
        encoding="utf-8",
    )
    install_fake_hub(monkeypatch, hub)

    assert MODULE.main(request["argv"]) == 0

    receipt = json.loads(request["output"].read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["commit_sha"] == "metadata-v2-sha"
    assert receipt["recovered_existing_head"] is True
    assert hub.create_calls == 0
    assert not pending.exists()


def test_existing_pending_journal_resumes_without_creating_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="metadata-sha",
        pending_path=pending,
    )
    hub.install_metadata_commit(
        [
            FakeCommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=request["readme"].read_bytes(),
            ),
            FakeCommitOperationAdd(
                path_in_repo="manifests/provenance.json",
                path_or_fileobj=request["provenance"].read_bytes(),
            ),
        ]
    )
    pending.write_text(
        MODULE.canonical_json(pending_payload(request, "metadata-sha")) + "\n",
        encoding="utf-8",
    )
    install_fake_hub(monkeypatch, hub)

    assert MODULE.main(request["argv"]) == 0

    receipt = json.loads(request["output"].read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["commit_sha"] == "metadata-sha"
    assert receipt["recovered_existing_head"] is True
    assert hub.create_calls == 0
    assert not pending.exists()


def test_unrelated_changed_head_is_rejected_and_pending_journal_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="unrelated-sha",
        pending_path=pending,
    )
    # Give the unrelated head the expected inventory and byte sizes, while
    # using different metadata bytes. Size-only checks would accept this;
    # downloaded SHA-256 verification must reject it.
    hub.install_metadata_commit(
        [
            FakeCommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=b"x" * len(request["readme"].read_bytes()),
            ),
            FakeCommitOperationAdd(
                path_in_repo="manifests/provenance.json",
                path_or_fileobj=b"y" * len(request["provenance"].read_bytes()),
            ),
        ]
    )
    install_fake_hub(monkeypatch, hub)

    with pytest.raises(RuntimeError, match="remote metadata checksum mismatch"):
        MODULE.main(request["argv"])

    journal = json.loads(pending.read_text(encoding="utf-8"))
    assert journal["status"] == "verifying"
    assert journal["commit_sha"] == "unrelated-sha"
    assert hub.create_calls == 0
    assert not request["output"].exists()


def test_retry_honors_429_retry_after_floor_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RateLimited(Exception):
        def __init__(self, retry_after: str) -> None:
            super().__init__("rate limited")
            self.response = SimpleNamespace(
                status_code=429, headers={"Retry-After": retry_after}
            )

    calls = 0
    waits: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RateLimited("17")
        return "done"

    monkeypatch.setattr(MODULE, "wait_retry", waits.append)

    assert MODULE.retry_hf(operation) == "done"
    assert waits == [310.0, 310.0]
    assert MODULE.retry_delay_seconds(RateLimited("725"), 0) == 725.0


def test_post_commit_verification_failure_leaves_resumable_pending_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = write_request(tmp_path)
    pending = request["output"].with_name(f".{request['output'].name}.pending")
    hub = FakeHub(
        base_revision="base-sha",
        commit_revision="metadata-sha",
        pending_path=pending,
    )
    hub.download_corrupt = True
    install_fake_hub(monkeypatch, hub)

    with pytest.raises(RuntimeError, match="remote metadata checksum mismatch"):
        MODULE.main(request["argv"])

    journal = json.loads(pending.read_text(encoding="utf-8"))
    assert journal["status"] == "verifying"
    assert journal["commit_sha"] == "metadata-sha"
    assert hub.create_calls == 1
    assert not request["output"].exists()

    # Once the remote read works, the same invocation resumes the journal and
    # finalizes without attempting a second commit.
    hub.download_corrupt = False
    assert MODULE.main(request["argv"]) == 0
    assert hub.create_calls == 1
    assert not pending.exists()
    receipt = json.loads(request["output"].read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["commit_sha"] == "metadata-sha"
