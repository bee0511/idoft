#!/usr/bin/env python3
"""Analyze commits for deleted or moved tests based on pr-data.csv entries."""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_TEST_DIRS: Sequence[str] = (
    "src/test/java",
    "src/test/groovy",
    "src/test/kotlin",
    "src/test/scala",
    "src/test/resources",
    "src/test/java9",
    "src/integration-test/java",
    "src/integration-test/kotlin",
    "src/integrationTest/java",
    "src/integrationTest/kotlin",
    "src/it/java",
    "src/it/groovy",
    "src/it/kotlin",
    "src/functionalTest/java",
    "src/functional-test/java",
    "src/acceptance-test/java",
    "src/component-test/java",
    "tests/java",
)

TEST_FILE_EXTENSIONS: Sequence[str] = ("java", "kt", "kts", "groovy", "scala")

CSV_TEST_NAME_KEY = "Fully-Qualified Test Name (packageName.ClassName.methodName)"


@dataclass
class DiffEntry:
    status: str
    path: Optional[str] = None
    old_path: Optional[str] = None
    new_path: Optional[str] = None

    @property
    def normalized_status(self) -> str:
        if not self.status:
            return "unknown"
        code = self.status[0]
        if code == "R":
            return "renamed"
        if code == "C":
            return "copied"
        if code == "D":
            return "deleted"
        if code == "A":
            return "added"
        if code == "M":
            return "modified"
        return self.status.lower()


@dataclass
class AnalysisResult:
    project_url: str
    commit: str
    parent: Optional[str]
    module_path: str
    test_name: str
    class_name: str
    method_name: str
    file_before: Optional[str]
    file_after: Optional[str]
    diff_entry: Optional[DiffEntry]
    method_before: Optional[bool]
    method_after: Optional[bool]
    notes: List[str]

    def summarize_chain(self) -> str:
        chain: List[str] = []
        if self.file_before:
            chain.append(self.file_before)
        if self.diff_entry and self.diff_entry.normalized_status == "renamed" and self.file_after:
            chain.append(self.file_after)
        if self.file_after is None and self.diff_entry and self.diff_entry.normalized_status == "deleted":
            chain.append("<deleted>")
        if not chain and self.file_after is None:
            chain.append("<missing>")
        return " -> ".join(chain) if chain else "(no file information)"

    def status_line(self) -> str:
        if not self.diff_entry:
            return "File untouched in commit"
        status = self.diff_entry.normalized_status
        if status == "renamed" and self.diff_entry.status:
            status = f"renamed ({self.diff_entry.status})"
        return f"File status: {status}"

    def deletion_summary(self) -> str:
        if self.method_before is False:
            return "Method not present in parent revision"
        if self.method_before and not self.method_after:
            if self.file_after is None:
                return "Method deleted because containing file was removed"
            return "Method removed from file"
        if self.method_before and self.method_after:
            return "Method still present after commit"
        if self.method_before is None:
            return "Unable to inspect parent file"
        return "No evidence of deletion"


class GitRepository:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise ValueError(f"Repository path does not exist: {self.root}")

    def _run(self, args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["git", "-C", self.root, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check,
        )
        return completed

    def run(self, *args: str) -> str:
        return self._run(list(args)).stdout

    def commit_exists(self, commit: str) -> bool:
        try:
            self._run(["cat-file", "-e", commit], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def parents(self, commit: str) -> List[str]:
        output = self.run("rev-list", "--parents", "-n", "1", commit).strip()
        if not output:
            return []
        parts = output.split()
        return parts[1:]

    @lru_cache(maxsize=None)
    def tree_files(self, commit: str) -> Tuple[str, ...]:
        output = self.run("ls-tree", "-r", "--full-tree", "--name-only", commit)
        files = tuple(line.strip() for line in output.splitlines() if line.strip())
        return files

    def has_file(self, commit: str, path: str) -> bool:
        try:
            self._run(["cat-file", "-e", f"{commit}:{path}"], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def show_file(self, commit: str, path: str) -> Optional[str]:
        try:
            return self.run("show", f"{commit}:{path}")
        except subprocess.CalledProcessError:
            return None

    def diff_name_status(self, parent: str, commit: str) -> List[DiffEntry]:
        completed = self._run(["diff", "--name-status", "-M", "-C", parent, commit])
        entries: List[DiffEntry] = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            status = parts[0]
            if status.startswith("R") or status.startswith("C"):
                if len(parts) >= 3:
                    entries.append(DiffEntry(status=status, old_path=parts[1], new_path=parts[2]))
            elif len(parts) >= 2:
                entries.append(DiffEntry(status=status, path=parts[1]))
        return entries


def join_path(*components: str) -> str:
    filtered = [c.strip("/") for c in components if c and c.strip("/")]
    return "/".join(filtered)


def split_test_name(fqtn: str) -> Tuple[str, str, List[str]]:
    cleaned = fqtn.strip()
    if not cleaned:
        raise ValueError("Empty test name")
    parts = cleaned.split('.')
    if len(parts) < 2:
        raise ValueError(f"Unexpected test name format: {fqtn}")
    method_name = parts[-1]
    class_name = parts[-2]
    package_parts = parts[:-2]
    return method_name, class_name, package_parts


def filter_records(rows: Iterable[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    for row in rows:
        commit = (row.get("SHA Detected") or "").strip()
        project = (row.get("Project URL") or "").strip()
        module = (row.get("Module Path") or "").strip()
        test_name = (row.get(CSV_TEST_NAME_KEY) or "").strip()
        status = (row.get("Status") or "").strip()
        category = (row.get("Category") or "").strip()

        if args.project and not any(p.lower() in project.lower() for p in args.project):
            continue
        if args.status and not any(s.lower() in status.lower() for s in args.status):
            continue
        if args.category and not any(c.lower() in category.lower() for c in args.category):
            continue
        if args.module and args.module.lower() not in module.lower():
            continue
        if args.test_pattern and args.test_pattern.lower() not in test_name.lower():
            continue
        if args.require_commit and not commit:
            continue
        filtered.append(row)
        if args.limit and len(filtered) >= args.limit:
            break
    return filtered


def strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"(?m)^\s*#.*$", "", code)
    return code


def method_exists(content: str, method_name: str) -> bool:
    if not content:
        return False
    sanitized = strip_comments(content)
    java_pattern = re.compile(
        rf"^\s*(?:public|protected|private|static|final|synchronized|abstract|default|native|transient|strictfp|\s)*(?:<[^>]+>\s*)?(?:[\w\[\]<>.,?]+\s+)+{re.escape(method_name)}\s*\(",
        re.MULTILINE,
    )
    kotlin_pattern = re.compile(
        rf"^\s*(?:suspend\s+)?fun\s+{re.escape(method_name)}\s*\(",
        re.MULTILINE,
    )
    scala_pattern = re.compile(
        rf"^\s*def\s+{re.escape(method_name)}\s*(?:\[.*?\])?\s*\(",
        re.MULTILINE,
    )
    return bool(java_pattern.search(sanitized) or kotlin_pattern.search(sanitized) or scala_pattern.search(sanitized))


def choose_best_candidate(
    candidates: Sequence[str],
    module_path: str,
    package_parts: Sequence[str],
    class_name: str,
) -> Optional[str]:
    if not candidates:
        return None
    module_lower = module_path.lower().strip("/")
    package_path = "/".join(package_parts).lower()
    package_segments = [segment.lower() for segment in package_parts]

    best_path: Optional[str] = None
    best_score = -1
    for candidate in candidates:
        candidate_lower = candidate.lower()
        if "/test" not in candidate_lower and "test/" not in candidate_lower:
            continue
        if not os.path.basename(candidate_lower).startswith(class_name.lower()):
            continue
        score = 0
        if module_lower:
            if candidate_lower.startswith(module_lower):
                score += 5
            elif f"/{module_lower}/" in candidate_lower:
                score += 4
        if package_path and package_path in candidate_lower:
            score += 5
        for segment in package_segments:
            if segment and segment in candidate_lower:
                score += 1
        if class_name.lower() in os.path.basename(candidate_lower):
            score += 3
        if score > best_score:
            best_score = score
            best_path = candidate
    return best_path


def find_candidate_file(
    repo: GitRepository,
    commit: str,
    module_path: str,
    package_parts: Sequence[str],
    class_name: str,
) -> Optional[str]:
    package_path = "/".join(package_parts)
    for test_dir in DEFAULT_TEST_DIRS:
        for extension in TEST_FILE_EXTENSIONS:
            candidate = join_path(module_path, test_dir, package_path, f"{class_name}.{extension}")
            if repo.has_file(commit, candidate):
                return candidate
    tree = repo.tree_files(commit)
    matching: List[str] = []
    for extension in TEST_FILE_EXTENSIONS:
        suffix = f"{class_name}.{extension}"
        matching.extend(path for path in tree if path.endswith(suffix))
    return choose_best_candidate(matching, module_path, package_parts, class_name)


def entry_matches_class(entry: DiffEntry, class_name: str) -> bool:
    class_lower = class_name.lower()
    for path in (entry.path, entry.old_path, entry.new_path):
        if path and os.path.basename(path).lower().startswith(class_lower):
            return True
    return False


def find_diff_entry(entries: Sequence[DiffEntry], file_before: Optional[str], class_name: str) -> Optional[DiffEntry]:
    if file_before:
        for entry in entries:
            if entry.normalized_status in {"renamed", "copied"}:
                if entry.old_path == file_before:
                    return entry
            elif entry.path == file_before:
                return entry
    for entry in entries:
        if entry_matches_class(entry, class_name):
            return entry
    return None


def determine_file_after(
    repo: GitRepository,
    commit: str,
    entry: Optional[DiffEntry],
    fallback: Optional[str],
) -> Optional[str]:
    if entry is None:
        if fallback and repo.has_file(commit, fallback):
            return fallback
        return None
    status = entry.normalized_status
    if status == "deleted":
        return None
    if status == "renamed":
        return entry.new_path
    path = entry.path or entry.new_path or fallback
    if path and repo.has_file(commit, path):
        return path
    return None


def analyze_against_parent(
    repo: GitRepository,
    row: Dict[str, str],
    commit: str,
    parent: str,
    diff_cache: Dict[Tuple[str, str], List[DiffEntry]],
) -> AnalysisResult:
    project = (row.get("Project URL") or "").strip()
    module = (row.get("Module Path") or "").strip()
    test_name = (row.get(CSV_TEST_NAME_KEY) or "").strip()
    method_name, class_name, package_parts = split_test_name(test_name)
    source_class_name = class_name.split('$', 1)[0]

    file_before = find_candidate_file(repo, parent, module, package_parts, source_class_name)
    cache_key = (parent, commit)
    if cache_key not in diff_cache:
        diff_cache[cache_key] = repo.diff_name_status(parent, commit)
    diff_entry = find_diff_entry(diff_cache[cache_key], file_before, source_class_name)
    file_after = determine_file_after(repo, commit, diff_entry, file_before)

    notes: List[str] = []
    method_before: Optional[bool] = None
    method_after: Optional[bool] = None

    if file_before:
        content_before = repo.show_file(parent, file_before)
        if content_before is None:
            notes.append(f"Unable to read {file_before} in parent {parent}")
        else:
            method_before = method_exists(content_before, method_name)
    else:
        notes.append("File not found in parent commit")

    if file_after:
        content_after = repo.show_file(commit, file_after)
        if content_after is None:
            notes.append(f"Unable to read {file_after} in commit {commit}")
        else:
            method_after = method_exists(content_after, method_name)
    else:
        method_after = False

    return AnalysisResult(
        project_url=project,
        commit=commit,
        parent=parent,
        module_path=module,
        test_name=test_name,
        class_name=class_name,
        method_name=method_name,
        file_before=file_before,
        file_after=file_after,
        diff_entry=diff_entry,
        method_before=method_before,
        method_after=method_after,
        notes=notes,
    )


def analyze_row(repo: GitRepository, row: Dict[str, str]) -> Tuple[List[AnalysisResult], List[str]]:
    commit = (row.get("SHA Detected") or "").strip()
    notes: List[str] = []
    if not commit:
        notes.append("Missing commit SHA")
        return [], notes
    if not repo.commit_exists(commit):
        notes.append(f"Commit {commit} not found in repository {repo.root}")
        return [], notes
    parents = repo.parents(commit)
    if not parents:
        notes.append(f"Commit {commit} has no parents; cannot determine deletions")
        return [], notes

    diff_cache: Dict[Tuple[str, str], List[DiffEntry]] = {}
    results = [analyze_against_parent(repo, row, commit, parent, diff_cache) for parent in parents]
    return results, notes


def format_analysis(result: AnalysisResult) -> str:
    lines = [f"Parent: {result.parent}"]
    if result.file_before:
        lines.append(f"  File before: {result.file_before}")
    else:
        lines.append("  File before: <not found>")
    if result.file_after and result.file_after != result.file_before:
        lines.append(f"  File after: {result.file_after}")
    lines.append(f"  {result.status_line()}")
    lines.append(f"  Rename chain: {result.summarize_chain()}")
    if result.method_before is not None:
        lines.append(f"  Method in parent: {'yes' if result.method_before else 'no'}")
    if result.method_after is not None:
        lines.append(f"  Method after commit: {'yes' if result.method_after else 'no'}")
    lines.append(f"  Result: {result.deletion_summary()}")
    for note in result.notes:
        lines.append(f"  Note: {note}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-data", required=True, help="Path to pr-data.csv")
    parser.add_argument("--repo", required=True, help="Path to a local clone of the project under analysis")
    parser.add_argument("--project", action="append", help="Case-insensitive substring to match project URLs")
    parser.add_argument("--status", action="append", help="Case-insensitive substring to match statuses")
    parser.add_argument("--category", action="append", help="Case-insensitive substring to match categories")
    parser.add_argument("--module", help="Substring to match module paths")
    parser.add_argument("--test-pattern", help="Substring to match fully-qualified test names")
    parser.add_argument("--limit", type=int, help="Maximum number of rows to process")
    parser.add_argument("--require-commit", action="store_true", help="Only include rows that specify a commit SHA")
    parser.add_argument("--output", help="Write results to the given file in addition to stdout")
    return parser.parse_args()


def load_csv(path: str) -> Iterable[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            yield row


def main() -> None:
    args = parse_args()
    repo = GitRepository(args.repo)
    rows = list(load_csv(args.pr_data))
    filtered_rows = filter_records(rows, args)

    if not filtered_rows:
        message = "No matching rows found in dataset"
        print(message)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(message + "\n")
        return

    outputs: List[str] = []
    total = len(filtered_rows)

    for index, row in enumerate(filtered_rows, start=1):
        project = (row.get("Project URL") or "").strip()
        commit = (row.get("SHA Detected") or "").strip()
        test_name = (row.get(CSV_TEST_NAME_KEY) or "").strip()
        header_lines = [
            f"=== Record {index}/{total} ===",
            f"Project: {project}",
            f"Commit: {commit or '<missing>'}",
            f"Test: {test_name or '<missing>'}",
        ]
        results, row_notes = analyze_row(repo, row)
        if row_notes:
            header_lines.extend(f"Note: {note}" for note in row_notes)
        if results:
            header_lines.append(f"Parents analyzed: {len(results)}")
            for result in results:
                header_lines.append(format_analysis(result))
        else:
            header_lines.append("No analysis available")
        outputs.append("\n".join(header_lines))

    final_output = "\n\n".join(outputs)
    print(final_output)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(final_output + "\n")


if __name__ == "__main__":
    main()
