## checkCommitForDeletion

`check_deletion.py` inspects commits referenced in `pr-data.csv` and reports how
individual test cases move or disappear across each commit.

### Requirements

- Python 3.8 or newer.
- A local clone of the repository that contains the commit you want to audit.
- Access to the `pr-data.csv` file shipped with this repository (or another
  dataset in the same format).

### Usage

```bash
# Run from the repository root
./auto-check-deletion/check_deletion.py \
    --pr-data pr-data.csv \
    --repo /path/to/shardingsphere-elasticjob \
    --project https://github.com/apache/shardingsphere-elasticjob \
    --status Deleted \
    --limit 5
```

Key options:

- `--project`, `--status`, `--category`, `--module`, and `--test-pattern` allow
  you to narrow the dataset by substring (case-insensitive). You may pass the
  same option multiple times to match several patterns.
- `--limit` restricts the number of matching rows that will be analysed.
- `--output` writes the formatted report to a file in addition to standard
  output.

### Output

For every matching entry the script prints the associated project, commit, and
fully-qualified test name. Each parent of the commit is inspected individually
so merges are handled explicitly. For every parent the tool reports:

- The path of the test file before and after the commit (following renames
  automatically).
- The git status of the file within the commit (modified, renamed, deleted,
  etc.).
- Whether the test method exists before and after the change.
- A textual summary that highlights whether the method was removed because the
  file vanished or because the method body was deleted.

The rename chain is displayed explicitly, for example
`module/src/test/java/.../MyTest.java -> module/src/test/java/.../MyTest.java -> <deleted>`
so you can follow when a file moves and where it ultimately disappears.
