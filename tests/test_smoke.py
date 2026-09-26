"""Quick smoke test for edge_cases validators."""
from edge_cases import (
    parse_github_url,
    normalize_github_url,
    InvalidRepoUrlError,
    LogEmptyError,
    LogMalformedError,
    detect_file_format,
    read_log,
    validate_log,
)

print("=== URL normalization ===")
assert normalize_github_url("github.com/foo/bar") == "https://github.com/foo/bar"
assert normalize_github_url("https://github.com/foo/bar") == "https://github.com/foo/bar"
print("ok")

print("=== URL parsing ===")
assert parse_github_url("github.com/foo/bar") == ("foo", "bar")
assert parse_github_url("https://github.com/foo/bar/") == ("foo", "bar")
assert parse_github_url("https://github.com/foo/bar.git") == ("foo", "bar")
try:
    parse_github_url("not a url")
    raise AssertionError("should have raised InvalidRepoUrlError")
except InvalidRepoUrlError:
    print("ok")

print("=== File format detection ===")
assert detect_file_format("log.csv") == "csv"
assert detect_file_format("log.jsonl") == "jsonl"
try:
    detect_file_format("log.txt")
    raise AssertionError("should have raised LogMalformedError")
except LogMalformedError:
    print("ok")

print("=== Empty log ===")
try:
    read_log(b"", "log.csv")
    raise AssertionError("should have raised LogEmptyError")
except LogEmptyError:
    print("ok")

print("=== Valid log ===")
csv_bytes = b"iteration,train_loss,val_loss,lr\n1,1.0,1.1,0.001\n2,0.9,1.0,0.001\n"
df = read_log(csv_bytes, "log.csv")
df = validate_log(df)
assert len(df) == 2
assert "train_loss" in df.columns
print("ok")

print("=== Column aliasing ===")
csv_bytes = b"step,loss,validation_loss,learning_rate\n1,1.0,1.1,0.001\n2,0.9,1.0,0.001\n"
df = read_log(csv_bytes, "log.csv")
df = validate_log(df)
assert "iteration" in df.columns
assert "train_loss" in df.columns
assert "val_loss" in df.columns
assert "lr" in df.columns
print("ok")

print()
print("ALL SMOKE TESTS PASSED")