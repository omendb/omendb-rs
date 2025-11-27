# Contributing to OmenDB

## Development Setup

```bash
# Clone
git clone https://github.com/omendb/omendb.git
cd omendb

# Rust tests
cargo test --lib

# Python tests
cd python
uv sync
uv run pytest tests/
```

## Code Style

- Rust: `cargo fmt && cargo clippy`
- Python: Follow existing patterns

## Submitting Changes

1. Fork the repository
2. Create a branch for your change
3. Run tests
4. Submit a PR with a clear description

## Reporting Issues

Open an issue with:
- OmenDB version
- Python version
- Minimal reproduction

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
