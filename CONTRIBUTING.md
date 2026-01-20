# Contributing to OmenDB

We welcome contributions! Before you contribute, please read this document.

## Contributor License Agreement (CLA)

By submitting a pull request, you agree to the following:

1. You grant the project maintainers a perpetual, worldwide, non-exclusive, royalty-free license to use, reproduce, modify, and distribute your contributions under any license.

2. You have the right to submit the contribution and it doesn't violate any other agreement.

3. You understand your contribution may be relicensed as the project evolves.

**We use [CLA Assistant](https://cla-assistant.io/) to manage this.** On your first PR, you'll be asked to sign the CLA by commenting on the PR. It takes 10 seconds.

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

# Node tests
cd node
npm install
npm run build
npm test
```

## Code Style

- Rust: `cargo fmt && cargo clippy`
- Python: Follow existing patterns
- TypeScript: Follow existing patterns

## Submitting Changes

1. Fork the repository
2. Create a branch for your change
3. Run tests
4. Submit a PR with a clear description

## Reporting Issues

Open an issue with:

- OmenDB version
- Python/Node version
- Minimal reproduction

## License

OmenDB is licensed under [PolyForm Shield 1.0.0](LICENSE). By contributing, you agree that your contributions will be licensed under the project's CLA terms.
