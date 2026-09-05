# trinity-lowering

`trinity-lowering` provides concrete implementation candidates and validated
physical tensor program graphs.

## Getting Started

```shell
git clone --recursive https://github.com/kaist-ina/trinity-lowering.git
```

## Contribution

```shell
# Install the Git hook and check all tracked files.
pre-commit install
pre-commit run --all-files

# Format changes and run the Rust checks directly.
cargo fmt -p trinity-lowering
cargo clippy -p trinity-lowering --all-targets --locked -- -D warnings
```
