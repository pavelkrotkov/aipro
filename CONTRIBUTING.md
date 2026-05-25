# Contributing

Install the project and development tools:

```sh
uv sync --all-extras --dev
```

Run the test suite:

```sh
uv run pytest
```

Run linting and formatting:

```sh
uv run ruff check .
uv run ruff format .
```

Run type checking:

```sh
uv run ty check .
```

Install pre-commit hooks:

```sh
uv run pre-commit install
```
