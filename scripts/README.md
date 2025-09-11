# Development Scripts

This directory contains helper scripts for development and testing.

## Scripts

### `setup.sh`
Initial setup script that:
- Activates the virtual environment
- Installs/updates dependencies
- Runs all tests

Usage:
```bash
./scripts/setup.sh
```

### `dev.sh`
Development helper script with common commands.

Usage:
```bash
./scripts/dev.sh <command>
```

Available commands:
- `test` - Run all tests
- `test-vector` - Run vector embedding tests
- `test-workflow` - Run workflow tests
- `server` - Start development server
- `install` - Install/update dependencies
- `lint` - Run linting

Examples:
```bash
# Run all tests
./scripts/dev.sh test

# Start development server
./scripts/dev.sh server

# Run vector tests
./scripts/dev.sh test-vector
```

## Cursor Rules

This project follows these cursor rules:

1. **Environment Setup**: Always run `source .venv/bin/activate` before Python commands
2. **Test Organization**: All tests go in `src/test/` folder
3. **Dependency Management**: Update `pyproject.toml` when adding new dependencies

## Virtual Environment

The project uses a virtual environment located at `.venv/`. Always activate it before running any Python commands:

```bash
source .venv/bin/activate
```
