# Repository Guidelines

## Commit And PR Titles

Commit messages and pull request titles should use the Conventional Commits style:

```text
type(scope): summary
```

The scope is optional.

Allowed types:

- `build`: Changes that affect the build system or external dependencies.
- `chore`: Routine maintenance that does not affect runtime behavior.
- `ci`: Changes to CI configuration or automation.
- `docs`: Documentation-only changes.
- `feat`: A new feature.
- `fix`: A bug fix.
- `perf`: A performance improvement.
- `refactor`: Code changes that neither fix a bug nor add a feature.
- `revert`: Reverts a previous change.
- `style`: Formatting or style changes that do not affect behavior.
- `test`: Adds or updates tests.

Examples:

- `feat: add voxel selection outline`
- `fix(worldgen): avoid cave cutoff at chunk edges`
- `chore: update contributor instructions`
