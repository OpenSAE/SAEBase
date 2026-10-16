# Maintainer Workflow

SAEBase is the shared software infrastructure for OpenSAE projects. Its own [Contributing Guide](../CONTRIBUTING.md) defines the repository-specific test and provenance expectations; the [OpenSAE Governance](https://github.com/OpenSAE/.github/blob/main/GOVERNANCE.md) and [Code of Conduct](https://github.com/OpenSAE/.github/blob/main/CODE_OF_CONDUCT.md) apply across the organization.

## Review and ownership

The primary maintainer is [@AtheavenMa](https://github.com/AtheavenMa). The [CODEOWNERS](../.github/CODEOWNERS) file assigns current ownership of critical API, experiment, infrastructure, and release paths. Additional maintainers should only be added to CODEOWNERS once they have agreed to the responsibility and have write access.

Code-owner designation requests reviewers; it does not by itself block a merge. Requiring separate code-owner approval is appropriate once a second independent reviewer is available. The author of a pull request cannot independently approve their own proposed changes.

## Pull request expectations

1. Identify the API, loader, data contract, or research behavior changed.
2. Run the deterministic offline regression suite, including packaging checks where relevant.
3. For new SAE or model support, document the original upstream source, exact hook, layer, normalization convention, access conditions and license.
4. Distinguish successful synthetic checks from actual checkpoint, model-pair and GPU acceptance. Provide immutable evidence for any new verification claims.
5. Preserve public API, persisted cache schemas and scientific protocols unless compatibility changes are deliberate and documented.
6. Merge only after applicable CI checks are successful, and verify the target branch has not moved.

Do not reimplement SAR, RSS or Input Localization metrics in SAEBase merely to produce duplicated scientific implementations; preserve independent research sources and integration contracts.

## Repository protection

GitHub branch protection and required checks are repository settings, not part of CODEOWNERS. They must be configured separately by an administrator. Start by requiring pull requests and successful standard CI checks for main, and disallow force pushes. Do not require a separate reviewer when there is only one maintainer; that would prevent self-authored PRs from merging.

Current standard CI jobs include offline-tests on Python 3.11 and 3.12, analysis-install, package, and localization-native. Long-running model or GPU acceptance runs remain a separate release or adapter validation responsibility.
