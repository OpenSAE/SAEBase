# Security policy

SAEBase is research software. It can download public checkpoints and run external
model code in environments configured by the user.

## Reporting a security concern

Use the GitHub **Report a vulnerability** option under the repository's Security
tab when enabled. For a private report, contact the repository maintainers
through the organization profile contact information and request a private
reporting channel. Please avoid sharing credentials or exploit details in public issues.

## Safe use

- Review checkpoint and dataset provenance before loading external assets.
- Use isolated environments and least-privilege credentials.
- Avoid executing untrusted model repository code.
- Evaluate GPU integrations on dedicated test machines and control their process ownership.
- Never commit tokens, private prompts, or personal data to the repository.

See `docs/PROVENANCE.md` for the release provenance checklist.
