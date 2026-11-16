# Security policy

## 目的和主要功能

说明第三方资源使用与安全问题的反馈方式，帮助维护者定位并处理实际安全问题。

## 适用范围

用于加载外部资源、处理凭据或报告安全问题；具体处置依据实际对象与当前授权。

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

See `doc/code/development/PROVENANCE.md` for the release provenance checklist.
