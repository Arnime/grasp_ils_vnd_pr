# Security policy

## Supported versions

The latest released version is supported with security fixes. Older
versions receive fixes only on a best-effort basis.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security reports.

Instead, use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
feature, or email the maintainer.

You should receive an acknowledgement within 7 days. We aim to publish a
fix and an advisory within 30 days for confirmed vulnerabilities.

## Disclosure policy

We follow a coordinated disclosure model:

1. The reporter contacts us privately.
2. We confirm and develop a fix.
3. We agree on a public disclosure date.
4. We publish a release and a GitHub Security Advisory.

## Vulnerability response process

Our documented process for handling vulnerability reports:

1. **Receipt**: Reporter contacts us via private vulnerability report or
   email.
2. **Acknowledgement** (within 7 days): We confirm receipt and assign a
   severity.
3. **Investigation** (within 14 days): We reproduce and assess impact.
4. **Fix development**: We develop, review, and test a patch in a private
   fork.
5. **Coordinated disclosure**: We agree a public disclosure date with the
   reporter.
6. **Release and advisory** (within 30 days of confirmation): We publish a
   patched release and a GitHub Security Advisory (GHSA).
7. **Credit**: We publicly credit the reporter in the GHSA and in the
   acknowledgement table below, unless they request anonymity.

## Reporter credit policy

We credit every reporter of a confirmed vulnerability in the GitHub
Security Advisory and in the table below, unless the reporter explicitly
requests anonymity. No vulnerability reports have been resolved in the
last 12 months.

## Acknowledgements and audit log

We commit to acknowledging valid vulnerability reports within 7 days and
aim to publish fixes and advisories within 30 days for confirmed issues.
To provide transparency, maintainers will record acknowledgements and the
fix timeline in this section (or link to the related GitHub Security
Advisory):

| Date | Reporter | Acknowledged | Advisory / GHSA | Fixed in |
|------|----------|--------------|----------------|---------|
| - | - | - | - | - |

(Maintainers: when responding to a report, append a new row with dates
and links to the advisory or release that fixes the issue.)

## Security review

A security review of the `givp` codebase was conducted by the maintainer
in April 2026 covering the security requirements and the trust boundary
of the library. The review considered:

- **Attack surface:** `givp` is a pure-Python numerical library that
  accepts user-supplied callables and numerical bounds. It performs no
  network I/O, filesystem access, or process execution. The trust boundary
  is entirely the Python process; no sandbox or privilege separation
  applies.
- **Threat model:** an adversary could supply a malicious objective
  function, but that function already executes in the caller's process
  with the caller's permissions — `givp` does not expand this attack
  surface. Denial-of-service via an infinite loop in the callback is
  possible but is the caller's responsibility to guard against.
- **Input validation:** all public parameters are validated via
  `GIVPConfig` at call time. Invalid types and out-of-range values
  raise typed exceptions (`InvalidConfigError`, `BoundsError`) before
  any computation begins.
- **Dependency review:** the only runtime dependency is `numpy`. All CI
  and release dependencies are hash-pinned and audited weekly with
  `pip-audit` via the scheduled security workflow.
- **Static analysis:** `mypy --strict` and `ruff` run on every commit.
  `bandit` and `semgrep` are run in the security CI workflow to catch
  common Python security anti-patterns.
- **Fuzzing:** a `fuzz/fuzz_givp.py` target exercises the public API with
  arbitrary inputs using `atheris` to detect unexpected crashes.
- **Findings:** no security vulnerabilities were identified. The complete
  security requirements and assurance case are documented in
  [`docs/security-requirements.md`](docs/security-requirements.md).

The review will be refreshed whenever a significant architectural change
is made or at least once every five years.
