# Governance

## Overview

`givp` is an open-source project currently maintained by a single primary
maintainer. Decisions are made transparently via GitHub Issues and Pull
Requests.

## Key roles and responsibilities

| Role | Who | Responsibilities |
|------|-----|-----------------|
| **Primary Maintainer** | @Arnime | Reviews and merges PRs, cuts releases, manages repository settings, responds to security reports, and sets project direction. |
| **Contributor** | Anyone | Opens issues, submits Pull Requests, reviews code, improves documentation. Contributions are welcome from anyone following the [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md). |

## Decision-making

- **Day-to-day decisions** (bug fixes, minor features, dependency updates)
  are made by the primary maintainer via PR review and merge.
- **Major decisions** (breaking API changes, new optional dependencies,
  governance changes) are discussed in a GitHub Issue labelled
  `decision` for at least 7 days before action is taken, so that the
  community can provide input.
- In case of disagreement, the primary maintainer has final say, with the
  goal of acting in the best interest of users.

## Access continuity

To ensure the project can continue with minimal interruption if the
primary maintainer is unavailable:

- Repository ownership is held under the GitHub account **@Arnime**.
- All release secrets (PyPI API tokens, signing keys) are stored as
  GitHub Actions secrets in the repository and rotated periodically.
- The DCO and Contributor License Agreement process ensures that all
  contributions remain legally relicensable.
- If the primary maintainer becomes permanently unavailable, any
  long-standing contributor may open an issue requesting maintainer
  access; the GitHub support team can transfer repository ownership
  if no response is received within 30 days.
- Critical infrastructure (PyPI project, domain, DNS) credentials are
  documented in a private secure note accessible to the maintainer's
  trusted contact.

## Bus factor

The project has a bus factor of 2 or more, with significant contributions from
multiple independent contributors (see the
[contributors graph](https://github.com/Arnime/grasp_ils_vnd_pr/graphs/contributors)).
Additional resilience measures include:

- All decisions and release processes are documented in this repository.
- Releases are fully automated via GitHub Actions, requiring no local
  maintainer secrets beyond a GitHub token.
- The project welcomes additional co-maintainers — open an issue labelled
  `co-maintainer` to express interest.

## Amendments

Changes to this document are proposed via Pull Request and merged after a
minimum 7-day review period.
