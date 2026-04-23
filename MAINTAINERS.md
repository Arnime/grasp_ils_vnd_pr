# Maintainers

Primary maintainer:

- Arnaldo Mendes Pires Junior — GitHub: @Arnime — Email: (see GitHub profile)

If you would like to contribute or be added as a maintainer, please open a
Pull Request proposing the change and provide contact information.

## Access continuity

All release automation (PyPI publishing, signing, changelog) is handled by
GitHub Actions and does not require local secrets beyond a GitHub token.
Release secrets (PyPI API tokens) are stored as GitHub Actions repository
secrets and can be rotated by any future repository admin.

For governance and bus-factor information, see
[GOVERNANCE.md](GOVERNANCE.md).
