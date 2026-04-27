# Security requirements and assurance case

This document covers:

1. [Security requirements](#security-requirements) — what the library guarantees
and does not guarantee.
2. [Threat model](#threat-model) — adversarial scenarios and trust boundaries.
3. [Assurance case](#assurance-case) — argument that the design and
implementation are secure.

---

## Security requirements

### What `givp` guarantees

- **Input validation**: every public API call validates parameter types
  and value ranges at runtime. Invalid inputs raise `ValueError` or
  `TypeError` with descriptive messages before any computation begins.
- **No network access**: the library never initiates network connections.
  It is a pure-Python numerical optimizer; all computation is local.
- **No file I/O**: the library does not read or write files. Objective
  functions and bounds are supplied by the caller in memory.
- **Deterministic output given a seed**: when `Config.seed` is set, the
  result is reproducible across identical environments (same Python
  version, same NumPy version).
- **No credential handling**: the library does not process passwords,
  tokens, private keys, or any authentication material.
- **Memory safety**: implemented in pure Python with NumPy; no native
  code is compiled, eliminating whole classes of memory corruption
  vulnerabilities (buffer overflows, use-after-free, etc.).

### What `givp` does *not* guarantee

- **Safety of the caller's objective function**: if the user-supplied
  objective function has side effects, accesses the network, or raises
  exceptions, `givp` propagates those effects. The library cannot
  sandbox untrusted objective functions.
- **Confidentiality of objective function inputs/outputs**: the library
  stores intermediate solutions in memory during a run. If the process
  memory is accessible to an attacker, those values may be readable.
- **Correctness on adversarially crafted inputs outside stated
  constraints**: inputs that pass validation but are semantically
  adversarial (e.g., bounds that trigger extremely long runtimes) are
  not mitigated beyond the `max_iter` and `time_limit` configuration
  options.

---

## Threat model

### Assets

| Asset | Description |
|-------|-------------|
| Objective function results | Numerical outputs returned by the caller's function |
| Best solution found | The `Result` object returned to the caller |
| Library source code | The `givp` package on PyPI and GitHub |

### Trust boundaries

```text
+----------------------------------+
|        Caller's process          |
|  (trusted: controls Python env)  |
|                                  |
|  +----------------------------+  |
|  |   givp library             |  |
|  |  - validates inputs        |  |
|  |  - calls objective fn      |  |
|  |  - returns Result          |  |
|  +----------------------------+  |
|         |           ^            |
|         v           |            |
|  Objective function (caller-     |
|  supplied, considered trusted    |
|  within the same process)        |
+----------------------------------+
         |
         v
   NumPy / Python stdlib
   (considered trusted third-party)
```

**External threats considered:**

| Threat | Likelihood | Mitigation |
|--------|-----------|-----------|
| Malicious PyPI package replacing `givp` | Low | Releases are signed with Sigstore; users should verify attestations |
| Supply-chain attack via `numpy` | Medium | `pip-audit` and Dependabot monitor CVEs weekly |
| Adversarial inputs causing DoS (infinite loop) | Low | `max_iter` and `time_limit` caps bound execution time |
| Code injection via objective function | N/A | The library never evaluates strings as code |

---

## Assurance case

### Claim

> `givp` is secure for use as a numerical optimisation library in
> Python applications and research workflows.

### Argument structure

#### 1. Secure design principles applied

- **Least privilege**: the library requests no OS permissions beyond
  what Python itself requires. It does not spawn processes, open
  sockets, or write files.
- **Fail-safe defaults**: if a caller omits configuration, sensible
  safe defaults are used (e.g., `seed=None` for true randomness,
  `max_iter=1000`). Invalid configurations raise exceptions immediately
  rather than producing silent incorrect results.
- **Economy of mechanism**: the public API surface is minimal — one
  function (`optimize`) and one configuration class (`Config`). Fewer
  entry points mean a smaller attack surface.
- **Complete mediation**: all public inputs pass through the `Config`
  validation layer before being used in computation.
- **Open design**: the full source is publicly available; security
  claims are not based on obscurity.

#### 2. Common implementation weaknesses countered

| CWE | Weakness | Counter-measure |
|-----|----------|----------------|
| CWE-20 | Improper input validation | `Config` validates type and range; `mypy --strict` prevents type errors at static analysis time |
| CWE-190 | Integer overflow | Python integers are arbitrary precision; NumPy operations validated against dtype limits |
| CWE-676 | Use of potentially dangerous function | `bandit` and `semgrep` scan for dangerous functions in CI |
| CWE-119/125/787 | Buffer errors | Pure Python + NumPy; no unsafe memory operations |
| CWE-327 | Use of broken cryptographic algorithm | Library does not implement cryptography |
| CWE-502 | Deserialization of untrusted data | Library does not deserialize external data |
| CWE-89/79 | Injection | Library does not construct SQL, shell commands, or HTML |

#### 3. Security tooling evidence

| Tool | What it checks | Frequency |
|------|---------------|-----------|
| `bandit` | Common Python security anti-patterns (hardcoded secrets, unsafe calls) | Every push/PR |
| `semgrep` | Broader static analysis rules including OWASP Top 10 patterns | Every push/PR |
| `mypy --strict` | Type safety; prevents entire classes of runtime errors | Every push/PR |
| `pip-audit` | Known CVEs in dependencies | Every push/PR + weekly |
| Dependabot | Automated dependency updates for CVE-affected versions | Weekly |
| Sigstore / GitHub Attestations | Cryptographic provenance of PyPI releases | Every release |

#### 4. Residual risks

- **Bus factor ≥ 2**: multiple independent contributors; fully automated
  release pipeline and access continuity plan documented in
  [GOVERNANCE.md](https://github.com/Arnime/grasp_ils_vnd_pr/blob/main/GOVERNANCE.md).
- **Caller-supplied objective function**: no mitigation possible within
  the library; documented as out-of-scope in the security requirements.

### Evidence links

- Source: <https://github.com/Arnime/grasp_ils_vnd_pr/tree/main/src/givp>
- CI security workflow: <https://github.com/Arnime/grasp_ils_vnd_pr/blob/main/.github/workflows/security.yml>
- Audit workflow: <https://github.com/Arnime/grasp_ils_vnd_pr/blob/main/.github/workflows/audit.yml>
- Release attestations: <https://github.com/Arnime/grasp_ils_vnd_pr/attestations>
- Input validation: <https://github.com/Arnime/grasp_ils_vnd_pr/blob/main/src/givp/_config.py>
