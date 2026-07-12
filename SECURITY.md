# Security policy

## Supported versions

The project is pre-1.0. Security fixes are applied to the latest code on `dev`
and included in the next promoted release. Historical development snapshots are
not maintained as separate supported lines.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/srgangaram-swe/comprehensive_ml/security/advisories/new)
with:

- affected revision and environment;
- minimal reproduction or proof of concept;
- expected impact and attack preconditions; and
- any known mitigation.

A report will be acknowledged as soon as practical, normally within three
business days. Remediation timing depends on severity and reproducibility.

## Artifact safety

Model files serialized by joblib/pickle must be treated as executable content.
Never load an artifact from an untrusted source. SHA-256 entries in a run
manifest detect unexpected changes; they do not make an unsafe serialization
format safe.
