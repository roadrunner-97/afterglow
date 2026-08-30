# Releases

## Versioning

Versions follow [Semantic Versioning](https://semver.org) with a **pre-1.0 ramp** — i.e. while the major version is `0`, anything goes and there are no backwards-compatibility promises. Tags are prefixed with `v`.

```
v MAJOR . MINOR . PATCH
   │       │       │
   │       │       └── bug fixes only       e.g. v0.1.0 → v0.1.1
   │       └────────── features / changes   e.g. v0.1.0 → v0.2.0
   └────────────────── reserved for "I'd be happy if a stranger downloaded this"
```

Rules of thumb while we're still on `0.x`:

- New effect, new UI feature, refactor that changes user-visible behaviour → bump **MINOR**
- Crash fix, rendering bug, build fix → bump **PATCH**
- Don't worry about MAJOR until the app feels stable enough to commit to APIs

## Cutting a release

```sh
git tag -a v0.5.0 -m "Afterglow 0.5.0"
git push origin v0.5.0
```

Pushing a `v*` tag fires `.github/workflows/release.yml`: it rebuilds on a
clean Ubuntu runner, runs the full test suite, and publishes both an AppImage
and a stripped system-Qt tarball, each with a SHA-256 checksum. Auto-generated
changelog notes are appended below the install instructions.

Releases land at <https://github.com/roadrunner-97/afterglow/releases>.

## Re-doing a release

If the workflow fails (e.g. an apt package got renamed) and you need to retry the same version, delete both the tag and the half-created GitHub Release before re-tagging:

```sh
git push --delete origin v0.2.0     # remove from the remote
git tag -d v0.2.0                   # remove locally
# delete the draft Release in the GitHub web UI
git tag v0.2.0 && git push origin v0.2.0
```

If the broken release already shipped to people, **don't reuse the version** — bump the patch (`v0.2.1`) and ship a fix.
