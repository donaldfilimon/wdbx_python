# Self-hosted macOS runner

The `lint` job in `.github/workflows/lint.yml` (checks `lint (3.9)`, `lint (3.10)` and `lint (3.11)`) runs on a macOS arm64 runner registered to this repository. GitHub-hosted jobs can't start while the account's Actions billing is locked, but self-hosted jobs still run.

## Registration

| Field | Value |
|-------|-------|
| Labels | `self-hosted`, `macOS`, `ARM64`, `wdbx_python` |
| Register at | [Settings → Actions → Runners → New self-hosted runner](https://github.com/donaldfilimon/wdbx_python/settings/actions/runners/new?arch=arm64) (macOS, ARM64) |

A runner is registered to one repository. If the same Mac already runs a runner for another repository (for example `abi`), install a second runner in its own directory (for example `~/actions-runner-wdbx_python`), give it the custom label `wdbx_python` when `./config.sh` asks for additional labels, then run `./svc.sh install && ./svc.sh start`.

Until a runner with these labels is online, same-repo lint jobs wait in the queue. With one runner the three matrix legs run one after another.

## Host requirements

- **Python tool cache.** On macOS, `actions/setup-python` always installs into `/Users/runner/hostedtoolcache`, because its prebuilt interpreters are not relocatable. Create that directory once and give the runner user write access:

  ```sh
  sudo mkdir -p /Users/runner/hostedtoolcache
  sudo chown "$(whoami)":staff /Users/runner/hostedtoolcache   # run as the runner user
  ```

  After that, the job needs no `sudo`. setup-python downloads the newest arm64 build of each matrix version (3.9.13, 3.10.11, 3.11.9 today) the first time and reuses it afterwards.
- Network access to github.com (for the Python builds and actions) and to PyPI (for `ruff`, `black`, `isort`, `autoflake` and the editable install of this package).
- Nothing else: no Xcode, Homebrew or Docker. The Xcode Command Line Tools are enough for `git`.

The lint steps install packages into the tool-cache interpreters, and those packages persist between runs. To start clean, delete `/Users/runner/hostedtoolcache/Python`.

## Security

This repository is public, so the self-hosted `lint` job runs only for `push` to `main` or `develop` and for pull requests from branches in this repository (`head.repo.full_name == github.repository`). It is also skipped in forks of the repository. Fork pull requests use the GitHub-hosted `lint-hosted` job (`lint (GitHub-hosted, fork PRs)`). Checkouts use `persist-credentials: false`, and the workflow token is `contents: read`.

Where you can, run the runner under a dedicated macOS user rather than your daily account, and keep no production secrets on the host.

## Jobs that stay hosted

- `lint-hosted` stays on `ubuntu-latest` by design: it runs untrusted fork code, which must never reach the self-hosted machine. It won't start while the billing lock lasts.

No other workflows exist in this repository.
