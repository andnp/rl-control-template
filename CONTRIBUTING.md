# Contributing Guide

## How to contribute
1. Fork this repository
2. Clone your fork `git clone https://github.com/<your user name>/rl-control-template.git`
3. Make a virtual environment named `.venv` and install the dev environment.
    ```
    python -m venv .venv
    source .venv/bin/activate
    pip install .[dev]
    ```
4. Run `sh dev-setup.sh` or manually setup the precommit hooks
5. Make small, targeted changes to the code in your clone. Make commits using the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style.
6. Open a PR against the `main` branch of the parent repo: `andnp/rl-control-template`


### Conventional Commits
This repo follows the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style for _every_ commit in the history.
This commit style allows the automated build pipeline to increment the version number of the library whenever changes are pushed based on whether a commit (a) fixes a bug, (b) adds a feature, or (c) makes a non-backwards-compatible change.

The basic idea behind conventional commits is to make commit messages using the following template
```
<commit type>: <short lower-case memo>

A longer description of the change, why the change is being made, links to github issues or PRs where appropriate using #12 where "12" is the issue/PR number on github, etc.
This longer description is usually no more than a couple of sentences, though in rare instances can extend to a couple of paragraphs depending on the complexity (not size) of the change.

<optional>
BREAKING CHANGE: a description of how this commit breaks the backwards compatibility of the library.
A few examples would be
1. Renaming a method
2. Changing a function's signature or contract
3. Reorganizing code to different import paths
```

There are several valid `<commit type>`s.
The most common are:
1. `fix` - fixing a bug, making small tweaks to the internals of the library, etc. Bumps the "patch" version (i.e. `1.3.1` -> `1.3.2`)
2. `feat` - adding new functionality to the library. Bumps the "minor" version (i.e. `1.3.1` -> `1.4.0`)
3. `ci` - changing the CI build and metadata. Does not change library version, as this does not impact end users
4. `style` and `refactor` - not fixing a bug or adding a feature, just changing _how_ the code is written. Library should function identically after applying this commit
5. `test` - adding, modifying, removing test code.
6. `docs` - any changes to documentation
7. `perf` - changes to code that do not observably change functionality, but improve performance

If any of the above include a sentence starting with `BREAKING CHANGE:` in the commit body, then a "major" version bump will occur (i.e. `1.3.1` -> `2.0.0`).
Naturally, including a `BREAKING CHANGE:` alongside some of these commit types will raise some red flags (a `docs` change better not also be a `BREAKING CHANGE:`, even if _technically_ the system allows it).
Best practice is to discuss potential `BREAKING CHANGE:` code with the maintainers before putting up the PR, typically in a github issue.

## Contact
Please feel free to reach out on Slack to discuss bugs, feature requests, ideas, etc.