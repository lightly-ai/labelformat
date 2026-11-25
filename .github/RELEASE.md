# Release

How to publish a new version of `labelformat` to [PyPI](https://pypi.org/project/labelformat/).

The [Publish to PyPI](https://github.com/lightly-ai/labelformat/actions/workflows/publish.yml)
workflow uploads the package with
[trusted publishing](https://docs.pypi.org/trusted-publishers/). Nobody needs a PyPI token or
password.

## Release process

### 1. Bump the version

Open a `Bump version to X.Y.Z` pull request:

- Set `version` in `pyproject.toml`.
- Run `uv sync` to update `uv.lock`. Use the uv version that CI uses, see
  [run-tests.yml](workflows/run-tests.yml).

Merge it.

### 2. Create the GitHub release

Create a [release](https://github.com/lightly-ai/labelformat/releases/new) from `main`:

1. Tag: `vX.Y.Z` > **Create new tag**. The tag must match the version in `pyproject.toml`.
2. Title: `vX.Y.Z`.
3. Click **Generate release notes**.
4. Select **Set as the latest release**.
5. Click **Publish release**.

### 3. Approve the deployment

Publishing the release starts
[Publish to PyPI](https://github.com/lightly-ai/labelformat/actions/workflows/publish.yml). It
builds the package from the tag and waits for approval of the `pypi` environment, if the
environment has required reviewers. Approve it.

### 4. Check the release

Make sure that the new version is on [PyPI](https://pypi.org/project/labelformat/).

## If the workflow fails

The workflow stops before the upload if the tag does not match the version in `pyproject.toml`.
Delete the release and the tag, then do step 2 again with the correct tag.

For other failures, fix the cause and run
[Publish to PyPI](https://github.com/lightly-ai/labelformat/actions/workflows/publish.yml) manually
with the tag and `target` set to `pypi`. Files that are already on PyPI are skipped.

## Rehearse on TestPyPI

Run [Publish to PyPI](https://github.com/lightly-ai/labelformat/actions/workflows/publish.yml)
manually with an existing tag and `target` set to `testpypi`. This does not need a GitHub release.
If the version is already on TestPyPI, the upload is skipped.
