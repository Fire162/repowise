"""Pick the tests a change needs to run, or say why it needs them all.

``repowise impacted-tests`` names the tests that cover or reach each changed
file. A CI job can only trust that as a subset when every part of the answer
is positively known, so :func:`select_tests` fails closed: anything it cannot
vouch for answers "run everything", and the answer always says why. A full run
is chosen when any of these hold:

1. there is no change to select from;
2. a changed or deleted file can change any test: a dependency lock or
   manifest no ecosystem scope covers, build configuration (a Makefile, a
   Dockerfile), a shared test helper, CI config, Repowise's own config, or a
   path listed in ``tests.full_run_on``;
3. a changed or deleted file sits in a test tree but is not code (data, a
   snapshot, a golden file), or is a helper module no test imports;
4. a changed file is neither code nor documentation, and no code names it and
   no package holds it; or a file's scope (below) finds no test, or a file
   naming it has no known test;
5. a changed code file has no known test, only a filename guess names one, or
   a route to it passes through a test helper no test imports, or any Python
   test helper while a conftest or pytest config loads plugins by name (its
   users are unknown);
6. the index is missing, was built before other files changed, disagrees with
   itself about its commit, or its graph could not be read;
7. the per-test map hit its stored row cap;
8. a deleted code file has no known test, or a test the index names is
   missing from the checkout.

Otherwise the subset is the covering tests, the tests the graph shows reaching
the changed files (a changed test, the call graph, the import graph) and
``tests.always_run``, plus every test the graph cannot see into (not indexed,
or with no resolved edge). A test package's ``__init__.py`` or a ``conftest.py``
(changed, deleted, or on a route to a changed file) stands for every test under
its directory. A helper module tests import stands for the tests that import
it, directly or through other helpers (basis ``helper-importers``), which are the files that run it: Python runs the package file
for each module in it, and pytest loads a conftest for each test at or below
it. Only documentation (``docs/`` and the root README,
CHANGELOG, LICENSE and the like, never code) that no code names is skipped
without a test.

A file that would run everything but whose reach is known runs a scope
(:func:`trigger_scopes`) instead: a manifest, lockfile or runner config runs
its ecosystem's tests (a JS package or Go module only those under its
directory, plus the tests reaching its code); a production ``__init__.py``
runs the tests reaching any module under its directory, since importing one
runs it; a non-code file runs the tests reaching the code that names it, and
else the tests of the package holding it (under its directory or reaching its
code). Ceilings: a test of another ecosystem that reads a manifest as data, and
code that globs a directory instead of naming a file, are not seen.

:func:`runner_args` renders a selection as arguments for one test runner; a
full run renders as the :data:`RUN_ALL` sentinel. pytest and go reject it as a
path; jest and vitest treat arguments as patterns and would match nothing, so
a pipeline must branch on the run-all flag rather than rely on the sentinel.
"""

from __future__ import annotations

import configparser
import re
import shlex
import tomllib
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Any

import pathspec

from ..support_paths import DOC_EXTENSIONS
from ..test_paths import is_test_path, is_test_related_path, is_test_support_path

#: Printed alone instead of arguments when every test must run.
RUN_ALL = ":all"

RUNNERS = ("auto", "pytest", "go", "jest", "files")

# (why it forces a full run, gitwildmatch patterns). The config extends these,
# never replaces them. Ceiling: no opt-out until someone needs one.
_FULL_RUN_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "dependencies can change any test",
        (
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "poetry.lock",
            "uv.lock",
            "Pipfile.lock",
            "Pipfile",
            "requirements*.txt",
            "constraints*.txt",
            "go.mod",
            "go.sum",
            "Cargo.lock",
            "Cargo.toml",
            "Gemfile.lock",
            "Gemfile",
            "composer.lock",
            "composer.json",
        ),
    ),
    (
        "build or test configuration can change any test",
        (
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "package.json",
            "tsconfig*.json",
            "jest.config.*",
            "vitest.config.*",
            "vite.config.*",
            "pytest.ini",
            "tox.ini",
            "noxfile.py",
            "Makefile",
            "CMakeLists.txt",
            "build.gradle*",
            "settings.gradle*",
            "pom.xml",
            "Dockerfile*",
            "docker-compose*.y*ml",
        ),
    ),
    ("shared test data can change any test", ("**/testdata/**", "**/fixtures/**")),
    (
        "CI configuration changed",
        (
            ".github/workflows/**",
            ".gitlab-ci.yml",
            "**/.gitlab-ci.yml",
            ".gitlab/**",
            ".circleci/**",
            "Jenkinsfile*",
            "azure-pipelines*.y*ml",
            ".azure-pipelines/**",
            ".buildkite/**",
            "bitbucket-pipelines.yml",
            ".travis.yml",
            ".drone.y*ml",
            ".woodpecker/**",
            ".woodpecker.y*ml",
            "appveyor.y*ml",
            ".teamcity/**",
        ),
    ),
    ("Repowise's configuration changed", (".repowise/config.yaml",)),
)

_DEFAULT_SPECS = tuple(
    (why, pathspec.PathSpec.from_lines("gitwildmatch", patterns))
    for why, patterns in _FULL_RUN_GROUPS
)

# Documentation only: ``docs/`` and root-level project meta. A doc or image
# anywhere else may be package data a test reads, so it is not skipped.
_DOCS_SPEC = pathspec.PathSpec.from_lines("gitwildmatch", ["/docs/**"])
_ROOT_META = (
    "readme",
    "changelog",
    "changes",
    "history",
    "license",
    "licence",
    "copying",
    "notice",
    "authors",
    "contributors",
    "contributing",
    "code_of_conduct",
    "security",
)
# A root file only reads as project meta with no extension or a prose one:
# ``history.json`` or ``changes.yaml`` is data a test may load.
_DOC_SUFFIXES = DOC_EXTENSIONS | {"", ".markdown"}

# Extensions a test runner collects tests from; anything else in a test tree
# (data, snapshots, golden files) is read by tests, not run.
_TEST_CODE_SUFFIXES = frozenset(
    {".py", ".go", ".java", ".kt", ".kts", ".scala", ".groovy", ".rb", ".cs", ".fs"}
    | {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}
    | {".rs", ".php", ".swift", ".ex", ".exs", ".dart", ".c", ".cc", ".cpp"}
)

_PYTHON = (".py",)
_JS = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts")
# coverage.py names each phase of a test as its own context.
_PHASES = ("|run", "|setup", "|teardown")


@dataclass(frozen=True)
class TestSelectionConfig:
    """``tests.full_run_on`` and ``tests.always_run`` from ``.repowise/config.yaml``."""

    __test__ = False  # not a pytest class, whatever its name says

    full_run_on: tuple[str, ...] = ()
    always_run: tuple[str, ...] = ()

    @classmethod
    def from_repo_config(cls, raw: Mapping[str, Any]) -> TestSelectionConfig:
        """Parse the ``tests`` block; raises ``ValueError`` naming what is wrong.

        A misspelt key is refused rather than ignored: a full-run trigger that
        silently stops applying would skip tests the change needs.
        """
        block = raw.get("tests")
        if block is None:
            return cls()
        if not isinstance(block, Mapping):
            raise ValueError(f"tests must be a mapping, got {type(block).__name__}.")
        unknown = sorted(set(block) - {"full_run_on", "always_run"})
        if unknown:
            raise ValueError(
                f"tests.{unknown[0]} is not a setting; use tests.full_run_on or "
                "tests.always_run."
            )
        return cls(
            full_run_on=_string_list(block, "full_run_on"),
            always_run=_string_list(block, "always_run"),
        )


def _string_list(block: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = block.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"tests.{key} must be a list of non-empty strings, got {value!r}.")
    return tuple(v.strip() for v in value)


@dataclass(frozen=True)
class Selection:
    """What to run for a change, and why.

    ``tests`` are what a runner that takes test ids (pytest) is given: covering
    test ids where coverage named them, else whole test files. ``test_files``
    are the files behind them, for runners that take files. ``packages`` are
    the directories of changed Go files that hold tests of their own. ``basis``
    names, per changed or deleted file, what decided it: ``full-run``,
    ``no-tests-needed``, ``deleted-test``, ``test-tree``, ``test-package``,
    ``conftest``, ``helper-importers``, ``coverage``,
    ``changed-test``, ``call-graph``, ``import-graph``, ``filename-pattern``,
    ``unknown``, ``none`` when nothing was asked (no index), or a scope's
    ``ecosystem``, ``package-importers``, ``named-by`` or ``owner-package``.
    """

    run_all: bool
    reasons: tuple[str, ...]
    tests: tuple[str, ...] = ()
    test_files: tuple[str, ...] = ()
    packages: tuple[str, ...] = ()
    always_run: tuple[str, ...] = ()
    skipped_files: tuple[str, ...] = ()
    basis: Mapping[str, str] = field(default_factory=dict)
    pytest_testpaths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_all": self.run_all,
            "reasons": list(self.reasons),
            "tests": list(self.tests),
            "test_files": list(self.test_files),
            "packages": list(self.packages),
            "always_run": list(self.always_run),
            "skipped_files": list(self.skipped_files),
            "basis": dict(self.basis),
        }


def is_documentation(path: str) -> bool:
    """``docs/**`` or a root-level README, CHANGELOG, LICENSE and the like.

    Never code: a ``docs/conf.py``, an example a test imports, or a root
    ``history.py`` is a module like any other, so the graph decides it.
    """
    if is_code_file(path):
        return False
    if _DOCS_SPEC.match_file(path):
        return True
    p = PurePosixPath(path)
    return (
        len(p.parts) == 1
        and p.suffix.lower() in _DOC_SUFFIXES
        and p.name.lower().startswith(_ROOT_META)
    )


# pytest config a doctest glob or a plugin can be declared in.
_PYTEST_CONFIGS = frozenset({"pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini"})
# ``pytest_plugins = [...]`` or ``-p name`` (not ``-p no:name``, which disables one).
_PLUGIN_DECLARATION = re.compile(r"\bpytest_plugins\b|(?:^|[\s\"'=])-p\s*(?!no:)[A-Za-z_]")


def is_scan_source(path: str) -> bool:
    """Code or pytest config: the files :func:`doc_readers` and :func:`plugin_loader` read."""
    name = PurePosixPath(path).name
    return is_code_file(path) or name in _PYTEST_CONFIGS


def doc_readers(docs: Collection[str], sources: Iterable[tuple[str, str]]) -> dict[str, str]:
    """``{doc: a file naming it}`` for each of *docs* some code or config names."""
    return {doc: names[0] for doc, names in file_namers(docs, sources).items()}


def file_namers(files: Collection[str], sources: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """``{file: the code naming it}`` for each of *files* some code or config names.

    Code that reads a file names it (``ROOT / "README.md"``), and a
    ``--doctest-glob`` turns every doc into a test, so the config declaring
    one names each doc. Ceiling: code that globs a directory names no file.
    """
    names = {f: PurePosixPath(f).name for f in files}
    docs = [f for f in files if PurePosixPath(f).suffix.lower() in _DOC_SUFFIXES]
    out: dict[str, list[str]] = {}
    for path, text in sources:
        # Config only for the glob, code only for names: pyproject's
        # ``readme = "README.md"`` is packaging, not a test.
        if not is_code_file(path):
            if "doctest-glob" in text:
                for doc in docs:
                    out.setdefault(doc, []).insert(0, path)
            continue
        for f, name in names.items():
            if name in text and f != path:
                out.setdefault(f, []).append(path)
    return out


def plugin_loader(sources: Iterable[tuple[str, str]]) -> str | None:
    """The first conftest or pytest config that loads a plugin module by name.

    Such a plugin's fixtures reach tests that never import it, so while one is
    declared, a test helper's importers are not all of its users.
    """
    for path, text in sources:
        name = PurePosixPath(path).name
        pytest_file = name == "conftest.py" or name in _PYTEST_CONFIGS
        if pytest_file and _PLUGIN_DECLARATION.search(text):
            return path
    return None


_PACKAGE_INIT_REASON = "every import of the package runs it, and those are not all tracked"


def scope_kind(path: str) -> str | None:
    """``test-package`` or ``conftest`` for a file every test under its directory runs.

    A test tree's ``__init__.py`` runs for each module in the package, and
    pytest loads a ``conftest.py`` for each test at or below its directory, so
    those tests are exactly its users.
    """
    name = PurePosixPath(path).name
    if name == "conftest.py":
        return "conftest"
    if name == "__init__.py" and is_test_related_path(path):
        return "test-package"
    return None


def is_code_file(path: str) -> bool:
    """A file with an extension a test runner loads, test or not."""
    return PurePosixPath(path).suffix.lower() in _TEST_CODE_SUFFIXES


def is_test_helper(path: str) -> bool:
    """Code in a test tree that is neither a test nor a directory-scoped file."""
    return (
        is_code_file(path)
        and is_test_related_path(path)
        and not is_runnable_test(path)
        and scope_kind(path) is None
    )


def is_runnable_test(path: str) -> bool:
    """A test-shaped file name with an extension a runner collects tests from."""
    p = PurePosixPath(path)
    return p.suffix.lower() in _TEST_CODE_SUFFIXES and is_test_path(p.name)


def full_run_reason(path: str, extra: Iterable[str] = ()) -> str | None:
    """Why a change to *path* can change any test, or ``None``."""
    specs = list(_DEFAULT_SPECS)
    if extra := tuple(extra):
        spec = pathspec.PathSpec.from_lines("gitwildmatch", extra)
        specs.append(("it matches tests.full_run_on", spec))
    for why, spec in specs:
        if spec.match_file(path):
            return why
    if PurePosixPath(path).name == "__init__.py" and not is_test_related_path(path):
        return _PACKAGE_INIT_REASON
    if is_test_support_path(path) and scope_kind(path) is None:
        return "a shared test helper can change any test"
    return None


# Manifests and lockfiles whose reach is one ecosystem's tests:
# (ecosystem, test suffixes, scoped to the file's directory, name globs). A
# directory-scoped one (a JS package, a Go module) runs that ecosystem's tests
# under its directory and every test reaching code there. The rest run the
# whole ecosystem: one environment serves every package (a venv, a bundle), or
# the file can apply far from its directory (tsconfig ``extends``, a runner
# config whose globs reach other trees). Cargo stays a full run: Rust unit
# tests live inside source files, which no selection lists.
_JVM = (".java", ".kt", ".kts", ".scala", ".groovy")
_ECOSYSTEMS: tuple[tuple[str, tuple[str, ...], bool, tuple[str, ...]], ...] = (
    ("JavaScript", _JS, True, ("package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml")),
    ("JavaScript", _JS, False, ("tsconfig*.json", "jest.config.*", "vitest.config.*", "vite.config.*")),
    (
        "Python",
        _PYTHON,
        False,
        (
            *("pyproject.toml", "setup.py", "setup.cfg", "poetry.lock", "uv.lock", "Pipfile"),
            *("Pipfile.lock", "requirements*.txt", "constraints*.txt", "pytest.ini", "tox.ini"),
            "noxfile.py",
        ),
    ),
    ("Go", (".go",), True, ("go.mod", "go.sum")),
    ("Ruby", (".rb",), False, ("Gemfile", "Gemfile.lock")),
    ("PHP", (".php",), False, ("composer.json", "composer.lock")),
    ("JVM", _JVM, False, ("build.gradle*", "settings.gradle*", "pom.xml")),
)
_SHARED_TEST_DATA = dict(_DEFAULT_SPECS)["shared test data can change any test"]


@dataclass(frozen=True)
class Scope:
    """The tests a changed non-code file runs in place of every test.

    Every known test under *root* whose suffix is in *suffixes* (any test when
    empty; none when *root* is ``None``), plus the tests reaching each file in
    *reach* and in *namers*. *reach* is production code under *root*, which
    may have no test of its own; a namer is code that names the file, so a
    namer no test reaches leaves the file's tests unknown.
    """

    why: str
    basis: str
    root: str | None = None
    suffixes: tuple[str, ...] = ()
    reach: tuple[str, ...] = ()
    namers: tuple[str, ...] = ()

    @property
    def routes(self) -> tuple[str, ...]:
        """The files whose reaching tests the selection needs from the graph."""
        return (*self.reach, *self.namers)


def needs_namers(path: str, config: TestSelectionConfig) -> bool:
    """Whether :func:`trigger_scopes` asks who names *path*: non-code outside a test tree."""
    return (
        not is_code_file(path)
        and not is_test_related_path(path)
        and full_run_reason(path, config.full_run_on) is None
    )


def trigger_scopes(
    paths: Iterable[str],
    tracked: Collection[str],
    namers: Mapping[str, Collection[str]],
    config: TestSelectionConfig,
) -> dict[str, Scope]:
    """The :class:`Scope` each of *paths* runs instead of every test, where it has one.

    *tracked* are the checkout's files; *namers* maps a file to the files
    naming it (:func:`file_namers`). A path with no scope keeps its rule: a
    full-run trigger runs everything, an unnamed doc runs nothing.
    """
    from ..ingestion.package_roots import package_roots_from_paths

    roots = package_roots_from_paths(set(tracked))
    out: dict[str, Scope] = {}
    for path in paths:
        if scope := _trigger_scope(path, tracked, roots, namers, config):
            out[path] = scope
    return out


def _trigger_scope(
    path: str,
    tracked: Collection[str],
    roots: set[str],
    namers: Mapping[str, Collection[str]],
    config: TestSelectionConfig,
) -> Scope | None:
    if is_test_related_path(path) or _SHARED_TEST_DATA.match_file(path):
        return None  # read by tests anywhere: the test-tree rule decides
    if config.full_run_on and pathspec.PathSpec.from_lines(
        "gitwildmatch", config.full_run_on
    ).match_file(path):
        return None  # the config asked for a full run
    if eco := _ecosystem(path):
        return _ecosystem_scope(path, eco, tracked)
    why = full_run_reason(path)
    if why == _PACKAGE_INIT_REASON:
        return _package_scope(path, tracked)
    if why is not None or is_code_file(path):
        return None
    return _named_scope(path, tracked, roots, tuple(namers.get(path, ())))


def _ecosystem(path: str) -> tuple[str, tuple[str, ...], bool] | None:
    name = PurePosixPath(PurePosixPath(path).name)
    for lang, suffixes, scoped, globs in _ECOSYSTEMS:
        if any(name.match(g) for g in globs):
            return lang, suffixes, scoped
    return None


def _ecosystem_scope(
    path: str, eco: tuple[str, tuple[str, ...], bool], tracked: Collection[str]
) -> Scope:
    lang, suffixes, scoped = eco
    root = str(PurePosixPath(path).parent) if scoped else "."
    if root == ".":
        return Scope(f"it can change any {lang} test", "ecosystem", ".", suffixes)
    return Scope(
        f"it can change the {lang} tests under {root}/ and those reaching its code",
        "ecosystem",
        root,
        suffixes,
        _code_under(root, tracked, suffixes),
    )


def _package_scope(path: str, tracked: Collection[str]) -> Scope | None:
    """A production ``__init__.py`` runs for every import of a module under its directory."""
    root = str(PurePosixPath(path).parent)
    if root == ".":
        return None
    return Scope(
        f"every import of a module under {root}/ runs it",
        "package-importers",
        root,
        _PYTHON,
        _code_under(root, tracked, _PYTHON),
    )


def _named_scope(
    path: str, tracked: Collection[str], roots: set[str], namers: tuple[str, ...]
) -> Scope | None:
    """Code naming *path* reads it; else the package holding it is the closest owner.

    A namer inside the owning package is trusted alone. A name written only
    outside it may be another file's (``layout.css`` is a common name), so the
    owner's tests run beside the namers'.
    """
    if namers and not all(is_code_file(n) for n in namers):
        return None  # a doctest glob: the doc is a test
    if not namers and is_documentation(path):
        return None  # a doc no code names needs no tests
    owner = _owner(path, roots)
    if namers and (owner is None or any(_is_under(n, owner) for n in namers)):
        return Scope(f"it is named by {_some(namers)}", "named-by", namers=namers)
    if owner is None:
        return None
    why = f"it belongs to the package at {owner}/"
    if namers:
        why += f" and is named by {_some(namers)}"
    return Scope(why, "owner-package", owner, (), _code_under(owner, tracked, ()), namers)


def _owner(path: str, roots: set[str]) -> str | None:
    """The deepest package root holding *path*, never the repository root."""
    from ..ingestion.package_roots import module_for

    owner = module_for(path, roots)
    return owner if owner in roots else None


def _some(paths: tuple[str, ...]) -> str:
    return paths[0] if len(paths) == 1 else f"{paths[0]} and {len(paths) - 1} more"


def _is_under(path: str, root: str) -> bool:
    return root == "." or path.startswith(f"{root}/")


def _code_under(root: str, tracked: Collection[str], suffixes: tuple[str, ...]) -> tuple[str, ...]:
    """Production code under *root* (in *suffixes*, any code when empty), sorted."""
    return tuple(
        sorted(
            p
            for p in tracked
            if _is_under(p, root)
            and is_code_file(p)
            and (not suffixes or p.endswith(suffixes))
            and not is_test_related_path(p)
        )
    )


@dataclass(frozen=True)
class SelectionInput:
    """Everything :func:`select_tests` decides from.

    *changed* are the paths present after the change, *deleted* those it
    removed, *label* names the change; *tiers* is ``impacted-tests``' result
    (``covered``, ``inferred``, ``unknown``, ``helper_importers``).
    *map_current* is false when the per-test map was measured at neither end of
    the change, so its tests were matched by file; *map_truncated* when it holds
    as many rows as the store keeps. *index_gap* lists the files that changed
    between the indexed commit and the change, ``None`` when that cannot be
    told; *index_problem* says why the index's own commit cannot be trusted.
    *graph_error* names a graph read that failed. *missing* are test files the
    tiers name that the checkout does not have; *go_test_dirs* the directories
    holding ``_test.go`` files; *known_tests* every runnable test in the
    checkout, which a test package's ``__init__.py`` or a ``conftest.py``
    expands to. *doc_readers* maps a changed doc to a file naming it
    (:func:`doc_readers`); *plugin_loader* is a file loading pytest plugins by
    name (:func:`plugin_loader`); *unplaced_tests* are tests the graph cannot
    see into (not indexed, or with no resolved edge), which every subset runs.
    """

    changed: Collection[str]
    deleted: Collection[str]
    tiers: Mapping[str, Any]
    config: TestSelectionConfig = field(default_factory=TestSelectionConfig)
    index_available: bool = True
    label: str = "the base"
    map_current: bool = True
    map_truncated: bool = False
    index_gap: Collection[str] | None = ()
    index_problem: str | None = None
    graph_error: str | None = None
    missing: Collection[str] = ()
    go_test_dirs: Collection[str] = ()
    known_tests: Collection[str] = ()
    doc_readers: Mapping[str, str] = field(default_factory=dict)
    plugin_loader: str | None = None
    unplaced_tests: Collection[str] = ()
    pytest_testpaths: tuple[str, ...] = ()
    # :func:`trigger_scopes`' answer; *tiers* also holds rows for each route.
    scopes: Mapping[str, Scope] = field(default_factory=dict)


@dataclass
class _Triage:
    """The changed paths sorted by what decides them, before any test is asked."""

    run_all: list[str] = field(default_factory=list)
    basis: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    code: list[str] = field(default_factory=list)


def select_tests(inp: SelectionInput) -> Selection:
    """Decide what a change must run (see the module docstring for the rules)."""
    deleted = set(inp.deleted)
    paths = sorted(set(inp.changed) | deleted)
    scoped = [p for p in paths if p in inp.scopes]
    triage = _triage([p for p in paths if p not in inp.scopes], inp.config, inp.doc_readers)
    run_all = [*_empty_change_reasons(paths, inp.label), *triage.run_all]
    walks = bool(triage.code) or any(inp.scopes[p].routes for p in scoped)
    run_all += _index_reasons(inp, has_code=walks)

    evidence = _Evidence.of(inp, deleted)
    per_file, file_reasons = _tests_per_file(triage.code, evidence)
    in_scope, scope_reasons, scope_notes = _scoped_tests(scoped, inp.scopes, evidence)
    if inp.index_available:
        run_all += file_reasons + scope_reasons
    basis = {**triage.basis, **{path: per_file[path][1] for path in triage.code}}
    basis.update({path: inp.scopes[path].basis for path in scoped})

    # The graph cannot say what an unplaced test reaches, so it always runs.
    unplaced = [(t, t) for t in inp.unplaced_tests] if triage.code or scoped else []
    tests, test_files = _runnable(
        [*(t for path in triage.code for t in per_file[path][0]), *in_scope, *unplaced]
    )
    return Selection(
        run_all=bool(run_all),
        reasons=tuple(run_all + scope_notes + _notes(inp, triage.skipped, bool(unplaced))),
        tests=tests,
        test_files=test_files,
        packages=_go_packages(triage.code, deleted, inp.go_test_dirs),
        always_run=inp.config.always_run,
        skipped_files=tuple(triage.skipped),
        basis=basis,
        pytest_testpaths=inp.pytest_testpaths,
    )


def _triage(
    paths: list[str], config: TestSelectionConfig, doc_readers: Mapping[str, str]
) -> _Triage:
    """Full-run triggers, documentation and test-tree data first; code for the tiers."""
    out = _Triage()
    for path in paths:
        if why := full_run_reason(path, config.full_run_on):
            out.run_all.append(f"{path} changed: {why}.")
            out.basis[path] = "full-run"
        elif is_documentation(path) and (reader := doc_readers.get(path)):
            out.run_all.append(
                f"{path} is named by {reader}, so a test may read it; the tests "
                "that do are not tracked."
            )
            out.basis[path] = "full-run"
        elif is_documentation(path):
            out.skipped.append(path)
            out.basis[path] = "no-tests-needed"
        elif is_test_related_path(path) and not is_code_file(path):
            out.run_all.append(
                f"{path} is in a test tree but is not code (data, a snapshot or a "
                "golden file); the tests that read it are not tracked."
            )
            out.basis[path] = "test-tree"
        else:
            out.code.append(path)
    return out


def _scoped_tests(
    scoped: list[str], scopes: Mapping[str, Scope], ev: _Evidence
) -> tuple[list[_TestRef], list[str], list[str]]:
    """The tests each scoped file runs, the run-all reasons found, and a note per file."""
    tests: list[_TestRef] = []
    reasons: list[str] = []
    notes: list[str] = []
    for path in scoped:
        scope = scopes[path]
        mine = [(t, t) for t in _scope_tests(scope, ev)]
        for route in scope.routes:
            found, _, why = _file_tests(route, ev, route_only=True)
            mine += found
            reasons += [f"{path} changed; {r}" for r in why]
            if route in scope.namers and not (found or why):
                reasons.append(f"{path} is named by {route}, and no test is known to reach it.")
        if not mine and not reasons:
            reasons.append(f"{path} changed: {scope.why}, and no such test is known.")
        tests += mine
        files = {f for _, f in mine if f}
        notes.append(f"{path} changed: {scope.why}; {len(files)} test file(s) run for it.")
    if len(notes) > _MAX_SCOPE_NOTES:
        more = len(notes) - _MAX_SCOPE_NOTES
        notes = [*notes[:_MAX_SCOPE_NOTES], f"{more} more changed file(s) run a scoped set."]
    return tests, reasons, notes


_MAX_SCOPE_NOTES = 5


def _scope_tests(scope: Scope, ev: _Evidence) -> list[str]:
    """Known tests under *scope*'s root in its suffixes; a test this change deletes is gone."""
    if scope.root is None:
        return []
    return [
        t
        for t in _under({scope.root}, ev.known_tests)
        if (not scope.suffixes or t.endswith(scope.suffixes)) and t not in ev.deleted
    ]


def _empty_change_reasons(paths: list[str], label: str) -> list[str]:
    return [] if paths else [f"No change against {label}; nothing to select from."]


def _index_reasons(inp: SelectionInput, *, has_code: bool) -> list[str]:
    """Why the index cannot vouch for the tiers; nothing to say without code."""
    if not has_code:
        return []
    if not inp.index_available:
        return [
            "No index: nothing records which tests reach the changed code "
            "(run `repowise init`, or restore a cached .repowise directory)."
        ]
    out = [inp.index_problem] if inp.index_problem else _gap_reasons(inp.index_gap)
    if inp.graph_error:
        out.append(
            f"The graph could not be read ({inp.graph_error}), so the tests reaching "
            "the changed files are unknown."
        )
    if inp.map_truncated:
        out.append(
            "The per-test map is at its stored row cap, so some tests' coverage "
            "was dropped when it was ingested."
        )
    return out


def _notes(inp: SelectionInput, skipped: list[str], unplaced: bool) -> list[str]:
    """Reasons that explain the selection without forcing a full run."""
    out = []
    if unplaced:
        out.append(
            f"{len(inp.unplaced_tests)} test file(s) the graph cannot see into run with "
            f"every selection (e.g. {sorted(inp.unplaced_tests)[0]})."
        )
    if not inp.map_current and inp.tiers.get("covered"):
        out.append(
            "The per-test map was measured at another commit, so covering tests "
            "are matched by file, not by changed line."
        )
    if skipped:
        out.append(f"{len(skipped)} changed file(s) are documentation: no tests needed.")
    return out


def _go_packages(
    code: list[str], deleted: set[str], go_test_dirs: Collection[str]
) -> tuple[str, ...]:
    """Directories of changed Go files that hold same-package tests."""
    dirs = {
        str(PurePosixPath(p).parent)
        for p in code
        if p.endswith(".go") and not p.endswith("_test.go") and p not in deleted
    }
    return tuple(sorted(dirs & set(go_test_dirs)))


def _gap_reasons(index_gap: Collection[str] | None) -> list[str]:
    if index_gap is None:
        return [
            "Cannot tell what changed since the index was built (its commit is not "
            "recorded, or not in this clone); run `repowise update` before selecting."
        ]
    unseen = sorted(p for p in index_gap if not is_documentation(p))
    if not unseen:
        return []
    return [
        f"The index predates {len(unseen)} changed file(s) outside this change "
        f"(e.g. {unseen[0]}), so its graph cannot see them; run `repowise update` "
        "before selecting."
    ]


_TestRef = tuple[str, str | None]  # (test id or file, the file it lives in)

# Bases that need no test of their own to be safe.
_SELF_SUFFICIENT = ("deleted-test", "test-package", "conftest")


@dataclass(frozen=True)
class _Evidence:
    """The tiers indexed by changed file, and what the checkout lacks."""

    covered: Mapping[str, list[_TestRef]]
    inferred: Mapping[str, list[tuple[str, str]]]
    unknown: frozenset[str]
    importers: Mapping[str, Collection[str]]
    missing: frozenset[str]
    deleted: frozenset[str]
    known_tests: tuple[str, ...]
    plugin_loader: str | None

    @classmethod
    def of(cls, inp: SelectionInput, deleted: set[str]) -> _Evidence:
        covered: dict[str, list[_TestRef]] = {}
        for test_id, info in (inp.tiers.get("covered") or {}).items():
            for source in info.get("source_files", ()):
                covered.setdefault(source, []).append((test_id, info.get("test_file") or None))
        inferred: dict[str, list[tuple[str, str]]] = {}
        for row in inp.tiers.get("inferred") or ():
            inferred.setdefault(row["source_file"], []).append((row["test_file"], row["via"]))
        return cls(
            covered=covered,
            inferred=inferred,
            unknown=frozenset(inp.tiers.get("unknown") or ()),
            importers=inp.tiers.get("helper_importers") or {},
            missing=frozenset(inp.missing),
            deleted=frozenset(deleted),
            known_tests=tuple(inp.known_tests),
            plugin_loader=inp.plugin_loader,
        )

    def found(self, path: str) -> list[_TestRef]:
        """What the graph named for *path*, a name-shaped guess excluded."""
        return [(t, t) for t, via in self.inferred.get(path, ()) if via != "filename-pattern"]

    def guesses(self, path: str) -> list[str]:
        return [t for t, via in self.inferred.get(path, ()) if via == "filename-pattern"]

    def present(self, tests: list[_TestRef]) -> list[_TestRef]:
        """*tests* the checkout still has; a test this change deletes is gone."""
        return [(t, f) for t, f in tests if f not in self.missing and f not in self.deleted]


def _tests_per_file(
    code: list[str], ev: _Evidence
) -> tuple[dict[str, tuple[list[_TestRef], str]], list[str]]:
    """``{path: ([(test id or file, test file)], basis)}`` and the run-all reasons."""
    reasons = _stale_index_reasons(ev)
    out: dict[str, tuple[list[_TestRef], str]] = {}
    for path in code:
        tests, basis, file_reasons = _file_tests(path, ev)
        out[path] = (tests, basis)
        reasons += file_reasons
    return out, reasons


def _stale_index_reasons(ev: _Evidence) -> list[str]:
    stale = sorted(ev.missing - ev.deleted)
    if not stale:
        return []
    return [
        f"{stale[0]} is a test the index names but the checkout does not have "
        f"({len(stale)} such), so the index is out of date; run `repowise update`."
    ]


def _file_tests(
    path: str, ev: _Evidence, *, route_only: bool = False
) -> tuple[list[_TestRef], str, list[str]]:
    """One changed file's tests, the evidence behind them, and any run-all reasons.

    With *route_only* the file only links tests to the change (it changed before
    the base), so having no test of its own is not a reason.
    """
    found = ev.found(path)
    tests = ev.present([*ev.covered.get(path, ()), *found])
    basis = _basis(path, ev.covered, ev.inferred, ev.unknown)
    scopes, basis = _scope_files(path, found, basis)
    tests = _expand_scopes(tests, scopes, ev)
    helpers = sorted(
        {f for _, f in found if not is_runnable_test(f)} - ev.deleted - scopes - {path}
    )
    drop = {*helpers, path} if is_test_helper(path) else set(helpers)
    tests = [(t, f) for t, f in tests if f is None or f not in drop]
    if is_test_helper(path):
        return tests, "helper-importers", _own_helper_reasons(path, ev)

    reasons = _helper_route_reasons(path, helpers, ev) + _unfiled_reasons(path, tests)
    # A deleted test needs no run of its own; the tests importing it do.
    if path in ev.deleted and is_runnable_test(path):
        basis = "deleted-test"
    if not (route_only or tests or helpers or basis in _SELF_SUFFICIENT):
        reasons.append(_no_test_reason(path, basis, ev))
    return tests, basis, reasons


def _scope_files(path: str, found: list[_TestRef], basis: str) -> tuple[set[str], str]:
    """Test-package ``__init__.py`` / ``conftest.py`` files on *path*'s routes, or itself.

    Each runs for every test under its directory, so it stands for those tests
    rather than for itself.
    """
    scopes = {f for _, f in found if f and scope_kind(f)}
    if kind := scope_kind(path):
        scopes.add(path)
        basis = kind
    return scopes, basis


def _expand_scopes(tests: list[_TestRef], scopes: set[str], ev: _Evidence) -> list[_TestRef]:
    kept = [(t, f) for t, f in tests if f not in scopes]
    return kept + [(t, t) for t in _tests_under(scopes, ev.known_tests) if t not in ev.deleted]


def _own_helper_reasons(path: str, ev: _Evidence) -> list[str]:
    """A changed helper stands for its importers; one no test imports is unknown."""
    if plugin := _plugin_reason(path, ev):
        return [plugin]
    if ev.importers.get(path):
        return []
    return [
        f"{path} is a test helper no test imports; tests that use it without an "
        "import are not tracked."
    ]


def _plugin_reason(helper: str, ev: _Evidence) -> str | None:
    """A Python helper may be a plugin, whose fixtures reach tests that never import it."""
    if not (ev.plugin_loader and helper.endswith(_PYTHON)):
        return None
    return (
        f"{helper} is a test helper and {ev.plugin_loader} loads pytest plugins by "
        "name, so tests that use its fixtures without an import are not tracked."
    )


def _helper_route_reasons(path: str, helpers: list[str], ev: _Evidence) -> list[str]:
    """A helper on a route stands for its importers, which the walk adds beside it.

    One no test imports is used some other way (a fixture, a plugin), so its
    users are unknown.
    """
    plugins = [r for h in helpers if (r := _plugin_reason(h, ev))]
    if plugins:
        return [f"{path} is reached through a test helper: {plugins[0]}"]
    unimported = [h for h in helpers if not ev.importers.get(h)]
    if not unimported:
        return []
    return [
        f"{path} is reached through the test helper {unimported[0]}, and no test "
        "imports that helper; tests that use it without an import are not tracked."
    ]


def _unfiled_reasons(path: str, tests: list[_TestRef]) -> list[str]:
    return [
        f"Coverage names {test_id} for {path}, but not its file."
        for test_id, test_file in tests
        if test_file is None
    ]


def _no_test_reason(path: str, basis: str, ev: _Evidence) -> str:
    """Why a changed file with no test left forces a full run."""
    if path in ev.deleted:
        return f"{path} was deleted and no test is known to have used it."
    if guesses := ev.guesses(path):
        return f"{path}: only a filename guess names a test ({guesses[0]})."
    if basis in ("unknown", "none"):
        return f"{path}: no coverage, no test reaching it in the graph, no paired test."
    return f"{path}: its only known tests are not in the checkout."


def _tests_under(inits: Collection[str], known_tests: Collection[str]) -> list[str]:
    """Every known test below the directory of each scope file."""
    return _under({str(PurePosixPath(i).parent) for i in inits}, known_tests)


def _under(dirs: Collection[str], files: Collection[str]) -> list[str]:
    return [f for f in files if any(_is_under(f, d) for d in dirs)]


def expand_test_scopes(tests: Iterable[str], test_files: Collection[str]) -> list[str]:
    """*tests* as a run list: each scope file replaced by the tests under its directory.

    A graph walk reports a ``conftest.py`` or test-package ``__init__.py`` it
    stopped at, but pytest collects nothing from either; the tests they run for
    are the runnable ones among *test_files* below their directory, the same
    expansion the selection makes. Order is kept and duplicates dropped; a scope
    with no runnable test under it drops out.
    """
    tests = list(tests)
    if not any(scope_kind(t) for t in tests):
        return list(dict.fromkeys(tests))
    runnable = sorted(t for t in test_files if is_runnable_test(t))
    out: dict[str, None] = {}
    for test in tests:
        for expanded in _tests_under([test], runnable) if scope_kind(test) else (test,):
            out.setdefault(expanded)
    return list(out)


def _basis(
    path: str,
    covered: Mapping[str, Any],
    inferred: Mapping[str, list[tuple[str, str]]],
    unknown: set[str],
) -> str:
    """The strongest evidence behind *path*'s tests."""
    if path in covered:
        return "coverage"
    vias = {via for _, via in inferred.get(path, ())}
    for via in ("changed-test", "call-graph", "import-graph", "filename-pattern"):
        if via in vias:
            return via
    return "unknown" if path in unknown else "none"


def _runnable(selected: list[_TestRef]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(tests, test_files)``: node ids and whole files, deduplicated, stable.

    A node id whose whole file is selected anyway is dropped.
    """
    files: dict[str, None] = {}
    whole: set[str] = set()
    ids: dict[str, None] = {}
    for test, test_file in selected:
        if test_file is None:
            continue
        files[test_file] = None
        node = _node_id(test, test_file)
        if node == test_file:
            whole.add(test_file)
        ids[node] = None
    tests = [t for t in ids if t in whole or t.split("::", 1)[0] not in whole]
    return tuple(tests), tuple(files)


def _node_id(test_id: str, test_file: str) -> str:
    """A runnable id: ``file::name`` rebased on the resolved file, else the file."""
    for phase in _PHASES:
        if test_id.endswith(phase):
            test_id = test_id[: -len(phase)]
            break
    if "::" not in test_id:
        return test_file
    return f"{test_file}::{test_id.split('::', 1)[1]}"


def _all_python(files: tuple[str, ...], packages: tuple[str, ...]) -> bool:
    return bool(files) and all(f.endswith(_PYTHON) for f in files)


def _all_go(files: tuple[str, ...], packages: tuple[str, ...]) -> bool:
    # A changed Go file's own package counts even with no selected test file.
    return bool(files or packages) and all(f.endswith("_test.go") for f in files)


def _all_js(files: tuple[str, ...], packages: tuple[str, ...]) -> bool:
    return bool(files) and all(f.endswith(_JS) for f in files)


# ``auto`` picks the first runner every selected test file belongs to.
_AUTO_RUNNERS = (("pytest", _all_python), ("go", _all_go), ("jest", _all_js))


def resolve_runner(selection: Selection, runner: str) -> str:
    """*runner*, or for ``auto`` the one every selected test file belongs to."""
    if runner != "auto":
        return runner
    files, packages = selection.test_files, selection.packages
    return next((name for name, fits in _AUTO_RUNNERS if fits(files, packages)), "files")


def _pytest_args(selection: Selection) -> list[str]:
    roots = selection.pytest_testpaths
    return [
        t
        for t in selection.tests
        if (f := t.split("::", 1)[0]).endswith(_PYTHON) and _pytest_collects(f, roots)
    ]


def _pytest_collects(path: str, testpaths: tuple[str, ...]) -> bool:
    """Whether a bare ``pytest`` would run *path*, so passing it adds nothing new.

    A test-shaped module outside ``testpaths`` and outside any test tree is
    production code (``core/test_paths.py``) that pytest would fail to collect;
    a full run never runs it, so neither does a subset.
    """
    p = PurePosixPath(path)
    # A neutral file name asks about the directory alone: in a test tree, keep it.
    if not testpaths or is_test_related_path(str(p.parent / "__init__.py")):
        return True
    candidates = [path, *(str(parent) for parent in p.parents)]
    return any(fnmatchcase(c, root) for root in testpaths for c in candidates)


# Where pytest reads ``testpaths`` from, in its own order: the first file that
# holds the section wins, and a ``pytest.ini`` wins even without one.
_PYTEST_SECTIONS = (
    ("pytest.ini", "pytest"),
    ("pyproject.toml", None),
    ("tox.ini", "pytest"),
    ("setup.cfg", "tool:pytest"),
)


def pytest_testpaths(read: Callable[[str], str | None]) -> tuple[str, ...]:
    """``testpaths`` from the root pytest config (*read* returns a file's text or None).

    ``()`` when unset or unreadable, which filters nothing.
    """
    for name, section in _PYTEST_SECTIONS:
        if (text := read(name)) is None:
            continue
        try:
            value = _testpaths_value(text, section, always=name == "pytest.ini")
        except (tomllib.TOMLDecodeError, configparser.Error):
            return ()
        if value is not None:
            paths = value.split() if isinstance(value, str) else value
            return tuple(str(v).strip("/").removeprefix("./") or "." for v in paths)
    return ()


def _testpaths_value(text: str, section: str | None, *, always: bool) -> Any:
    """The raw ``testpaths`` of one config, ``None`` when it has no pytest section."""
    if section is None:
        block = tomllib.loads(text).get("tool", {}).get("pytest", {}).get("ini_options")
        return None if block is None else block.get("testpaths", ())
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(text)
    if not parser.has_section(section):
        return "" if always else None
    return parser.get(section, "testpaths", fallback="")


def _go_args(selection: Selection) -> list[str]:
    go_tests = [f for f in selection.test_files if f.endswith("_test.go")]
    dirs = {str(PurePosixPath(f).parent) for f in go_tests} | set(selection.packages)
    return sorted("." if d == "." else f"./{d}" for d in dirs)


def _jest_args(selection: Selection) -> list[str]:
    return [f for f in selection.test_files if f.endswith(_JS)]


def _file_args(selection: Selection) -> list[str]:
    return list(selection.test_files)


_RUNNER_ARGS = {"pytest": _pytest_args, "go": _go_args, "jest": _jest_args}


def runner_args(selection: Selection, runner: str) -> list[str]:
    """Arguments for *runner* (already resolved); ``[RUN_ALL]`` for a full run.

    Each runner gets only the tests it can run, so a job per language can share
    one selection. ``tests.always_run`` entries are appended as written. jest
    and vitest get file paths, meant for ``--runTestsByPath``.
    """
    if selection.run_all:
        return [RUN_ALL]
    args = _RUNNER_ARGS.get(runner, _file_args)(selection)
    return list(dict.fromkeys([*args, *selection.always_run]))


def format_args(args: Iterable[str]) -> str:
    """One shell-quoted line."""
    return shlex.join(args)
