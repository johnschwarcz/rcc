"""The README has to actually work.

Two things rot independently: the code blocks, and the argument tables. Both are
checked here against the running library rather than by reading.

Blocks are executed in one shared namespace, in order, exactly as a reader would
follow them — so a block that uses ``chain`` gets the ``chain`` an earlier block
defined.
"""

import dataclasses
import re
from pathlib import Path
import pytest
import torch
from rcc import RCCConfig

README = Path(__file__).resolve().parent.parent / "README.md"
TEXT = README.read_text(encoding="utf-8")

#: Names the README refers to as the reader's own. Supplying them is better than
#: skipping the blocks: the surrounding calls still have to run and typecheck
#: against the real signatures.
READER_SUPPLIED = {
    # A target belief, shaped like the quickstart's chain emits.
    "target_belief": torch.softmax(torch.randn(16, 10, 2, 4), -1),
    # Some other source of per-channel context, for the standalone classifier.
    "my_own_context_vector": torch.randn(16, 3, 2),
    # What the controller's chosen actions turned out to be worth.
    "rates_of_those_actions": torch.rand(16, 3),
    "preferences": torch.rand(3).round(),
}


def _python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", TEXT, re.S)


def _table_rows(header: str) -> dict[str, str]:
    """Parse the markdown table that follows ``header`` into ``{name: cell}``.

    Returns ``{}`` when the table is absent. These tests check that what the
    README *does* say is true; they must not force it to keep saying anything.
    """
    if header not in TEXT:
        return {}
    body = TEXT.split(header, 1)[1]
    rows: dict[str, str] = {}
    seen_a_row = False
    for line in body.splitlines():
        if not line.startswith("|"):
            # Stop at the end of *this* table, whether or not any cell parsed.
            # Waiting for a match instead would run on into the next table.
            if seen_a_row:
                break
            continue
        seen_a_row = True
        cells = [c.strip() for c in line.strip("|").split("|")]
        # The first cell is either a bare name (`n_vars`) or a call
        # (`reward_loss(belief, ...)`); both identify the thing being documented.
        name = re.fullmatch(r"`([a-z_]+)(?:\(.*\))?`", cells[0])
        if name:
            rows[name.group(1)] = cells[1].strip("`")
    return rows


def _same_default(shown: str, actual: object) -> bool:
    """Whether a README cell states ``actual``, ignoring quoting style."""
    normalise = str.maketrans("", "", "`'\"")
    expected = "None" if actual is None else str(actual)
    return shown.translate(normalise) == expected.translate(normalise)


# --------------------------------------------------------------------------- #
# the code blocks run
# --------------------------------------------------------------------------- #
def test_every_python_block_executes():
    """Run the README top to bottom in one namespace, as a reader would."""
    namespace: dict = dict(READER_SUPPLIED)
    ran = 0
    for block in _python_blocks():
        exec(compile(block, "README.md", "exec"), namespace)  # noqa: S102
        ran += 1
    assert ran, "no runnable python blocks found — did the extraction break?"


def test_the_quickstart_produces_the_shape_it_claims():
    """The quickstart states its own output shape in a comment; hold it to that."""
    namespace: dict = dict(READER_SUPPLIED)
    quickstart = next(b for b in _python_blocks() if "belief.shape" in b)
    exec(compile(quickstart, "README.md", "exec"), namespace)  # noqa: S102

    claimed = re.search(r"belief\.shape\s*#\s*\(([\d, ]+)\)", quickstart)
    assert claimed, "the quickstart no longer states the shape it produces"
    expected = tuple(int(n) for n in claimed.group(1).split(","))
    assert tuple(namespace["belief"].shape) == expected


# --------------------------------------------------------------------------- #
# the tables describe the real API
# --------------------------------------------------------------------------- #
def test_config_table_matches_the_dataclass():
    """Every field documented, every documented field real, defaults correct."""
    documented = _table_rows("| Field | Default | Meaning |")
    if not documented:
        pytest.skip("README has no config table")
    actual = {f.name: f.default for f in dataclasses.fields(RCCConfig)}

    assert set(documented) == set(actual), (
        f"undocumented: {sorted(set(actual) - set(documented))}; "
        f"invented: {sorted(set(documented) - set(actual))}"
    )
    for name, shown in documented.items():
        assert _same_default(shown, actual[name]), (
            f"{name}: README says {shown!r}, actual default is {actual[name]!r}"
        )


def test_derived_properties_named_in_the_readme_exist():
    listed = re.search(r"Derived properties: (.+?)\.\n", TEXT, re.S)
    if not listed:
        pytest.skip("README does not list derived properties")
    cfg = RCCConfig()
    for name in re.findall(r"`([a-z_]+)`", listed.group(1)):
        assert hasattr(cfg, name), f"README names a derived property {name!r}"


def test_objective_table_names_real_functions():
    documented = _table_rows("| Function | Trains | Needs |")
    if not documented:
        pytest.skip("README has no objective table")
    import rcc

    for call in documented:
        assert hasattr(rcc, call), f"README documents {call!r}, which rcc does not export"


def test_stage_table_names_real_modules():
    import rcc

    body = TEXT.split("| Stage | Module | Reads | Produces |", 1)
    if len(body) == 1:
        pytest.skip("README has no stage table")
    for name in re.findall(r"`([A-Z][A-Za-z]+)`", body[1].split("\n\n", 1)[0]):
        assert hasattr(rcc, name), f"README documents {name!r}, which rcc does not export"


# --------------------------------------------------------------------------- #
# links and images
# --------------------------------------------------------------------------- #
def test_every_referenced_image_exists():
    for src in re.findall(r'src="([^"]+)"', TEXT):
        assert (README.parent / src).exists(), f"README references missing {src}"


def test_every_relative_link_resolves():
    for target in re.findall(r"\]\((?!https?:|#)([^)]+)\)", TEXT):
        assert (README.parent / target).exists(), f"README links to missing {target}"


def test_every_example_the_readme_lists_exists():
    for script in re.findall(r"python (examples/[\w_]+\.py)", TEXT):
        assert (README.parent / script).exists(), f"README runs missing {script}"
