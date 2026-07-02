"""Brew-payload encoding tests.

These pin the ``@TP:`` start-command encoding against two vectors that
were validated live on a JURA E6 (a real coffee was brewed from each).
The logic under test is framework-free (pure stdlib), so this module
imports nothing from Home Assistant and makes no network calls.
"""

from __future__ import annotations

import pytest

from custom_components.jura.brew import ProductArg, ProductDef, build_start_command


def _product(code: str, args: list[ProductArg]) -> ProductDef:
    return ProductDef(code=code, name="x", picture=None, pos_csv=None, args=tuple(args))


def test_vector_1_coffee_with_overrides() -> None:
    """Coffee(0x03), strength 2, 130 ml, temp Normal(0x01).

    130 // 5 == 26 == 0x1A.
    """
    product = _product(
        "03",
        [
            ProductArg(kind="COFFEE_STRENGTH", argument="F3", index=2, default=5),
            ProductArg(kind="WATER_AMOUNT", argument="F4", index=3, default=120, min=25, max=240, step=5),
            ProductArg(kind="TEMPERATURE", argument="F7", index=6, default=2),
        ],
    )
    assert build_start_command(product, strength=2, water_ml=130, temp=1) == "@TP:0300021A000001000100000000000000"


def test_vector_2_espresso_defaults() -> None:
    """Default Espresso: code 0x02, strength 8, 45 ml (-> 9), temp 0x02."""
    product = _product(
        "02",
        [
            ProductArg(kind="COFFEE_STRENGTH", argument="F3", index=2, default=8),
            ProductArg(kind="WATER_AMOUNT", argument="F4", index=3, default=45, min=25, max=240, step=5),
            ProductArg(kind="TEMPERATURE", argument="F7", index=6, default=2),
        ],
    )
    assert build_start_command(product) == "@TP:02000809000002000100000000000000"


def test_payload_is_uppercase_and_16_bytes() -> None:
    product = _product(
        "03",
        [ProductArg(kind="WATER_AMOUNT", argument="F4", index=3, default=255, step=1)],
    )
    out = build_start_command(product, water_ml=255)
    assert out.startswith("@TP:")
    hex_part = out.removeprefix("@TP:")
    assert len(hex_part) == 32
    assert hex_part == hex_part.upper()


# ---------------------------------------------------------------------------
# Clamping. JURA WiFi takes per-byte literals (no sentinels): an out-of-range
# water byte floods the machine (0xFF -> 255*step ml) rather than meaning
# "max". So every value MUST be clamped to the product's XML range before it
# goes on the wire.
# ---------------------------------------------------------------------------


def _byte(payload: str, index: int) -> str:
    hex_part = payload.removeprefix("@TP:")
    return hex_part[index * 2 : index * 2 + 2]


def test_water_clamps_to_max_not_overflow() -> None:
    """A huge water request clamps to Max, it must NOT wrap mod 256.

    Max 240 ml / step 5 -> 48 == 0x30. The old ``& 0xFF`` masking would
    have yielded 99999 // 5 == 19999 -> 0x1F (a silent, wrong value).
    """
    product = _product(
        "03",
        [ProductArg(kind="WATER_AMOUNT", argument="F4", index=3, default=120, min=25, max=240, step=5)],
    )
    out = build_start_command(product, water_ml=99999)
    assert _byte(out, 3) == "30"
    assert _byte(out, 3) != "FF"
    assert _byte(out, 3) != "1F"


def test_water_clamps_to_min() -> None:
    """Below Min clamps up to Min (25 ml / step 5 -> 5 == 0x05)."""
    product = _product(
        "03",
        [ProductArg(kind="WATER_AMOUNT", argument="F4", index=3, default=120, min=25, max=240, step=5)],
    )
    out = build_start_command(product, water_ml=0)
    assert _byte(out, 3) == "05"


def test_strength_clamps_to_item_set() -> None:
    """Strength above the highest ITEM clamps to it (items 1..10 -> 0x0A)."""
    product = _product(
        "03",
        [
            ProductArg(
                kind="COFFEE_STRENGTH",
                argument="F3",
                index=2,
                default=5,
                items=tuple((str(n), n) for n in range(1, 11)),
            )
        ],
    )
    out = build_start_command(product, strength=99)
    assert _byte(out, 2) == "0A"


def test_temp_clamps_to_item_set() -> None:
    """Temperature outside the ITEM set clamps into it (items 0..2 -> 0x02)."""
    product = _product(
        "03",
        [
            ProductArg(
                kind="TEMPERATURE",
                argument="F7",
                index=6,
                default=1,
                items=(("Low", 0), ("Normal", 1), ("High", 2)),
            )
        ],
    )
    out = build_start_command(product, temp=7)
    assert _byte(out, 6) == "02"


def test_no_byte_exceeds_0xff_without_max() -> None:
    """With no Max defined, the encoded byte is still capped at 0xFF."""
    product = _product(
        "03",
        [ProductArg(kind="WATER_AMOUNT", argument="F4", index=3, default=10, step=1)],
    )
    out = build_start_command(product, water_ml=99999)
    assert _byte(out, 3) == "FF"


def test_factory_default_emits_xml_default_payload() -> None:
    """``None`` params == Factory Default == the product's XML default bytes.

    Strength None -> default 8 (0x08), water None -> default 45//5 (0x09),
    temp None -> default 2 (0x02): the validated Espresso default vector.
    """
    product = _product(
        "02",
        [
            ProductArg(kind="COFFEE_STRENGTH", argument="F3", index=2, default=8),
            ProductArg(kind="WATER_AMOUNT", argument="F4", index=3, default=45, min=25, max=240, step=5),
            ProductArg(kind="TEMPERATURE", argument="F7", index=6, default=2),
        ],
    )
    assert build_start_command(product, strength=None, water_ml=None, temp=None) == (
        "@TP:02000809000002000100000000000000"
    )


def test_milk_foam_amount_f6_default_encoded() -> None:
    """MILK_FOAM_AMOUNT (F6, index 5) brews at its default: 22 // 1 -> 0x16.

    Derived from the EF1030 XML (not yet live-brewed): a milk drink previously
    left its milk-foam byte at 0x00 because only F3/F4/F7 were encoded.
    """
    product = _product(
        "08",
        [ProductArg(kind="MILK_FOAM_AMOUNT", argument="F6", index=5, default=22, min=1, max=45, step=1)],
    )
    assert build_start_command(product) == "@TP:08000000001600000100000000000000"


def test_milk_foam_amount_override_encoded() -> None:
    """An explicit milk_foam override lands in byte 5 as ``value // step``.

    Cappuccino-style product with milk foam 1..45 step 1: milk_foam=10 -> 0x0A.
    Water must stay at its own default — the MinMax args are independent.
    """
    product = _product(
        "04",
        [
            ProductArg(kind="WATER_AMOUNT", argument="F4", index=3, default=120, min=25, max=240, step=5),
            ProductArg(kind="MILK_FOAM_AMOUNT", argument="F6", index=5, default=22, min=1, max=45, step=1),
        ],
    )
    payload = build_start_command(product, milk_foam=10)
    assert _byte(payload, 5) == "0A"
    assert _byte(payload, 3) == "18"  # water default 120 // 5


def test_milk_amount_f5_default_and_override_encoded() -> None:
    """MILK_AMOUNT (F5, index 4) — the milk *liquid* phase, distinct from foam.

    EF545 (Z10) Milkcoffee: milk default 7, range 1..45 step 1. Previously the
    kind was not parsed at all, so byte 4 went out as 0x00 — a from-panel
    milk coffee would skip its milk phase entirely.
    """
    product = _product(
        "05",
        [
            ProductArg(kind="MILK_AMOUNT", argument="F5", index=4, default=7, min=1, max=45, step=1),
            ProductArg(kind="MILK_FOAM_AMOUNT", argument="F6", index=5, default=3, min=0, max=45, step=1),
        ],
    )
    defaults = build_start_command(product)
    assert _byte(defaults, 4) == "07"
    assert _byte(defaults, 5) == "03"
    # Overrides steer each phase independently.
    both = build_start_command(product, milk=20, milk_foam=10)
    assert _byte(both, 4) == "14"
    assert _byte(both, 5) == "0A"


def test_milk_foam_amount_override_clamps_to_range() -> None:
    """Milk overrides clamp to the product's [Min, Max] like water does."""
    product = _product(
        "04",
        [ProductArg(kind="MILK_FOAM_AMOUNT", argument="F6", index=5, default=22, min=1, max=45, step=1)],
    )
    assert _byte(build_start_command(product, milk_foam=999), 5) == "2D"  # max 45
    assert _byte(build_start_command(product, milk_foam=0), 5) == "01"  # min 1


def test_milk_override_ignored_by_water_only_product() -> None:
    """Milk overrides must not leak into products without milk args."""
    product = _product(
        "02",
        [ProductArg(kind="WATER_AMOUNT", argument="F4", index=3, default=45, min=25, max=240, step=5)],
    )
    assert build_start_command(product, milk=15, milk_foam=30) == build_start_command(product)


def test_bypass_f10_default_encoded() -> None:
    """BYPASS (F10, index 9) brews at its default: 40 // 5 -> 0x08."""
    product = _product(
        "28",
        [ProductArg(kind="BYPASS", argument="F10", index=9, default=40, min=0, max=240, step=5)],
    )
    assert _byte(build_start_command(product), 9) == "08"


def test_brew_xml_path_end_to_end() -> None:
    """The XML sourcing path: resolve jura_connect's bundled data and
    encode a real product. Validates importlib.resources wiring +
    encoding shape without asserting machine-specific defaults.
    """
    pytest.importorskip("jura_connect")
    from custom_components.jura.brew import jura_connect_xml_dir, load_definition

    base_dir = jura_connect_xml_dir()
    assert base_dir is not None
    definition = load_definition("EF1091", base_dir=base_dir)
    assert definition is not None
    assert definition.products
    for product in definition.products:
        payload = build_start_command(product)
        assert payload.startswith("@TP:")
        assert len(payload.removeprefix("@TP:")) == 32
