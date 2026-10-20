"""spartan/make_worklist.py: current names, idempotence, validation, listing check."""

import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "make_worklist", os.path.join(os.path.dirname(__file__), "..", "spartan", "make_worklist.py"))
mw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mw)

OLD = ["level1/thermal/candlebark/Curve_1_100_kW_60_cm.nc\tcandlebark/E_rubida/100kW/curve/60cm/1\t0\t1\ts\n",
       "level1/thermal/candlebark/Flat_2_50_kW_20_cm.nc\tcandlebark/E_rubida/50kW/flat/20cm/2\t0\t1\ts\n",
       "level1/thermal/branchlet/Acacia/51.nc\tbranchlet/Acacia/150kW/Twigs/51\t0\t1\ts\n"]


def test_maps_both_candlebark_columns_to_current_names():
    rows = mw.build(OLD)
    assert rows[0] == ("level1/thermal/candlebark/Curve1_100kW_60cm.nc",
                       "candlebark/E_rubida/100kW/cylindrical/60cm/cylindrical1_100kW_60cm")
    assert rows[1] == ("level1/thermal/candlebark/Flat2_50kW_20cm.nc",
                       "candlebark/E_rubida/50kW/flat/20cm/flat2_50kW_20cm")
    assert rows[2] == ("level1/thermal/branchlet/Acacia/51.nc", "branchlet/Acacia/150kW/Twigs/51")


def test_is_idempotent():
    once = mw.build(OLD)
    twice = mw.build([f"{a}\t{b}\n" for a, b in once])
    assert once == twice


@pytest.mark.parametrize("bad", [
    "nc_path\tlevel2_path\n",                                   # a header
    "level1/thermal/x.nc\t\n",                                  # empty label
    "\tbranchlet/Acacia/150kW/Twigs/51\n",                      # empty NetCDF
    "level1/thermal/x.nc\tsomething/else/1\n",                  # unknown family
    "level1/thermal/x.nc\tbranchlet/../../etc\n",               # path traversal
])
def test_rejects_malformed_lines(bad):
    with pytest.raises(ValueError):
        mw.build([bad])


def test_skips_blank_and_comment_lines_and_strips_fields():
    rows = mw.build(["\n", "# comment\n", " level1/thermal/branchlet/Acacia/51.nc \t branchlet/Acacia/150kW/Twigs/51 \n"])
    assert rows == [("level1/thermal/branchlet/Acacia/51.nc", "branchlet/Acacia/150kW/Twigs/51")]


def test_listing_must_match_exactly():
    rows_ok = {"level1/thermal/candlebark/Curve1_100kW_60cm.nc", "level1/thermal/candlebark/Flat2_50kW_20cm.nc",
               "level1/thermal/branchlet/Acacia/51.nc"}
    assert len(mw.build(OLD, rows_ok)) == 3
    with pytest.raises(ValueError):
        mw.build(OLD, rows_ok - {"level1/thermal/branchlet/Acacia/51.nc"})           # missing on MediaFlux
    with pytest.raises(ValueError):
        mw.build(OLD, rows_ok | {"level1/thermal/branchlet/Acacia/52.nc"})           # listed but unused
