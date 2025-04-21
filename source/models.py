import pandas as pd
import pandera as pa
from pandera.typing import Series

schema = pa.DataFrameSchema(
    {
        "PolNum": pa.Column(pa.Int),
        "CalYear": pa.Column(
            pa.Int, checks=pa.Check(lambda s: (s >= 2000) & (s <= 2025))
        ),
        "Gender": pa.Column(pa.String, checks=pa.Check.isin(["Male", "Female"])),
        "Type": pa.Column(
            pa.String, checks=pa.Check.isin(["A", "B", "C", "D", "E", "F"])
        ),
        "Category": pa.Column(
            pa.String, checks=pa.Check.isin(["Medium", "Large", "Small"])
        ),
        "Occupation": pa.Column(
            pa.String,
            checks=pa.Check.isin(
                ["Employed", "Self-employed", "Housewife", "Unemployed", "Retired"]
            ),
        ),
        "Age": pa.Column(pa.Int, checks=pa.Check(lambda s: s >= 0)),
        "Group1": pa.Column(pa.Int),
        "Bonus": pa.Column(pa.Int),
        "Poldur": pa.Column(pa.Int, checks=pa.Check(lambda s: s >= 0)),
        "Value": pa.Column(pa.Int, checks=pa.Check(lambda s: s >= 0)),
        "Adind": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),
        "SubGroup2": pa.Column(pa.String),
        "Group2": pa.Column(pa.String),
        "Density": pa.Column(pa.Float, checks=pa.Check(lambda s: s >= 0)),
        "Exppdays": pa.Column(pa.Int, checks=pa.Check(lambda s: (s >= 0) & (s <= 366))),
        "Numtppd": pa.Column(pa.Int, checks=pa.Check(lambda s: s >= 0), required=False),
        "Numtpbi": pa.Column(pa.Int, checks=pa.Check(lambda s: s >= 0), required=False),
        "Indtppd": pa.Column(
            pa.Float, checks=pa.Check(lambda s: s >= 0), required=False
        ),
        "Indtpbi": pa.Column(
            pa.Float, checks=pa.Check(lambda s: s >= 0), required=False
        ),
    },
    strict=True,
    coerce=True,
    ordered=True,
    name="Schema",
)
