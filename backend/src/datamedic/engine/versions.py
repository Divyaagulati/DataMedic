"""Library versions recorded on every version and profile (reproducibility)."""

from importlib.metadata import PackageNotFoundError, version

TRACKED = (
    "pandas",
    "numpy",
    "pyarrow",
    "openpyxl",
    "xlrd",
    "scipy",
    "scikit-learn",
    "imbalanced-learn",
)


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRACKED:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out
