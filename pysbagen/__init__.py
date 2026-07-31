"""pysbagen public package."""

from .api import (
    RenderResult,
    inspect_artifact,
    render_schedule,
    render_sleep,
    render_specs,
    write_audio,
)
from .compatibility import CompatibilityState, ImportReport, RenderDisposition
from .importers import ImportedArtifact, import_artifact, import_drg, import_sbg
from .interoperability import EngineDiscrepancy, InteroperabilityReport, inspect_with_sbagenx
from .sbagenx_backend import BackendCapability, SBaGenXProbe, probe_sbagenx
from .sbagenx_native import (
    NativeDiagnostic,
    NativeValidationReport,
    SBaGenXNative,
    SBaGenXNativeError,
    UnsupportedSBaGenXAPI,
    validate_sbagenx_source,
)
from .sbgf import import_sbgf
from .sleep import SleepLayers, SleepRecipe, SleepRequest, build_sleep_recipe

__all__ = [
    "BackendCapability",
    "CompatibilityState",
    "EngineDiscrepancy",
    "ImportReport",
    "ImportedArtifact",
    "InteroperabilityReport",
    "NativeDiagnostic",
    "NativeValidationReport",
    "RenderDisposition",
    "RenderResult",
    "SBaGenXNative",
    "SBaGenXNativeError",
    "SBaGenXProbe",
    "SleepLayers",
    "SleepRecipe",
    "SleepRequest",
    "UnsupportedSBaGenXAPI",
    "build_sleep_recipe",
    "import_artifact",
    "import_drg",
    "import_sbg",
    "import_sbgf",
    "inspect_artifact",
    "inspect_with_sbagenx",
    "probe_sbagenx",
    "render_schedule",
    "render_sleep",
    "render_specs",
    "validate_sbagenx_source",
    "write_audio",
]
__version__ = "0.4.0"
