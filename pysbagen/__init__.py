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
from .living_sessions import (
    AffectSnapshot,
    LivingSessionArchive,
    LivingSessionPlan,
    SessionEvent,
    SessionMutation,
    SessionOutcome,
    StoredSession,
    create_child_sleep_plan,
    create_sleep_plan,
    recommend_child_mode,
    sleep_request_from_manifest,
)
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
    "AffectSnapshot",
    "BackendCapability",
    "CompatibilityState",
    "EngineDiscrepancy",
    "ImportReport",
    "ImportedArtifact",
    "InteroperabilityReport",
    "LivingSessionArchive",
    "LivingSessionPlan",
    "NativeDiagnostic",
    "NativeValidationReport",
    "RenderDisposition",
    "RenderResult",
    "SBaGenXNative",
    "SBaGenXNativeError",
    "SBaGenXProbe",
    "SessionEvent",
    "SessionMutation",
    "SessionOutcome",
    "SleepLayers",
    "SleepRecipe",
    "SleepRequest",
    "StoredSession",
    "UnsupportedSBaGenXAPI",
    "build_sleep_recipe",
    "create_child_sleep_plan",
    "create_sleep_plan",
    "import_artifact",
    "import_drg",
    "import_sbg",
    "import_sbgf",
    "inspect_artifact",
    "inspect_with_sbagenx",
    "probe_sbagenx",
    "recommend_child_mode",
    "render_schedule",
    "render_sleep",
    "render_specs",
    "sleep_request_from_manifest",
    "validate_sbagenx_source",
    "write_audio",
]
__version__ = "0.4.0"
