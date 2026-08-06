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
from .constellation import (
    ConstellationEdge,
    ConstellationGraph,
    ConstellationNode,
    build_constellation,
    write_constellation_json,
)
from .constellation_render import (
    install_constellation_renderer,
    render_constellation_html,
    write_constellation_html,
)
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
from .living_session_policy import install_living_session_policy
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

install_living_session_policy()
install_constellation_renderer()

__all__ = [
    "AffectSnapshot",
    "BackendCapability",
    "CompatibilityState",
    "ConstellationEdge",
    "ConstellationGraph",
    "ConstellationNode",
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
    "build_constellation",
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
    "render_constellation_html",
    "render_schedule",
    "render_sleep",
    "render_specs",
    "sleep_request_from_manifest",
    "validate_sbagenx_source",
    "write_audio",
    "write_constellation_html",
    "write_constellation_json",
]
__version__ = "0.4.0"
