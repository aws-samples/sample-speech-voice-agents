import functools
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from strands.models import BedrockModel

# Load the repo-root .env (copied from .env.example) so AWS_PROFILE, AWS_REGION
# and STRANDS_MODEL_* are set once for every lab. Variables already exported in
# the shell win; the file only fills in what is missing. report_aws_target()
# prints the result and warns when the shell overrode the file.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

# Known-good active model. Used when auto mode is off or discovery fails.
FALLBACK_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Region used when neither AWS_REGION nor the AWS profile sets one.
DEFAULT_REGION = "us-east-1"

# Preferred families for auto mode, tried in order. The first family with an
# active model wins; within a family the newest (by creation date) is chosen.
_PREFERRED_FAMILIES = ("claude-sonnet", "claude-opus", "claude-haiku")


def get_region() -> str:
    """Return the AWS region every lab should use.

    Resolution order matches boto3/BedrockModel: ``AWS_REGION`` (shell or
    .env), then the active AWS profile's configured region, then
    ``DEFAULT_REGION``. Labs with a service-specific regional constraint
    (for example Nova Sonic in lab 06) call this first and then validate.
    """
    try:
        import boto3

        return (
            os.environ.get("AWS_REGION")
            or boto3.session.Session().region_name
            or DEFAULT_REGION
        )
    except Exception:
        return os.environ.get("AWS_REGION", DEFAULT_REGION)


_target_reported = False


def report_aws_target(model_id: str | None = None) -> None:
    """Print, once per process, the region and model this run will use.

    Shown before any agent output so "which model was I on?" is never a
    guess and a region mismatch is visible before the first Bedrock call.
    Also warns when the shell's ``AWS_REGION`` overrode the value in ``.env``,
    the one case where the file a user just edited is not what takes effect.
    """
    global _target_reported
    if _target_reported:
        return
    _target_reported = True

    region = get_region()
    line = f"[aws] region={region}"
    if model_id:
        line += f"  model={model_id}"
    print(line)

    wanted = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    if wanted.get("AWS_REGION") and wanted["AWS_REGION"] != region:
        print(
            f"[aws] note: .env sets AWS_REGION={wanted['AWS_REGION']} but the shell's "
            f"AWS_REGION won. Unset it to use the .env value."
        )


@functools.lru_cache(maxsize=1)
def _auto_model_id() -> str:
    """Discover the newest active (non-Legacy) preferred model on Bedrock.

    Returns ``FALLBACK_MODEL_ID`` if discovery cannot run for any reason.
    """
    try:
        import boto3

        client = boto3.client("bedrock", region_name=get_region())

        # Foundation models the provider still marks ACTIVE (not Legacy/EOL).
        active_model_ids = {
            m["modelId"]
            for m in client.list_foundation_models(byProvider="anthropic")[
                "modelSummaries"
            ]
            if m.get("modelLifecycle", {}).get("status") == "ACTIVE"
        }

        # All system-defined inference profiles (paginated).
        profiles = []
        next_token = None
        while True:
            kwargs = {"maxResults": 100}
            if next_token:
                kwargs["nextToken"] = next_token
            resp = client.list_inference_profiles(**kwargs)
            profiles.extend(resp.get("inferenceProfileSummaries", []))
            next_token = resp.get("nextToken")
            if not next_token:
                break

        def underlying_active(profile) -> bool:
            for m in profile.get("models", []):
                model_id = m.get("modelArn", "").rsplit("/", 1)[-1]
                if model_id in active_model_ids:
                    return True
            return False

        usable = [
            p
            for p in profiles
            if p.get("status") == "ACTIVE"
            and p.get("type") == "SYSTEM_DEFINED"
            and p.get("inferenceProfileId", "").startswith("us.anthropic.")
            and p.get("createdAt") is not None
            and underlying_active(p)
        ]

        for family in _PREFERRED_FAMILIES:
            matches = [p for p in usable if family in p["inferenceProfileId"]]
            if matches:
                newest = max(matches, key=lambda p: p["createdAt"])
                return newest["inferenceProfileId"]
    except Exception:
        # No credentials, missing permissions, throttling, API change, etc.
        pass
    return FALLBACK_MODEL_ID


def resolve_model_id() -> str:
    """Resolve the model id: env override, then auto mode, then fallback."""
    override = os.environ.get("STRANDS_MODEL_ID")
    if override:
        return override
    if os.environ.get("STRANDS_MODEL_AUTO", "1") == "0":
        return FALLBACK_MODEL_ID
    return _auto_model_id()


def get_model(**kwargs) -> BedrockModel:
    """Return a BedrockModel using the resolved (auto/env/fallback) model id.

    Any keyword arguments (``temperature``, ``max_tokens``, ``region_name``,
    or an explicit ``model_id`` override) are passed through to BedrockModel.

    ``region_name`` defaults to :func:`get_region`. BedrockModel on its own
    prefers the profile's configured region over ``AWS_REGION``, so without
    this the model discovery (which uses get_region) and the actual Bedrock
    calls could land in two different regions.
    """
    kwargs.setdefault("model_id", resolve_model_id())
    if "boto_session" not in kwargs:
        kwargs.setdefault("region_name", get_region())
    report_aws_target(kwargs["model_id"])
    return BedrockModel(**kwargs)
