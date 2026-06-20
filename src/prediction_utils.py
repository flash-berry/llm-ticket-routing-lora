import json

from pydantic import ValidationError

from src.schemas import TicketRoutingOutput


def parse_and_validate_response(raw_response: str) -> dict:
    result = {
        "raw_response": raw_response,
        "parsed_response": None,
        "json_valid": False,
        "schema_valid": False,
        "prediction": None,
        "error": None,
    }

    try:
        parsed_response = json.loads(raw_response)
        result["json_valid"] = True
        result["parsed_response"] = parsed_response
    except json.decoder.JSONDecodeError as exc:
        result["error"] = f"JSON error: {exc.msg}"
        return result

    if not isinstance(parsed_response, dict):
        result["error"] = "Schema error: JSON response must be an object"
        return result

    try:
        validated_prediction = TicketRoutingOutput.model_validate(parsed_response)
        result["schema_valid"] = True
        result["prediction"] = validated_prediction.model_dump()
    except ValidationError as exc:
        result["error"] = f"Schema error: {exc}"

    return result