from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


TicketType = Literal['Incident', 'Request', 'Problem', 'Change']
Urgency = Literal['low', 'medium', 'high', 'critical']


class TicketRoutingOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    ticket_type: TicketType
    topic: str = Field(min_length=1)
    urgency: Urgency
    tags: list[str]

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        cleaned = []

        for tag in value:
            tag = str(tag).strip()

            if tag:
                cleaned.append(tag)

        return cleaned