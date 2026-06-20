from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TicketType = Literal["Incident", "Request", "Problem", "Change"]

Urgency = Literal["low", "medium", "high"]

Topic = Literal[
    "Billing and Payments",
    "Customer Service",
    "General Inquiry",
    "Human Resources",
    "IT Support",
    "Product Support",
    "Returns and Exchanges",
    "Sales and Pre-Sales",
    "Service Outages and Maintenance",
    "Technical Support",
]


class TicketRoutingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_type: TicketType
    topic: Topic
    urgency: Urgency
    tags: list[str] = Field(max_length=8)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        cleaned_tags = []

        for tag in value:
            cleaned_tag = tag.strip()

            if not cleaned_tag:
                raise ValueError("tags must not contain empty strings")

            cleaned_tags.append(cleaned_tag)

        if len(cleaned_tags) != len(set(cleaned_tags)):
            raise ValueError("tags must not contain duplicates")

        return cleaned_tags