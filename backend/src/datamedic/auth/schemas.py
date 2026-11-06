import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class StrictModel(BaseModel):
    # Unknown fields (e.g. a client-supplied "user_id" or "owner_id") are rejected.
    model_config = ConfigDict(extra="forbid")


class SignupIn(StrictModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)
    display_name: str | None = Field(default=None, max_length=100)


class LoginIn(StrictModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)


class TokenIn(StrictModel):
    token: str = Field(min_length=10, max_length=200)


class PasswordResetRequestIn(StrictModel):
    email: EmailStr


class PasswordResetConfirmIn(StrictModel):
    token: str = Field(min_length=10, max_length=200)
    new_password: str = Field(min_length=1, max_length=1024)


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str | None
    email_verified: bool
    plan: str


class AuthStateOut(BaseModel):
    user: UserOut
    csrf_token: str


class CsrfOut(BaseModel):
    csrf_token: str


class SessionOut(BaseModel):
    id: uuid.UUID
    created_at: datetime
    last_seen_at: datetime
    user_agent: str | None
    current: bool


class SessionListOut(BaseModel):
    items: list[SessionOut]


class AcceptedOut(BaseModel):
    status: str = "accepted"
