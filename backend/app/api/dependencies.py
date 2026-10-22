"""
dependencies.py
---------------
Reusable authentication and role-based access control dependencies for FastAPI.

Features:
- Validates Authorization Bearer tokens against the JWT signing key and expiration.
- Loads user state from SQLite database as the single source of truth.
- Verifies account active status.
- Enforces role-based permissions (HTTP 403 Forbidden on role mismatch).
"""

from typing import Any, Callable, Dict, List, Optional


from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

from app.auth import decode_access_token, get_user_by_id
from app.db.database import get_db

# HTTPBearer security scheme (auto_error=False allows returning custom 401 response)
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Dict[str, Any]:
    """
    Dependency that authenticates an incoming request via JWT Bearer token.

    Validates:
    - Presence of Authorization header with Bearer scheme.
    - Valid signature and expiration time.
    - Valid user ID corresponding to an active user in the database.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (jwt.PyJWTError, ValueError) as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or malformed authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing user subject claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = int(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token subject must be a valid integer user ID.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    with get_db() as conn:
        user = get_user_by_id(conn, user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.get("is_active", 1):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive or has been disabled.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Return database record (ensuring role and roll_no from DB are the source of truth)
    return user


def require_roles(*allowed_roles: str) -> Callable[..., Dict[str, Any]]:
    """
    Factory dependency that enforces role-based access control.
    Returns HTTP 403 Forbidden if the authenticated user's database role
    is not in allowed_roles.
    """
    allowed_set = set(allowed_roles)

    def role_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user),
    ) -> Dict[str, Any]:
        user_role = current_user.get("role", "")
        if user_role not in allowed_set:
            roles_str = ", ".join(sorted(allowed_set))
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Access requires one of the following roles: [{roles_str}]. Current role: '{user_role}'.",
            )
        return current_user

    return role_dependency
