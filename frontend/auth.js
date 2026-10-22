/**
 * auth.js
 * Centralized authentication and session management helper for SmartAttendance frontend.
 *
 * Rules:
 * - Stores JWT access token exclusively in sessionStorage (current tab session).
 * - Never stores passwords or tokens in URLs or localStorage.
 * - Attaches Authorization: Bearer <token> to protected requests.
 * - Verifies user identity via GET /api/auth/me on page load.
 * - Handles 401 (session expired) and 403 (access denied) gracefully.
 */

const BACKEND_URL = 'http://127.0.0.1:8000';
const TOKEN_KEY = 'smartattendance_token';
const USER_KEY = 'smartattendance_user';

/**
 * Retrieve the current access token from sessionStorage.
 */
function getToken() {
    return sessionStorage.getItem(TOKEN_KEY);
}

/**
 * Retrieve the cached user safe profile from sessionStorage.
 */
function getStoredUser() {
    try {
        const u = sessionStorage.getItem(USER_KEY);
        return u ? JSON.parse(u) : null;
    } catch (e) {
        return null;
    }
}

/**
 * Store access token and user profile in sessionStorage.
 */
function setSession(token, user) {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    if (user) sessionStorage.setItem(USER_KEY, JSON.stringify(user));
}

/**
 * Clear session storage.
 */
function clearSession() {
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(USER_KEY);
}

/**
 * Log out user: clears session and redirects to login page.
 */
function logout() {
    clearSession();
    window.location.href = 'index.html?logged_out=1';
}

/**
 * Helper to build headers with Authorization Bearer token attached.
 */
function getAuthHeaders(extraHeaders = {}) {
    const token = getToken();
    const headers = { ...extraHeaders };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
}

/**
 * Verify session on dashboard load.
 * Calls GET /api/auth/me to confirm token validity and verify role permissions.
 * Redirects on 401 or displays access-denied UI on 403.
 *
 * @param {string[]} allowedRoles - List of allowed roles, e.g. ['student'] or ['teacher', 'admin'].
 * @returns {Promise<Object|null>} Safe user profile if valid; null if rejected.
 */
async function requireAuth(allowedRoles = []) {
    const token = getToken();
    if (!token) {
        clearSession();
        window.location.replace('index.html?session_expired=1');
        return null;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);

    try {
        const resp = await fetch(`${BACKEND_URL}/api/auth/me`, {
            method: 'GET',
            headers: { 'Authorization': `Bearer ${token}` },
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (resp.status === 401) {
            clearSession();
            window.location.replace('index.html?session_expired=1');
            return null;
        }

        if (!resp.ok) {
            throw new Error(`Session check failed with HTTP ${resp.status}`);
        }

        const user = await resp.json();
        sessionStorage.setItem(USER_KEY, JSON.stringify(user));

        if (allowedRoles && allowedRoles.length > 0 && !allowedRoles.includes(user.role)) {
            showAccessDenied(user.role, allowedRoles);
            return null;
        }

        return user;
    } catch (err) {
        clearTimeout(timeoutId);
        console.error('[SmartAttendance Auth] Session check error:', err);
        // If network issue with backend, check if we have a stored user
        const stored = getStoredUser();
        if (stored && allowedRoles.length > 0 && !allowedRoles.includes(stored.role)) {
            showAccessDenied(stored.role, allowedRoles);
            return null;
        }
        return stored;
    }
}

/**
 * Handle API response errors: redirects on 401, shows alert on 403.
 *
 * @param {Response} response - Fetch Response object
 * @returns {boolean} True if handled (error occurred), false if OK
 */
function handleAuthResponseError(response) {
    if (response.status === 401) {
        clearSession();
        window.location.replace('index.html?session_expired=1');
        return true;
    }
    if (response.status === 403) {
        alert('Access Denied (403): You do not have permission to perform this action.');
        return true;
    }
    return false;
}

/**
 * Render an in-page 403 Forbidden Access Denied screen.
 */
function showAccessDenied(currentRole, allowedRoles) {
    const rolesStr = allowedRoles.join(', ');
    document.body.innerHTML = `
        <div style="font-family:'Segoe UI', sans-serif; display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:80vh; text-align:center; padding:20px;">
            <div style="font-size:4rem; margin-bottom:10px;">🚫</div>
            <h1 style="color:#d82b2b; margin:0 0 10px;">Access Denied (403 Forbidden)</h1>
            <p style="color:#555; max-width:520px; font-size:16px; line-height:1.5;">
                Your account role is <strong>${escapeHtml(currentRole)}</strong>. This dashboard requires one of the following roles: <strong>${escapeHtml(rolesStr)}</strong>.
            </p>
            <div style="margin-top:25px; display:flex; gap:15px;">
                <a href="index.html" class="btn" style="background:#2b6099; text-decoration:none; padding:10px 20px; border-radius:4px; color:white; font-weight:bold;">Return to Login</a>
                <button onclick="logout()" class="btn" style="background:#d82b2b; border:none; padding:10px 20px; border-radius:4px; color:white; font-weight:bold; cursor:pointer;">Log Out</button>
            </div>
        </div>
    `;
}

/**
 * Escape HTML special characters for safe insertion into DOM.
 */
function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
