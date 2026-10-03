"""
Automated unit & integration test for Authentication System (Signup, Login, Me, Logout)
"""
import sys
import os

backend_dir = os.path.abspath(os.path.dirname(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_auth_system():
    print("==================================================")
    print("TESTING AUTHENTICATION SYSTEM (SIGNUP & SIGNIN)")
    print("==================================================")

    import uuid
    unique_suffix = uuid.uuid4().hex[:6]
    test_email = f"user_{unique_suffix}@stockpredict.ai"
    test_username = f"trader_{unique_suffix}"
    test_pwd = "StrongPassword@123"

    # 1. Test Register
    reg_payload = {
        "email": test_email,
        "username": test_username,
        "password": test_pwd
    }
    r = client.post("/auth/register", json=reg_payload)
    print(f"\n1. POST /auth/register -> Status {r.status_code}")
    assert r.status_code == 200, f"Register failed: {r.text}"
    reg_data = r.json()
    token = reg_data["token"]
    assert "token" in reg_data and len(token) == 64
    assert reg_data["user"]["email"] == test_email
    print(f"   Registered user '{reg_data['user']['username']}' with session token.")

    # 2. Test Duplicate Registration (Expected 400)
    r_dup = client.post("/auth/register", json=reg_payload)
    print(f"\n2. Duplicate registration attempt -> Status {r_dup.status_code} (Expected 400)")
    assert r_dup.status_code == 400
    print(f"   Error detail: {r_dup.json()['detail']}")

    # 3. Test Login with Username
    login_payload_username = {
        "username_or_email": test_username,
        "password": test_pwd
    }
    r_log_u = client.post("/auth/login", json=login_payload_username)
    print(f"\n3. POST /auth/login (via username) -> Status {r_log_u.status_code}")
    assert r_log_u.status_code == 200
    new_token = r_log_u.json()["token"]
    print(f"   Login successful via username.")

    # 4. Test Login with Email
    login_payload_email = {
        "username_or_email": test_email,
        "password": test_pwd
    }
    r_log_e = client.post("/auth/login", json=login_payload_email)
    print(f"\n4. POST /auth/login (via email) -> Status {r_log_e.status_code}")
    assert r_log_e.status_code == 200
    print(f"   Login successful via email.")

    # 5. Test Login with Wrong Password (Expected 401)
    r_bad = client.post("/auth/login", json={"username_or_email": test_email, "password": "WrongPassword"})
    print(f"\n5. Login with invalid password -> Status {r_bad.status_code} (Expected 401)")
    assert r_bad.status_code == 401
    print(f"   Error detail: {r_bad.json()['detail']}")

    # 6. Test GET /auth/me with Bearer token
    headers = {"Authorization": f"Bearer {new_token}"}
    r_me = client.get("/auth/me", headers=headers)
    print(f"\n6. GET /auth/me -> Status {r_me.status_code}")
    assert r_me.status_code == 200
    assert r_me.json()["user"]["username"] == test_username
    print(f"   Session authenticated for user: {r_me.json()['user']['username']}")

    # 7. Test Logout
    r_out = client.post("/auth/logout", headers=headers)
    print(f"\n7. POST /auth/logout -> Status {r_out.status_code}")
    assert r_out.status_code == 200

    # 8. Test Session Invalidated
    r_me_after = client.get("/auth/me", headers=headers)
    print(f"\n8. GET /auth/me after logout -> Status {r_me_after.status_code} (Expected 401)")
    assert r_me_after.status_code == 401
    print(f"   Token invalidation verified.")

    print("\n[SUCCESS] All 8 Authentication tests PASSED.")
    print("==================================================")

if __name__ == "__main__":
    test_auth_system()
