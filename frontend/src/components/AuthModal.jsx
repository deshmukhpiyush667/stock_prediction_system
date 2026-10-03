import React, { useState } from 'react';
import { X, Lock, Mail, User, AlertCircle, Loader2 } from 'lucide-react';

export default function AuthModal({ isOpen, onClose, onAuthSuccess, apiBase }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);

    // Frontend validation
    if (!isLogin) {
      if (!username.trim() || username.trim().length < 3) {
        setError("Username must be at least 3 characters.");
        return;
      }
      if (!email.trim() || !email.includes('@')) {
        setError("Please enter a valid email address.");
        return;
      }
      if (password.length < 6) {
        setError("Password must be at least 6 characters.");
        return;
      }
      if (password !== confirmPassword) {
        setError("Passwords do not match.");
        return;
      }
    } else {
      if (!username.trim()) {
        setError("Please enter your username or email.");
        return;
      }
      if (!password) {
        setError("Please enter your password.");
        return;
      }
    }

    setLoading(true);

    try {
      const url = isLogin ? `${apiBase}/auth/login` : `${apiBase}/auth/register`;
      const payload = isLogin
        ? { username_or_email: username.trim(), password }
        : { email: email.trim(), username: username.trim(), password };

      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || (isLogin ? "Login failed" : "Registration failed"));
      }

      // Success
      if (data.token && data.user) {
        localStorage.setItem("stock_token", data.token);
        localStorage.setItem("stock_user", JSON.stringify(data.user));
        onAuthSuccess(data.user, data.token);
        onClose();
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function toggleMode(mode) {
    setIsLogin(mode);
    setError(null);
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {isLogin ? "Sign In to Terminal" : "Create Account"}
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginTop: 3 }}>
              {isLogin
                ? "Access personalized model forecasts & market predictions"
                : "Register a trader account to track stock classification models"}
            </p>
          </div>
          <button className="modal-close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        {/* Tab Switcher */}
        <div className="auth-tabs">
          <button
            type="button"
            className={`auth-tab ${isLogin ? 'active' : ''}`}
            onClick={() => toggleMode(true)}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`auth-tab ${!isLogin ? 'active' : ''}`}
            onClick={() => toggleMode(false)}
          >
            Register (Sign Up)
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="auth-error-alert">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          {/* Username / Email field for Login OR Username for Signup */}
          <div className="auth-field">
            <label>{isLogin ? "Username or Email" : "Username"}</label>
            <div className="input-with-icon">
              <User size={16} className="input-icon" />
              <input
                type="text"
                placeholder={isLogin ? "e.g. trader123 or user@mail.com" : "e.g. quant_trader"}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoFocus
                required
              />
            </div>
          </div>

          {/* Email field (only for Signup) */}
          {!isLogin && (
            <div className="auth-field">
              <label>Email Address</label>
              <div className="input-with-icon">
                <Mail size={16} className="input-icon" />
                <input
                  type="email"
                  placeholder="e.g. name@domain.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>
          )}

          {/* Password field */}
          <div className="auth-field">
            <label>Password</label>
            <div className="input-with-icon">
              <Lock size={16} className="input-icon" />
              <input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </div>

          {/* Confirm Password (only for Signup) */}
          {!isLogin && (
            <div className="auth-field">
              <label>Confirm Password</label>
              <div className="input-with-icon">
                <Lock size={16} className="input-icon" />
                <input
                  type="password"
                  placeholder="••••••••"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                />
              </div>
            </div>
          )}

          <button type="submit" className="auth-submit-btn" disabled={loading}>
            {loading ? (
              <>
                <Loader2 size={16} className="animate-spin" style={{ marginRight: 8 }} />
                {isLogin ? "Authenticating..." : "Creating Account..."}
              </>
            ) : (
              isLogin ? "Sign In" : "Create Trader Account"
            )}
          </button>
        </form>

        <div className="modal-footer">
          {isLogin ? (
            <p>
              Don't have an account?{" "}
              <button type="button" className="auth-switch-link" onClick={() => toggleMode(false)}>
                Register here
              </button>
            </p>
          ) : (
            <p>
              Already registered?{" "}
              <button type="button" className="auth-switch-link" onClick={() => toggleMode(true)}>
                Sign in here
              </button>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
