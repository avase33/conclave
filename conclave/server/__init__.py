"""Offline web dashboard for execution traces."""

from .dashboard import render_dashboard
from .app import serve_trace

__all__ = ["render_dashboard", "serve_trace"]
