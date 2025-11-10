# Copyright 2025 Brandon Anderson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Live progress dashboard for parameter optimization.

Displays real-time updates during fit_limb() optimization, showing:
- Current parameter estimates (radius, altitude, focal length)
- Optimization progress and convergence
- Loss function evolution
- Smart warnings and educational hints

Works in: terminal, Jupyter notebooks, IPython
Updates in-place (no scrolling spam)

Example
-------
>>> obs = LimbObservation("photo.jpg", "config.yaml")
>>> obs.detect_limb(method="manual")
>>> obs.fit_limb(dashboard=True)  # Shows live dashboard during fitting
"""

import time
import sys

from typing import Dict, List, Optional, TextIO
from collections import deque
from contextlib import contextmanager
import threading


# Planet radii for reference (meters)
PLANET_RADII = {
    "earth": 6371000,
    "mars": 3389500,
    "jupiter": 69911000,
    "saturn": 58232000,
    "moon": 1737400,
    "pluto": 1188300,
}


class OutputCapture:
    """
    Capture stdout/stderr for display in dashboard.

    Thread-safe ring buffer that stores recent output lines.
    Designed to integrate with FitDashboard for live output display.

    Parameters
    ----------
    max_lines : int, default 5
        Maximum number of recent lines to keep
    line_width : int, default 55
        Maximum width before wrapping (dashboard constraint)
    capture_stderr : bool, default True
        Also capture stderr (warnings, errors)

    Examples
    --------
    >>> capture = OutputCapture(max_lines=5)
    >>> with capture:
    ...     print("Iteration 100: loss = 123.45")
    ...     print("Warning: parameter drift detected")
    >>> lines = capture.get_lines()
    """

    def __init__(
        self, max_lines: int = 5, line_width: int = 55, capture_stderr: bool = True
    ):
        self.max_lines = max_lines
        self.line_width = line_width
        self.capture_stderr = capture_stderr

        # Ring buffer for recent output
        self.lines: deque = deque(maxlen=max_lines)
        self._lock = threading.Lock()

        # Original streams (for restoration)
        self._original_stdout = None
        self._original_stderr = None

        # Active capture state
        self._capturing = False

    def write(self, text: str) -> None:
        """Write text to buffer (called by sys.stdout redirect)."""
        if not text or text == "\n":
            return

        # Split on newlines and wrap long lines
        for line in text.splitlines():
            if not line.strip():
                continue

            # Wrap line if too long
            wrapped = self._wrap_line(line.strip())

            with self._lock:
                for wrapped_line in wrapped:
                    self.lines.append(wrapped_line)

    def flush(self) -> None:
        """Flush (no-op, required for file-like interface)."""
        pass

    def _wrap_line(self, line: str) -> List[str]:
        """Wrap a long line to fit dashboard width."""
        if len(line) <= self.line_width:
            return [line]

        # Simple word-aware wrapping
        wrapped = []
        current = ""

        for word in line.split():
            if len(current) + len(word) + 1 <= self.line_width:
                current += (" " if current else "") + word
            else:
                if current:
                    wrapped.append(current)
                current = word

        if current:
            wrapped.append(current)

        return wrapped or [line[: self.line_width]]

    def get_lines(self) -> List[str]:
        """Get recent output lines (thread-safe)."""
        with self._lock:
            return list(self.lines)

    def clear(self) -> None:
        """Clear captured output."""
        with self._lock:
            self.lines.clear()

    def __enter__(self):
        """Start capturing (context manager)."""
        self._original_stdout = sys.stdout
        if self.capture_stderr:
            self._original_stderr = sys.stderr

        sys.stdout = self
        if self.capture_stderr:
            sys.stderr = self

        self._capturing = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop capturing and restore streams."""
        sys.stdout = self._original_stdout
        if self.capture_stderr and self._original_stderr:
            sys.stderr = self._original_stderr

        self._capturing = False
        return False


def is_jupyter() -> bool:
    """Check if running in Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class FitDashboard:
    """
    Live dashboard for optimization progress.

    Shows real-time parameter estimates, convergence status, and helpful
    warnings during fit_limb() optimization.

    Parameters
    ----------
    initial_params : dict
        Initial parameter values (r, h, f, w, etc.)
    target_planet : str, default 'earth'
        Reference planet for comparison ('earth', 'mars', etc.)
    max_iter : int, optional
        Maximum iterations (if known)
    free_params : list of str, optional
        Which parameters are being optimized
    total_stages : int, default 1
        Number of optimization stages (for multi-resolution)
    cumulative_max_iter : int, optional
        Total iterations across all stages
    min_refresh_delay : float, default 0.0
        Minimum time (seconds) between refreshes (0.0 enables adaptive)
    refresh_frequency : int, default 1
        Refresh every N iterations
    output_capture : OutputCapture, optional
        Output capture instance for displaying print/log statements
    show_output : bool, default True
        Show output section in dashboard (requires output_capture)
    max_output_lines : int, default 3
        Maximum number of output lines to display
    max_warnings : int, default 3
        Number of warning message slots
    max_hints : int, default 3
        Number of hint message slots
    min_message_display_time : float, default 3.0
        Minimum time (seconds) to display warnings/hints before removal
    width : int, default 63
        Dashboard width in characters (includes borders)

    Attributes
    ----------
    iteration : int
        Current iteration number
    loss_history : list of float
        Loss values over time
    param_history : list of dict
        Parameter values over time
    """

    def __init__(
        self,
        initial_params: Dict[str, float],
        target_planet: str = "earth",
        max_iter: Optional[int] = None,
        free_params: Optional[List[str]] = None,
        total_stages: int = 1,
        cumulative_max_iter: Optional[int] = None,
        min_refresh_delay: float = 0.00,
        refresh_frequency: int = 1,
        output_capture: Optional[OutputCapture] = None,
        show_output: bool = True,
        max_output_lines: int = 3,
        max_warnings: int = 3,
        max_hints: int = 3,
        min_message_display_time: float = 3.0,
        width: int = 63,
    ):
        self.initial_params = initial_params.copy()
        self.target_planet = target_planet.lower()
        self.target_radius = PLANET_RADII.get(self.target_planet, 6371000)
        self.max_iter = max_iter or 1000
        self.free_params = free_params or ["r", "h", "f", "w"]

        # Dashboard dimensions
        self.width = width
        self.content_width = width - 4  # "│ " + content + " │"
        self.bullet_content_width = width - 9  # "│   • " + content + " │"
        self.output_content_width = width - 6  # "│   " + content + " │"

        # Multi-stage tracking
        self.total_stages = total_stages
        self.current_stage = 1
        self.resolution_label = "full" if total_stages == 1 else "1/4x"
        self.stage_start_iteration = 0
        self.cumulative_max_iter = cumulative_max_iter or max_iter

        # Tracking
        self.start_time = time.time()
        self.iteration = 0
        self.cumulative_iteration = 0
        self.loss_history: List[float] = []
        self.param_history: List[Dict[str, float]] = []
        self.last_render_time = 0
        self.min_refresh_delay = min_refresh_delay
        self.refresh_frequency = refresh_frequency

        # Adaptive refresh system
        self._enable_adaptive_refresh = (
            min_refresh_delay == 0.0
        )  # Only if not manually set
        self._adaptive_delay = 0.05  # Start at 50ms (20 Hz)
        self._loss_velocity_window = 10  # Window for loss velocity calculation
        self._last_loss_velocity = 0.0

        # For proper loss reduction calculation across stages
        self.stage_start_loss = None
        self.initial_loss = None

        # State
        self.converged = False
        self.in_jupyter = is_jupyter()

        # Output capture integration
        self.output_capture = output_capture
        self.show_output = show_output and output_capture is not None
        self.max_output_lines = max_output_lines

        # Message display config
        self.max_warnings = max_warnings
        self.max_hints = max_hints
        self.min_message_display_time = min_message_display_time

        # Message tracking (message -> first_seen_time)
        self._warnings_tracking: Dict[str, float] = {}
        self._hints_tracking: Dict[str, float] = {}

        # Dynamic dashboard height based on features
        # Core dashboard: title(3) + progress(1-2) + params(7) + quality(5) + time(1) = 16-17
        base_height = 16 if total_stages == 1 else 17

        # Add warning section (blank + header + slots)
        base_height += 2 + max_warnings

        # Add hint section (blank + header + slots)
        base_height += 2 + max_hints

        # Add output section if enabled (blank + header + slots)
        if self.show_output:
            base_height += 2 + max_output_lines

        # Closing border
        base_height += 1

        self.dashboard_height = base_height
        self._display_handle = None  # For Jupyter display updates

        # Initial render
        self._first_render = True

    def _border_top(self) -> str:
        """Generate top border."""
        return "┌" + "─" * (self.width - 2) + "┐"

    def _border_middle(self) -> str:
        """Generate middle border."""
        return "├" + "─" * (self.width - 2) + "┤"

    def _border_bottom(self) -> str:
        """Generate bottom border."""
        return "└" + "─" * (self.width - 2) + "┘"

    def _line(self, content: str) -> str:
        """Generate content line with proper padding."""
        return f"│ {content:<{self.content_width}} │"

    def _line_bullet(self, content: str) -> str:
        """Generate bullet line with proper padding."""
        return f"│    - {content:<{self.bullet_content_width}} │"
        # return f"│   • {content:<{self.bullet_content_width}} │"

    def _line_indent(self, content: str) -> str:
        """Generate indented line (for parameters/output)."""
        return f"│   {content:<{self.output_content_width}} │"

    def _blank_line(self) -> str:
        """Generate blank line."""
        return "│" + " " * (self.width - 2) + "│"

    def update(self, current_params: Dict[str, float], loss: float) -> None:
        """
        Update dashboard with current optimization state.

        Called by optimizer callback at each iteration.

        Parameters
        ----------
        current_params : dict
            Current parameter values (r, h, f, w, etc.)
        loss : float
            Current loss function value
        """
        self.iteration += 1
        self.cumulative_iteration += 1
        self.loss_history.append(loss)
        self.param_history.append(current_params.copy())

        # Capture initial loss for percentage calculation
        if self.initial_loss is None:
            self.initial_loss = loss
        if self.stage_start_loss is None:
            self.stage_start_loss = loss

        # Update adaptive refresh rate if enabled
        if self._enable_adaptive_refresh:
            self._update_adaptive_delay()

        # Throttle rendering with adaptive or fixed delay
        current_time = time.time()
        effective_delay = (
            self._adaptive_delay
            if self._enable_adaptive_refresh
            else self.min_refresh_delay
        )

        should_render = (
            current_time - self.last_render_time > effective_delay
            and not self.iteration % self.refresh_frequency
        ) or self._first_render

        if should_render:
            self._render()
            self.last_render_time = current_time

    def finalize(self, success: bool = True) -> None:
        """
        Show final summary after optimization completes.

        Parameters
        ----------
        success : bool, default True
            Whether optimization converged successfully
        """
        self.converged = success
        self._render()  # Final render

        # Add completion message
        if self.in_jupyter:
            from IPython.display import display, HTML

            # Add message below the dashboard
            if success:
                display(
                    HTML(
                        '<div style="color: green; font-weight: bold; margin-top: 10px;">✓ Optimization completed successfully!</div>'
                    )
                )
            else:
                display(
                    HTML(
                        '<div style="color: orange; font-weight: bold; margin-top: 10px;">⚠ Optimization stopped (may not have converged)</div>'
                    )
                )
        else:
            # Terminal: show cursor and add completion message
            print("\033[?25h", end="", flush=True)  # Show cursor
            print()  # Move to new line after dashboard
            if success:
                print("✓ Optimization completed successfully!")
            else:
                print("⚠ Optimization stopped (may not have converged)")

    def start_stage(self, stage_num: int, resolution_label: str, max_iter: int) -> None:
        """
        Start a new optimization stage (for multi-resolution).

        Parameters
        ----------
        stage_num : int
            Stage number (1-indexed)
        resolution_label : str
            Resolution label (e.g., "1/4x", "1/2x", "full")
        max_iter : int
            Maximum iterations for this stage
        """
        self.current_stage = stage_num
        self.resolution_label = resolution_label
        self.max_iter = max_iter
        self.iteration = 0
        self.stage_start_iteration = self.cumulative_iteration
        self.stage_start_loss = None  # Will be set on first update of new stage

    def _update_adaptive_delay(self) -> None:
        """
        Dynamically adjust refresh rate based on optimization state.

        Strategy:
        - Fast updates (20 Hz) when loss changing rapidly
        - Slow updates (2-5 Hz) when converging/stable
        - Balance responsiveness vs CPU efficiency
        """
        if len(self.loss_history) < self._loss_velocity_window:
            # Early iterations: keep it snappy
            self._adaptive_delay = 0.05  # 20 Hz
            return

        # Calculate loss velocity (relative change per iteration)
        recent_losses = self.loss_history[-self._loss_velocity_window :]
        if recent_losses[0] == 0:
            loss_velocity = 0.0
        else:
            loss_velocity = abs(
                (recent_losses[-1] - recent_losses[0]) / recent_losses[0]
            )

        self._last_loss_velocity = loss_velocity

        # Determine refresh rate based on activity level
        # High activity (>1% change) → 20 Hz (50ms)
        # Medium activity (0.1-1%) → 10 Hz (100ms)
        # Low activity (0.01-0.1%) → 5 Hz (200ms)
        # Very low (<0.01%) → 2 Hz (500ms)

        if loss_velocity > 0.01:
            # Rapid changes: update frequently
            self._adaptive_delay = 0.05  # 20 Hz
        elif loss_velocity > 0.001:
            # Moderate changes
            self._adaptive_delay = 0.10  # 10 Hz
        elif loss_velocity > 0.0001:
            # Slow changes
            self._adaptive_delay = 0.20  # 5 Hz
        else:
            # Nearly converged
            self._adaptive_delay = 0.50  # 2 Hz

        # Additional adjustment: slow down for very long runs
        elapsed = time.time() - self.start_time
        if elapsed > 60:  # After 1 minute
            self._adaptive_delay = max(self._adaptive_delay, 0.20)  # Max 5 Hz
        if elapsed > 300:  # After 5 minutes
            self._adaptive_delay = max(self._adaptive_delay, 0.50)  # Max 2 Hz

    def _render(self) -> None:
        """Draw/update the dashboard."""
        dashboard_text = self._build_dashboard()

        if self.in_jupyter:
            # Jupyter: use display handle for flicker-free updates
            from IPython.display import display, HTML

            # Wrap in <pre> tag to preserve formatting
            html_content = f'<pre style="font-family: monospace; line-height: 1.2;">{dashboard_text}</pre>'

            if self._first_render:
                # First render: create display with handle
                self._display_handle = display(HTML(html_content), display_id=True)
                self._first_render = False
            else:
                # Subsequent renders: update existing display
                self._display_handle.update(HTML(html_content))
        else:
            # Terminal: update in-place
            if self._first_render:
                # First time: hide cursor and print dashboard
                print("\033[?25l", end="", flush=True)  # Hide cursor
                print(dashboard_text, end="", flush=True)  # No trailing newline
                self._first_render = False
            else:
                # Subsequent renders: move up, clear, and reprint
                # Move cursor to beginning of line, then up to start of dashboard
                print("\r", end="", flush=True)  # Move to beginning of current line
                print(
                    f"\033[{self.dashboard_height - 1}A", end="", flush=True
                )  # Move up (dashboard_height - 1) lines
                # Clear from cursor to end of screen
                print("\033[J", end="", flush=True)
                # Print updated dashboard (no trailing newline)
                print(dashboard_text, end="", flush=True)

    def _build_dashboard(self) -> str:
        """Build the dashboard text."""
        if not self.param_history:
            return "Initializing optimization..."

        # Get current state
        current_params = self.param_history[-1]
        current_loss = self.loss_history[-1]

        # Calculate metrics
        radius_km = current_params.get("r", 0) / 1000
        altitude_km = current_params.get("h", 0) / 1000
        focal_mm = current_params.get("f", 0) * 1000
        sensor_mm = current_params.get("w", 0) * 1000

        # Loss change (from stage start)
        if len(self.loss_history) > 1 and self.stage_start_loss is not None:
            loss_reduction = (
                (self.stage_start_loss - current_loss) / self.stage_start_loss
            ) * 100
        else:
            loss_reduction = 0

        # Convergence assessment
        convergence_status = self._assess_convergence()

        # Time estimates
        elapsed = time.time() - self.start_time
        if self.cumulative_iteration > 0:
            time_per_iter = elapsed / self.cumulative_iteration
            remaining_iters = max(
                0, self.cumulative_max_iter - self.cumulative_iteration
            )
            estimated_remaining = time_per_iter * remaining_iters
        else:
            estimated_remaining = 0

        # Warnings and hints
        current_time = time.time()
        warnings_list = self._check_warnings(current_params)
        hints_list = self._generate_hints()

        # Filter messages based on time visible
        warnings_list = self._filter_messages_by_time(
            warnings_list, self._warnings_tracking, current_time
        )
        hints_list = self._filter_messages_by_time(
            hints_list, self._hints_tracking, current_time
        )

        # Build dashboard
        lines = []

        # Title varies based on single vs multi-stage
        if self.total_stages > 1:
            title = f"STAGE {self.current_stage}/{self.total_stages}: {self.resolution_label.upper()} RESOLUTION"
            lines.extend(
                [
                    self._border_top(),
                    self._line(title),
                    self._border_middle(),
                ]
            )
        else:
            lines.extend(
                [
                    self._border_top(),
                    self._line("FITTING PLANETARY RADIUS"),
                    self._border_middle(),
                ]
            )

        # Progress bars
        if self.total_stages > 1:
            # Stage progress
            stage_pct = (self.iteration / self.max_iter) * 100
            stage_bar = self._make_progress_bar(stage_pct, width=18)
            stage_text = f"Stage:   {stage_bar} {self.iteration}/{self.max_iter} iter"
            lines.append(self._line(stage_text))

            # Overall progress
            overall_pct = (self.cumulative_iteration / self.cumulative_max_iter) * 100
            overall_bar = self._make_progress_bar(overall_pct, width=18)
            overall_text = f"Overall: {overall_bar} {self.cumulative_iteration}/{self.cumulative_max_iter} iter"
            lines.append(self._line(overall_text))
        else:
            # Single stage - just one progress bar
            progress_pct = (self.iteration / self.max_iter) * 100
            progress_bar = self._make_progress_bar(progress_pct)
            prog_text = (
                f"Progress: {progress_bar} {self.iteration}/{self.max_iter} iterations"
            )
            lines.append(self._line(prog_text))

        lines.append(self._blank_line())
        lines.append(self._line("Current Best Estimate:"))

        # Parameters - use _line_indent for indented content
        radius_text = f"Radius:       {radius_km:>7,.0f} km"
        lines.append(self._line_indent(radius_text))

        altitude_text = f"Altitude:     {altitude_km:>7.2f} km"
        lines.append(self._line_indent(altitude_text))

        focal_text = f"Focal length: {focal_mm:>7.1f} mm"
        lines.append(self._line_indent(focal_text))

        sensor_text = f"Sensor width: {sensor_mm:>7.2e} mm"
        lines.append(self._line_indent(sensor_text))

        lines.append(self._blank_line())
        lines.append(self._line("Fit Quality:"))

        loss_text = (
            f"Loss:         {current_loss:>7,.0f} (↓{loss_reduction:>4.0f}% from start)"
        )
        lines.append(self._line_indent(loss_text))

        conv_text = f"Convergence:  {convergence_status}"
        lines.append(self._line_indent(conv_text))

        lines.append(self._blank_line())

        # Time line with optional refresh rate info
        if (
            self._enable_adaptive_refresh
            and len(self.loss_history) >= self._loss_velocity_window
        ):
            refresh_hz = 1.0 / self._adaptive_delay if self._adaptive_delay > 0 else 0
            time_text = f"Time: {self._format_time(elapsed)} | ~{self._format_time(estimated_remaining)} left | {refresh_hz:.0f}Hz"
        else:
            time_text = f"Time: {self._format_time(elapsed)} | ~{self._format_time(estimated_remaining)} remaining"
        lines.append(self._line(time_text))

        # Warnings section
        lines.append(self._blank_line())
        # lines.append(self._line("⚠ Warnings:"))
        lines.append(self._line("Warnings:"))

        for i in range(self.max_warnings):
            if i < len(warnings_list):
                warning = warnings_list[i]
                if len(warning) > self.bullet_content_width:
                    warning = warning[: self.bullet_content_width - 3] + "..."
                lines.append(self._line_bullet(warning))
            else:
                lines.append(self._blank_line())

        # Hints section
        lines.append(self._blank_line())
        # lines.append(self._line("ℹ Hints:"))
        lines.append(self._line("Hints:"))

        for i in range(self.max_hints):
            if i < len(hints_list):
                hint = hints_list[i]
                if len(hint) > self.bullet_content_width:
                    hint = hint[: self.bullet_content_width - 3] + "..."
                lines.append(self._line_bullet(hint))
            else:
                lines.append(self._blank_line())

        # Output section (if enabled)
        if self.show_output:
            lines.append(self._blank_line())
            lines.append(self._line("Recent Output:"))

            output_lines = (
                self.output_capture.get_lines() if self.output_capture else []
            )

            # Get the last N lines (most recent output)
            recent_output = (
                output_lines[-self.max_output_lines :]
                if len(output_lines) > self.max_output_lines
                else output_lines
            )

            for i in range(self.max_output_lines):
                if i < len(recent_output):
                    output_text = recent_output[i]
                    if len(output_text) > self.output_content_width:
                        output_text = (
                            output_text[: self.output_content_width - 3] + "..."
                        )
                    lines.append(self._line_indent(output_text))
                else:
                    lines.append(self._blank_line())

        lines.append(self._border_bottom())

        # Verify all lines are exactly self.width chars (for perfect alignment)
        for i, line in enumerate(lines):
            if len(line) != self.width:
                # Debug info if alignment is off
                import sys

                print(
                    f"\nDEBUG: Line {i} has {len(line)} chars (expected {self.width})",
                    file=sys.stderr,
                )
                print(f"Line: '{line}'", file=sys.stderr)

        # Verify we have exactly the right number of lines
        assert (
            len(lines) == self.dashboard_height
        ), f"Dashboard has {len(lines)} lines but should have {self.dashboard_height}"

        return "\n".join(lines)

    def _make_progress_bar(self, percent: float, width: int = 20) -> str:
        """Create ASCII progress bar."""
        percent = min(100, max(0, percent))  # Clamp to 0-100
        filled = int(width * percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"{bar} {percent:>3.0f}%"

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs:02d}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins:02d}m"

    def _filter_messages_by_time(
        self, messages: List[str], tracking: Dict[str, float], current_time: float
    ) -> List[str]:
        """
        Filter messages based on time visible.

        Keeps messages that have been visible for at least min_message_display_time,
        then uses FIFO to limit to available slots.
        """
        # Update tracking: add new messages, track their first appearance
        for msg in messages:
            if msg not in tracking:
                tracking[msg] = current_time

        # Remove messages no longer in current list
        tracking_keys = list(tracking.keys())
        for msg in tracking_keys:
            if msg not in messages:
                del tracking[msg]

        # Filter: keep messages visible for min time + fill remaining with newest
        time_threshold = current_time - self.min_message_display_time

        # Messages that have been visible long enough (must keep)
        persistent = [
            msg for msg in messages if tracking.get(msg, current_time) <= time_threshold
        ]

        # If persistent messages exceed max slots, use FIFO (oldest first)
        max_slots = (
            self.max_warnings if tracking is self._warnings_tracking else self.max_hints
        )
        if len(persistent) >= max_slots:
            # Sort by first appearance time (oldest first), take the max_slots oldest
            persistent.sort(key=lambda msg: tracking[msg])
            return persistent[:max_slots]

        # Fill remaining slots with newest messages
        remaining_slots = max_slots - len(persistent)
        new_messages = [msg for msg in messages if msg not in persistent]

        # Combine persistent + newest
        return persistent + new_messages[:remaining_slots]

    def _assess_convergence(self) -> str:
        """Assess convergence status."""
        if self.iteration < 10:
            return "Initializing..."

        if self.iteration < 50:
            return "Exploring parameter space"

        # Look at recent loss trend
        recent_window = 20
        if len(self.loss_history) < recent_window:
            return "Early stage"

        recent_losses = self.loss_history[-recent_window:]
        loss_change = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]

        if loss_change > 0.1:
            return "Improving rapidly"
        elif loss_change > 0.01:
            return "Improving steadily"
        elif loss_change > 0.001:
            return "Converging"
        elif loss_change > 0:
            return "Nearly converged"
        else:
            return "Stalled (may be converged)"

    def _check_warnings(self, params: Dict[str, float]) -> List[str]:
        """Generate smart warnings based on parameter evolution."""
        warnings_list = []

        # Check altitude units (common error)
        altitude_m = params.get("h", 0)
        if altitude_m < 1000:  # < 1 km
            warnings_list.append(
                "Altitude very low (<1km). Ground photos may not show curvature"
            )

        # Check radius is reasonable
        radius_km = params.get("r", 0) / 1000
        if radius_km < 1000:
            warnings_list.append(
                f"Radius very small ({radius_km:.0f} km). Check horizon visible"
            )
        elif radius_km > 100000:
            warnings_list.append(
                f"Radius very large ({radius_km:.0f} km). Check altitude/camera"
            )

        # Check for parameter drift (if enough history)
        if len(self.param_history) > 200 and "f" in self.free_params:
            focal_drift = self._calc_parameter_drift("f")
            if focal_drift > 0.5:  # 50% drift
                warnings_list.append(
                    "Focal length drifting. Consider fixing if known from EXIF"
                )

        # Check for stalling
        if self.iteration > 300:
            recent_improvement = self._recent_improvement(window=50)
            if recent_improvement < 0.001:  # < 0.1% improvement
                warnings_list.append(
                    "Optimization stalling. May have converged or local minimum"
                )

        return warnings_list

    def _generate_hints(self) -> List[str]:
        """Generate educational hints during optimization."""
        hints = []

        if self.iteration < 50:
            hints.append("Optimizer exploring parameter space to find best fit")
            return hints

        # Convergence hints
        if len(self.loss_history) > 50:
            recent_trend = self._loss_trend()
            if recent_trend == "decreasing":
                hints.append("Loss decreasing steadily - optimization healthy")
            elif recent_trend == "stable":
                hints.append("Loss stabilizing - approaching convergence")

        # Radius proximity hints
        if self.param_history:
            radius_km = self.param_history[-1].get("r", 0) / 1000
            target_km = self.target_radius / 1000
            error = abs(radius_km - target_km)

            if error < 500:  # Within 500 km
                error_pct = (error / target_km) * 100
                hints.append(
                    f"Within {error_pct:.1f}% of {self.target_planet.title()}'s radius!"
                )
            elif error < 1000:
                hints.append(
                    f"Approaching {self.target_planet.title()}'s size ({target_km:,.0f} km)"
                )

        # Adaptive refresh hint
        if self._enable_adaptive_refresh and self._adaptive_delay > 0.2:
            # Only show when refresh has slowed to 5 Hz or less
            hints.append(f"Updates slowed to save CPU (loss changing slowly)")

        # Iteration hints
        if self.iteration > self.max_iter * 0.8:
            hints.append("Approaching max iterations - fit should complete soon")

        return hints

    def _calc_parameter_drift(self, param_name: str) -> float:
        """Calculate relative drift in a parameter."""
        if len(self.param_history) < 100:
            return 0.0

        early_values = [
            p[param_name] for p in self.param_history[:50] if param_name in p
        ]
        recent_values = [
            p[param_name] for p in self.param_history[-50:] if param_name in p
        ]

        if not early_values or not recent_values:
            return 0.0

        early_mean = sum(early_values) / len(early_values)
        recent_mean = sum(recent_values) / len(recent_values)

        if early_mean == 0:
            return 0.0

        return abs((recent_mean - early_mean) / early_mean)

    def _recent_improvement(self, window: int = 50) -> float:
        """Calculate recent improvement in loss."""
        if len(self.loss_history) < window:
            return 1.0  # Assume improving

        recent = self.loss_history[-window:]
        if recent[0] == 0:
            return 0.0

        improvement = (recent[0] - recent[-1]) / recent[0]
        return max(0, improvement)

    def _loss_trend(self, window: int = 50) -> str:
        """Determine recent loss trend."""
        if len(self.loss_history) < window:
            return "unknown"

        recent = self.loss_history[-window:]

        # Calculate trend
        improvement = (recent[0] - recent[-1]) / recent[0]

        if improvement > 0.05:
            return "decreasing"
        elif improvement > -0.01:
            return "stable"
        else:
            return "increasing"
