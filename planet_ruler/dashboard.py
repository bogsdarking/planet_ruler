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
from typing import Dict, List, Optional


# Planet radii for reference (meters)
PLANET_RADII = {
    'earth': 6371000,
    'mars': 3389500,
    'jupiter': 69911000,
    'saturn': 58232000,
    'moon': 1737400,
    'pluto': 1188300,
}


def is_jupyter() -> bool:
    """Check if running in Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


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
        target_planet: str = 'earth',
        max_iter: Optional[int] = None,
        free_params: Optional[List[str]] = None
    ):
        self.initial_params = initial_params.copy()
        self.target_planet = target_planet.lower()
        self.target_radius = PLANET_RADII.get(self.target_planet, 6371000)
        self.max_iter = max_iter or 1000
        self.free_params = free_params or ['r', 'h', 'f', 'w']
        
        # Tracking
        self.start_time = time.time()
        self.iteration = 0
        self.loss_history: List[float] = []
        self.param_history: List[Dict[str, float]] = []
        self.last_render_time = 0
        
        # State
        self.converged = False
        self.in_jupyter = is_jupyter()
        self.dashboard_height = 25  # Fixed height: always exactly 25 lines
        self._display_handle = None  # For Jupyter display updates
        
        # Initial render
        self._first_render = True
    
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
        self.loss_history.append(loss)
        self.param_history.append(current_params.copy())
        
        # Throttle rendering to avoid flicker (max 10 Hz)
        current_time = time.time()
        if current_time - self.last_render_time > 0.1 or self._first_render:
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
                display(HTML('<div style="color: green; font-weight: bold; margin-top: 10px;">✓ Optimization completed successfully!</div>'))
            else:
                display(HTML('<div style="color: orange; font-weight: bold; margin-top: 10px;">⚠ Optimization stopped (may not have converged)</div>'))
        else:
            # Terminal: show cursor and add completion message
            print('\033[?25h', end='', flush=True)  # Show cursor
            print()  # Move to new line after dashboard
            if success:
                print("✓ Optimization completed successfully!")
            else:
                print("⚠ Optimization stopped (may not have converged)")
    
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
                print('\033[?25l', end='', flush=True)  # Hide cursor
                print(dashboard_text, end='', flush=True)  # No trailing newline
                self._first_render = False
            else:
                # Subsequent renders: move up, clear, and reprint
                # Move cursor to beginning of line, then up to start of dashboard
                print('\r', end='', flush=True)  # Move to beginning of current line
                print(f'\033[{self.dashboard_height - 1}A', end='', flush=True)  # Move up (dashboard_height - 1) lines
                # Clear from cursor to end of screen
                print('\033[J', end='', flush=True)
                # Print updated dashboard (no trailing newline)
                print(dashboard_text, end='', flush=True)
    
    def _build_dashboard(self) -> str:
        """Build the dashboard text."""
        if not self.param_history:
            return "Initializing optimization..."
        
        # Get current state
        current_params = self.param_history[-1]
        current_loss = self.loss_history[-1]
        
        # Calculate metrics
        radius_km = current_params.get('r', 0) / 1000
        altitude_km = current_params.get('h', 0) / 1000
        focal_mm = current_params.get('f', 0) * 1000
        sensor_mm = current_params.get('w', 0) * 1000
        
        # Progress
        progress_pct = (self.iteration / self.max_iter) * 100
        progress_bar = self._make_progress_bar(progress_pct)
        
        # Loss change
        if len(self.loss_history) > 1:
            initial_loss = self.loss_history[0]
            loss_reduction = ((initial_loss - current_loss) / initial_loss) * 100
        else:
            loss_reduction = 0
        
        # Convergence assessment
        convergence_status = self._assess_convergence()
        
        # Time estimates
        elapsed = time.time() - self.start_time
        if self.iteration > 0:
            time_per_iter = elapsed / self.iteration
            remaining_iters = max(0, self.max_iter - self.iteration)
            estimated_remaining = time_per_iter * remaining_iters
        else:
            estimated_remaining = 0
        
        # Warnings and hints
        warnings_list = self._check_warnings(current_params)
        hints_list = self._generate_hints()
        
        # Build dashboard - all content must be exactly 61 chars wide
        lines = [
            "┌─────────────────────────────────────────────────────────────┐",
            "│ FITTING PLANETARY RADIUS                                    │",
            "├─────────────────────────────────────────────────────────────┤",
        ]
        
        # Progress line
        prog_text = f"Progress: {progress_bar} {self.iteration}/{self.max_iter} iterations"
        lines.append(f"│ {prog_text:<59} │")
        
        lines.append("│                                                             │")
        lines.append("│ Current Best Estimate:                                      │")
        
        # Radius line
        radius_text = f"Radius:       {radius_km:>7,.0f} km  (target: ~{self.target_radius/1000:,.0f} km)"
        lines.append(f"│   {radius_text:<57} │")
        
        # Altitude line
        altitude_text = f"Altitude:     {altitude_km:>7.2f} km"
        lines.append(f"│   {altitude_text:<57} │")
        
        # Always show all parameters (for fixed height)
        focal_text = f"Focal length: {focal_mm:>7.1f} mm"
        lines.append(f"│   {focal_text:<57} │")
        
        sensor_text = f"Sensor width: {sensor_mm:>7.2f} mm"
        lines.append(f"│   {sensor_text:<57} │")
        
        lines.append("│                                                             │")
        lines.append("│ Fit Quality:                                                │")
        
        # Loss line
        loss_text = f"Loss:         {current_loss:>7,.0f} (↓{loss_reduction:>4.0f}% from start)"
        lines.append(f"│   {loss_text:<57} │")
        
        # Convergence line
        conv_text = f"Convergence:  {convergence_status}"
        lines.append(f"│   {conv_text:<57} │")
        
        lines.append("│                                                             │")
        
        # Time line
        time_text = f"Time: {self._format_time(elapsed)} | ~{self._format_time(estimated_remaining)} remaining"
        lines.append(f"│ {time_text:<59} │")
        
        # Always reserve exactly 2 slots for warnings (fixed height)
        lines.append("│                                                             │")
        lines.append("│ ⚠ Warnings:                                                 │")
        
        # Always add exactly 2 warning lines
        for i in range(2):
            if i < len(warnings_list):
                warning = warnings_list[i]
                # Truncate to fit (56 chars for content after bullet)
                if len(warning) > 56:
                    warning = warning[:53] + "..."
                lines.append(f"│   • {warning:<56}│")
            else:
                # Empty slot
                lines.append("│                                                             │")
        
        # Always reserve exactly 2 slots for hints (fixed height)
        lines.append("│                                                             │")
        lines.append("│ ℹ Hints:                                                    │")
        
        # Always add exactly 2 hint lines
        for i in range(2):
            if i < len(hints_list):
                hint = hints_list[i]
                # Truncate to fit (56 chars for content after bullet)
                if len(hint) > 56:
                    hint = hint[:53] + "..."
                lines.append(f"│   • {hint:<56}│")
            else:
                # Empty slot
                lines.append("│                                                             │")
        
        lines.append("└─────────────────────────────────────────────────────────────┘")
        
        # Verify all lines are exactly 63 chars (for perfect alignment)
        for i, line in enumerate(lines):
            if len(line) != 63:
                # Debug info if alignment is off
                import sys
                print(f"\nDEBUG: Line {i} has {len(line)} chars (expected 63)", file=sys.stderr)
                print(f"Line: '{line}'", file=sys.stderr)
        
        # Verify we have exactly the right number of lines
        assert len(lines) == self.dashboard_height, f"Dashboard has {len(lines)} lines but should have {self.dashboard_height}"
        
        return '\n'.join(lines)
    
    def _make_progress_bar(self, percent: float, width: int = 20) -> str:
        """Create ASCII progress bar."""
        filled = int(width * percent / 100)
        bar = '█' * filled + '░' * (width - filled)
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
        altitude_m = params.get('h', 0)
        if altitude_m > 50000:  # > 50 km
            warnings_list.append(
                "Altitude very high (>50km). Check units (meters vs feet)?"
            )
        elif altitude_m < 1000:  # < 1 km
            warnings_list.append(
                "Altitude very low (<1km). Ground photos won't show curvature"
            )
        
        # Check radius is reasonable
        radius_km = params.get('r', 0) / 1000
        if radius_km < 1000:
            warnings_list.append(
                f"Radius very small ({radius_km:.0f} km). Check horizon visible"
            )
        elif radius_km > 100000:
            warnings_list.append(
                f"Radius very large ({radius_km:.0f} km). Check altitude/camera"
            )
        elif radius_km < 3000 or radius_km > 70000:
            # Outside typical planet range
            warnings_list.append(
                f"Radius ({radius_km:.0f} km) outside typical planet range"
            )
        
        # Check for parameter drift (if enough history)
        if len(self.param_history) > 200 and 'f' in self.free_params:
            focal_drift = self._calc_parameter_drift('f')
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
            hints.append(
                "Optimizer exploring parameter space to find best fit"
            )
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
            radius_km = self.param_history[-1].get('r', 0) / 1000
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
        
        # Iteration hints
        if self.iteration > self.max_iter * 0.8:
            hints.append(
                "Approaching max iterations - fit should complete soon"
            )
        
        return hints
    
    def _calc_parameter_drift(self, param_name: str) -> float:
        """Calculate relative drift in a parameter."""
        if len(self.param_history) < 100:
            return 0.0
        
        early_values = [p[param_name] for p in self.param_history[:50] 
                        if param_name in p]
        recent_values = [p[param_name] for p in self.param_history[-50:] 
                         if param_name in p]
        
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