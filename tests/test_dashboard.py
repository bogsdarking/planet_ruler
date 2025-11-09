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
Tests for dashboard module.

Tests the FitDashboard and OutputCapture classes including:
- Output capture functionality
- Adaptive refresh rate
- Message persistence
- Configurable dimensions
"""

import pytest
import time
import threading
import numpy as np
from planet_ruler.dashboard import FitDashboard, OutputCapture


class TestOutputCapture:
    """Tests for OutputCapture class."""
    
    def test_creation(self):
        """Test OutputCapture instantiation."""
        capture = OutputCapture(max_lines=20, line_width=55, capture_stderr=True)
        assert capture.max_lines == 20
        assert capture.line_width == 55
        assert capture.capture_stderr is True
        assert len(capture.lines) == 0
    
    def test_context_manager(self):
        """Test OutputCapture as context manager."""
        capture = OutputCapture(max_lines=5)
        
        with capture:
            print("Test line 1")
            print("Test line 2")
            print("Test line 3")
        
        lines = capture.get_lines()
        assert len(lines) == 3
        assert "Test line 1" in lines[0]
        assert "Test line 2" in lines[1]
        assert "Test line 3" in lines[2]
    
    def test_ring_buffer(self):
        """Test that ring buffer correctly limits size."""
        capture = OutputCapture(max_lines=3)
        
        with capture:
            for i in range(10):
                print(f"Line {i}")
        
        lines = capture.get_lines()
        assert len(lines) == 3
        # Should have last 3 lines
        assert "Line 7" in lines[0]
        assert "Line 8" in lines[1]
        assert "Line 9" in lines[2]
    
    def test_line_wrapping(self):
        """Test that long lines are wrapped correctly."""
        capture = OutputCapture(max_lines=5, line_width=30)
        
        with capture:
            print("This is a very long line that should be wrapped automatically")
        
        lines = capture.get_lines()
        assert len(lines) > 1  # Should be wrapped
        for line in lines:
            assert len(line) <= 30
    
    def test_clear(self):
        """Test clearing captured output."""
        capture = OutputCapture(max_lines=5)
        
        with capture:
            print("Test")
        
        assert len(capture.get_lines()) == 1
        capture.clear()
        assert len(capture.get_lines()) == 0
    
    def test_thread_safety(self):
        """Test that OutputCapture is thread-safe."""
        capture = OutputCapture(max_lines=50)
        
        def worker(worker_id):
            with capture:
                for i in range(10):
                    print(f"Worker {worker_id}: message {i}")
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        lines = capture.get_lines()
        assert len(lines) > 0  # Should have captured messages


class TestFitDashboard:
    """Tests for FitDashboard class."""
    
    @pytest.fixture
    def basic_dashboard(self):
        """Create a basic dashboard for testing."""
        params = {'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036}
        return FitDashboard(
            initial_params=params,
            target_planet='earth',
            max_iter=100,
        )
    
    def test_creation(self, basic_dashboard):
        """Test FitDashboard instantiation."""
        assert basic_dashboard.max_iter == 100
        assert basic_dashboard.target_planet == 'earth'
        assert basic_dashboard.iteration == 0
    
    def test_output_capture_integration(self):
        """Test FitDashboard with OutputCapture integration."""
        capture = OutputCapture(max_lines=10)
        
        dashboard = FitDashboard(
            initial_params={'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036},
            target_planet='earth',
            max_iter=100,
            output_capture=capture,
            show_output=True,
            max_output_lines=3,
        )
        
        assert dashboard.output_capture is capture
        assert dashboard.show_output is True
        assert dashboard.max_output_lines == 3
    
    def test_update(self, basic_dashboard):
        """Test updating dashboard state."""
        params = {'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036}
        loss = 1000.0
        
        basic_dashboard.update(params, loss)
        
        assert basic_dashboard.iteration == 1
        assert len(basic_dashboard.loss_history) == 1
        assert basic_dashboard.loss_history[0] == loss
        assert len(basic_dashboard.param_history) == 1
    
    def test_adaptive_refresh_enabled_by_default(self):
        """Test that adaptive refresh is enabled by default."""
        dashboard = FitDashboard(
            initial_params={'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036},
            max_iter=100,
        )
        assert dashboard._enable_adaptive_refresh is True
    
    def test_adaptive_refresh_disabled_with_fixed_delay(self):
        """Test that adaptive refresh is disabled with fixed delay."""
        dashboard = FitDashboard(
            initial_params={'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036},
            max_iter=100,
            min_refresh_delay=0.1,  # Fixed 10 Hz
        )
        assert dashboard._enable_adaptive_refresh is False
    
    def test_adaptive_refresh_adjusts_to_velocity(self):
        """Test that adaptive refresh rate adjusts based on loss velocity."""
        dashboard = FitDashboard(
            initial_params={'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036},
            max_iter=100,
        )
        
        params = {'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036}
        
        # Simulate rapid descent (high velocity)
        for i in range(20):
            loss = 10000 * (1 - i/20)**2
            dashboard.update(params, loss)
        
        # Should have fast refresh during rapid descent
        assert dashboard._adaptive_delay <= 0.1
        
        # Simulate true convergence (completely stable loss)
        for i in range(50):
            loss = 100.0  # No change at all
            dashboard.update(params, loss)
        
        # Should have slow refresh during convergence
        assert dashboard._adaptive_delay >= 0.2
    
    def test_configurable_message_slots(self):
        """Test configurable warning and hint slots."""
        dashboard = FitDashboard(
            initial_params={'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036},
            max_iter=100,
            max_warnings=5,
            max_hints=4,
        )
        
        assert dashboard.max_warnings == 5
        assert dashboard.max_hints == 4
    
    def test_configurable_width(self):
        """Test configurable dashboard width."""
        dashboard = FitDashboard(
            initial_params={'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036},
            max_iter=100,
            width=80,
        )
        
        assert dashboard.width == 80
        assert dashboard.content_width == 76  # 80 - 4
        assert dashboard.bullet_content_width == 71  # 80 - 9
        assert dashboard.output_content_width == 74  # 80 - 6
    
    def test_message_persistence(self):
        """Test that message filtering tracks display time."""
        dashboard = FitDashboard(
            initial_params={'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036},
            max_iter=100,
            min_message_display_time=0.1,  # 0.1 seconds
            max_warnings=2,
        )
        
        # Use dashboard's actual tracking dict
        tracking = dashboard._warnings_tracking
        current_time = time.time()
        messages = ["Warning 1", "Warning 2", "Warning 3"]
        
        # First call - all messages are new
        filtered = dashboard._filter_messages_by_time(messages, tracking, current_time)
        
        # Should track all messages
        assert len(tracking) == 3
        
        # Should limit to max slots (2)
        assert len(filtered) == 2
        
        # Wait for min display time
        time.sleep(0.15)
        
        # Now only keep first message, add new one
        new_messages = ["Warning 1", "Warning 4"]
        new_time = time.time()
        
        filtered = dashboard._filter_messages_by_time(new_messages, tracking, new_time)
        
        # Warning 1 should persist (old enough), Warning 4 is new
        assert "Warning 1" in filtered
    
    def test_dashboard_height_calculation(self):
        """Test that dashboard height is calculated correctly."""
        # Single stage, no output
        dashboard = FitDashboard(
            initial_params={'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036},
            max_iter=100,
            max_warnings=3,
            max_hints=3,
        )
        expected = 16 + (2 + 3) + (2 + 3) + 1  # base + warnings + hints + border
        assert dashboard.dashboard_height == expected
        
        # Multi-stage, with output
        dashboard_multi = FitDashboard(
            initial_params={'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036},
            max_iter=100,
            total_stages=3,
            max_warnings=3,
            max_hints=3,
            output_capture=OutputCapture(),
            show_output=True,
            max_output_lines=3,
        )
        expected_multi = 17 + (2 + 3) + (2 + 3) + (2 + 3) + 1
        assert dashboard_multi.dashboard_height == expected_multi
    
    def test_border_helpers(self):
        """Test border and line formatting helpers."""
        dashboard = FitDashboard(
            initial_params={'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036},
            max_iter=100,
            width=63,
        )
        
        # Test borders
        assert len(dashboard._border_top()) == 63
        assert len(dashboard._border_middle()) == 63
        assert len(dashboard._border_bottom()) == 63
        
        # Test lines
        assert len(dashboard._line("Test")) == 63
        assert len(dashboard._line_bullet("Test")) == 63
        assert len(dashboard._line_indent("Test")) == 63
        assert len(dashboard._blank_line()) == 63
    
    def test_finalize(self, basic_dashboard):
        """Test dashboard finalization."""
        params = {'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036}
        basic_dashboard.update(params, 1000)
        
        basic_dashboard.finalize(success=True)
        assert basic_dashboard.converged is True


class TestDashboardKwargsIntegration:
    """Tests for dashboard_kwargs passthrough to fit_limb."""
    
    def test_kwargs_passthrough(self):
        """Test that dashboard_kwargs are properly passed through."""
        from planet_ruler import LimbObservation
        import inspect
        
        # Check fit_limb signature
        sig = inspect.signature(LimbObservation.fit_limb)
        assert 'dashboard_kwargs' in sig.parameters
        assert sig.parameters['dashboard_kwargs'].default is None
    
    def test_kwargs_applied_to_dashboard(self):
        """Test that kwargs are applied when creating dashboard."""
        dashboard_kwargs = {
            'max_warnings': 5,
            'max_hints': 4,
            'min_message_display_time': 5.0,
            'width': 80,
        }
        
        dashboard = FitDashboard(
            initial_params={'r': 6371000, 'h': 30000, 'f': 0.025, 'w': 0.036},
            max_iter=100,
            **dashboard_kwargs
        )
        
        assert dashboard.max_warnings == 5
        assert dashboard.max_hints == 4
        assert dashboard.min_message_display_time == 5.0
        assert dashboard.width == 80


if __name__ == '__main__':
    pytest.main([__file__, '-v'])