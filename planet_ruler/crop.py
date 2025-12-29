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
Image Cropping Tool for Planet Ruler

Allows users to select a rectangular region to crop from a horizon image,
automatically adjusting camera parameters (detector size, principal point, dimensions).
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


logger = logging.getLogger(__name__)


class TkImageCropper:
    """
    Tkinter-based interactive tool for cropping images with automatic parameter scaling.

    Features:
    - Drag to select rectangular crop region
    - Zoom with scroll wheel
    - Scrollable canvas for navigation
    - Automatic parameter scaling for detector_size (w), principal point (x0, y0), and dimensions
    - Save cropped image and adjusted parameters
    """

    def __init__(
        self,
        image_path: str,
        initial_parameters: Optional[Dict] = None,
        initial_zoom: Optional[float] = None,
    ):
        """
        Initialize the crop tool.

        Args:
            image_path: Path to the image to crop
            initial_parameters: Dictionary of current parameters (must include 'w' if provided)
            initial_zoom: Initial zoom level (None = auto-fit)
        """
        self.image_path = image_path
        self.initial_parameters = initial_parameters or {}

        # Load image
        self.original_image = Image.open(image_path)
        self.width, self.height = self.original_image.size

        # Crop rectangle state (in original coordinates)
        self.crop_rect = None  # (x_start, y_start, x_end, y_end)
        self.drag_start = None  # (x, y) start of current drag
        self.rect_id = None  # Canvas rectangle object

        # Results (set after crop is confirmed)
        self.cropped_image = None
        self.scaled_parameters = None

        # Setup main window
        self.root = tk.Tk()
        self.root.title(f"Planet Ruler - Crop Image - {Path(image_path).name}")
        self.root.geometry("1400x900")

        # Zoom level
        self.zoom_level = initial_zoom if initial_zoom is not None else 1.0

        # Create UI
        self.create_widgets()

        # Auto-fit zoom if requested
        if initial_zoom is None:
            self.auto_fit_zoom()
        else:
            self.update_display()

    def create_widgets(self):
        """Create all UI widgets."""
        # Top frame for controls and status
        top_frame = tk.Frame(self.root, bg="lightgray", pady=8, padx=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Status label
        self.status_label = tk.Label(
            top_frame,
            text=self.get_status_text(),
            font=("Arial", 11),
            bg="lightgray",
            fg="black",
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Buttons
        button_config = {
            "font": ("Arial", 11, "bold"),
            "bg": "#4CAF50",
            "fg": "black",
            "activebackground": "#45a049",
            "activeforeground": "black",
            "relief": tk.RAISED,
            "bd": 3,
            "padx": 12,
            "pady": 6,
        }

        btn_crop = tk.Button(
            top_frame,
            text="Crop & Save",
            command=self.crop_and_save,
            **button_config,
        )
        btn_crop.pack(side=tk.LEFT, padx=3)
        
        btn_crop_only = tk.Button(
            top_frame,
            text="Crop (No Save)",
            command=self.crop_only,
            **button_config,
        )
        btn_crop_only.pack(side=tk.LEFT, padx=3)

        clear_config = button_config.copy()
        clear_config.update({"bg": "#f44336", "activebackground": "#da190b"})
        btn_clear = tk.Button(
            top_frame, text="Clear Selection", command=self.clear_selection, **clear_config
        )
        btn_clear.pack(side=tk.LEFT, padx=3)

        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas frame with scrollbars
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", cursor="crosshair")

        # Scrollbars
        v_scroll = tk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        h_scroll = tk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )

        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Right side controls
        controls_frame = tk.Frame(main_frame, bg="lightgray", padx=15, pady=10)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Zoom controls
        tk.Label(
            controls_frame, text="ZOOM", font=("Arial", 12, "bold"), bg="lightgray"
        ).pack(pady=(0, 10))

        self.zoom_label = tk.Label(
            controls_frame, text="100%", font=("Arial", 11, "bold"), bg="lightgray"
        )
        self.zoom_label.pack(pady=5)

        zoom_btn_config = {
            "font": ("Arial", 11, "bold"),
            "bg": "#2196F3",
            "fg": "black",
            "activebackground": "#0b7dda",
            "activeforeground": "black",
            "width": 12,
            "pady": 6,
            "relief": tk.RAISED,
            "bd": 3,
        }

        btn_zoom_in = tk.Button(
            controls_frame,
            text="Zoom In (+)",
            command=lambda: self.adjust_zoom(1.2),
            **zoom_btn_config,
        )
        btn_zoom_in.pack(pady=3)

        btn_zoom_out = tk.Button(
            controls_frame,
            text="Zoom Out (-)",
            command=lambda: self.adjust_zoom(1 / 1.2),
            **zoom_btn_config,
        )
        btn_zoom_out.pack(pady=3)

        btn_fit = tk.Button(
            controls_frame,
            text="Fit to Window",
            command=self.auto_fit_zoom,
            **zoom_btn_config,
        )
        btn_fit.pack(pady=3)

        btn_100 = tk.Button(
            controls_frame,
            text="100% (1:1)",
            command=lambda: self.set_zoom(1.0),
            **zoom_btn_config,
        )
        btn_100.pack(pady=3)

        # Instructions
        instructions = (
            "Click & Drag: Select crop region  |  "
            "Scroll Wheel: Zoom  |  "
            "Crop & Save: Save to disk  |  "
            "Crop (No Save): In-memory only"
        )
        tk.Label(
            self.root,
            text=instructions,
            relief=tk.SUNKEN,
            font=("Arial", 10),
            bg="white",
        ).pack(side=tk.BOTTOM, fill=tk.X)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)

        # Bind scroll wheel for zoom
        self.canvas.bind("<MouseWheel>", self.on_scroll_zoom)
        self.canvas.bind("<Button-4>", self.on_scroll_zoom)
        self.canvas.bind("<Button-5>", self.on_scroll_zoom)

        # Keyboard shortcuts
        self.root.bind("<plus>", lambda e: self.adjust_zoom(1.2))
        self.root.bind("<minus>", lambda e: self.adjust_zoom(1 / 1.2))
        self.root.bind("<equal>", lambda e: self.adjust_zoom(1.2))
        self.root.bind("<Escape>", lambda e: self.clear_selection())

    def auto_fit_zoom(self):
        """Automatically set zoom to fit image in window."""
        canvas_width = max(800, self.canvas.winfo_width())
        canvas_height = max(600, self.canvas.winfo_height())

        zoom_x = canvas_width / self.width
        zoom_y = canvas_height / self.height

        self.zoom_level = max(0.05, min(zoom_x, zoom_y, 1.0))
        self.update_display()

    def set_zoom(self, zoom: float):
        """Set absolute zoom level."""
        self.zoom_level = max(0.05, min(5.0, zoom))
        self.update_display()

    def adjust_zoom(self, factor: float):
        """Adjust zoom by a multiplicative factor."""
        self.set_zoom(self.zoom_level * factor)

    def on_scroll_zoom(self, event):
        """Handle scroll wheel for zooming."""
        if event.num == 4 or event.delta > 0:
            factor = 1.1
        elif event.num == 5 or event.delta < 0:
            factor = 1 / 1.1
        else:
            return

        self.adjust_zoom(factor)

    def update_display(self):
        """Update the displayed image with current zoom."""
        # Calculate dimensions
        zoomed_width = max(1, int(self.width * self.zoom_level))
        zoomed_height = max(1, int(self.height * self.zoom_level))

        # Resize image
        try:
            resized = self.original_image.resize(
                (zoomed_width, zoomed_height), Image.Resampling.LANCZOS
            )
        except AttributeError:
            resized = self.original_image.resize(
                (zoomed_width, zoomed_height), Image.LANCZOS
            )

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(resized)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo, tags="image")
        self.canvas.configure(scrollregion=(0, 0, zoomed_width, zoomed_height))

        # Update labels
        self.zoom_label.config(text=f"{self.zoom_level*100:.0f}%")

        # Redraw crop rectangle if exists
        self.redraw_crop_rect()
        self.update_status()

    def redraw_crop_rect(self):
        """Redraw crop rectangle at current zoom."""
        # Delete old crop rectangle first (prevents echoes during drag)
        self.canvas.delete("crop_rect")
        
        if self.crop_rect is None:
            return

        x1, y1, x2, y2 = self.crop_rect

        # Convert to display coordinates
        x1_disp = x1 * self.zoom_level
        y1_disp = y1 * self.zoom_level
        x2_disp = x2 * self.zoom_level
        y2_disp = y2 * self.zoom_level

        # Draw rectangle
        self.rect_id = self.canvas.create_rectangle(
            x1_disp,
            y1_disp,
            x2_disp,
            y2_disp,
            outline="yellow",
            width=3,
            tags="crop_rect",
        )

        # Draw corner handles
        r = 5  # handle radius
        for x, y in [(x1_disp, y1_disp), (x2_disp, y1_disp), 
                     (x1_disp, y2_disp), (x2_disp, y2_disp)]:
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill="yellow",
                outline="black",
                width=2,
                tags="crop_rect",
            )

    def on_drag_start(self, event):
        """Start dragging to define crop rectangle."""
        # Get canvas coordinates
        x_display = self.canvas.canvasx(event.x)
        y_display = self.canvas.canvasy(event.y)

        # Convert to original coordinates
        x_orig = x_display / self.zoom_level
        y_orig = y_display / self.zoom_level

        # Validate coordinates
        if 0 <= x_orig < self.width and 0 <= y_orig < self.height:
            self.drag_start = (x_orig, y_orig)

    def on_drag_motion(self, event):
        """Update crop rectangle during drag."""
        if self.drag_start is None:
            return

        # Get current position
        x_display = self.canvas.canvasx(event.x)
        y_display = self.canvas.canvasy(event.y)

        x_orig = x_display / self.zoom_level
        y_orig = y_display / self.zoom_level

        # Clamp to image bounds
        x_orig = max(0, min(self.width - 1, x_orig))
        y_orig = max(0, min(self.height - 1, y_orig))

        # Update crop rectangle
        x1, y1 = self.drag_start
        x2, y2 = x_orig, y_orig

        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        self.crop_rect = (x1, y1, x2, y2)
        self.redraw_crop_rect()
        self.update_status()

    def on_drag_end(self, event):
        """Finish dragging."""
        self.drag_start = None

    def clear_selection(self):
        """Clear the crop selection."""
        self.crop_rect = None
        self.drag_start = None
        self.canvas.delete("crop_rect")
        self.update_status()
    
    def update_status(self):
        """Update status text."""
        self.status_label.config(text=self.get_status_text())

    def get_status_text(self) -> str:
        """Generate status text."""
        base = f"Original: {self.width}×{self.height}px | Zoom: {self.zoom_level*100:.0f}%"
        
        if self.crop_rect:
            x1, y1, x2, y2 = self.crop_rect
            w = int(x2 - x1)
            h = int(y2 - y1)
            base += f" | Crop: {w}×{h}px ({100*w/self.width:.1f}% × {100*h/self.height:.1f}%)"
        else:
            base += " | No selection"
        
        return base

    def calculate_scaled_parameters(self) -> Dict:
        """
        Calculate scaled parameters after cropping.

        Returns:
            Dictionary of scaled parameters
        """
        if self.crop_rect is None:
            raise ValueError("No crop region selected")

        x1, y1, x2, y2 = self.crop_rect
        crop_width = x2 - x1
        crop_height = y2 - y1

        # Calculate scale factors
        scale_x = crop_width / self.width
        scale_y = crop_height / self.height

        # Start with initial parameters
        scaled = self.initial_parameters.copy()

        # Update pixel dimensions
        scaled['n_pix_x'] = int(crop_width)
        scaled['n_pix_y'] = int(crop_height)

        # Update principal point (shift and scale)
        if 'x0' in scaled:
            scaled['x0'] = (scaled['x0'] - x1)
        else:
            # Default principal point is center
            scaled['x0'] = crop_width / 2

        if 'y0' in scaled:
            scaled['y0'] = (scaled['y0'] - y1)
        else:
            scaled['y0'] = crop_height / 2

        # CRITICAL: Update detector size (effective field of view)
        # The detector width represents the physical sensor dimension
        # When cropping, we're using a fraction of that sensor
        if 'w' in scaled:
            # Scale detector width by crop fraction
            scaled['w'] = scaled['w'] * scale_x
            logger.info(
                f"Scaled detector width: {self.initial_parameters['w']*1000:.3f}mm "
                f"→ {scaled['w']*1000:.3f}mm (crop ratio: {scale_x:.3f})"
            )
        
        # Similarly for height if provided
        if 'h_detector' in scaled:
            scaled['h_detector'] = scaled['h_detector'] * scale_y

        # Focal length (f) does NOT change - it's a lens property
        # Field of view will automatically update via: fov = 2 * arctan(w / (2*f))

        return scaled

    def crop_and_save(self):
        """Perform crop and optionally save results."""
        if self.crop_rect is None:
            messagebox.showwarning(
                "No Selection", "Please select a crop region first"
            )
            return

        x1, y1, x2, y2 = self.crop_rect

        # Ensure integer coordinates
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # Perform crop
        self.cropped_image = self.original_image.crop((x1, y1, x2, y2))

        # Calculate scaled parameters
        self.scaled_parameters = self.calculate_scaled_parameters()

        # Save cropped image
        output_path = (
            Path(self.image_path).parent / f"{Path(self.image_path).stem}_cropped.jpg"
        )
        self.cropped_image.save(output_path, quality=95)

        # Prepare message
        msg = (
            f"Cropped image saved!\n\n"
            f"Original: {self.width}×{self.height}px\n"
            f"Cropped: {x2-x1}×{y2-y1}px\n"
            f"Crop region: ({x1}, {y1}) to ({x2}, {y2})\n\n"
            f"Saved to: {output_path}\n\n"
        )

        if 'w' in self.scaled_parameters:
            msg += (
                f"Scaled Parameters:\n"
                f"  detector_width: {self.initial_parameters['w']*1000:.3f}mm "
                f"→ {self.scaled_parameters['w']*1000:.3f}mm\n"
                f"  n_pix_x: {self.width} → {self.scaled_parameters['n_pix_x']}\n"
                f"  n_pix_y: {self.height} → {self.scaled_parameters['n_pix_y']}\n"
                f"  x0: {self.initial_parameters.get('x0', self.width/2):.1f} "
                f"→ {self.scaled_parameters['x0']:.1f}\n"
                f"  y0: {self.initial_parameters.get('y0', self.height/2):.1f} "
                f"→ {self.scaled_parameters['y0']:.1f}\n"
            )

        messagebox.showinfo("Crop Complete", msg)
        
        # Don't close window - let user close manually like annotate tool
    
    def crop_only(self):
        """Perform crop without saving to disk (in-memory only)."""
        if self.crop_rect is None:
            messagebox.showwarning(
                "No Selection", "Please select a crop region first"
            )
            return

        x1, y1, x2, y2 = self.crop_rect

        # Ensure integer coordinates
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # Perform crop
        self.cropped_image = self.original_image.crop((x1, y1, x2, y2))

        # Calculate scaled parameters
        self.scaled_parameters = self.calculate_scaled_parameters()

        # Prepare message
        msg = (
            f"Crop complete (not saved to disk)\n\n"
            f"Original: {self.width}×{self.height}px\n"
            f"Cropped: {x2-x1}×{y2-y1}px\n"
            f"Crop region: ({x1}, {y1}) to ({x2}, {y2})\n\n"
        )

        if 'w' in self.scaled_parameters:
            msg += (
                f"Scaled Parameters:\n"
                f"  detector_width: {self.initial_parameters['w']*1000:.3f}mm "
                f"→ {self.scaled_parameters['w']*1000:.3f}mm\n"
                f"  n_pix_x: {self.width} → {self.scaled_parameters['n_pix_x']}\n"
                f"  n_pix_y: {self.height} → {self.scaled_parameters['n_pix_y']}\n"
                f"  x0: {self.initial_parameters.get('x0', self.width/2):.1f} "
                f"→ {self.scaled_parameters['x0']:.1f}\n"
                f"  y0: {self.initial_parameters.get('y0', self.height/2):.1f} "
                f"→ {self.scaled_parameters['y0']:.1f}\n"
            )

        messagebox.showinfo("Crop Complete", msg)
        
        # Don't close window - let user close manually like annotate tool

    def run(self):
        """Run the crop tool (blocking)."""
        self.root.mainloop()

    def get_crop_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """Get crop bounds in original coordinates."""
        if self.crop_rect is None:
            return None
        
        x1, y1, x2, y2 = self.crop_rect
        return (int(x1), int(y1), int(x2), int(y2))

    def get_scaled_parameters(self) -> Optional[Dict]:
        """Get scaled parameters after cropping."""
        return self.scaled_parameters


# Convenience function for use in LimbObservation
def crop_observation_image(
    image_path: str,
    initial_parameters: Dict,
) -> Tuple[Image.Image, Dict, Tuple[int, int, int, int]]:
    """
    Interactively crop an observation image and get scaled parameters.

    Args:
        image_path: Path to image file
        initial_parameters: Current observation parameters (must include 'w')

    Returns:
        Tuple of (cropped_image, scaled_parameters, crop_bounds)
        crop_bounds is (x1, y1, x2, y2) in original coordinates
    """
    cropper = TkImageCropper(image_path, initial_parameters)
    cropper.run()

    if cropper.cropped_image is None:
        raise ValueError("Crop operation cancelled - no region selected")

    return (
        cropper.cropped_image,
        cropper.scaled_parameters,
        cropper.get_crop_bounds(),
    )


# Example usage
if __name__ == "__main__":
    import sys

    print("Tkinter Image Crop Tool")
    print("=" * 60)
    print("\nControls:")
    print("  Click & Drag:    Select crop region")
    print("  Scroll Wheel:    Zoom in/out")
    print("  +/- keys:        Zoom in/out")
    print("  Esc:             Clear selection")
    print("\nButtons:")
    print("  Zoom: Fit to Window, 100%, +/-")
    print("  Crop & Save: Apply crop and save results")
    print("  Clear Selection: Remove crop region")
    print("\n" + "=" * 60)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("\nUsage: python crop.py <image_path>")
        print("\nOr in your code:")
        print("  from crop import TkImageCropper")
        print("  cropper = TkImageCropper('image.jpg', {'w': 0.036})")
        print("  cropper.run()")
        sys.exit(0)

    # Example parameters
    example_params = {
        'w': 0.0236,  # 23.6mm detector width
        'n_pix_x': 4000,
        'n_pix_y': 3000,
        'x0': 2000,
        'y0': 1500,
    }

    cropper = TkImageCropper(image_path, example_params)
    cropper.run()

    if cropper.cropped_image:
        print(f"\nCrop successful!")
        print(f"Scaled parameters: {cropper.scaled_parameters}")