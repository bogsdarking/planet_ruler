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
Manual Limb Annotation Tool for Planet Ruler

Allows users to click points on a horizon image and generate a sparse target
for fitting with the existing planet_ruler pipeline.
"""

import numpy as np
import json
import logging
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


class ToolTip:
    """Simple tooltip that appears on hover."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None

        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return

        # Position tooltip near the widget
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            fg="black",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Arial", 9),
            padx=5,
            pady=3,
        )
        label.pack()

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


def create_tooltip(widget, text):
    """Helper function to create tooltip for a widget."""
    return ToolTip(widget, text)


class TkLimbAnnotator:
    """
    Tkinter-based interactive tool for manually annotating planet limbs.

    Features:
    - Zoom with scroll wheel (fit large images in window)
    - Vertical stretch buttons (stretch pixels vertically for precision)
    - Scrollable canvas for navigating
    - Click to add points, right-click to undo
    - Save/load points to JSON
    - Generate sparse target array for CostFunction
    """

    def __init__(self, image_path, initial_stretch=1.0, initial_zoom=None):
        """
        Initialize the annotation tool.

        Args:
            image_path (str): Path to the image to annotate
            initial_stretch (float): Initial vertical stretch factor
            initial_zoom (float): Initial zoom level (None = auto-fit)
        """
        self.image_path = image_path
        self.vertical_stretch = initial_stretch

        # Load image
        self.original_image = Image.open(image_path)
        self.width, self.height = self.original_image.size

        # Store clicked points (in original coordinates)
        self.points = []  # List of (x, y) tuples

        # Setup main window
        self.root = tk.Tk()
        self.root.title(f"Planet Ruler - Limb Annotation - {Path(image_path).name}")
        self.root.geometry("1400x900")

        # Zoom level - start at 1.0, will be adjusted after widgets created
        self.zoom_level = initial_zoom if initial_zoom is not None else 1.0

        # Create UI
        self.create_widgets()

        # Auto-fit zoom if requested
        if initial_zoom is None:
            self.auto_fit_zoom()
        else:
            self.update_stretched_image()

    def create_widgets(self):
        """Create all UI widgets."""
        # Top frame for controls
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

        # Buttons with explicit styling and better visibility
        button_config = {
            "font": ("Arial", 11, "bold"),
            "bg": "#4CAF50",
            "fg": "black",  # Changed to black for better contrast
            "activebackground": "#45a049",
            "activeforeground": "black",
            "relief": tk.RAISED,
            "bd": 3,
            "padx": 12,
            "pady": 6,
            "highlightbackground": "#2E7D32",
            "highlightthickness": 2,
            "highlightcolor": "#2E7D32",
        }

        btn_generate = tk.Button(
            top_frame,
            text="Generate Target",
            command=self.generate_target,
            **button_config,
        )
        btn_generate.pack(side=tk.LEFT, padx=3)
        create_tooltip(
            btn_generate,
            "Create sparse target array from annotated points\n"
            "for use with CostFunction",
        )

        btn_save = tk.Button(
            top_frame, text="Save Points", command=self.save_points, **button_config
        )
        btn_save.pack(side=tk.LEFT, padx=3)
        create_tooltip(
            btn_save, "Save current points to JSON file\n" "for later loading"
        )

        btn_load = tk.Button(
            top_frame, text="Load Points", command=self.load_points, **button_config
        )
        btn_load.pack(side=tk.LEFT, padx=3)
        create_tooltip(btn_load, "Load previously saved points from JSON file")

        clear_config = button_config.copy()
        clear_config.update(
            {
                "bg": "#f44336",
                "activebackground": "#da190b",
                "highlightbackground": "#b71c1c",
                "fg": "black",
                "activeforeground": "black",
            }
        )
        btn_clear = tk.Button(
            top_frame, text="Clear All", command=self.clear_all, **clear_config
        )
        btn_clear.pack(side=tk.LEFT, padx=3)
        create_tooltip(btn_clear, "Remove all annotated points")

        # Main container for canvas and controls
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

        # Grid layout for canvas and scrollbars
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Right side controls frame
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
            "highlightbackground": "#1565C0",
            "highlightthickness": 2,
        }

        btn_zoom_in = tk.Button(
            controls_frame,
            text="Zoom In (+)",
            command=lambda: self.adjust_zoom(1.2),
            **zoom_btn_config,
        )
        btn_zoom_in.pack(pady=3)
        create_tooltip(btn_zoom_in, "Zoom in to see more detail\n(or use scroll wheel)")

        btn_zoom_out = tk.Button(
            controls_frame,
            text="Zoom Out (-)",
            command=lambda: self.adjust_zoom(1 / 1.2),
            **zoom_btn_config,
        )
        btn_zoom_out.pack(pady=3)
        create_tooltip(
            btn_zoom_out, "Zoom out to see more of image\n(or use scroll wheel)"
        )

        btn_fit = tk.Button(
            controls_frame,
            text="Fit to Window",
            command=self.auto_fit_zoom,
            **zoom_btn_config,
        )
        btn_fit.pack(pady=3)
        create_tooltip(btn_fit, "Auto-fit entire image to window")

        btn_100 = tk.Button(
            controls_frame,
            text="100% (1:1)",
            command=lambda: self.set_zoom(1.0),
            **zoom_btn_config,
        )
        btn_100.pack(pady=3)
        create_tooltip(btn_100, "Reset to actual image size (1 pixel = 1 pixel)")

        tk.Label(
            controls_frame,
            text="(or use scroll wheel)",
            font=("Arial", 8),
            bg="lightgray",
        ).pack(pady=5)

        # Separator
        tk.Frame(controls_frame, height=2, bg="gray").pack(fill=tk.X, pady=15)

        # Vertical stretch controls
        tk.Label(
            controls_frame,
            text="VERTICAL\nSTRETCH",
            font=("Arial", 12, "bold"),
            bg="lightgray",
            justify=tk.CENTER,
        ).pack(pady=(0, 10))

        self.stretch_label = tk.Label(
            controls_frame, text="1.0x", font=("Arial", 11, "bold"), bg="lightgray"
        )
        self.stretch_label.pack(pady=5)

        stretch_btn_config = {
            "font": ("Arial", 11, "bold"),
            "bg": "#FF9800",
            "fg": "black",
            "activebackground": "#e68900",
            "activeforeground": "black",
            "width": 12,
            "pady": 6,
            "relief": tk.RAISED,
            "bd": 3,
            "highlightbackground": "#E65100",
            "highlightthickness": 2,
        }

        btn_stretch_inc = tk.Button(
            controls_frame,
            text="Increase (+)",
            command=lambda: self.adjust_stretch(0.5),
            **stretch_btn_config,
        )
        btn_stretch_inc.pack(pady=3)
        create_tooltip(
            btn_stretch_inc,
            "Increase vertical stretch\n" "Makes subtle horizon curves easier to see",
        )

        btn_stretch_dec = tk.Button(
            controls_frame,
            text="Decrease (-)",
            command=lambda: self.adjust_stretch(-0.5),
            **stretch_btn_config,
        )
        btn_stretch_dec.pack(pady=3)
        create_tooltip(btn_stretch_dec, "Decrease vertical stretch")

        btn_stretch_reset = tk.Button(
            controls_frame,
            text="Reset (1x)",
            command=lambda: self.set_stretch(1.0),
            **stretch_btn_config,
        )
        btn_stretch_reset.pack(pady=3)
        create_tooltip(btn_stretch_reset, "Reset to normal aspect ratio (no stretch)")

        tk.Label(
            controls_frame,
            text="Stretches height\nfor precision",
            font=("Arial", 8),
            bg="lightgray",
            justify=tk.CENTER,
        ).pack(pady=10)

        # Instructions at bottom
        instructions = (
            "Left Click: Add point  |  Right Click: Undo  |  "
            "Scroll Wheel: Zoom  |  Click & Drag: Pan"
        )
        tk.Label(
            self.root,
            text=instructions,
            relief=tk.SUNKEN,
            font=("Arial", 10),
            bg="white",
        ).pack(side=tk.BOTTOM, fill=tk.X)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        # Bind scroll wheel for zoom
        self.canvas.bind("<MouseWheel>", self.on_scroll_zoom)  # Windows/Mac
        self.canvas.bind("<Button-4>", self.on_scroll_zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_scroll_zoom)  # Linux scroll down

        # Bind keyboard shortcuts
        self.root.bind("<plus>", lambda e: self.adjust_zoom(1.2))
        self.root.bind("<minus>", lambda e: self.adjust_zoom(1 / 1.2))
        self.root.bind("<equal>", lambda e: self.adjust_zoom(1.2))  # + without shift

    def auto_fit_zoom(self):
        """Automatically set zoom to fit image in window."""
        # Get canvas size (use reasonable defaults if not yet rendered)
        canvas_width = max(800, self.canvas.winfo_width())
        canvas_height = max(600, self.canvas.winfo_height())

        # Calculate zoom to fit
        zoom_x = canvas_width / self.width
        zoom_y = canvas_height / (self.height * self.vertical_stretch)

        # Use smaller zoom to ensure both dimensions fit
        # Ensure minimum zoom of 0.05 to prevent 0-size images
        self.zoom_level = max(0.05, min(zoom_x, zoom_y, 1.0))

        self.update_stretched_image()

    def set_zoom(self, zoom):
        """Set absolute zoom level."""
        self.zoom_level = max(0.05, min(5.0, zoom))  # Clamp to 5%-500%
        self.update_stretched_image()

    def adjust_zoom(self, factor):
        """Adjust zoom by a multiplicative factor."""
        self.set_zoom(self.zoom_level * factor)

    def on_scroll_zoom(self, event):
        """Handle scroll wheel for zooming."""
        # Determine scroll direction
        if event.num == 4 or event.delta > 0:  # Scroll up = zoom in
            factor = 1.1
        elif event.num == 5 or event.delta < 0:  # Scroll down = zoom out
            factor = 1 / 1.1
        else:
            return

        self.adjust_zoom(factor)

    def set_stretch(self, stretch):
        """Set absolute stretch level."""
        self.vertical_stretch = max(1.0, min(20.0, stretch))  # Clamp to 1-20x
        self.update_stretched_image()

    def adjust_stretch(self, delta):
        """Adjust stretch by an additive amount."""
        self.set_stretch(self.vertical_stretch + delta)

    def update_stretched_image(self):
        """Update the displayed image with current zoom and stretch."""
        # Calculate final dimensions: zoom first, then stretch
        zoomed_width = max(1, int(self.width * self.zoom_level))
        zoomed_height = max(1, int(self.height * self.zoom_level))
        stretched_height = max(1, int(zoomed_height * self.vertical_stretch))

        # Resize image (zoom, then stretch)
        try:
            # Try new PIL.Image.Resampling.LANCZOS first
            zoomed = self.original_image.resize(
                (zoomed_width, zoomed_height), Image.Resampling.LANCZOS
            )
            stretched = zoomed.resize(
                (zoomed_width, stretched_height), Image.Resampling.LANCZOS
            )
        except AttributeError:
            # Fall back to older PIL.Image.LANCZOS for compatibility
            zoomed = self.original_image.resize(
                (zoomed_width, zoomed_height), Image.LANCZOS
            )
            stretched = zoomed.resize((zoomed_width, stretched_height), Image.LANCZOS)

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(stretched)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo, tags="image")

        # Set scrollable region
        self.canvas.configure(scrollregion=(0, 0, zoomed_width, stretched_height))

        # Update labels
        self.zoom_label.config(text=f"{self.zoom_level*100:.0f}%")
        self.stretch_label.config(text=f"{self.vertical_stretch:.1f}x")

        # Redraw all points
        self.redraw_points()
        self.update_status()

    def redraw_points(self):
        """Redraw all annotation points at current zoom and stretch."""
        self.canvas.delete("point")
        self.canvas.delete("label")

        for i, (x, y_orig) in enumerate(self.points):
            # Convert to display coordinates (zoom + stretch)
            x_display = x * self.zoom_level
            y_display = y_orig * self.zoom_level * self.vertical_stretch

            # Draw point (size scales with zoom)
            r = max(4, int(5 * self.zoom_level))
            self.canvas.create_oval(
                x_display - r,
                y_display - r,
                x_display + r,
                y_display + r,
                fill="red",
                outline="yellow",
                width=2,
                tags="point",
            )

            # Draw label
            font_size = max(9, int(11 * self.zoom_level))
            self.canvas.create_text(
                x_display,
                y_display - 15,
                text=str(i + 1),
                fill="yellow",
                font=("Arial", font_size, "bold"),
                tags="label",
            )

    def on_left_click(self, event):
        """Add a point at click location."""
        # Get canvas coordinates (accounting for scroll)
        x_display = self.canvas.canvasx(event.x)
        y_display = self.canvas.canvasy(event.y)

        # Convert from display to original coordinates
        x_original = x_display / self.zoom_level
        y_original = y_display / (self.zoom_level * self.vertical_stretch)

        # Validate coordinates
        if 0 <= x_original < self.width and 0 <= y_original < self.height:
            self.points.append((x_original, y_original))
            self.redraw_points()
            self.update_status()

    def on_right_click(self, event):
        """Undo last point."""
        if self.points:
            self.points.pop()
            self.redraw_points()
            self.update_status()

    def clear_all(self):
        """Clear all points."""
        if self.points and messagebox.askyesno("Clear All", "Remove all points?"):
            self.points = []
            self.redraw_points()
            self.update_status()

    def update_status(self):
        """Update status text."""
        self.status_label.config(text=self.get_status_text())

    def get_status_text(self):
        """Generate status text."""
        return (
            f"Points: {len(self.points)} | "
            f"Image: {self.width}×{self.height}px | "
            f"Zoom: {self.zoom_level*100:.0f}% | "
            f"Stretch: {self.vertical_stretch:.1f}x"
        )

    def generate_target(self):
        """Generate sparse target array."""
        if len(self.points) < 3:
            messagebox.showwarning(
                "Insufficient Points", "Need at least 3 points to generate target"
            )
            return

        # Create sparse target array (in ORIGINAL coordinates)
        target = np.full(self.width, np.nan)

        # Fill in clicked positions
        for x, y in self.points:
            x_idx = int(round(x))
            if 0 <= x_idx < self.width:
                target[x_idx] = y

        # Save target to file
        output_path = (
            Path(self.image_path).parent / f"{Path(self.image_path).stem}_target.npy"
        )
        np.save(output_path, target)

        # Report statistics
        n_valid = np.sum(~np.isnan(target))
        coverage = 100 * n_valid / self.width

        msg = (
            f"Generated sparse target array\n\n"
            f"Shape: {target.shape}\n"
            f"Valid points: {n_valid}/{self.width} ({coverage:.1f}%)\n"
            f"Y range: [{np.nanmin(target):.1f}, {np.nanmax(target):.1f}]\n\n"
            f"Saved to: {output_path}\n\n"
            f"Usage:\n"
            f"target = np.load('{output_path.name}')\n"
            f"cost_fn = CostFunction(target=target, ...)"
        )

        messagebox.showinfo("Target Generated", msg)

        return target

    def save_points(self):
        """Save points to JSON."""
        if not self.points:
            messagebox.showwarning("No Points", "No points to save")
            return

        output_path = (
            Path(self.image_path).parent
            / f"{Path(self.image_path).stem}_limb_points.json"
        )

        data = {
            "image_path": str(self.image_path),
            "image_size": [self.width, self.height],
            "points": [(float(x), float(y)) for x, y in self.points],
            "n_points": len(self.points),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        messagebox.showinfo(
            "Saved", f"Saved {len(self.points)} points to:\n{output_path}"
        )

    def load_points(self):
        """Load points from JSON."""
        json_path = filedialog.askopenfilename(
            title="Load Points",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not json_path:
            return

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            self.points = [(x, y) for x, y in data["points"]]
            self.redraw_points()
            self.update_status()

            messagebox.showinfo(
                "Loaded", f"Loaded {len(self.points)} points from:\n{json_path}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load points:\n{str(e)}")

    def get_target(self):
        """Get the current sparse target array."""
        if len(self.points) < 3:
            return None

        target = np.full(self.width, np.nan)
        for x, y in self.points:
            x_idx = int(round(x))
            if 0 <= x_idx < self.width:
                target[x_idx] = y
        return target

    def run(self):
        """Start the application."""
        # Show instructions on startup
        instructions = (
            "Welcome to the Limb Annotation Tool!\n\n"
            "Quick Instructions:\n\n"
            "1. Click 5-10 points along the horizon/limb\n"
            "   • Left click to add a point\n"
            "   • Right click to undo last point\n\n"
            "2. Use ZOOM (scroll wheel or buttons) to navigate large images\n\n"
            "3. Use STRETCH (buttons) to exaggerate vertical curvature\n"
            "   • Makes subtle horizon curves easier to see and click accurately\n"
            "   • All coordinates are saved in original image space\n\n"
            "4. When satisfied, click 'Generate Target'\n\n"
            "5. Close the window when done\n\n"
            "Hover over any button for more details!"
        )

        messagebox.showinfo("Limb Annotation Instructions", instructions)

        self.root.mainloop()


class TkMaskSelector:
    """
    Interactive mask classification tool for segmentation results.

    Works with ANY segmentation method - completely backend-agnostic.
    Allows users to classify masks as 'planet', 'sky', or 'exclude' for
    horizon detection. Aligns with Planet Ruler's educational philosophy
    of transparency over black-box automation.

    Args:
        image: Original image array (H x W x 3)
        masks: List of masks in any of these formats:
            - List of np.ndarray (boolean H x W arrays)
            - List of dicts with 'mask' or 'segmentation' key
            - Mixed formats are OK
        initial_zoom: Initial zoom level (None = auto-fit to window)
    """

    def __init__(self, image: np.ndarray, masks: list, initial_zoom: float = None):
        self.image = image
        self.height, self.width = image.shape[:2]

        # Normalize masks to consistent internal format
        self.masks = self._normalize_masks(masks)

        # Classification state - default first two to planet/sky
        self.mask_classifications = {i: "exclude" for i in range(len(self.masks))}
        if len(self.masks) >= 1:
            self.mask_classifications[0] = "planet"
        if len(self.masks) >= 2:
            self.mask_classifications[1] = "sky"

        self.selected_mask = 0 if len(self.masks) > 0 else None

        # Colors for visualization (lighter colors for readability)
        self.colors = {
            "planet": (100, 200, 100),  # Green
            "sky": (100, 200, 255),  # Light cyan (readable on dark bg)
            "exclude": (128, 128, 128),  # Gray
        }

        # Text colors for listbox (readable on white)
        self.text_colors = {
            "planet": "darkgreen",
            "sky": "blue",  # Standard blue on white
            "exclude": "gray",
        }

        # Highlight colors based on classification
        self.highlight_colors = {
            "planet": (150, 255, 150),  # Light green
            "sky": (150, 200, 255),  # Light cyan
            "exclude": (255, 255, 100),  # Yellow (default/unclassified)
        }

        # GUI state
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.is_panning = False
        self.is_finished = False  # Track if user clicked Done

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Mask Classification - Planet Ruler")

        # Handle window close properly
        self.root.protocol("WM_DELETE_WINDOW", self.finish)

        # Calculate window size and zoom to fit image
        target_canvas_width = 1000
        target_canvas_height = 700

        if initial_zoom is None:
            # Auto-fit: calculate zoom to show full image
            zoom_w = target_canvas_width / self.width
            zoom_h = target_canvas_height / self.height
            self.zoom_level = min(zoom_w, zoom_h, 1.0)  # Don't zoom in, only out
        else:
            self.zoom_level = initial_zoom

        # Set window size based on zoom
        window_width = min(1600, int(self.width * self.zoom_level) + 450)
        window_height = min(1000, int(self.height * self.zoom_level) + 100)
        self.root.geometry(f"{window_width}x{window_height}")

        self._build_gui()
        self._create_overlay_image()
        self.update_canvas()

    def _normalize_masks(self, masks: list) -> list:
        """
        Convert various mask formats to consistent internal format.

        Accepts:
        - np.ndarray: boolean mask
        - dict with 'mask' key: {'mask': array, ...}
        - dict with 'segmentation' key: {'segmentation': array, ...} (SAM format)

        Returns:
        - List of dicts: [{'mask': array, 'area': int, 'id': int}, ...]
        """
        normalized = []

        for idx, m in enumerate(masks):
            if isinstance(m, np.ndarray):
                # Just a boolean array
                normalized.append(
                    {"mask": m.astype(bool), "area": int(np.sum(m)), "id": idx}
                )
            elif isinstance(m, dict):
                # Dict format - extract mask
                mask_array = m.get("mask", m.get("segmentation"))
                if mask_array is None:
                    raise ValueError(
                        f"Mask {idx} dict has no 'mask' or 'segmentation' key"
                    )

                normalized.append(
                    {
                        "mask": mask_array.astype(bool),
                        "area": m.get("area", int(np.sum(mask_array))),
                        "id": idx,
                        "original": m,  # Keep original for reference
                    }
                )
            else:
                raise TypeError(f"Mask {idx} must be ndarray or dict, got {type(m)}")

        # Sort by area (largest first) - universal heuristic
        normalized.sort(key=lambda x: x["area"], reverse=True)

        return normalized

    def _build_gui(self):
        """Build the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - controls
        left_panel = ttk.Frame(main_frame, width=400)  # Even wider
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_panel.pack_propagate(False)

        # Title
        title = ttk.Label(
            left_panel, text="Mask Classification", font=("Arial", 16, "bold")
        )
        title.pack(pady=(0, 10))

        # Instructions
        instructions = ttk.Label(
            left_panel,
            text=(
                "Select mask:\n"
                "  • Click directly on image\n"
                "  • Click in list below\n"
                "  • Arrow keys (↑/↓) to cycle\n\n"
                "Classify selected mask:\n"
                "  P = Planet (green)\n"
                "  S = Sky (cyan)\n"
                "  X = Exclude (gray)\n\n"
                "Navigate:\n"
                "  Middle click = Pan\n"
                "  Scroll = Zoom"
            ),
            justify=tk.LEFT,
            font=("Arial", 11),
        )
        instructions.pack(pady=(0, 15))

        # Mask list
        list_label = ttk.Label(
            left_panel, text="Masks (by area):", font=("Arial", 13, "bold")
        )
        list_label.pack()

        # Scrollable mask list
        list_frame = ttk.Frame(left_panel)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.mask_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Arial", 13),  # Large, readable font
        )
        self.mask_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.mask_listbox.yview)

        self.mask_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        # Action buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Done", command=self.finish).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(
            button_frame, text="Reset All", command=self.reset_classifications
        ).pack(fill=tk.X, pady=2)

        # Status
        self.status_label = ttk.Label(
            left_panel, text="", font=("Arial", 11), foreground="blue"
        )
        self.status_label.pack(pady=5)

        # Right panel - canvas
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas with scrollbars
        self.canvas = tk.Canvas(right_panel, bg="gray")
        h_scrollbar = ttk.Scrollbar(
            right_panel, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        v_scrollbar = ttk.Scrollbar(
            right_panel, orient=tk.VERTICAL, command=self.canvas.yview
        )

        self.canvas.config(
            xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set
        )

        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-2>", self.start_pan)  # Middle click
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.root.bind("<Key>", self.on_key_press)

        self.update_mask_list()

    def _create_overlay_image(self):
        """Create overlay image with color-coded highlights and proper edge detection."""
        import cv2

        # Start with original image
        overlay_array = self.image.copy().astype(np.float32)

        # Alpha blending parameter
        alpha = 0.4

        # Apply masks using vectorized operations
        for mask_idx, mask_dict in enumerate(self.masks):
            mask_bool = mask_dict["mask"]
            classification = self.mask_classifications[mask_idx]

            if classification != "exclude":
                color = self.colors[classification]
                r, g, b = color

                # Vectorized blend: overlay = alpha * color + (1-alpha) * original
                overlay_array[mask_bool, 0] = (
                    alpha * r + (1 - alpha) * overlay_array[mask_bool, 0]
                )
                overlay_array[mask_bool, 1] = (
                    alpha * g + (1 - alpha) * overlay_array[mask_bool, 1]
                )
                overlay_array[mask_bool, 2] = (
                    alpha * b + (1 - alpha) * overlay_array[mask_bool, 2]
                )

        # Add color-coded highlight border for selected mask
        if self.selected_mask is not None:
            selected_mask_bool = self.masks[self.selected_mask]["mask"].astype(np.uint8)
            classification = self.mask_classifications[self.selected_mask]

            # Get highlight color based on classification
            highlight_r, highlight_g, highlight_b = self.highlight_colors[
                classification
            ]

            # Create thick border by dilating mask
            kernel = np.ones((15, 15), np.uint8)  # Very thick border
            dilated = cv2.dilate(selected_mask_bool, kernel, iterations=1)
            border = (dilated > 0) & (~selected_mask_bool.astype(bool))

            # Highlight edges ONLY where mask actually touches
            edge_highlight = np.zeros_like(selected_mask_bool, dtype=bool)

            # Top edge - highlight only columns where mask touches top
            top_touching = selected_mask_bool[0, :]
            if np.any(top_touching):
                for x in np.where(top_touching)[0]:
                    edge_highlight[0:10, x] = True

            # Bottom edge
            bottom_touching = selected_mask_bool[-1, :]
            if np.any(bottom_touching):
                for x in np.where(bottom_touching)[0]:
                    edge_highlight[-10:, x] = True

            # Left edge - highlight only rows where mask touches left
            left_touching = selected_mask_bool[:, 0]
            if np.any(left_touching):
                for y in np.where(left_touching)[0]:
                    edge_highlight[y, 0:10] = True

            # Right edge
            right_touching = selected_mask_bool[:, -1]
            if np.any(right_touching):
                for y in np.where(right_touching)[0]:
                    edge_highlight[y, -10:] = True

            # Combine border and edge highlights
            full_highlight = border | edge_highlight

            # Apply color-coded highlight
            overlay_array[full_highlight, 0] = highlight_r
            overlay_array[full_highlight, 1] = highlight_g
            overlay_array[full_highlight, 2] = highlight_b

        # Convert to uint8
        overlay_array = np.clip(overlay_array, 0, 255).astype(np.uint8)

        # Convert to PIL Image
        self.overlay_image = Image.fromarray(overlay_array)

    def update_canvas(self):
        """Update canvas with current overlay at current zoom level."""
        # Calculate zoomed size
        display_width = int(self.width * self.zoom_level)
        display_height = int(self.height * self.zoom_level)

        # Ensure overlay_image exists (defensive programming for testing)
        if not hasattr(self, "overlay_image") or self.overlay_image is None:
            self._create_overlay_image()

        # Resize overlay image
        resized = self.overlay_image.resize(
            (display_width, display_height), Image.Resampling.LANCZOS
        )

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(resized)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=(0, 0, display_width, display_height))

    def update_mask_list(self):
        """Update the mask listbox with text labels."""
        self.mask_listbox.delete(0, tk.END)

        for idx, mask_dict in enumerate(self.masks):
            area = mask_dict["area"]
            classification = self.mask_classifications[idx]

            # Text labels instead of symbols
            label_text = {"planet": "Planet", "sky": "Sky", "exclude": "Exclude"}[
                classification
            ]

            label = f"[{label_text:>7}] #{idx}: {area:,}px"
            self.mask_listbox.insert(tk.END, label)

            # Color code with readable colors
            self.mask_listbox.itemconfig(idx, fg=self.text_colors[classification])

        # Highlight selected
        if self.selected_mask is not None:
            self.mask_listbox.selection_set(self.selected_mask)
            self.mask_listbox.see(self.selected_mask)

    def on_listbox_select(self, event):
        """Handle mask selection from listbox."""
        selection = self.mask_listbox.curselection()
        if selection:
            self.selected_mask = selection[0]
            self._create_overlay_image()
            self.update_canvas()
            self.status_label.config(text=f"Selected mask #{self.selected_mask}")

    def on_canvas_click(self, event):
        """Handle click on canvas to select mask."""
        # Convert canvas coords to image coords
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        img_x = int(canvas_x / self.zoom_level)
        img_y = int(canvas_y / self.zoom_level)

        # Check if within image bounds
        if not (0 <= img_x < self.width and 0 <= img_y < self.height):
            return

        # Find which mask was clicked (check from smallest to largest area)
        clicked_mask = None
        for idx in reversed(range(len(self.masks))):
            if self.masks[idx]["mask"][img_y, img_x]:
                clicked_mask = idx
                break

        if clicked_mask is not None:
            self.selected_mask = clicked_mask
            self.mask_listbox.selection_clear(0, tk.END)
            self.mask_listbox.selection_set(clicked_mask)
            self.mask_listbox.see(clicked_mask)
            self._create_overlay_image()
            self.update_canvas()
            self.status_label.config(text=f"Selected mask #{clicked_mask}")

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        key = event.keysym.lower()

        # Arrow keys to cycle through masks
        if key in ("up", "down"):
            if self.selected_mask is None:
                self.selected_mask = 0
            else:
                if key == "up":
                    self.selected_mask = (self.selected_mask - 1) % len(self.masks)
                else:  # down
                    self.selected_mask = (self.selected_mask + 1) % len(self.masks)

            self.mask_listbox.selection_clear(0, tk.END)
            self.mask_listbox.selection_set(self.selected_mask)
            self.mask_listbox.see(self.selected_mask)
            self._create_overlay_image()
            self.update_canvas()
            self.status_label.config(text=f"Selected mask #{self.selected_mask}")
            return

        # Classification shortcuts
        if self.selected_mask is None:
            return

        key_char = event.char.lower()

        if key_char == "p":
            self.mask_classifications[self.selected_mask] = "planet"
            self._create_overlay_image()
            self.update_canvas()
            self.update_mask_list()
            self.status_label.config(text=f"Mask #{self.selected_mask} → Planet")
        elif key_char == "s":
            self.mask_classifications[self.selected_mask] = "sky"
            self._create_overlay_image()
            self.update_canvas()
            self.update_mask_list()
            self.status_label.config(text=f"Mask #{self.selected_mask} → Sky")
        elif key_char == "x":
            self.mask_classifications[self.selected_mask] = "exclude"
            self._create_overlay_image()
            self.update_canvas()
            self.update_mask_list()
            self.status_label.config(text=f"Mask #{self.selected_mask} → Exclude")

    def start_pan(self, event):
        """Start panning."""
        self.is_panning = True
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        self.canvas.config(cursor="fleur")

    def do_pan(self, event):
        """Pan the canvas."""
        if self.is_panning:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            self.canvas.xview_scroll(-dx, tk.UNITS)
            self.canvas.yview_scroll(-dy, tk.UNITS)
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y

    def end_pan(self, event):
        """End panning."""
        self.is_panning = False
        self.canvas.config(cursor="")

    def on_mousewheel(self, event):
        """Zoom with mouse wheel."""
        # Zoom factor
        if event.delta > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1

        self.zoom_level = max(0.1, min(5.0, self.zoom_level))
        self._create_overlay_image()
        self.update_canvas()

    def reset_classifications(self):
        """Reset all to exclude except first two."""
        self.mask_classifications = {i: "exclude" for i in range(len(self.masks))}
        if len(self.masks) >= 1:
            self.mask_classifications[0] = "planet"
        if len(self.masks) >= 2:
            self.mask_classifications[1] = "sky"
        self._create_overlay_image()
        self.update_canvas()
        self.update_mask_list()
        self.status_label.config(text="Reset to defaults")

    def finish(self):
        """Close window - nuclear option for Jupyter compatibility."""
        self.is_finished = True

        try:
            # Disconnect from display
            self.root.withdraw()

            # Give it a moment
            self.root.update()

            # Force quit the mainloop
            self.root.quit()

        except Exception as e:
            logging.warning(f"Error during finish: {e}")

        finally:
            # Destroy in a separate try block
            try:
                self.root.destroy()
            except:
                pass

    def run(self):
        """Run the interactive selector with proper cleanup."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            # Ensure window is destroyed even if mainloop exits abnormally
            try:
                if self.root.winfo_exists():
                    self.root.destroy()
            except:
                pass

    def get_classified_masks(self):
        """Return dictionary of classified masks with original mask objects."""
        return {
            "planet": [
                self.masks[i]
                for i, c in self.mask_classifications.items()
                if c == "planet"
            ],
            "sky": [
                self.masks[i]
                for i, c in self.mask_classifications.items()
                if c == "sky"
            ],
            "exclude": [
                self.masks[i]
                for i, c in self.mask_classifications.items()
                if c == "exclude"
            ],
        }


# Example usage
if __name__ == "__main__":
    import sys

    print("Tkinter Manual Limb Annotation Tool")
    print("=" * 60)
    print("\nControls:")
    print("  Left Click:      Add point")
    print("  Right Click:     Undo last point")
    print("  Scroll Wheel:    Zoom in/out")
    print("  +/- keys:        Zoom in/out")
    print("  Scroll bars:     Navigate image")
    print("\nButtons:")
    print("  Zoom: Fit to Window, 100%, +/-")
    print("  Stretch: Increase/Decrease/Reset (makes curves easier to see)")
    print("  Generate Target: Create sparse array (.npy)")
    print("  Save/Load Points: Persist annotations (.json)")
    print("\n" + "=" * 60)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("\nUsage: python tk_annotator.py <image_path>")
        print("\nOr in your code:")
        print("  from tk_annotator import TkLimbAnnotator")
        print("  annotator = TkLimbAnnotator('image.jpg')")
        print("  annotator.run()")
        sys.exit(0)

    # Create and run
    annotator = TkLimbAnnotator(image_path, initial_stretch=1.0)
    annotator.run()

    # After closing, get target if points were added
    if len(annotator.points) >= 3:
        target = annotator.get_target()
        print(f"\n✓ Generated target with {np.sum(~np.isnan(target))} points")
