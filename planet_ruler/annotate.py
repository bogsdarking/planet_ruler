"""
Manual Limb Annotation Tool for Planet Ruler

Allows users to click points on a horizon image and generate a sparse target
for fitting with the existing planet_ruler pipeline.
"""

import numpy as np
import json
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
