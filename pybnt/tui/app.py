"""PyBrainViewer Terminal User Interface."""

from textual.app import App
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Tree, Static, Button, Label, Log
from pathlib import Path


class FileBrowser(Tree):
    """File browser widget for navigating brain data files."""

    def __init__(self):
        super().__init__("Files")
        self.root.expand()

    def on_mount(self):
        self._populate(self.root, Path.cwd())

    def _populate(self, node, path: Path):
        try:
            for child in sorted(path.iterdir()):
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    sub = node.add(child.name, expand=False)
                    sub.data = child
                    # Add children lazily
                    self._populate(sub, child)
                elif child.suffix in (
                    ".nv",
                    ".node",
                    ".edge",
                    ".nii",
                    ".nii.gz",
                    ".dcm",
                    ".png",
                    ".jpg",
                    ".pial",
                ):
                    node.add_leaf(child.name, data=child)
        except PermissionError:
            pass


class ProcessingPanel(Vertical):
    """Panel with processing options."""

    def compose(self):
        yield Label("Processing Options", classes="panel-title")
        yield Button("Generate Surface", id="gen-surface", variant="primary")
        yield Button("Compute Connectivity", id="compute-conn", variant="primary")
        yield Button("Register Images", id="register", variant="primary")
        yield Button("Super-Resolution", id="superres", variant="primary")
        yield Button("AI Segmentation", id="ai-seg", variant="default")
        yield Button("AI Denoise", id="ai-denoise", variant="default")
        yield Label("Status", classes="panel-title")
        yield Log(id="status-log", max_lines=100)


class BrainViewTUI(App):
    """PyBrainViewer Terminal User Interface."""

    CSS = """
    Screen {
        background: $surface;
    }
    #main-container {
        height: 100%;
    }
    #file-panel {
        width: 30%;
        min-width: 20;
        border-right: solid $primary;
    }
    #content-area {
        width: 70%;
    }
    #processing-panel {
        height: 100%;
        border-left: solid $primary;
        padding: 1;
    }
    #info-panel {
        height: 60%;
    }
    #status-panel {
        height: 40%;
        border-top: solid $primary;
    }
    .panel-title {
        text-style: bold;
        color: $accent;
        padding: 0 1;
    }
    Button {
        margin: 0 1;
        width: 100%;
    }
    Log {
        height: 100%;
    }
    """

    BINDINGS = [
        ("o", "open_file", "Open Selected"),
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Dark Mode"),
        ("r", "refresh", "Refresh"),
    ]

    def compose(self):
        yield Header(show_clock=True)
        with Container(id="main-container"):
            with Horizontal():
                yield FileBrowser()
                with Vertical(id="content-area"):
                    with Vertical(id="info-panel"):
                        yield Static("Select a file to begin", id="file-info")
                    with Vertical(id="status-panel"):
                        yield Log(id="log", max_lines=50)
            yield ProcessingPanel()
        yield Footer()

    def on_mount(self):
        self.query_one("#log", Log).write_line("PyBrainViewer TUI started. Open a file to begin.")

    def action_open_file(self):
        """Handle file open action."""
        tree = self.query_one(FileBrowser)
        cursor = tree.cursor_node
        if cursor and cursor.data and cursor.data.is_file():
            path = cursor.data
            self.query_one("#file-info", Static).update(f"Selected: {path}")
            log = self.query_one("#log", Log)
            log.write_line(f"Opened: {path}")
            self._process_file(path)

    def _process_file(self, path: Path):
        """Process the selected file based on extension."""
        ext = path.suffix.lower()
        log = self.query_one("#log", Log)

        if ext == ".nv":
            log.write_line(f"Surface file detected: {path.name}")
            log.write_line("  Use 'Visualize Surface' in GUI for 3D rendering")
        elif ext == ".node":
            log.write_line(f"Node file detected: {path.name}")
            from pybnt.core.nodes import parse_node_file

            nodes = parse_node_file(str(path))
            log.write_line(f"  Nodes: {len(nodes)}")
        elif ext == ".edge":
            log.write_line(f"Edge file detected: {path.name}")
            from pybnt.core.edges import parse_edge_file

            edges = parse_edge_file(str(path))
            log.write_line(f"  Matrix shape: {edges.shape}")
        elif ext in (".nii", ".gz"):
            log.write_line(f"NIfTI file detected: {path.name}")
            import nibabel as nib

            img = nib.load(str(path))
            log.write_line(f"  Shape: {img.shape}, Data type: {img.get_data_dtype()}")
        elif ext == ".dcm":
            log.write_line(f"DICOM file detected: {path.name}")
        else:
            log.write_line(f"File: {path.name}")

    def on_button_pressed(self, event: Button.Pressed):
        log = self.query_one("#log", Log)
        log.write_line(f"Action: {event.button.id} triggered")

        if event.button.id == "gen-surface":
            log.write_line("Surface generation requires NIfTI input. Use FreeSurfer pipeline.")
        elif event.button.id == "compute-conn":
            log.write_line("Connectivity: Select a 4D NIfTI or image directory first.")
        elif event.button.id == "register":
            log.write_line("Registration: Need fixed and moving images.")
        elif event.button.id == "superres":
            log.write_line("Super-resolution: Select an image directory.")
        elif event.button.id == "ai-seg":
            log.write_line("AI Segmentation: Select a brain NIfTI file.")
        elif event.button.id == "ai-denoise":
            log.write_line("AI Denoise: Select an image file.")

    def action_refresh(self):
        """Refresh the file browser."""
        self.query_one("#log", Log).write_line("Refreshed file browser.")

    def action_toggle_dark(self):
        self.dark = not self.dark


def run_tui():
    """Launch the TUI application."""
    app = BrainViewTUI()
    app.run()


if __name__ == "__main__":
    run_tui()
