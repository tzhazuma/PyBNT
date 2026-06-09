"""PyBrainViewer - PyQt6 Graphical User Interface."""
import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QGroupBox, QCheckBox, QGridLayout, QProgressBar, QStatusBar,
    QMenuBar, QToolBar, QMenu, QTextEdit, QFrame, QSplitter,
    QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QIcon, QFont


class ProcessingThread(QThread):
    """Background thread for long-running processing tasks."""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, processing_func, *args, **kwargs):
        super().__init__()
        self.processing_func = processing_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.status.emit("Processing started...")
            self.progress.emit(10)
            result = self.processing_func(*self.args, **self.kwargs)
            self.progress.emit(100)
            self.status.emit("Processing complete!")
            self.finished.emit(True, str(result))
        except Exception as e:
            self.finished.emit(False, str(e))


class Ui_MainWindow(QMainWindow):
    """Main window for PyBrainViewer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyBrainViewer v0.2.0")
        self.setMinimumSize(900, 700)
        self.save_path = None
        self.current_file = None
        self._dark_mode = False
        self._setup_menu_bar()
        self._setup_tool_bar()
        self._setup_central_widget()
        self._setup_status_bar()

    # ------------------------------------------------------------------ #
    #  Menu bar
    # ------------------------------------------------------------------ #
    def _setup_menu_bar(self):
        """Create the menu bar with File, Process, View, AI, Help menus."""
        menubar = self.menuBar()

        # ---- File menu -------------------------------------------------
        file_menu = menubar.addMenu("&File")

        open_surface = QAction("Open Surface (.nv)...", self)
        open_surface.setShortcut("Ctrl+O")
        open_surface.triggered.connect(lambda: self._browse_file("surface"))
        file_menu.addAction(open_surface)

        open_node = QAction("Open Node (.node)...", self)
        open_node.setShortcut("Ctrl+N")
        open_node.triggered.connect(lambda: self._browse_file("node"))
        file_menu.addAction(open_node)

        open_edge = QAction("Open Edge (.edge)...", self)
        open_edge.setShortcut("Ctrl+E")
        open_edge.triggered.connect(lambda: self._browse_file("edge"))
        file_menu.addAction(open_edge)

        open_image = QAction("Open Image (NIfTI/DICOM)...", self)
        open_image.setShortcut("Ctrl+I")
        open_image.triggered.connect(lambda: self._browse_file("image"))
        file_menu.addAction(open_image)

        file_menu.addSeparator()

        save_action = QAction("Save Screenshot...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_screenshot)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ---- Process menu ----------------------------------------------
        process_menu = menubar.addMenu("&Process")

        conn_action = QAction("Compute Connectivity...", self)
        conn_action.triggered.connect(self._compute_connectivity)
        process_menu.addAction(conn_action)

        register_action = QAction("Register Images...", self)
        register_action.triggered.connect(self._register_images)
        process_menu.addAction(register_action)

        superres_action = QAction("Super-Resolution...", self)
        superres_action.triggered.connect(self._run_superres)
        process_menu.addAction(superres_action)

        # ---- View menu ------------------------------------------------
        view_menu = menubar.addMenu("&View")

        layouts = [
            "Lateral Left", "Lateral Right", "Medial",
            "Ventral", "Dorsal", "Anterior", "Posterior",
        ]
        for layout in layouts:
            action = QAction(layout, self)
            action.triggered.connect(
                lambda checked, l=layout: self._set_layout(
                    l.lower().replace(" ", "_")
                )
            )
            view_menu.addAction(action)

        view_menu.addSeparator()

        dark_action = QAction("Toggle Dark Mode", self)
        dark_action.setShortcut("Ctrl+D")
        dark_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(dark_action)

        # ---- AI menu --------------------------------------------------
        ai_menu = menubar.addMenu("&AI")

        seg_action = QAction("Brain Segmentation...", self)
        seg_action.triggered.connect(self._ai_segmentation)
        ai_menu.addAction(seg_action)

        denoise_action = QAction("Denoise Image...", self)
        denoise_action.triggered.connect(self._ai_denoise)
        ai_menu.addAction(denoise_action)

        ask_action = QAction("Ask AI Assistant...", self)
        ask_action.setShortcut("Ctrl+A")
        ask_action.triggered.connect(self._ask_ai)
        ai_menu.addAction(ask_action)

        # ---- Help menu ------------------------------------------------
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("About PyBrainViewer", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        guide_action = QAction("Quick Guide", self)
        guide_action.triggered.connect(self._show_guide)
        help_menu.addAction(guide_action)

    # ------------------------------------------------------------------ #
    #  Toolbar
    # ------------------------------------------------------------------ #
    def _setup_tool_bar(self):
        """Create the main toolbar with common actions."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        toolbar.addAction("Open Surface", lambda: self._browse_file("surface"))
        toolbar.addAction("Open Node", lambda: self._browse_file("node"))
        toolbar.addAction("Open Edge", lambda: self._browse_file("edge"))
        toolbar.addAction("Open Image", lambda: self._browse_file("image"))
        toolbar.addSeparator()
        toolbar.addAction("Visualize", self._visualize)
        toolbar.addAction("Save", self._save_screenshot)

    # ------------------------------------------------------------------ #
    #  Central widget  (main content area)
    # ------------------------------------------------------------------ #
    def _setup_central_widget(self):
        """Create the main content area with input panels and controls."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ----- Left panel ------------------------------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        input_group = QGroupBox("Input Files")
        input_grid = QGridLayout()

        input_grid.addWidget(QLabel("Surface (.nv):"), 0, 0)
        self.surface_path = QLineEdit()
        input_grid.addWidget(self.surface_path, 0, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file("surface"))
        input_grid.addWidget(btn, 0, 2)

        input_grid.addWidget(QLabel("Node (.node):"), 1, 0)
        self.node_path = QLineEdit()
        input_grid.addWidget(self.node_path, 1, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file("node"))
        input_grid.addWidget(btn, 1, 2)

        input_grid.addWidget(QLabel("Edge (.edge):"), 2, 0)
        self.edge_path = QLineEdit()
        input_grid.addWidget(self.edge_path, 2, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file("edge"))
        input_grid.addWidget(btn, 2, 2)

        input_grid.addWidget(QLabel("Image (NIfTI):"), 3, 0)
        self.image_path = QLineEdit()
        input_grid.addWidget(self.image_path, 3, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file("image"))
        input_grid.addWidget(btn, 3, 2)

        input_group.setLayout(input_grid)
        left_layout.addWidget(input_group)

        # Processing options
        proc_group = QGroupBox("Processing Options")
        proc_grid = QGridLayout()

        self.chk_freesurfer = QCheckBox("FreeSurfer surface extraction")
        proc_grid.addWidget(self.chk_freesurfer, 0, 0)

        self.chk_connectivity = QCheckBox("Compute connectivity matrix")
        proc_grid.addWidget(self.chk_connectivity, 1, 0)

        self.chk_registration = QCheckBox("Image registration (elastix)")
        proc_grid.addWidget(self.chk_registration, 2, 0)

        self.chk_superres = QCheckBox("AI Super-Resolution")
        proc_grid.addWidget(self.chk_superres, 3, 0)

        self.chk_ai_split = QCheckBox("AI brain area segmentation")
        proc_grid.addWidget(self.chk_ai_split, 4, 0)

        self.chk_ai_correct = QCheckBox("AI image correction")
        proc_grid.addWidget(self.chk_ai_correct, 5, 0)

        proc_group.setLayout(proc_grid)
        left_layout.addWidget(proc_group)

        # Action buttons
        action_layout = QHBoxLayout()

        viz_btn = QPushButton("Visualize")
        viz_btn.clicked.connect(self._visualize)
        viz_btn.setMinimumHeight(40)
        action_layout.addWidget(viz_btn)

        run_btn = QPushButton("Run Processing")
        run_btn.clicked.connect(self._run_processing)
        run_btn.setMinimumHeight(40)
        run_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )
        action_layout.addWidget(run_btn)

        left_layout.addLayout(action_layout)
        left_layout.addStretch()

        # ----- Right panel (output / log) --------------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        right_layout.addWidget(QLabel("Output Log:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self.log_output)

        self.progress_bar = QProgressBar()
        right_layout.addWidget(self.progress_bar)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([450, 450])

        main_layout.addWidget(splitter)

    # ------------------------------------------------------------------ #
    #  Status bar
    # ------------------------------------------------------------------ #
    def _setup_status_bar(self):
        """Create status bar with progress information."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _browse_file(self, file_type):
        """Open file browser for different file types."""
        filters = {
            "surface": "Surface files (*.nv *.pial *.gii *.obj);;All files (*)",
            "node": "Node files (*.node *.txt);;All files (*)",
            "edge": "Edge files (*.edge *.txt);;All files (*)",
            "image": (
                "Medical images (*.nii *.nii.gz *.dcm *.dicom *.png *.jpg)"
                ";;All files (*)"
            ),
        }
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Open {file_type}",
            "",
            filters.get(file_type, "All files (*)"),
        )
        if path:
            if file_type == "surface":
                self.surface_path.setText(path)
            elif file_type == "node":
                self.node_path.setText(path)
            elif file_type == "edge":
                self.edge_path.setText(path)
            elif file_type == "image":
                self.image_path.setText(path)
            self.log(f"Loaded {file_type}: {path}")

    def log(self, message):
        """Add message to log output."""
        self.log_output.append(message)
        self.status_bar.showMessage(message)

    # ------------------------------------------------------------------ #
    #  Visualization
    # ------------------------------------------------------------------ #
    def _visualize(self):
        """Render the loaded data using pyvista."""
        surface = self.surface_path.text()
        node = self.node_path.text()
        edge = self.edge_path.text()

        self.log("Starting 3D visualization...")

        try:
            if node:
                from pybnt.visualization.plotter import draw_nodes

                draw_nodes(node, edge or None, surface or None, show=True)
                self.log("Node visualization complete.")
            elif surface:
                from pybnt.visualization.plotter import draw_surface

                draw_surface(surface, show=True)
                self.log("Surface visualization complete.")
            else:
                QMessageBox.warning(
                    self, "Error",
                    "Please select at least a surface or node file.",
                )
        except Exception as e:
            self.log(f"Visualization error: {e}")
            QMessageBox.critical(self, "Error", f"Visualization failed: {e}")

    # ------------------------------------------------------------------ #
    #  Background processing
    # ------------------------------------------------------------------ #
    def _run_processing(self):
        """Execute selected processing options in background thread."""
        if not any(
            [
                self.chk_freesurfer.isChecked(),
                self.chk_connectivity.isChecked(),
                self.chk_registration.isChecked(),
                self.chk_superres.isChecked(),
                self.chk_ai_split.isChecked(),
                self.chk_ai_correct.isChecked(),
            ]
        ):
            QMessageBox.information(
                self, "Info",
                "Please select at least one processing option.",
            )
            return

        self.log("Starting processing...")
        self.progress_bar.setValue(0)
        self.processing_thread = ProcessingThread(self._process_actions)
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.status.connect(self.log)
        self.processing_thread.finished.connect(self._on_processing_finished)
        self.processing_thread.start()

    def _process_actions(self):
        """Execute all selected processing actions."""
        image = self.image_path.text()
        if not image:
            raise ValueError("Please select an image file for processing.")

        if self.chk_freesurfer.isChecked():
            from pybnt.external.freesurfer import split_freesurfer

            split_freesurfer(
                self.image_path.text(), "subject", "./output"
            )

        if self.chk_connectivity.isChecked() and self.edge_path.text():
            from pybnt.processing.connectivity import (
                compute_correlation_matrix,
            )
            import numpy as np

            matrix = compute_correlation_matrix(image)
            np.savetxt(self.edge_path.text(), matrix)

        if self.chk_registration.isChecked():
            from pybnt.external.elastix import register_elastix

            register_elastix(image, image, "./registration_output")

        if self.chk_superres.isChecked():
            from pybnt.processing.superres import run_superres

            run_superres(image, "./superres_output", "./superres_output")

        return "Processing completed successfully."

    def _on_processing_finished(self, success, message):
        """Callback when background processing completes."""
        if success:
            self.log(f"Processing succeeded: {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.log(f"Processing failed: {message}")
            QMessageBox.critical(self, "Error", message)

    # ------------------------------------------------------------------ #
    #  Menu actions
    # ------------------------------------------------------------------ #
    def _compute_connectivity(self):
        """Menu action: compute connectivity matrix."""
        image = self.image_path.text()
        if not image:
            QMessageBox.warning(
                self, "Warning",
                "Please load an image file first.",
            )
            return
        self.log("Computing connectivity matrix...")
        self.chk_connectivity.setChecked(True)
        self._run_processing()

    def _register_images(self):
        """Menu action: register images."""
        self.log("Image registration selected.")
        QMessageBox.information(
            self, "Registration",
            "Select fixed and moving images in the input panel.",
        )

    def _run_superres(self):
        """Menu action: run super-resolution."""
        self.log("Super-resolution selected.")
        self.chk_superres.setChecked(True)
        self._run_processing()

    def _set_layout(self, layout_name):
        """Set view layout."""
        self.log(f"Layout set to: {layout_name}")

    def _toggle_theme(self):
        """Toggle between light and dark theme."""
        if self._dark_mode:
            self.setStyleSheet("")
            self._dark_mode = False
            self.log("Switched to light theme")
        else:
            self.setStyleSheet(
                """
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QLabel {
                    color: #ffffff;
                }
                QGroupBox {
                    color: #ffffff;
                    border: 1px solid #555;
                }
                QLineEdit {
                    background-color: #3c3c3c;
                    color: #ffffff;
                }
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #00ff00;
                }
                QCheckBox {
                    color: #ffffff;
                }
                """
            )
            self._dark_mode = True
            self.log("Switched to dark theme")

    def _ai_segmentation(self):
        """Run AI brain segmentation."""
        self.chk_ai_split.setChecked(True)
        self._run_processing()

    def _ai_denoise(self):
        """Run AI image denoising."""
        self.chk_ai_correct.setChecked(True)
        self._run_processing()

    def _ask_ai(self):
        """Open AI assistant dialog."""
        question, ok = QInputDialog.getText(
            self, "AI Assistant", "Ask about brain imaging:"
        )
        if ok and question:
            from pybnt.ai.llm import ask_llm

            response = ask_llm([{"role": "user", "content": question}])
            if response:
                QMessageBox.information(
                    self, "AI Response", str(response)
                )
            else:
                QMessageBox.warning(
                    self, "Error",
                    "Failed to get AI response. Check API key.",
                )

    def _save_screenshot(self):
        """Save current visualization as image."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff)",
        )
        if path:
            self.save_path = path
            self.log(f"Screenshot saved to: {path}")
            try:
                from pybnt.visualization.plotter import draw_surface
                surface = self.surface_path.text()
                node = self.node_path.text()
                if node:
                    from pybnt.visualization.plotter import draw_nodes
                    edge = self.edge_path.text()
                    ax = draw_nodes(node, edge or None, surface or None, show=False)
                elif surface:
                    ax = draw_surface(surface, show=False)
                else:
                    self.log("No surface/node loaded. Using empty view.")
                    import pyvista as pv
                    ax = pv.Plotter()
                    ax.background_color = "white"
                ax.screenshot(path)
                ax.close()
                self.log(f"✅ Screenshot saved to {path}")
            except Exception as e:
                self.log(f"❌ Screenshot failed: {e}")

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About PyBrainViewer",
            """<h2>PyBrainViewer v0.2.0</h2>
            <p>AI-Powered Multi-Function Brain Visual Tool</p>
            <p>By Tang Zhihao (ShanghaiTech University)</p>
            <p>Compatible with BrainNet Viewer formats.</p>
            <p>Built with: PyQt6, pyvista, nibabel, nilearn</p>""",
        )

    def _show_guide(self):
        """Show quick guide."""
        QMessageBox.information(
            self,
            "Quick Guide",
            """<h3>How to Use</h3>
            <ol>
            <li>Load surface (.nv), node (.node), and edge (.edge) files</li>
            <li>Click 'Visualize' to render 3D brain view</li>
            <li>Use processing options for advanced analysis</li>
            <li>AI features require API keys or model weights</li>
            </ol>""",
        )


def main():
    """Launch the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("PyBrainViewer")
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec())


def run_gui():
    """Entry point for GUI launch."""
    main()


if __name__ == "__main__":
    main()
