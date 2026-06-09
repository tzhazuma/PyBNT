"""AI Settings dialog for provider/model selection and question input."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QComboBox, QLineEdit, QTextEdit,
    QDialogButtonBox, QFormLayout, QMessageBox,
)


class AISettingsDialog(QDialog):
    """Dialog for selecting AI provider, model, API key, and entering a question.

    Provides a unified interface for all AI interactions in the GUI.
    Use the properties (provider, model, api_key, question) after exec().
    """

    PROVIDERS = ["opencodego", "dashscope"]
    DEFAULT_MODELS = {
        "opencodego": "mimo-v2.5",
        "dashscope": "qwen-plus",
    }

    def __init__(self, parent=None, title="Ask AI Assistant",
                 prompt_label="Your question:", show_question=True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(450)
        self._show_question = show_question
        self._prompt_label = prompt_label
        self._setup_ui()

    def _setup_ui(self):
        """Build the dialog layout."""
        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Provider selector
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(self.PROVIDERS)
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        form.addRow("Provider:", self.provider_combo)

        # Model name
        self.model_edit = QLineEdit(self.DEFAULT_MODELS["opencodego"])
        form.addRow("Model:", self.model_edit)

        # API key (optional)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("Leave blank to use env variable")
        form.addRow("API Key:", self.api_key_edit)

        layout.addLayout(form)

        # Question / prompt text
        if self._show_question:
            layout.addWidget(QLabel(self._prompt_label))
            self.question_edit = QTextEdit()
            self.question_edit.setPlaceholderText("Enter your question or prompt here...")
            self.question_edit.setMinimumHeight(80)
            layout.addWidget(self.question_edit)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_provider_changed(self, provider):
        """Auto-fill default model when provider changes."""
        default = self.DEFAULT_MODELS.get(provider, "")
        current = self.model_edit.text()
        if not current or current in self.DEFAULT_MODELS.values():
            self.model_edit.setText(default)

    def _validate_and_accept(self):
        """Validate inputs before accepting."""
        if self._show_question:
            if not self.question_edit.toPlainText().strip():
                QMessageBox.warning(
                    self, "Input Required",
                    "Please enter a question or prompt.",
                )
                return
        self.accept()

    @property
    def provider(self) -> str:
        """Selected provider name."""
        return self.provider_combo.currentText()

    @property
    def model(self) -> str:
        """Selected model name."""
        text = self.model_edit.text().strip()
        return text or self.DEFAULT_MODELS.get(self.provider, "mimo-v2.5")

    @property
    def api_key(self) -> str:
        """Entered API key (may be empty to use env variable)."""
        return self.api_key_edit.text().strip()

    @property
    def question(self) -> str:
        """Entered question or prompt text."""
        if self._show_question:
            return self.question_edit.toPlainText().strip()
        return ""
