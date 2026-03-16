import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextBrowser
from PyQt6.QtCore import Qt

class DocsTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self._main_window = main_window
        
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        # Use a distinct background to look like a document viewer
        self.browser.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4; font-size: 14px; padding: 10px;")
        layout.addWidget(self.browser)
        
        self._load_docs()

    def _load_docs(self):
        project = self._main_window.current_project()
        docs_path = project.get("docs_path", "")
        
        if not docs_path:
            self.browser.setMarkdown("# Documentation\\n\\nNo `docs_path` provided for this project in the registry.")
            return
            
        full_path = self._main_window.resolve_path(docs_path)
        
        if not os.path.exists(full_path):
            self.browser.setMarkdown(f"# Documentation Error\\n\\nFile not found: `{full_path}`")
            return
            
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            self.browser.setMarkdown(md_content)
        except Exception as e:
            self.browser.setMarkdown(f"# Error Reading Docs\\n\\n`{str(e)}`")
