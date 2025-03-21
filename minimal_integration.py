"""
Minimal Grok-like UI integration for NeoCortex AI
This file integrates the minimalist UI with the existing NeoCortex backend
"""

import sys
import os
from datetime import datetime
import getpass

from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QIcon, QPalette
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox, QScrollArea, QFrame,
    QStatusBar
)

# Import from the main ki.py file
from ki import NeoCortex, ReasoningThread as OriginalReasoningThread

# Color scheme for UI styling
COLOR_SCHEME = {
    "background": "#131314",    # Very dark background (Grok style)
    "input_bg": "#232427",      # Input area background
    "text": "#ffffff",          # White text color
    "secondary_text": "#ececf1", # Slightly dimmer text
    "muted_text": "#9ca3af",    # Muted text
    "accent": "#10a37f",        # Accent color
    "accent_hover": "#0d8c6d",  # Hover color
    "error": "#f43f5e",         # Error color
}

class MinimalReasoningThread(QThread):
    """Thread for processing queries and returning responses."""
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str, int)
    
    def __init__(self, query, model_type="Standard"):
        super().__init__()
        self.query = query
        self.model_type = model_type
        self.neocortex = NeoCortex()
        
    def run(self):
        try:
            # Determine which processing mode to use based on model type
            show_work = self.model_type != "Fast"
            use_agents = self.model_type == "Advanced"
            
            # Use the actual NeoCortex processor
            self.progress_update.emit("Processing query...", 10)
            
            # Process with the appropriate method
            result = self.neocortex.solve(
                query=self.query,
                show_work=show_work,
                use_agents=use_agents
            )
            
            # Get the final answer from the result
            final_answer = result.get("final_answer", "I couldn't process your query.")
            
            # Emit the final answer
            self.result_ready.emit(final_answer)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class MinimalNeoCortexGUI(QMainWindow):
    """A minimal Grok-like GUI for NeoCortex."""
    
    def __init__(self):
        super().__init__()
        self.neocortex = NeoCortex()
        self.conversation_history = []
        self.thinking_dots = 0
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components."""
        # Set window properties
        self.setWindowTitle("NeoCortex AI")
        self.setGeometry(100, 100, 1000, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {COLOR_SCHEME['background']};")
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create the conversation area (top part)
        self.conversation_area = QScrollArea()
        self.conversation_area.setWidgetResizable(True)
        self.conversation_area.setFrameShape(QFrame.NoFrame)
        self.conversation_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.conversation_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLOR_SCHEME['background']};
                border: none;
            }}
        """)
        
        # Container for messages
        self.conversation_container = QWidget()
        self.conversation_layout = QVBoxLayout(self.conversation_container)
        self.conversation_layout.setAlignment(Qt.AlignCenter)
        self.conversation_layout.setContentsMargins(20, 20, 20, 20)
        self.conversation_layout.setSpacing(30)
        
        # Add welcome message
        self.add_welcome_message()
        
        # Add stretch to push content to top
        self.conversation_layout.addStretch()
        
        self.conversation_area.setWidget(self.conversation_container)
        main_layout.addWidget(self.conversation_area, 1)
        
        # Create the input area (bottom part)
        input_container = QWidget()
        input_container.setStyleSheet(f"background-color: {COLOR_SCHEME['background']};")
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(20, 10, 20, 20)
        
        # Input box with rounded corners
        self.query_container = QWidget()
        self.query_container.setStyleSheet(f"""
            background-color: {COLOR_SCHEME['input_bg']};
            border-radius: 12px;
        """)
        query_container_layout = QVBoxLayout(self.query_container)
        query_container_layout.setContentsMargins(5, 5, 5, 5)
        
        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText("What do you want to know?")
        self.query_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLOR_SCHEME['input_bg']};
                color: {COLOR_SCHEME['text']};
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-size: 14pt;
                font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
            }}
        """)
        self.query_edit.setMinimumHeight(60)
        self.query_edit.setMaximumHeight(150)
        query_container_layout.addWidget(self.query_edit)
        
        # Bottom buttons area
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(10, 5, 10, 5)
        
        # Model selector
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Standard",
            "Fast",
            "Advanced"
        ])
        self.model_selector.setCurrentIndex(0)
        self.model_selector.setStyleSheet(f"""
            QComboBox {{
                background-color: transparent;
                color: {COLOR_SCHEME['muted_text']};
                border: none;
                padding: 5px;
                font-size: 10pt;
                min-width: 150px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
        """)
        
        # Add model selector and spacer
        buttons_layout.addWidget(self.model_selector)
        buttons_layout.addStretch()
        
        # Send button
        self.send_button = QPushButton("→")
        self.send_button.setFont(QFont('Arial', 16))
        self.send_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_SCHEME['accent']};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px;
                min-width: 40px;
                min-height: 40px;
                max-width: 40px;
                max-height: 40px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_SCHEME['accent_hover']};
            }}
        """)
        self.send_button.clicked.connect(self.process_query)
        
        buttons_layout.addWidget(self.send_button)
        query_container_layout.addLayout(buttons_layout)
        
        input_layout.addWidget(self.query_container)
        
        # Add quick prompt buttons
        quick_prompts_layout = QHBoxLayout()
        quick_prompts_layout.setSpacing(10)
        quick_prompts_layout.setContentsMargins(10, 5, 10, 0)
        
        prompt_button_style = f"""
            QPushButton {{
                background-color: rgba(100, 100, 100, 0.2);
                color: {COLOR_SCHEME['secondary_text']};
                border: none;
                border-radius: 15px;
                padding: 8px 15px;
                font-size: 10pt;
            }}
            QPushButton:hover {{
                background-color: rgba(100, 100, 100, 0.3);
            }}
        """
        
        # Add some quick prompt options that match the NeoCortex capabilities
        for prompt in ["Analyze this", "Help me understand", "Solve a problem"]:
            button = QPushButton(prompt)
            button.setStyleSheet(prompt_button_style)
            button.clicked.connect(lambda _, p=prompt: self.set_prompt(p))
            quick_prompts_layout.addWidget(button)
        
        quick_prompts_layout.addStretch()
        input_layout.addLayout(quick_prompts_layout)
        
        main_layout.addWidget(input_container, 0)
        
        # Set up thinking animation timer
        self.thinking_timer = QTimer()
        self.thinking_timer.timeout.connect(self.update_thinking_animation)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLOR_SCHEME['background']};
                color: {COLOR_SCHEME['muted_text']};
                border-top: none;
            }}
        """)
        self.setStatusBar(self.status_bar)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Connect enter key to send
        self.query_edit.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        """Filter events to capture Enter key in the text edit."""
        if obj == self.query_edit and event.type() == 7:  # QEvent.KeyPress
            if event.key() == 16777220 and not event.modifiers() & 33554432:  # Enter key without Shift
                self.process_query()
                return True
        return super().eventFilter(obj, event)
    
    def set_prompt(self, prompt):
        """Set the text of the query edit to the selected prompt."""
        self.query_edit.setText(prompt + ": ")
        self.query_edit.setFocus()
        cursor = self.query_edit.textCursor()
        cursor.movePosition(1)  # QTextCursor.End
        self.query_edit.setTextCursor(cursor)
    
    def add_welcome_message(self):
        """Add a welcome message to the conversation."""
        # Get the user's name
        username = getpass.getuser().capitalize()
        
        # Get the time of day
        hour = datetime.now().hour
        if 5 <= hour < 12:
            greeting = "Good morning"
        elif 12 <= hour < 18:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
        
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_layout.setAlignment(Qt.AlignCenter)
        welcome_layout.setContentsMargins(0, 100, 0, 0)
        
        # Welcome message
        welcome_label = QLabel(f"{greeting}, {username}.")
        welcome_label.setStyleSheet(f"""
            color: {COLOR_SCHEME['text']};
            font-size: 24pt;
            font-weight: normal;
            font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
        """)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(welcome_label)
        
        # Subtitle
        subtitle_label = QLabel("How can I help you today?")
        subtitle_label.setStyleSheet(f"""
            color: {COLOR_SCHEME['secondary_text']};
            font-size: 16pt;
            font-weight: normal;
            font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
            margin-top: 5px;
        """)
        subtitle_label.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(subtitle_label)
        
        self.conversation_layout.addWidget(welcome_widget)
    
    def add_user_message(self, text):
        """Add a user message to the conversation."""
        # Remove the welcome message if this is the first message
        if len(self.conversation_history) == 0:
            # Clear the layout first
            for i in reversed(range(self.conversation_layout.count())):
                item = self.conversation_layout.itemAt(i)
                if item.widget():
                    item.widget().deleteLater()
        
        message_widget = QWidget()
        message_layout = QHBoxLayout(message_widget)
        message_layout.setContentsMargins(0, 0, 0, 0)
        
        # Push content to the right
        message_layout.addStretch()
        
        # Message bubble
        text_container = QWidget()
        text_container.setStyleSheet(f"""
            background-color: {COLOR_SCHEME['accent']};
            border-radius: 18px;
            padding: 10px;
            margin: 5px;
            max-width: 600px;
        """)
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(10, 10, 10, 10)
        
        text_label = QLabel(text)
        text_label.setStyleSheet(f"""
            color: {COLOR_SCHEME['text']};
            font-size: 12pt;
            font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
            margin: 5px;
            padding: 5px;
        """)
        text_label.setWordWrap(True)
        text_layout.addWidget(text_label)
        
        message_layout.addWidget(text_container)
        
        self.conversation_layout.addWidget(message_widget)
        self.conversation_history.append({"role": "user", "content": text})
        
        # Scroll to bottom
        self.conversation_area.verticalScrollBar().setValue(
            self.conversation_area.verticalScrollBar().maximum()
        )
    
    def add_assistant_message(self, text):
        """Add an assistant message to the conversation."""
        message_widget = QWidget()
        message_layout = QHBoxLayout(message_widget)
        message_layout.setContentsMargins(0, 0, 0, 0)
        
        # Message bubble
        text_container = QWidget()
        text_container.setStyleSheet(f"""
            background-color: {COLOR_SCHEME['input_bg']};
            border-radius: 18px;
            padding: 10px;
            margin: 5px;
            max-width: 600px;
        """)
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(10, 10, 10, 10)
        
        text_label = QLabel(text)
        text_label.setStyleSheet(f"""
            color: {COLOR_SCHEME['text']};
            font-size: 12pt;
            font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
            margin: 5px;
            padding: 5px;
        """)
        text_label.setWordWrap(True)
        text_label.setTextFormat(Qt.RichText)
        text_label.setOpenExternalLinks(True)
        text_layout.addWidget(text_label)
        
        message_layout.addWidget(text_container)
        
        # Push content to the right
        message_layout.addStretch()
        
        self.conversation_layout.addWidget(message_widget)
        self.conversation_history.append({"role": "assistant", "content": text})
        
        # Scroll to bottom
        self.conversation_area.verticalScrollBar().setValue(
            self.conversation_area.verticalScrollBar().maximum()
        )
    
    def add_thinking_indicator(self):
        """Add a thinking indicator to the conversation."""
        self.thinking_widget = QWidget()
        thinking_layout = QHBoxLayout(self.thinking_widget)
        thinking_layout.setContentsMargins(0, 0, 0, 0)
        
        # Thinking bubble
        text_container = QWidget()
        text_container.setStyleSheet(f"""
            background-color: {COLOR_SCHEME['input_bg']};
            border-radius: 18px;
            padding: 10px;
            margin: 5px;
            max-width: 600px;
        """)
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(10, 10, 10, 10)
        
        self.thinking_label = QLabel("Thinking")
        self.thinking_label.setStyleSheet(f"""
            color: {COLOR_SCHEME['secondary_text']};
            font-size: 12pt;
            font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
            margin: 5px;
            padding: 5px;
        """)
        text_layout.addWidget(self.thinking_label)
        
        thinking_layout.addWidget(text_container)
        thinking_layout.addStretch()
        
        self.conversation_layout.addWidget(self.thinking_widget)
        
        # Start the thinking animation
        self.thinking_timer.start(500)
        
        # Scroll to bottom
        self.conversation_area.verticalScrollBar().setValue(
            self.conversation_area.verticalScrollBar().maximum()
        )
    
    def remove_thinking_indicator(self):
        """Remove the thinking indicator from the conversation."""
        if hasattr(self, 'thinking_widget'):
            self.thinking_timer.stop()
            self.thinking_widget.deleteLater()
            self.thinking_widget = None
    
    def update_thinking_animation(self):
        """Update the thinking animation dots."""
        if hasattr(self, 'thinking_label'):
            self.thinking_dots = (self.thinking_dots + 1) % 4
            dots = "." * self.thinking_dots
            self.thinking_label.setText(f"Thinking{dots}")
    
    def update_progress(self, status, progress):
        """Update progress status in the status bar."""
        self.status_bar.showMessage(f"Processing: {status} ({progress}%)")
    
    def process_query(self):
        """Process the user's query."""
        query = self.query_edit.toPlainText().strip()
        if not query:
            return
        
        # Add user message to conversation
        self.add_user_message(query)
        
        # Clear input field
        self.query_edit.clear()
        
        # Add thinking indicator
        self.add_thinking_indicator()
        
        # Disable send button during processing
        self.send_button.setEnabled(False)
        
        # Get model setting from selector
        model_type = self.model_selector.currentText()
        
        # Create reasoning thread
        self.reasoning_thread = MinimalReasoningThread(query, model_type)
        self.reasoning_thread.result_ready.connect(self.handle_result)
        self.reasoning_thread.error_occurred.connect(self.handle_error)
        self.reasoning_thread.progress_update.connect(self.update_progress)
        
        # Start processing
        self.reasoning_thread.start()
    
    def handle_result(self, response):
        """Handle the reasoning result."""
        # Remove thinking indicator
        self.remove_thinking_indicator()
        
        # Add assistant's response to conversation
        self.add_assistant_message(response)
        
        # Re-enable send button
        self.send_button.setEnabled(True)
        
        # Update status
        self.status_bar.showMessage("Ready", 3000)
    
    def handle_error(self, error_msg):
        """Handle errors in the reasoning process."""
        # Remove thinking indicator
        self.remove_thinking_indicator()
        
        # Add error message to conversation
        self.add_assistant_message(f"<span style='color: {COLOR_SCHEME['error']};'>Error: {error_msg}</span>")
        
        # Re-enable send button
        self.send_button.setEnabled(True)
        
        # Update status
        self.status_bar.showMessage("Error occurred", 3000)

def apply_dark_theme(app):
    """Apply dark theme styling to the application."""
    # Create dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(COLOR_SCHEME["background"]))
    dark_palette.setColor(QPalette.WindowText, QColor(COLOR_SCHEME["text"]))
    dark_palette.setColor(QPalette.Base, QColor(COLOR_SCHEME["input_bg"]))
    dark_palette.setColor(QPalette.AlternateBase, QColor(COLOR_SCHEME["background"]))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(COLOR_SCHEME["input_bg"]))
    dark_palette.setColor(QPalette.ToolTipText, QColor(COLOR_SCHEME["text"]))
    dark_palette.setColor(QPalette.Text, QColor(COLOR_SCHEME["text"]))
    dark_palette.setColor(QPalette.Button, QColor(COLOR_SCHEME["input_bg"]))
    dark_palette.setColor(QPalette.ButtonText, QColor(COLOR_SCHEME["text"]))
    dark_palette.setColor(QPalette.BrightText, QColor(COLOR_SCHEME["accent"]))
    dark_palette.setColor(QPalette.Link, QColor(COLOR_SCHEME["accent"]))
    dark_palette.setColor(QPalette.Highlight, QColor(COLOR_SCHEME["accent"]))
    dark_palette.setColor(QPalette.HighlightedText, QColor(COLOR_SCHEME["text"]))
    
    # Apply the palette
    app.setPalette(dark_palette)
    
    # Set global application style
    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
        }
    """)

def main():
    """Main entry point for the minimal GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("NeoCortex AI")
    
    apply_dark_theme(app)
    window = MinimalNeoCortexGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 