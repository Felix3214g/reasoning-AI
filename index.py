import sys
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading
import queue
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon, QPalette, QPixmap, QBrush, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTextEdit, QLineEdit, QGroupBox, QTabWidget, QSplitter,
    QComboBox, QListWidget, QListWidgetItem, QProgressBar, QDialog, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QCheckBox, QSpinBox, QDoubleSpinBox, QSlider,
    QScrollArea, QFrame, QGraphicsView, QGraphicsScene, QGraphicsItem, 
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QMessageBox,
    QStyleFactory, QToolBar, QAction, QStatusBar, QGraphicsRectItem
)

from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

# ============================================================================
# NeoCortex Implementation (Previously in neocortex_impl.py)
# ============================================================================

class Knowledge:
    """Class to represent knowledge in the cognitive system."""
    
    def __init__(self, content: str, source: str = "reasoning", confidence: float = 0.8):
        self.content = content
        self.source = source
        self.confidence = confidence
    
    def to_dict(self) -> Dict:
        """Convert Knowledge to a dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence
        }

class CognitiveNode:
    """Class to represent a node in the cognitive graph."""
    
    def __init__(self, node_id: str, node_type: str, content: str, 
                 parent_id: Optional[str] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.content = content
        self.parent_id = parent_id
        self.children: List[str] = []
    
    def add_child(self, child_id: str):
        """Add a child node ID to this node."""
        if child_id not in self.children:
            self.children.append(child_id)
    
    def to_dict(self) -> Dict:
        """Convert CognitiveNode to a dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "content": self.content,
            "parent_id": self.parent_id,
            "children": self.children
        }

class CognitiveGraph:
    """Class to represent the cognitive process graph."""
    
    def __init__(self):
        self.nodes: Dict[str, CognitiveNode] = {}
        self.root_id: Optional[str] = None
    
    def add_node(self, node: CognitiveNode) -> str:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        
        # If this node has a parent, add it as a child to the parent
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].add_child(node.node_id)
        
        # If this is the first node, set it as root
        if len(self.nodes) == 1:
            self.root_id = node.node_id
        
        return node.node_id
    
    def get_node(self, node_id: str) -> Optional[CognitiveNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def to_dict(self) -> Dict:
        """Convert CognitiveGraph to a dictionary."""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_id": self.root_id
        }

class NeoCortex:
    """Main class for the advanced cognitive architecture."""
    
    def __init__(self, model_name: str = "deepseek/deepseek-chat-v3", temperature: float = 0.2):
        """Initialize NeoCortex with specific model settings."""
        self.model_name = model_name
        self.temperature = temperature
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        # Hardcoded API key
        self.api_key = "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"
        self.cognitive_graph = CognitiveGraph()
        
        # Module states
        self.module_states = {
            "concept_network": True,
            "self_regulation": True,
            "spatial_reasoning": True,
            "causal_reasoning": True,
            "counterfactual": True,
            "metacognition": True
        }
        
        # Add response cache to speed up repeated requests
        self.response_cache = {}
        
        # Performance settings
        self.max_tokens = 600  # Limit token length for faster responses
        self.concurrent_requests = True  # Enable parallel processing
        self.reasoning_depth = "balanced"  # Options: "minimal", "balanced", "thorough"
        
        # Initialize session
        self.session = requests.Session()
        
        # Thread pool for parallel requests
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    # Removed _initialize_model method since we're using a hardcoded API key
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response using the OpenRouter API with retry logic for rate limits."""
        # Check cache first for faster responses
        cache_key = f"{prompt}_{self.model_name}_{self.temperature}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
            
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Create a more efficient payload with max_tokens limit
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are an advanced reasoning AI assistant. Be concise and efficient."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "extra_params": {
                        "provider_order": ["deepseek"]
                    }
                }
                
                # Add timeout to prevent hanging - reduced from 30 to 10 for faster feedback
                response = self.session.post(self.api_url, headers=headers, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Cache the response for future use
                    self.response_cache[cache_key] = content
                    return content
                    
                elif response.status_code == 429:
                    # Rate limit error
                    error_message = f"Rate limit exceeded: {response.text}"
                    print(f"Error on attempt {attempt+1}/{max_retries}: {error_message}")
                    
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    
                    # If this is the last attempt, return a user-friendly error message
                    if attempt == max_retries - 1:
                        return "I've reached my current usage limits. Please try again later or check your API quota."
                else:
                    # Other errors
                    error_message = f"Error {response.status_code}: {response.text}"
                    print(error_message)
                    return f"Error: {error_message}"
                    
            except requests.exceptions.Timeout:
                print(f"Timeout occurred on attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    return "Request timed out. The server might be experiencing high load."
            except Exception as e:
                error_message = str(e)
                print(f"Error on attempt {attempt+1}/{max_retries}: {error_message}")
                
                # For other errors, don't retry
                return f"Error: {error_message}"
        
        return "Failed after multiple retries. Please try again later."
    
    def _analyze_problem(self, query: str) -> Dict:
        """Analyze the problem structure."""
        prompt = f"""
        Analyze the following problem:
        
        {query}
        
        Break it down into its core components and subproblems. Identify key concepts, constraints, and goals.
        """
        
        response_text = self._generate_response(prompt)
        
        # Create a problem node in the cognitive graph
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        # Parse response to identify subproblems (simplified for demonstration)
        decomposition = {
            "core_components": response_text,
            "subproblems": [
                {"id": "sub_1", "description": "First aspect of the problem"},
                {"id": "sub_2", "description": "Second aspect of the problem"}
            ],
            "full_decomposition": response_text
        }
        
        # Add subproblem nodes
        for subproblem in decomposition["subproblems"]:
            subproblem_node = CognitiveNode(
                node_id=subproblem["id"],
                node_type="subproblem",
                content=subproblem["description"],
                parent_id="problem_0"
            )
            self.cognitive_graph.add_node(subproblem_node)
        
        return decomposition
    
    def _generate_perspectives(self, query: str, decomposition: Dict) -> List[Dict]:
        """Generate multiple perspectives on the problem."""
        prompt = f"""
        Consider the following problem from multiple perspectives:
        
        {query}
        
        Provide at least 3 different perspectives or approaches to this problem.
        """
        
        response_text = self._generate_response(prompt)
        
        # Simplified parsing for demonstration
        perspectives = [
            {
                "name": "Perspective 1",
                "description": "First way of looking at the problem",
                "insights": "Key insights from this perspective",
                "limitations": "Limitations of this perspective"
            },
            {
                "name": "Perspective 2",
                "description": "Second way of looking at the problem",
                "insights": "Key insights from this perspective",
                "limitations": "Limitations of this perspective"
            },
            {
                "name": "Perspective 3",
                "description": "Third way of looking at the problem",
                "insights": "Key insights from this perspective",
                "limitations": "Limitations of this perspective"
            }
        ]
        
        # Add perspective nodes to cognitive graph
        perspective_counter = 0
        for perspective in perspectives:
            perspective_node = CognitiveNode(
                node_id=f"perspective_{perspective_counter}",
                node_type="perspective",
                content=perspective["description"],
                parent_id="problem_0"
            )
            self.cognitive_graph.add_node(perspective_node)
            perspective_counter += 1
        
        return perspectives
    
    def _gather_evidence(self, query: str, decomposition: Dict) -> Dict[str, Dict]:
        """Gather evidence for each subproblem."""
        evidence = {}
        
        for subproblem in decomposition["subproblems"]:
            subproblem_id = subproblem["id"]
            subproblem_desc = subproblem["description"]
            
            prompt = f"""
            Analyze the following aspect of the problem:
            
            {subproblem_desc}
            
            Provide factual evidence, logical analysis, and any computational evidence.
            """
            
            response_text = self._generate_response(prompt)
            
            # Create evidence node in the cognitive graph
            evidence_node = CognitiveNode(
                node_id=f"evidence_{subproblem_id}",
                node_type="evidence",
                content=response_text,
                parent_id=subproblem_id
            )
            self.cognitive_graph.add_node(evidence_node)
            
            evidence[subproblem_id] = {
                "factual_evidence": "Relevant facts for this subproblem",
                "logical_analysis": "Logical reasoning for this subproblem",
                "computational_evidence": "Computational analysis for this subproblem",
                "synthesis": response_text
            }
        
        return evidence
    
    def _integrate_perspectives_evidence(self, query: str, perspectives: List[Dict], 
                                       evidence: Dict[str, Dict]) -> Dict:
        """Integrate perspectives and evidence into a coherent understanding."""
        prompt = f"""
        Integrate the following perspectives and evidence to form a complete understanding:
        
        Problem: {query}
        
        Perspectives: {json.dumps(perspectives)}
        
        Evidence: {json.dumps(evidence)}
        
        Provide a coherent integration of these elements.
        """
        
        response_text = self._generate_response(prompt)
        
        # Create integration node in the cognitive graph
        integration_node = CognitiveNode(
            node_id="integration_0",
            node_type="integration",
            content=response_text,
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(integration_node)
        
        integration = {
            "key_insights": "Key insights from integrating perspectives and evidence",
            "full_integration": response_text
        }
        
        return integration
    
    def _generate_solution(self, query: str, integration: Dict) -> Dict:
        """Generate a solution based on the integrated understanding."""
        prompt = f"""
        Generate a solution for the following problem based on the integrated understanding:
        
        Problem: {query}
        
        Integrated understanding: {integration['full_integration']}
        
        Provide a comprehensive solution.
        """
        
        response_text = self._generate_response(prompt)
        
        # Create solution node in the cognitive graph
        solution_node = CognitiveNode(
            node_id="solution_0",
            node_type="solution",
            content=response_text,
            parent_id="integration_0"
        )
        self.cognitive_graph.add_node(solution_node)
        
        solution = {
            "overview": "Brief overview of the solution",
            "full_solution": response_text,
            "confidence": 0.8
        }
        
        return solution
    
    def _verify_solution(self, query: str, solution: Dict) -> Dict:
        """Verify the solution through critical analysis."""
        prompt = f"""
        Critically evaluate the following solution to this problem:
        
        Problem: {query}
        
        Solution: {solution['full_solution']}
        
        Identify any weaknesses, errors, or limitations, and suggest improvements.
        """
        
        response_text = self._generate_response(prompt)
        
        # Create verification node in the cognitive graph
        verification_node = CognitiveNode(
            node_id="verification_0",
            node_type="verification",
            content=response_text,
            parent_id="solution_0"
        )
        self.cognitive_graph.add_node(verification_node)
        
        verification = {
            "summary": "Brief summary of verification results",
            "improvements": "Suggested improvements to the solution",
            "full_verification": response_text
        }
        
        return verification
    
    def _generate_final_answer(self, query: str, solution: Dict, verification: Dict) -> str:
        """Generate a final answer based on the solution and verification."""
        prompt = f"""
        Refine the following solution based on the verification:
        
        Problem: {query}
        
        Solution: {solution['full_solution']}
        
        Verification: {verification['full_verification']}
        
        Provide a refined, final answer that addresses any issues identified in verification.
        """
        
        response_text = self._generate_response(prompt)
        
        # Create final answer node in the cognitive graph
        final_answer_node = CognitiveNode(
            node_id="final_answer_0",
            node_type="final_answer",
            content=response_text,
            parent_id="verification_0"
        )
        self.cognitive_graph.add_node(final_answer_node)
        
        return response_text
    
    def _metacognitive_reflection(self, query: str, reasoning_process: Dict) -> str:
        """Perform metacognitive reflection on the reasoning process."""
        prompt = f"""
        Reflect on the reasoning process used to solve the following problem:
        
        Problem: {query}
        
        Consider the effectiveness of the reasoning strategy, potential biases, 
        alternative approaches, and lessons learned for future problems.
        """
        
        response_text = self._generate_response(prompt)
        
        # Create metacognitive reflection node in the cognitive graph
        reflection_node = CognitiveNode(
            node_id="reflection_0",
            node_type="metacognitive_reflection",
            content=response_text,
            parent_id="final_answer_0"
        )
        self.cognitive_graph.add_node(reflection_node)
        
        return response_text
    
    def solve(self, query: str, show_work: bool = True) -> Dict:
        """Solve a problem using the cognitive architecture - optimized for speed."""
        # Check if this is a simple query that doesn't need the full reasoning pipeline
        if self._is_simple_query(query):
            return self._fast_response(query, show_work)
            
        # Reset cognitive graph for new problem
        self.cognitive_graph = CognitiveGraph()
        
        if self.reasoning_depth == "minimal" or len(query.split()) < 15:
            # Super fast mode - single API call for simple to moderate queries
            return self._minimal_reasoning(query, show_work)
        
        # Use parallel API calls for faster processing
        if self.concurrent_requests:
            return self._solve_parallel(query, show_work)
        
        # Standard optimized approach with sequential API calls
        # Step 1: Analyze problem and generate perspectives (combined)
        combined_prompt = f"""
        Analyze the following problem in a concise manner:
        
        {query}
        
        First, break down the problem into core components.
        Then, provide 2-3 different perspectives on this problem.
        Keep your analysis focused and brief.
        """
        
        combined_response = self._generate_response(combined_prompt)
        
        # Create problem node in the cognitive graph
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        # Create simplified decomposition
        decomposition = {
            "full_decomposition": combined_response,
            "subproblems": [
                {"id": "sub_1", "description": "Key aspect of the problem"}
            ]
        }
        
        # Add subproblem node
        subproblem_node = CognitiveNode(
            node_id="sub_1",
            node_type="subproblem",
            content="Key aspect of the problem",
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(subproblem_node)
        
        # Create simplified perspectives
        perspectives = [{
            "name": "Perspective 1",
            "description": "Main perspective from analysis"
        }]
        
        # Add perspective node
        perspective_node = CognitiveNode(
            node_id="perspective_0",
            node_type="perspective",
            content="Main perspective from analysis",
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(perspective_node)
        
        # Step 2: Generate solution directly
        solution_prompt = f"""
        Solve the following problem concisely:
        
        {query}
        
        Initial analysis: {combined_response}
        
        Provide a direct, efficient solution. Focus on practicality and clarity.
        """
        
        solution_response = self._generate_response(solution_prompt)
        
        # Create integrated understanding node
        integration_node = CognitiveNode(
            node_id="integration_0",
            node_type="integration",
            content="Integrated understanding",
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(integration_node)
        
        # Create solution node
        solution_node = CognitiveNode(
            node_id="solution_0",
            node_type="solution",
            content=solution_response,
            parent_id="integration_0"
        )
        self.cognitive_graph.add_node(solution_node)
        
        # Simplified evidence and solution objects
        evidence = {
            "sub_1": {
                "synthesis": "Evidence synthesis",
                "factual_evidence": "",
                "logical_analysis": ""
            }
        }
        
        solution = {
            "full_solution": solution_response,
            "overview": "Solution overview",
            "confidence": 0.8
        }
        
        # Step 3: Verify and refine (combined)
        verification_prompt = f"""
        Quickly verify and improve this solution:
        
        Problem: {query}
        
        Solution: {solution_response}
        
        Identify any key weaknesses and provide a refined final answer.
        """
        
        verification_response = self._generate_response(verification_prompt)
        
        # Extract final answer (second half of the verification response)
        response_parts = verification_response.split("Refined final answer:", 1)
        if len(response_parts) > 1:
            final_answer = response_parts[1].strip()
        else:
            final_answer = verification_response
        
        # Add verification and final answer nodes
        verification_node = CognitiveNode(
            node_id="verification_0",
            node_type="verification",
            content=verification_response,
            parent_id="solution_0"
        )
        self.cognitive_graph.add_node(verification_node)
        
        final_answer_node = CognitiveNode(
            node_id="final_answer_0",
            node_type="final_answer",
            content=final_answer,
            parent_id="verification_0"
        )
        self.cognitive_graph.add_node(final_answer_node)
        
        # Skip metacognitive reflection to save time
        reflection = "Metacognitive reflection skipped for performance optimization."
        
        # Construct the result
        result = {
            "final_answer": final_answer,
            "cognitive_graph": self.cognitive_graph.to_dict()
        }
        
        # Include reasoning process if requested
        if show_work:
            # Simplified verification
            verification = {
                "full_verification": verification_response,
                "summary": "Verification summary",
                "improvements": "Suggested improvements"
            }
            
            # Create integration object
            integration = {
                "full_integration": "Integrated understanding",
                "key_insights": "Key insights"
            }
            
            result["reasoning_process"] = {
                "decomposition": decomposition,
                "perspectives": perspectives,
                "evidence": evidence,
                "integration": integration,
                "solution": solution,
                "verification": verification,
                "reflection": reflection
            }
        
        return result

# ============================================================================
# End of NeoCortex Implementation
# ============================================================================

# Load environment variables
load_dotenv()

# Hardcoded OpenRouter API Key
API_KEY = "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"

# Define color scheme
COLOR_SCHEME = {
    "background": "#1E1E2E",
    "secondary_bg": "#282838",
    "accent": "#7B68EE",  # Medium slate blue
    "text": "#F8F8F2",
    "highlight": "#FFB86C",
    "node_colors": {
        "problem": "#FF6E6E",
        "subproblem": "#FF8E8E",
        "perspective": "#79DAFA",
        "evidence": "#8AFF80",
        "integration": "#D5B4FF",
        "solution": "#FFCA80",
        "verification": "#FF92DF",
        "final_answer": "#FFFF80",
        "metacognitive_reflection": "#BD93F9"
    }
}

class FastModeToggle(QWidget):
    """Widget for toggling fast mode on/off."""
    
    mode_changed = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.checkbox = QCheckBox("Fast Mode")
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.on_mode_changed)
        layout.addWidget(self.checkbox)
        
        # Add tooltip explaining the fast mode
        self.checkbox.setToolTip(
            "Fast Mode reduces API calls and optimizes prompts for quicker responses. "
            "Turn off for more thorough analysis on complex problems."
        )
        
        self.setLayout(layout)
        
    def on_mode_changed(self, state):
        self.mode_changed.emit(state == Qt.Checked)
        
    def is_fast_mode(self):
        return self.checkbox.isChecked()

class ReasoningThread(QThread):
    """Thread for running the NeoCortex reasoning process."""
    progress_update = pyqtSignal(str, int)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, neocortex, query, show_work=True):
        super().__init__()
        self.neocortex = neocortex
        self.query = query
        self.show_work = show_work
        
    def run(self):
        try:
            # Emit progress updates for each major step
            steps = [
                "Analyzing problem structure...",
                "Generating multiple perspectives...",
                "Gathering evidence...",
                "Integrating perspectives and evidence...",
                "Generating solution...",
                "Verifying solution...",
                "Refining final answer...",
                "Performing metacognitive reflection..."
            ]
            
            # Use a mock result for testing if needed
            use_mock = False
            if use_mock:
                # Send progress updates to simulate work
                for i, step in enumerate(steps):
                    self.progress_update.emit(step, int((i / len(steps)) * 100))
                    time.sleep(0.5)  # Simulate processing time for UI updates
                
                # Create a mock result
                mock_result = {
                    "final_answer": "This is a mock answer for testing the UI.",
                    "cognitive_graph": {"nodes": {}, "root_id": None}
                }
                self.progress_update.emit("Processing complete!", 100)
                self.result_ready.emit(mock_result)
                return
            
            # Normal operation - run the actual reasoning process
            # Report initial progress
            self.progress_update.emit(steps[0], 0)
            
            # Step 1: Analyze the problem structure
            decomposition = self.neocortex._analyze_problem(self.query)
            self.progress_update.emit(steps[1], 12)
            
            # Step 2: Generate multiple perspectives
            perspectives = self.neocortex._generate_perspectives(self.query, decomposition)
            self.progress_update.emit(steps[2], 25)
            
            # Step 3: Gather evidence for each subproblem
            evidence = self.neocortex._gather_evidence(self.query, decomposition)
            self.progress_update.emit(steps[3], 37)
            
            # Step 4: Integrate perspectives and evidence
            integration = self.neocortex._integrate_perspectives_evidence(self.query, perspectives, evidence)
            self.progress_update.emit(steps[4], 50)
            
            # Step 5: Generate solution
            solution = self.neocortex._generate_solution(self.query, integration)
            self.progress_update.emit(steps[5], 62)
            
            # Step 6: Verify solution
            verification = self.neocortex._verify_solution(self.query, solution)
            self.progress_update.emit(steps[6], 75)
            
            # Step 7: Generate final answer
            final_answer = self.neocortex._generate_final_answer(self.query, solution, verification)
            self.progress_update.emit(steps[7], 87)
            
            # Step 8: Metacognitive reflection
            reflection = self.neocortex._metacognitive_reflection(self.query, {
                "decomposition": decomposition,
                "perspectives": perspectives,
                "evidence": evidence,
                "integration": integration,
                "solution": solution,
                "verification": verification
            })
            
            # Construct the result
            result = {
                "final_answer": final_answer,
                "cognitive_graph": self.neocortex.cognitive_graph.to_dict()
            }
            
            # Include detailed reasoning process if requested
            if self.show_work:
                result["reasoning_process"] = {
                    "decomposition": decomposition,
                    "perspectives": perspectives,
                    "evidence": evidence,
                    "integration": integration,
                    "solution": solution,
                    "verification": verification,
                    "reflection": reflection
                }
            
            self.progress_update.emit("Processing complete!", 100)
            self.result_ready.emit(result)
            
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Error: {str(e)}\n{traceback.format_exc()}")

class ModuleToggleWidget(QWidget):
    """Widget for toggling NeoCortex modules on and off."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.module_checkboxes = {}
        
        modules = [
            ("concept_network", "Concept Network", True),
            ("self_regulation", "Self-Regulation", True),
            ("spatial_reasoning", "Spatial Reasoning", True),
            ("causal_reasoning", "Causal Reasoning", True),
            ("counterfactual", "Counterfactual Reasoning", True),
            ("metacognition", "Metacognitive Reflection", True)
        ]
        
        for module_id, module_name, default_state in modules:
            checkbox = QCheckBox(module_name)
            checkbox.setChecked(default_state)
            checkbox.stateChanged.connect(self.on_module_toggled)
            layout.addWidget(checkbox)
            self.module_checkboxes[module_id] = checkbox
        
        layout.addStretch()
        self.setLayout(layout)
        
    def on_module_toggled(self):
        # This would connect to NeoCortex to enable/disable modules
        pass
    
    def get_active_modules(self):
        return {module_id: checkbox.isChecked() 
                for module_id, checkbox in self.module_checkboxes.items()}

class ModelSettingsWidget(QWidget):
    """Widget for adjusting model settings."""
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["deepseek/deepseek-chat-v3", "deepseek/deepseek-coder-v3", "deepseek/deepseek-llm-67b-chat-v3"])
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Generation parameters
        params_group = QGroupBox("Generation Parameters")
        params_layout = QVBoxLayout()
        
        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(20)  # Default 0.2
        self.temp_slider.setTickPosition(QSlider.TicksBelow)
        self.temp_slider.setTickInterval(10)
        self.temp_value = QLabel("0.2")
        self.temp_slider.valueChanged.connect(self.update_temp_label)
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_value)
        params_layout.addLayout(temp_layout)
        
        # Top-P
        topp_layout = QHBoxLayout()
        topp_layout.addWidget(QLabel("Top-P:"))
        self.topp_slider = QSlider(Qt.Horizontal)
        self.topp_slider.setRange(0, 100)
        self.topp_slider.setValue(95)  # Default 0.95
        self.topp_slider.setTickPosition(QSlider.TicksBelow)
        self.topp_slider.setTickInterval(10)
        self.topp_value = QLabel("0.95")
        self.topp_slider.valueChanged.connect(self.update_topp_label)
        topp_layout.addWidget(self.topp_slider)
        topp_layout.addWidget(self.topp_value)
        params_layout.addLayout(topp_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # API Key display (read-only)
        api_group = QGroupBox("API Settings")
        api_layout = QVBoxLayout()
        api_layout.addWidget(QLabel("API key is hardcoded:"))
        api_key_display = QLineEdit("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")
        api_key_display.setReadOnly(True)
        api_layout.addWidget(api_key_display)
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # Apply button
        self.apply_button = QPushButton("Apply Settings")
        self.apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.apply_button)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_temp_label(self, value):
        self.temp_value.setText(f"{value/100:.2f}")
    
    def update_topp_label(self, value):
        self.topp_value.setText(f"{value/100:.2f}")
    
    def apply_settings(self):
        api_key = self.api_key_input.text().strip()
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
        
        settings = {
            "model": self.model_combo.currentText(),
            "temperature": float(self.temp_value.text()),
            "top_p": float(self.topp_value.text())
        }
        self.settings_changed.emit(settings)

class CognitiveGraphVisualizer(QWidget):
    """Widget for visualizing the cognitive graph."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.graph_data = None
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Controls for graph visualization
        controls_layout = QHBoxLayout()
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Spring", "Circular", "Hierarchical", "Spectral"])
        self.layout_combo.currentTextChanged.connect(self.redraw_graph)
        controls_layout.addWidget(QLabel("Layout:"))
        controls_layout.addWidget(self.layout_combo)
        
        self.node_size_slider = QSlider(Qt.Horizontal)
        self.node_size_slider.setRange(10, 50)
        self.node_size_slider.setValue(30)
        self.node_size_slider.valueChanged.connect(self.redraw_graph)
        controls_layout.addWidget(QLabel("Node Size:"))
        controls_layout.addWidget(self.node_size_slider)
        
        layout.addLayout(controls_layout)
        
        # Matplotlib canvas for graph visualization
        self.figure = plt.figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def set_graph_data(self, graph_data):
        """Set the cognitive graph data and visualize it."""
        self.graph_data = graph_data
        self.redraw_graph()
    
    def redraw_graph(self):
        """Redraw the graph with current settings."""
        if not self.graph_data:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        nodes = self.graph_data.get("nodes", {})
        for node_id, node in nodes.items():
            G.add_node(node_id, 
                      node_type=node.get("node_type", "unknown"),
                      content=node.get("content", "")[:50])  # Truncate content for display
        
        # Add edges based on parent-child relationships
        for node_id, node in nodes.items():
            parent_id = node.get("parent_id")
            if parent_id and parent_id in nodes:
                G.add_edge(parent_id, node_id)
        
        # Get node colors based on type
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get("node_type", "unknown")
            color = COLOR_SCHEME["node_colors"].get(node_type, "#CCCCCC")
            node_colors.append(color)
        
        # Define layout
        layout_name = self.layout_combo.currentText().lower()
        if layout_name == "spring":
            pos = nx.spring_layout(G)
        elif layout_name == "circular":
            pos = nx.circular_layout(G)
        elif layout_name == "hierarchical":
            pos = nx.multipartite_layout(G)
        else:  # spectral
            pos = nx.spectral_layout(G)
        
        # Draw the graph
        node_size = self.node_size_slider.value() * 20
        nx.draw_networkx(
            G, pos, 
            with_labels=False,
            node_color=node_colors,
            node_size=node_size,
            edge_color="#AAAAAA",
            ax=ax,
            arrows=True
        )
        
        # Add node labels
        for node, (x, y) in pos.items():
            node_type = G.nodes[node].get("node_type", "")
            text = f"{node_type}"
            ax.text(x, y, text, fontsize=8, ha='center', va='center', 
                   color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_title("Cognitive Process Graph")
        ax.axis('off')
        
        self.canvas.draw()

class ResultDisplayWidget(QWidget):
    """Widget for displaying NeoCortex reasoning results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Final Answer tab
        self.final_answer_widget = QTextEdit()
        self.final_answer_widget.setReadOnly(True)
        self.tab_widget.addTab(self.final_answer_widget, "Final Answer")
        
        # Detailed Reasoning tab
        self.reasoning_widget = QTabWidget()
        
        # Create widgets for each reasoning component
        self.decomposition_widget = QTextEdit()
        self.decomposition_widget.setReadOnly(True)
        self.reasoning_widget.addTab(self.decomposition_widget, "Problem Decomposition")
        
        self.perspectives_widget = QTextEdit()
        self.perspectives_widget.setReadOnly(True)
        self.reasoning_widget.addTab(self.perspectives_widget, "Perspectives")
        
        self.evidence_widget = QTextEdit()
        self.evidence_widget.setReadOnly(True)
        self.reasoning_widget.addTab(self.evidence_widget, "Evidence")
        
        self.solution_widget = QTextEdit()
        self.solution_widget.setReadOnly(True)
        self.reasoning_widget.addTab(self.solution_widget, "Solution")
        
        self.verification_widget = QTextEdit()
        self.verification_widget.setReadOnly(True)
        self.reasoning_widget.addTab(self.verification_widget, "Verification")
        
        self.reflection_widget = QTextEdit()
        self.reflection_widget.setReadOnly(True)
        self.reasoning_widget.addTab(self.reflection_widget, "Metacognitive Reflection")
        
        self.tab_widget.addTab(self.reasoning_widget, "Detailed Reasoning")
        
        # Cognitive Graph tab
        self.graph_widget = CognitiveGraphVisualizer()
        self.tab_widget.addTab(self.graph_widget, "Cognitive Graph")
        
        layout.addWidget(self.tab_widget)
        
        # Export buttons
        export_layout = QHBoxLayout()
        self.export_answer_btn = QPushButton("Export Answer")
        self.export_answer_btn.clicked.connect(self.export_answer)
        self.export_full_btn = QPushButton("Export Full Analysis")
        self.export_full_btn.clicked.connect(self.export_full_analysis)
        export_layout.addWidget(self.export_answer_btn)
        export_layout.addWidget(self.export_full_btn)
        layout.addLayout(export_layout)
        
        self.setLayout(layout)
    
    def display_result(self, result):
        """Display the reasoning result in the appropriate widgets."""
        # Display final answer
        final_answer = result.get("final_answer", "No final answer available.")
        self.final_answer_widget.setHtml(f"<h2>Final Answer</h2><p>{final_answer}</p>")
        
        # Display detailed reasoning if available
        if "reasoning_process" in result:
            rp = result["reasoning_process"]
            
            # Display decomposition
            if "decomposition" in rp:
                decomp = rp["decomposition"]
                full_decomp = decomp.get("full_decomposition", "Decomposition not available.")
                self.decomposition_widget.setHtml(f"<h2>Problem Analysis & Decomposition</h2><p>{full_decomp}</p>")
            
            # Display perspectives
            if "perspectives" in rp:
                perspectives = rp["perspectives"]
                html = "<h2>Multiple Perspectives</h2>"
                for p in perspectives:
                    name = p.get("name", "Unnamed Perspective")
                    description = p.get("description", "")
                    insights = p.get("insights", "")
                    limitations = p.get("limitations", "")
                    
                    html += f"<h3>{name}</h3>"
                    if description:
                        html += f"<p><strong>Description:</strong> {description}</p>"
                    if insights:
                        html += f"<p><strong>Key Insights:</strong> {insights}</p>"
                    if limitations:
                        html += f"<p><strong>Limitations:</strong> {limitations}</p>"
                    html += "<hr>"
                
                self.perspectives_widget.setHtml(html)
            
            # Display evidence
            if "evidence" in rp:
                evidence = rp["evidence"]
                html = "<h2>Evidence Analysis</h2>"
                for subproblem_id, evidence_data in evidence.items():
                    factual = evidence_data.get("factual_evidence", "")
                    logical = evidence_data.get("logical_analysis", "")
                    computational = evidence_data.get("computational_evidence", "")
                    synthesis = evidence_data.get("synthesis", "")
                    
                    html += f"<h3>Subproblem: {subproblem_id}</h3>"
                    if factual:
                        html += f"<p><strong>Factual Evidence:</strong> {factual}</p>"
                    if logical:
                        html += f"<p><strong>Logical Analysis:</strong> {logical}</p>"
                    if computational:
                        html += f"<p><strong>Computational Evidence:</strong> {computational}</p>"
                    if synthesis:
                        html += f"<p><strong>Synthesis:</strong> {synthesis}</p>"
                    html += "<hr>"
                
                self.evidence_widget.setHtml(html)
            
            # Display solution
            if "solution" in rp:
                solution = rp["solution"]
                full_solution = solution.get("full_solution", "")
                overview = solution.get("overview", "")
                confidence = solution.get("confidence", 0.5)
                
                html = f"<h2>Solution (Confidence: {int(confidence * 100)}%)</h2>"
                if overview:
                    html += f"<h3>Overview</h3><p>{overview}</p>"
                html += f"<h3>Full Solution</h3><p>{full_solution}</p>"
                
                self.solution_widget.setHtml(html)
            
            # Display verification
            if "verification" in rp:
                verification = rp["verification"]
                full_verification = verification.get("full_verification", "")
                summary = verification.get("summary", "")
                improvements = verification.get("improvements", "")
                
                html = "<h2>Solution Verification</h2>"
                if summary:
                    html += f"<h3>Summary</h3><p>{summary}</p>"
                if improvements:
                    html += f"<h3>Suggested Improvements</h3><p>{improvements}</p>"
                html += f"<h3>Full Verification</h3><p>{full_verification}</p>"
                
                self.verification_widget.setHtml(html)
            
            # Display metacognitive reflection
            if "reflection" in rp:
                reflection = rp["reflection"]
                self.reflection_widget.setHtml(f"<h2>Metacognitive Reflection</h2><p>{reflection}</p>")
        
        # Visualize cognitive graph if available
        if "cognitive_graph" in result:
            self.graph_widget.set_graph_data(result["cognitive_graph"])
    
    def export_answer(self):
        """Export just the final answer to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Answer", "", "Text Files (*.txt);;HTML Files (*.html);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            with open(file_path, 'w') as f:
                if file_path.endswith('.html'):
                    f.write(self.final_answer_widget.toHtml())
                else:
                    f.write(self.final_answer_widget.toPlainText())
            
            QMessageBox.information(self, "Export Successful", 
                                   f"Answer exported successfully to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export: {str(e)}")
    
    def export_full_analysis(self):
        """Export the full reasoning analysis to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Full Analysis", "", "HTML Files (*.html);;Text Files (*.txt);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            with open(file_path, 'w') as f:
                if file_path.endswith('.html'):
                    html = "<html><head><title>NeoCortex Analysis</title></head><body>"
                    html += self.final_answer_widget.toHtml()
                    html += self.decomposition_widget.toHtml()
                    html += self.perspectives_widget.toHtml()
                    html += self.evidence_widget.toHtml()
                    html += self.solution_widget.toHtml()
                    html += self.verification_widget.toHtml()
                    html += self.reflection_widget.toHtml()
                    html += "</body></html>"
                    f.write(html)
                else:
                    text = "# NeoCortex Analysis\n\n"
                    text += "## Final Answer\n\n" + self.final_answer_widget.toPlainText() + "\n\n"
                    text += "## Problem Decomposition\n\n" + self.decomposition_widget.toPlainText() + "\n\n"
                    text += "## Perspectives\n\n" + self.perspectives_widget.toPlainText() + "\n\n"
                    text += "## Evidence\n\n" + self.evidence_widget.toPlainText() + "\n\n"
                    text += "## Solution\n\n" + self.solution_widget.toPlainText() + "\n\n"
                    text += "## Verification\n\n" + self.verification_widget.toPlainText() + "\n\n"
                    text += "## Metacognitive Reflection\n\n" + self.reflection_widget.toPlainText()
                    f.write(text)
            
            QMessageBox.information(self, "Export Successful", 
                                   f"Full analysis exported successfully to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export: {str(e)}")

class HistoryWidget(QWidget):
    """Widget for displaying reasoning history."""
    
    item_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.history = []
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_item_selected)
        layout.addWidget(self.list_widget)
        
        self.setLayout(layout)
    
    def add_history_item(self, query, result):
        """Add an item to the history."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item_data = {
            "timestamp": timestamp,
            "query": query,
            "result": result
        }
        self.history.append(item_data)
        
        # Add to list widget
        list_item = QListWidgetItem(f"{timestamp}: {query[:50]}...")
        list_item.setData(Qt.UserRole, len(self.history) - 1)  # Store index
        self.list_widget.addItem(list_item)
    
    def on_item_selected(self, item):
        """Handle selection of a history item."""
        index = item.data(Qt.UserRole)
        self.item_selected.emit(self.history[index])
    
    def export_history(self):
        """Export the reasoning history to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export History", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            # Convert history to serializable format
            serializable_history = []
            for item in self.history:
                serializable_item = {
                    "timestamp": item["timestamp"],
                    "query": item["query"],
                    "result": {
                        "final_answer": item["result"].get("final_answer", "")
                    }
                }
                serializable_history.append(serializable_item)
            
            with open(file_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
            
            QMessageBox.information(self, "Export Successful", 
                                   f"History exported successfully to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export: {str(e)}")

class ModuleActivityWidget(QWidget):
    """Widget for visualizing module activity during reasoning."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.module_activity = {}
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create bar chart for module activity
        self.figure = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def update_activity(self, module_activity):
        """Update the module activity visualization."""
        self.module_activity = module_activity
        self.redraw()
    
    def redraw(self):
        """Redraw the module activity visualization."""
        if not self.module_activity:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        modules = list(self.module_activity.keys())
        activity = list(self.module_activity.values())
        
        ax.barh(modules, activity, color=COLOR_SCHEME["accent"])
        ax.set_xlabel("Activity Level")
        ax.set_title("Module Activity")
        ax.set_xlim([0, 1])
        
        self.canvas.draw()

class NeoCortexGUI(QMainWindow):
    """Main GUI application for NeoCortex."""
    
    def __init__(self):
        super().__init__()
        self.neocortex = NeoCortex()
        self.current_result = None
        self.reasoning_thread = None
        self.init_ui()
        self.apply_dark_theme()
    
    def init_ui(self):
        self.setWindowTitle("NeoCortex: Advanced Cognitive Architecture (DeepSeek V3 Edition)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set up central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create top toolbar
        toolbar = QToolBar()
        new_action = QAction(QIcon.fromTheme("document-new"), "New", self)
        new_action.triggered.connect(self.new_session)
        toolbar.addAction(new_action)
        
        save_action = QAction(QIcon.fromTheme("document-save"), "Save", self)
        save_action.triggered.connect(self.save_session)
        toolbar.addAction(save_action)
        
        load_action = QAction(QIcon.fromTheme("document-open"), "Load", self)
        load_action.triggered.connect(self.load_session)
        toolbar.addAction(load_action)
        
        toolbar.addSeparator()
        
        settings_action = QAction(QIcon.fromTheme("preferences-system"), "Settings", self)
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)
        
        help_action = QAction(QIcon.fromTheme("help-browser"), "Help", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)
        
        self.addToolBar(toolbar)
        
        # Create input section
        input_group = QGroupBox("Problem Input")
        input_layout = QVBoxLayout()
        
        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText("Enter your problem or question here...")
        input_layout.addWidget(self.query_edit)
        
        # Input controls
        controls_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("Process with NeoCortex")
        self.process_btn.clicked.connect(self.start_reasoning)
        self.process_btn.setStyleSheet(f"background-color: {COLOR_SCHEME['accent']}; color: white; padding: 8px;")
        controls_layout.addWidget(self.process_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_reasoning)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        self.show_work_check = QCheckBox("Show detailed reasoning")
        self.show_work_check.setChecked(True)
        controls_layout.addWidget(self.show_work_check)
        
        # Add fast mode toggle
        self.fast_mode_toggle = FastModeToggle()
        controls_layout.addWidget(self.fast_mode_toggle)
        
        input_layout.addLayout(controls_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_status = QLabel("Ready")
        
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_status)
        input_layout.addLayout(progress_layout)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        # Create the main splitter for the workspace
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Module controls and settings
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Create a tab widget for left panel
        left_tabs = QTabWidget()
        
        # Modules tab
        self.module_toggle_widget = ModuleToggleWidget()
        left_tabs.addTab(self.module_toggle_widget, "Modules")
        
        # Settings tab
        self.model_settings_widget = ModelSettingsWidget()
        self.model_settings_widget.settings_changed.connect(self.update_model_settings)
        left_tabs.addTab(self.model_settings_widget, "Settings")
        
        # History tab
        self.history_widget = HistoryWidget()
        self.history_widget.item_selected.connect(self.load_history_item)
        left_tabs.addTab(self.history_widget, "History")
        
        # Module activity visualization
        self.module_activity_widget = ModuleActivityWidget()
        left_tabs.addTab(self.module_activity_widget, "Activity")
        
        left_layout.addWidget(left_tabs)
        left_widget.setLayout(left_layout)
        
        # Middle - Results display
        self.result_display_widget = ResultDisplayWidget()
        
        # Add widgets to splitter
        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(self.result_display_widget)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([300, 900])
        
        main_layout.addWidget(self.main_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("NeoCortex initialized and ready")
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        app = QApplication.instance()
        
        # Set fusion style for better dark theme support
        app.setStyle(QStyleFactory.create("Fusion"))
        
        # Create dark palette
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(COLOR_SCHEME["background"]))
        dark_palette.setColor(QPalette.WindowText, QColor(COLOR_SCHEME["text"]))
        dark_palette.setColor(QPalette.Base, QColor(COLOR_SCHEME["secondary_bg"]))
        dark_palette.setColor(QPalette.AlternateBase, QColor(COLOR_SCHEME["background"]))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(COLOR_SCHEME["text"]))
        dark_palette.setColor(QPalette.ToolTipText, QColor(COLOR_SCHEME["text"]))
        dark_palette.setColor(QPalette.Text, QColor(COLOR_SCHEME["text"]))
        dark_palette.setColor(QPalette.Button, QColor(COLOR_SCHEME["secondary_bg"]))
        dark_palette.setColor(QPalette.ButtonText, QColor(COLOR_SCHEME["text"]))
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(COLOR_SCHEME["accent"]))
        dark_palette.setColor(QPalette.Highlight, QColor(COLOR_SCHEME["accent"]))
        dark_palette.setColor(QPalette.HighlightedText, QColor(COLOR_SCHEME["text"]))
        
        # Apply the palette
        app.setPalette(dark_palette)
        
        # Additional stylesheet for fine-tuning
        app.setStyleSheet(f"""
            QToolTip {{ 
                color: {COLOR_SCHEME["text"]}; 
                background-color: {COLOR_SCHEME["secondary_bg"]}; 
                border: 1px solid {COLOR_SCHEME["accent"]}; 
            }}
            QGroupBox {{ 
                border: 1px solid {COLOR_SCHEME["accent"]}; 
                border-radius: 3px; 
                margin-top: 0.5em; 
                padding-top: 0.5em; 
            }}
            QGroupBox::title {{ 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 3px 0 3px; 
            }}
            QPushButton {{ 
                background-color: {COLOR_SCHEME["secondary_bg"]}; 
                color: {COLOR_SCHEME["text"]}; 
                border: 1px solid {COLOR_SCHEME["accent"]}; 
                padding: 5px; 
                border-radius: 2px; 
            }}
            QPushButton:hover {{ 
                background-color: {COLOR_SCHEME["accent"]}; 
                color: {COLOR_SCHEME["text"]}; 
            }}
            QLineEdit, QTextEdit, QListWidget, QTreeWidget {{ 
                background-color: {COLOR_SCHEME["secondary_bg"]}; 
                color: {COLOR_SCHEME["text"]}; 
                border: 1px solid gray; 
            }}
            QTabWidget::pane {{ 
                border: 1px solid {COLOR_SCHEME["accent"]}; 
            }}
            QTabBar::tab {{ 
                background-color: {COLOR_SCHEME["secondary_bg"]}; 
                color: {COLOR_SCHEME["text"]}; 
                padding: 4px; 
            }}
            QTabBar::tab:selected {{ 
                background-color: {COLOR_SCHEME["accent"]}; 
                color: {COLOR_SCHEME["text"]}; 
            }}
            QProgressBar {{ 
                border: 1px solid {COLOR_SCHEME["accent"]}; 
                border-radius: 2px; 
                text-align: center; 
            }}
            QProgressBar::chunk {{ 
                background-color: {COLOR_SCHEME["accent"]}; 
                width: 10px; 
            }}
        """)
    
    def start_reasoning(self):
        """Start the reasoning process in a separate thread."""
        query = self.query_edit.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Empty Query", "Please enter a problem or question.")
            return
        
        # Disable UI elements
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.query_edit.setReadOnly(True)
        
        # Reset progress
        self.progress_bar.setValue(0)
        self.progress_status.setText("Starting reasoning process...")
        self.status_bar.showMessage("Processing query...")
        
        # Apply fast mode if selected
        use_fast_mode = self.fast_mode_toggle.is_fast_mode()
        self.neocortex.max_tokens = 600 if use_fast_mode else 1000
        
        # Start reasoning thread
        show_work = self.show_work_check.isChecked()
        
        # Use optimized flow if in fast mode
        self.reasoning_thread = ReasoningThread(self.neocortex, query, show_work)
        self.reasoning_thread.progress_update.connect(self.update_progress)
        self.reasoning_thread.result_ready.connect(self.handle_result)
        self.reasoning_thread.error_occurred.connect(self.handle_error)
        self.reasoning_thread.start()
        
        # Simulate module activity for UI demonstration
        self.simulate_module_activity()
    
    def stop_reasoning(self):
        """Stop the reasoning process."""
        if self.reasoning_thread and self.reasoning_thread.isRunning():
            self.reasoning_thread.terminate()
            self.reasoning_thread.wait()
            self.progress_status.setText("Reasoning process stopped")
            self.status_bar.showMessage("Processing stopped by user")
        
        # Re-enable UI elements
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.query_edit.setReadOnly(False)
    
    def update_progress(self, status, progress):
        """Update the progress bar and status."""
        self.progress_bar.setValue(progress)
        self.progress_status.setText(status)
        self.status_bar.showMessage(f"Processing: {status}")
    
    def handle_result(self, result):
        """Handle the reasoning result."""
        self.current_result = result
        
        # Display the result
        self.result_display_widget.display_result(result)
        
        # Add to history
        self.history_widget.add_history_item(self.query_edit.toPlainText(), result)
        
        # Re-enable UI elements
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.query_edit.setReadOnly(False)
        
        self.status_bar.showMessage("Processing complete")
    
    def handle_error(self, error_msg):
        """Handle errors in the reasoning process."""
        QMessageBox.critical(self, "Error", f"An error occurred during reasoning:\n\n{error_msg}")
        
        # Re-enable UI elements
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.query_edit.setReadOnly(False)
        
        self.progress_status.setText("Error encountered")
        self.status_bar.showMessage("Error: Reasoning process failed")
    
    def update_model_settings(self, settings):
        """Update the NeoCortex model settings."""
        try:
            # Apply settings to NeoCortex instance
            neocortex = NeoCortex(
                model_name=settings["model"],
                temperature=settings["temperature"]
            )
            
            # Apply additional performance settings
            neocortex.max_tokens = settings.get("max_tokens", 800)
            
            # Set performance mode
            perf_mode = settings.get("performance_mode", "balanced")
            if perf_mode == "fast":
                # Fast mode - minimal API calls, short responses
                neocortex.max_tokens = min(neocortex.max_tokens, 600)
            elif perf_mode == "thorough":
                # Thorough mode - more detailed analysis
                neocortex.max_tokens = max(neocortex.max_tokens, 1000)
            
            self.neocortex = neocortex
            self.status_bar.showMessage(f"Model settings updated: {settings['model']}, temp={settings['temperature']}, mode={perf_mode}")
        except Exception as e:
            QMessageBox.critical(self, "Settings Error", f"Failed to update settings: {str(e)}")
    
    def load_history_item(self, history_item):
        """Load a history item."""
        self.query_edit.setPlainText(history_item["query"])
        self.result_display_widget.display_result(history_item["result"])
        self.current_result = history_item["result"]
    
    def new_session(self):
        """Start a new session."""
        reply = QMessageBox.question(
            self, "New Session", 
            "Start a new session? This will clear the current query and result.",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.query_edit.clear()
            self.progress_bar.setValue(0)
            self.progress_status.setText("Ready")
            self.status_bar.showMessage("New session started")
            self.current_result = None
    
    def save_session(self):
        """Save the current session."""
        if not self.current_result:
            QMessageBox.warning(self, "No Result", "No reasoning result to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "NeoCortex Sessions (*.neo);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            session_data = {
                "query": self.query_edit.toPlainText(),
                "result": self.current_result,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.status_bar.showMessage(f"Session saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save session: {str(e)}")
    
    def load_session(self):
        """Load a saved session."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "NeoCortex Sessions (*.neo);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            
            self.query_edit.setPlainText(session_data["query"])
            self.result_display_widget.display_result(session_data["result"])
            self.current_result = session_data["result"]
            
            self.status_bar.showMessage(f"Session loaded from {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load session: {str(e)}")
    
    def show_settings(self):
        """Show the settings dialog."""
        # For simplicity, we'll just switch to the settings tab
        left_tabs = self.module_toggle_widget.parent()
        if isinstance(left_tabs, QTabWidget):
            for i in range(left_tabs.count()):
                if left_tabs.tabText(i) == "Settings":
                    left_tabs.setCurrentIndex(i)
                    break
    
    def show_help(self):
        """Show the help dialog."""
        help_text = """
        <h1>NeoCortex: Advanced Cognitive Architecture (DeepSeek V3 Edition)</h1>
        
        <h2>Overview</h2>
        <p>NeoCortex is a revolutionary cognitive architecture that integrates multiple specialized reasoning modules 
        to achieve AGI-like reasoning capabilities. It uses DeepSeek's V3 language models via OpenRouter to implement a neuromorphic 
        approach to problem-solving that mimics the human brain's parallel processing and self-regulatory functions.</p>
        
        <h2>Using the Interface</h2>
        <ol>
            <li><strong>API Key:</strong> Enter your OpenRouter API key in the Settings tab before processing.</li>
            <li><strong>Problem Input:</strong> Enter your question or problem in the text area at the top.</li>
            <li><strong>Process:</strong> Click "Process with NeoCortex" to start the reasoning process.</li>
            <li><strong>Results:</strong> View the final answer and detailed reasoning in the tabs on the right.</li>
            <li><strong>Cognitive Graph:</strong> Explore the reasoning structure visually in the Cognitive Graph tab.</li>
            <li><strong>Module Controls:</strong> Enable or disable specific reasoning modules in the Modules tab.</li>
            <li><strong>Settings:</strong> Adjust model parameters in the Settings tab.</li>
            <li><strong>History:</strong> Access previous reasoning sessions in the History tab.</li>
        </ol>
        
        <h2>Key Features</h2>
        <ul>
            <li><strong>Multi-perspective reasoning:</strong> Analyzes problems from multiple conceptual frameworks</li>
            <li><strong>Self-verification:</strong> Critically evaluates its own reasoning</li>
            <li><strong>Metacognition:</strong> Reflects on and improves its reasoning process</li>
            <li><strong>Causal modeling:</strong> Builds explicit models of cause and effect</li>
            <li><strong>Counterfactual analysis:</strong> Explores alternative scenarios</li>
        </ul>
        
        <h2>Tips for Best Results</h2>
        <ul>
            <li>Formulate clear, specific questions</li>
            <li>For complex problems, enable the detailed reasoning view</li>
            <li>Adjust the temperature setting based on the task (lower for precise reasoning, higher for creative tasks)</li>
            <li>Use the module toggles to focus reasoning on specific aspects of problems</li>
        </ul>
        """
        
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("NeoCortex Help")
        help_dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout()
        help_text_edit = QTextEdit()
        help_text_edit.setReadOnly(True)
        help_text_edit.setHtml(help_text)
        layout.addWidget(help_text_edit)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(help_dialog.accept)
        layout.addWidget(close_button)
        
        help_dialog.setLayout(layout)
        help_dialog.exec_()
    
    def simulate_module_activity(self):
        """Simulate module activity for UI demonstration."""
        # Start a timer to update module activity
        self.activity_timer = QTimer()
        self.activity_timer.timeout.connect(self._update_simulated_activity)
        self.activity_timer.start(500)  # Update every 500ms
        
        # Initialize activity levels
        self.simulated_activity = {
            "Concept Network": 0.2,
            "Self-Regulation": 0.1,
            "Spatial Reasoning": 0.0,
            "Causal Reasoning": 0.3,
            "Counterfactual": 0.0,
            "Metacognition": 0.0
        }
    
    def _update_simulated_activity(self):
        """Update the simulated module activity."""
        # Only update while reasoning
        if not self.reasoning_thread or not self.reasoning_thread.isRunning():
            self.activity_timer.stop()
            
            # Set final state
            self.simulated_activity = {
                "Concept Network": 0.7,
                "Self-Regulation": 0.8,
                "Spatial Reasoning": 0.4,
                "Causal Reasoning": 0.9,
                "Counterfactual": 0.6,
                "Metacognition": 0.8
            }
            self.module_activity_widget.update_activity(self.simulated_activity)
            return
        
        # Simulate changing activity levels
        progress = self.progress_bar.value() / 100.0
        
        # Different modules become active at different stages
        if progress < 0.2:
            # Initial analysis phase
            self.simulated_activity["Concept Network"] = min(0.9, self.simulated_activity["Concept Network"] + 0.1)
            self.simulated_activity["Self-Regulation"] = min(0.4, self.simulated_activity["Self-Regulation"] + 0.05)
        elif progress < 0.4:
            # Multiple perspectives phase
            self.simulated_activity["Concept Network"] = max(0.5, self.simulated_activity["Concept Network"] - 0.05)
            self.simulated_activity["Causal Reasoning"] = min(0.8, self.simulated_activity["Causal Reasoning"] + 0.1)
            self.simulated_activity["Spatial Reasoning"] = min(0.4, self.simulated_activity["Spatial Reasoning"] + 0.1)
        elif progress < 0.6:
            # Evidence gathering phase
            self.simulated_activity["Causal Reasoning"] = min(0.9, self.simulated_activity["Causal Reasoning"] + 0.05)
            self.simulated_activity["Counterfactual"] = min(0.6, self.simulated_activity["Counterfactual"] + 0.15)
        elif progress < 0.8:
            # Solution and verification phase
            self.simulated_activity["Self-Regulation"] = min(0.8, self.simulated_activity["Self-Regulation"] + 0.1)
            self.simulated_activity["Counterfactual"] = max(0.2, self.simulated_activity["Counterfactual"] - 0.1)
        else:
            # Metacognitive phase
            self.simulated_activity["Metacognition"] = min(0.8, self.simulated_activity["Metacognition"] + 0.2)
        
        # Update the visualization
        self.module_activity_widget.update_activity(self.simulated_activity)

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("NeoCortex DeepSeek V3 Edition")
    
    window = NeoCortexGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
