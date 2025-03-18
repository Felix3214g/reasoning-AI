import sys
import os
import json
import time
import asyncio
import hashlib
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from datetime import datetime
import threading
import queue
import requests
import aiohttp
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
    QStyleFactory, QToolBar, QAction, QStatusBar, QGraphicsRectItem, QRadioButton,
    QInputDialog, QMenu
)

from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

# ============================================================================
# Enhanced NeoCortex Implementation with Advanced Capabilities
# ============================================================================

class Knowledge:
    """Class to represent knowledge in the cognitive system."""
    
    def __init__(self, content: str, source: str = "reasoning", confidence: float = 0.8,
                 domain: str = "general", timestamp: Optional[datetime] = None):
        self.content = content
        self.source = source
        self.confidence = confidence
        self.domain = domain
        self.timestamp = timestamp or datetime.now()
        self.uses = 0
        self.success_rate = 0.0
    
    def record_use(self, success: bool):
        """Record use of this knowledge piece and update success rate."""
        self.uses += 1
        if success:
            self.success_rate = ((self.success_rate * (self.uses - 1)) + 1) / self.uses
        else:
            self.success_rate = (self.success_rate * (self.uses - 1)) / self.uses
    
    def to_dict(self) -> Dict:
        """Convert Knowledge to a dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "domain": self.domain,
            "timestamp": self.timestamp.isoformat(),
            "uses": self.uses,
            "success_rate": self.success_rate
        }

class CognitiveNode:
    """Enhanced class to represent a node in the cognitive graph."""
    
    def __init__(self, node_id: str, node_type: str, content: str, 
                 parent_id: Optional[str] = None, domain: str = "general",
                 confidence: float = 0.8, metadata: Optional[Dict] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.content = content
        self.parent_id = parent_id
        self.children: List[str] = []
        self.domain = domain
        self.confidence = confidence
        self.metadata = metadata or {}
        self.creation_time = datetime.now()
        self.relationships = {}  # Map relationship types to node_ids
    
    def add_child(self, child_id: str, relationship_type: str = "child"):
        """Add a child node ID to this node with a specific relationship type."""
        if child_id not in self.children:
            self.children.append(child_id)
        
        # Add to relationship map
        if relationship_type not in self.relationships:
            self.relationships[relationship_type] = []
        if child_id not in self.relationships[relationship_type]:
            self.relationships[relationship_type].append(child_id)
    
    def get_children_by_relationship(self, relationship_type: str) -> List[str]:
        """Get children nodes with a specific relationship type."""
        return self.relationships.get(relationship_type, [])
    
    def add_metadata(self, key: str, value: Any):
        """Add a metadata key-value pair to this node."""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict:
        """Convert CognitiveNode to a dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "content": self.content,
            "parent_id": self.parent_id,
            "children": self.children,
            "domain": self.domain,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "creation_time": self.creation_time.isoformat(),
            "relationships": self.relationships
        }

class CognitiveGraph:
    """Enhanced class to represent the cognitive process graph."""
    
    def __init__(self):
        self.nodes: Dict[str, CognitiveNode] = {}
        self.root_id: Optional[str] = None
        self.domain_indices: Dict[str, List[str]] = {}  # Map domains to node_ids
        self.type_indices: Dict[str, List[str]] = {}  # Map node types to node_ids
        self.relationship_indices: Dict[str, List[Tuple[str, str]]] = {}  # Map relationship types to (source, target) pairs
    
    def add_node(self, node: CognitiveNode) -> str:
        """Add a node to the graph with enhanced indexing."""
        self.nodes[node.node_id] = node
        
        # Update domain index
        if node.domain not in self.domain_indices:
            self.domain_indices[node.domain] = []
        self.domain_indices[node.domain].append(node.node_id)
        
        # Update type index
        if node.node_type not in self.type_indices:
            self.type_indices[node.node_type] = []
        self.type_indices[node.node_type].append(node.node_id)
        
        # If this node has a parent, add it as a child to the parent
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].add_child(node.node_id)
            # Update relationship index
            if "child" not in self.relationship_indices:
                self.relationship_indices["child"] = []
            self.relationship_indices["child"].append((node.parent_id, node.node_id))
        
        # If this is the first node, set it as root
        if len(self.nodes) == 1:
            self.root_id = node.node_id
        
        return node.node_id
    
    def get_node(self, node_id: str) -> Optional[CognitiveNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_domain(self, domain: str) -> List[CognitiveNode]:
        """Get all nodes of a specific domain."""
        node_ids = self.domain_indices.get(domain, [])
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
    
    def get_nodes_by_type(self, node_type: str) -> List[CognitiveNode]:
        """Get all nodes of a specific type."""
        node_ids = self.type_indices.get(node_type, [])
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
    
    def get_relationships(self, relationship_type: str) -> List[Tuple[CognitiveNode, CognitiveNode]]:
        """Get all relationships of a specific type."""
        pairs = self.relationship_indices.get(relationship_type, [])
        return [(self.nodes[source], self.nodes[target]) 
                for source, target in pairs 
                if source in self.nodes and target in self.nodes]
    
    def add_relationship(self, source_id: str, target_id: str, relationship_type: str) -> bool:
        """Add a relationship between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        # Add relationship to source node
        self.nodes[source_id].add_child(target_id, relationship_type)
        
        # Update relationship index
        if relationship_type not in self.relationship_indices:
            self.relationship_indices[relationship_type] = []
        self.relationship_indices[relationship_type].append((source_id, target_id))
        
        return True
    
    def find_path(self, start_id: str, end_id: str) -> List[str]:
        """Find a path between two nodes in the graph."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return []
        
        # Simple BFS to find a path
        visited = {start_id}
        queue = [[start_id]]
        
        while queue:
            path = queue.pop(0)
            node_id = path[-1]
            
            if node_id == end_id:
                return path
            
            for child_id in self.nodes[node_id].children:
                if child_id not in visited:
                    visited.add(child_id)
                    new_path = list(path)
                    new_path.append(child_id)
                    queue.append(new_path)
        
        return []  # No path found
    
    def to_dict(self) -> Dict:
        """Convert CognitiveGraph to a dictionary."""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_id": self.root_id,
            "domain_indices": self.domain_indices,
            "type_indices": self.type_indices,
            "relationship_indices": self.relationship_indices
        }
    
    def from_dict(self, data: Dict) -> None:
        """Load CognitiveGraph from a dictionary."""
        self.nodes = {}
        for node_id, node_data in data.get("nodes", {}).items():
            node = CognitiveNode(
                node_id=node_data["node_id"],
                node_type=node_data["node_type"],
                content=node_data["content"],
                parent_id=node_data.get("parent_id"),
                domain=node_data.get("domain", "general"),
                confidence=node_data.get("confidence", 0.8),
                metadata=node_data.get("metadata", {})
            )
            node.children = node_data.get("children", [])
            node.relationships = node_data.get("relationships", {})
            if "creation_time" in node_data:
                node.creation_time = datetime.fromisoformat(node_data["creation_time"])
            self.nodes[node_id] = node
        
        self.root_id = data.get("root_id")
        self.domain_indices = data.get("domain_indices", {})
        self.type_indices = data.get("type_indices", {})
        self.relationship_indices = data.get("relationship_indices", {})


class SpecializedModule:
    """Base class for specialized reasoning modules."""
    
    def __init__(self, name: str, domain: str, description: str = ""):
        self.name = name
        self.domain = domain
        self.description = description
        self.performance_history = []
        self.success_rate = 0.0
        self.calls = 0
        self.active = True
    
    def can_handle(self, query: str) -> float:
        """Determine if this module can handle the query and return confidence score (0-1)."""
        return 0.0  # Base implementation always returns 0
    
    async def process(self, query: str, context: Dict = None) -> Dict:
        """Process a query and return a result."""
        raise NotImplementedError("Specialized modules must implement process method")
    
    def record_performance(self, success: bool, metadata: Dict = None):
        """Record the performance of this module."""
        self.calls += 1
        record = {
            "timestamp": datetime.now(),
            "success": success,
            "metadata": metadata or {}
        }
        self.performance_history.append(record)
        
        # Update success rate
        if success:
            self.success_rate = ((self.success_rate * (self.calls - 1)) + 1) / self.calls
        else:
            self.success_rate = (self.success_rate * (self.calls - 1)) / self.calls
    
    def to_dict(self) -> Dict:
        """Convert module to a dictionary."""
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "success_rate": self.success_rate,
            "calls": self.calls,
            "active": self.active
        }


class MathematicalModule(SpecializedModule):
    """Specialized module for mathematical reasoning."""
    
    def __init__(self):
        super().__init__(
            name="Mathematical Reasoning",
            domain="mathematics",
            description="Specialized module for mathematical problem-solving"
        )
        self.math_keywords = {
            "calculate", "compute", "solve", "equation", "formula", "mathematics", 
            "math", "algebra", "calculus", "geometry", "statistics", "probability",
            "derivative", "integral", "theorem", "proof", "sum", "difference", 
            "product", "quotient", "divide", "multiply", "add", "subtract",
            "square root", "exponent", "logarithm", "function", "variable",
            "triangle", "circle", "rectangle", "polygon", "angle"
        }
    
    def can_handle(self, query: str) -> float:
        """Determine if this module can handle mathematical queries."""
        query_lower = query.lower()
        
        # Count occurrences of math keywords
        keyword_count = sum(1 for keyword in self.math_keywords if keyword in query_lower)
        
        # Check for mathematical expressions
        has_equations = "=" in query or "+" in query or "-" in query or "*" in query or "/" in query
        has_numbers = any(char.isdigit() for char in query)
        
        # Calculate confidence score (0-1)
        base_score = min(0.1 + (keyword_count * 0.1), 0.5)
        if has_equations:
            base_score += 0.3
        if has_numbers:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    async def process(self, query: str, context: Dict = None) -> Dict:
        """Process a mathematical query with specialized handling."""
        system_prompt = """You are an expert mathematical reasoning system. Analyze and solve the following mathematical problem step-by-step.
        
        Important guidelines:
        1. Show your work clearly with each calculation explained
        2. Use proper mathematical notation when appropriate
        3. Consider multiple approaches if relevant
        4. Verify your final answer
        5. Present results in a clear, structured format
        
        THINKING: Begin each step of your mathematical reasoning with "THINKING:" and show your detailed work.
        """
        
        result = await self._call_api(system_prompt, query)
        
        # Extract the final answer if possible
        final_answer = result.get("text", "")
        steps = self._extract_steps(final_answer)
        
        return {
            "domain": "mathematics",
            "query": query,
            "response": final_answer,
            "steps": steps,
            "module": self.name
        }
    
    async def _call_api(self, system_prompt: str, query: str) -> Dict:
        """Call the API with mathematical specialization."""
        # This would typically use the API client from the parent class
        # For now we'll return a mock response
        await asyncio.sleep(0.5)  # Simulate API call
        
        return {
            "text": f"THINKING: I'll solve this math problem systematically.\n\nTHINKING: First, I'll identify the key variables and equations.\n\nTHINKING: Now I'll apply the appropriate mathematical techniques.\n\n### Refined Comprehensive Solution:\nThe solution to the problem is X=42."
        }
    
    def _extract_steps(self, text: str) -> List[str]:
        """Extract mathematical reasoning steps from the response."""
        steps = []
        for line in text.split('\n'):
            if line.strip().startswith("THINKING:"):
                steps.append(line.strip()[9:].strip())
        return steps


class EthicalModule(SpecializedModule):
    """Specialized module for ethical reasoning."""
    
    def __init__(self):
        super().__init__(
            name="Ethical Reasoning",
            domain="ethics",
            description="Specialized module for ethical dilemmas and considerations"
        )
        self.ethics_keywords = {
            "ethical", "moral", "right", "wrong", "good", "bad", "fair", "unfair",
            "just", "unjust", "harm", "benefit", "duty", "obligation", "virtue",
            "vice", "dilemma", "principle", "values", "rights", "responsibilities",
            "consequences", "utilitarian", "deontological", "justice", "autonomy",
            "consent", "privacy", "dignity", "equality", "equity", "diversity"
        }
        self.ethical_frameworks = [
            "utilitarian", "deontological", "virtue ethics", "care ethics",
            "justice", "rights-based", "pluralistic"
        ]
    
    def can_handle(self, query: str) -> float:
        """Determine if this module can handle ethical queries."""
        query_lower = query.lower()
        
        # Count occurrences of ethics keywords
        keyword_count = sum(1 for keyword in self.ethics_keywords if keyword in query_lower)
        
        # Check for explicit ethical questions
        has_ethical_question = "ethical" in query_lower or "moral" in query_lower
        has_ethical_framework = any(framework in query_lower for framework in self.ethical_frameworks)
        
        # Calculate confidence score (0-1)
        base_score = min(0.1 + (keyword_count * 0.08), 0.5)
        if has_ethical_question:
            base_score += 0.3
        if has_ethical_framework:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    async def process(self, query: str, context: Dict = None) -> Dict:
        """Process an ethical query with specialized handling."""
        system_prompt = """You are an expert ethical reasoning system. Analyze the following ethical question or dilemma in a balanced, nuanced way.
        
        Important guidelines:
        1. Consider multiple ethical frameworks (utilitarian, deontological, virtue ethics, etc.)
        2. Identify key stakeholders and their interests
        3. Explore potential consequences of different actions
        4. Acknowledge complexity and trade-offs
        5. Avoid absolutist judgments unless clearly warranted
        
        THINKING: Begin each step of your ethical reasoning with "THINKING:" and explore the question from multiple perspectives.
        """
        
        result = await self._call_api(system_prompt, query)
        
        # Extract perspectives from the response
        final_answer = result.get("text", "")
        perspectives = self._extract_perspectives(final_answer)
        
        return {
            "domain": "ethics",
            "query": query,
            "response": final_answer,
            "perspectives": perspectives,
            "module": self.name
        }
    
    async def _call_api(self, system_prompt: str, query: str) -> Dict:
        """Call the API with ethical specialization."""
        # This would typically use the API client from the parent class
        # For now we'll return a mock response
        await asyncio.sleep(0.5)  # Simulate API call
        
        return {
            "text": f"THINKING: I'll analyze this ethical question using multiple frameworks.\n\nTHINKING: From a utilitarian perspective, we should consider the consequences for all stakeholders.\n\nTHINKING: From a deontological perspective, we must consider duties and principles.\n\n### Refined Comprehensive Solution:\nThis ethical situation involves balancing competing values. The most ethical approach would be to..."
        }
    
    def _extract_perspectives(self, text: str) -> List[Dict]:
        """Extract ethical perspectives from the response."""
        perspectives = []
        current_framework = None
        current_content = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for framework mentions
            for framework in self.ethical_frameworks:
                if framework in line.lower():
                    # Save previous framework if exists
                    if current_framework and current_content:
                        perspectives.append({
                            "framework": current_framework,
                            "content": " ".join(current_content)
                        })
                        current_content = []
                    
                    current_framework = framework
                    current_content.append(line)
                    break
            else:
                if current_framework:
                    current_content.append(line)
        
        # Add the last perspective if exists
        if current_framework and current_content:
            perspectives.append({
                "framework": current_framework,
                "content": " ".join(current_content)
            })
        
        return perspectives


class LogicalModule(SpecializedModule):
    """Specialized module for logical reasoning and problem-solving."""
    
    def __init__(self):
        super().__init__(
            name="Logical Reasoning",
            domain="logic",
            description="Specialized module for logical problems and puzzles"
        )
        self.logic_keywords = {
            "logic", "puzzle", "riddle", "deduction", "inference", "syllogism", 
            "premise", "conclusion", "valid", "invalid", "fallacy", "argument",
            "contradiction", "consistent", "inconsistent", "necessary", "sufficient",
            "if-then", "implies", "proof", "disprove", "counterexample", "solve",
            "solution", "constraints", "conditions", "rules", "game theory"
        }
    
    def can_handle(self, query: str) -> float:
        """Determine if this module can handle logical queries."""
        query_lower = query.lower()
        
        # Count occurrences of logic keywords
        keyword_count = sum(1 for keyword in self.logic_keywords if keyword in query_lower)
        
        # Check for puzzle indicators
        has_puzzle = "puzzle" in query_lower or "riddle" in query_lower
        has_constraints = "if" in query_lower and "then" in query_lower
        has_logical_elements = "all" in query_lower or "some" in query_lower or "none" in query_lower
        
        # Calculate confidence score (0-1)
        base_score = min(0.1 + (keyword_count * 0.08), 0.5)
        if has_puzzle:
            base_score += 0.3
        if has_constraints:
            base_score += 0.2
        if has_logical_elements:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    async def process(self, query: str, context: Dict = None) -> Dict:
        """Process a logical query with specialized handling."""
        system_prompt = """You are an expert logical reasoning system. Analyze and solve the following logical problem with precise, step-by-step deduction.
        
        Important guidelines:
        1. Identify key premises and constraints
        2. Use symbolic notation if helpful
        3. Apply logical rules systematically
        4. Check for counterexamples to verify your reasoning
        5. Ensure your conclusion follows necessarily from the premises
        
        THINKING: Begin each step of your logical reasoning with "THINKING:" and show your deductive process clearly.
        """
        
        result = await self._call_api(system_prompt, query)
        
        # Extract the reasoning steps and solution
        final_answer = result.get("text", "")
        steps = self._extract_steps(final_answer)
        
        return {
            "domain": "logic",
            "query": query,
            "response": final_answer,
            "steps": steps,
            "module": self.name
        }
    
    async def _call_api(self, system_prompt: str, query: str) -> Dict:
        """Call the API with logical specialization."""
        # This would typically use the API client from the parent class
        # For now we'll return a mock response
        await asyncio.sleep(0.5)  # Simulate API call
        
        return {
            "text": f"THINKING: I'll analyze this logical puzzle by identifying the key constraints.\n\nTHINKING: Let me represent each rule symbolically and consider the logical implications.\n\nTHINKING: Now I can deduce the correct solution by eliminating contradictions.\n\n### Refined Comprehensive Solution:\nBased on the given constraints, the solution to the puzzle is..."
        }
    
    def _extract_steps(self, text: str) -> List[str]:
        """Extract logical reasoning steps from the response."""
        steps = []
        for line in text.split('\n'):
            if line.strip().startswith("THINKING:"):
                steps.append(line.strip()[9:].strip())
        return steps


class CreativeModule(SpecializedModule):
    """Specialized module for creative thinking and generation."""
    
    def __init__(self):
        super().__init__(
            name="Creative Thinking",
            domain="creativity",
            description="Specialized module for creative tasks and brainstorming"
        )
        self.creative_keywords = {
            "creative", "create", "generate", "design", "invent", "brainstorm", 
            "ideate", "imagine", "story", "poem", "fiction", "narrative", "art",
            "novel", "innovative", "unique", "original", "concept", "idea",
            "write", "author", "compose", "craft", "scenario", "character",
            "plot", "setting", "dialogue", "metaphor", "analogy"
        }
    
    def can_handle(self, query: str) -> float:
        """Determine if this module can handle creative queries."""
        query_lower = query.lower()
        
        # Count occurrences of creative keywords
        keyword_count = sum(1 for keyword in self.creative_keywords if keyword in query_lower)
        
        # Check for explicit creative requests
        has_creative_request = "creative" in query_lower or "create" in query_lower
        has_writing_request = "write" in query_lower or "story" in query_lower or "poem" in query_lower
        
        # Calculate confidence score (0-1)
        base_score = min(0.1 + (keyword_count * 0.08), 0.5)
        if has_creative_request:
            base_score += 0.3
        if has_writing_request:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    async def process(self, query: str, context: Dict = None) -> Dict:
        """Process a creative query with specialized handling."""
        system_prompt = """You are an expert creative thinking system. Approach the following request with originality, imagination, and artistry.
        
        Important guidelines:
        1. Generate truly unique and novel ideas
        2. Use vivid, sensory language and imagery
        3. Draw inspiration from diverse sources
        4. Balance innovation with coherence and quality
        5. Consider emotional impact and audience engagement
        
        THINKING: Begin each step of your creative process with "THINKING:" to show how your ideas develop and evolve.
        """
        
        result = await self._call_api(system_prompt, query)
        
        # Extract the creative process and final creation
        final_answer = result.get("text", "")
        process_notes = self._extract_process(final_answer)
        
        return {
            "domain": "creativity",
            "query": query,
            "response": final_answer,
            "process_notes": process_notes,
            "module": self.name
        }
    
    async def _call_api(self, system_prompt: str, query: str) -> Dict:
        """Call the API with creative specialization."""
        # This would typically use the API client from the parent class
        # For now we'll return a mock response
        await asyncio.sleep(0.5)  # Simulate API call
        
        return {
            "text": f"THINKING: I'll approach this creative task by gathering inspiration from diverse sources.\n\nTHINKING: Let me explore unusual metaphors and imagery that could make this piece unique.\n\nTHINKING: Now I'll craft the structure to maximize emotional impact.\n\n### Refined Comprehensive Solution:\n[Creative content would appear here]"
        }
    
    def _extract_process(self, text: str) -> List[str]:
        """Extract creative process notes from the response."""
        process_notes = []
        for line in text.split('\n'):
            if line.strip().startswith("THINKING:"):
                process_notes.append(line.strip()[9:].strip())
        return process_notes


class TemporalReasoningModule(SpecializedModule):
    """Specialized module for temporal and causal reasoning."""
    
    def __init__(self):
        super().__init__(
            name="Temporal Reasoning",
            domain="temporal",
            description="Specialized module for time-based and causal reasoning"
        )
        self.temporal_keywords = {
            "time", "sequence", "order", "cause", "effect", "before", "after",
            "during", "while", "when", "until", "since", "timeline", "history",
            "future", "past", "present", "prediction", "forecast", "projection",
            "trend", "pattern", "development", "evolution", "change", "process",
            "phase", "stage", "step", "causal", "because", "therefore", "thus",
            "consequently", "hence", "chronology", "simultaneous"
        }
    
    def can_handle(self, query: str) -> float:
        """Determine if this module can handle temporal queries."""
        query_lower = query.lower()
        
        # Count occurrences of temporal keywords
        keyword_count = sum(1 for keyword in self.temporal_keywords if keyword in query_lower)
        
        # Check for temporal indicators
        has_temporal_question = any(word in query_lower for word in ["when", "before", "after", "during", "while", "until"])
        has_causal_question = any(word in query_lower for word in ["why", "cause", "effect", "because", "reason"])
        
        # Calculate confidence score (0-1)
        base_score = min(0.1 + (keyword_count * 0.08), 0.5)
        if has_temporal_question:
            base_score += 0.3
        if has_causal_question:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    async def process(self, query: str, context: Dict = None) -> Dict:
        """Process a temporal reasoning query with specialized handling."""
        system_prompt = """You are an expert temporal and causal reasoning system. Analyze the following question with precise attention to sequences, causality, and time-based relationships.
        
        Important guidelines:
        1. Carefully identify temporal order and dependencies
        2. Distinguish correlation from causation
        3. Consider multiple causal pathways when appropriate
        4. Identify key events and turning points
        5. Explain both proximate and ultimate causes
        
        THINKING: Begin each step of your temporal analysis with "THINKING:" and clearly articulate causal relationships.
        """
        
        result = await self._call_api(system_prompt, query)
        
        # Extract the timeline and causal relationships
        final_answer = result.get("text", "")
        causal_chains = self._extract_causal_chains(final_answer)
        
        return {
            "domain": "temporal",
            "query": query,
            "response": final_answer,
            "causal_chains": causal_chains,
            "module": self.name
        }
    
    async def _call_api(self, system_prompt: str, query: str) -> Dict:
        """Call the API with temporal reasoning specialization."""
        # This would typically use the API client from the parent class
        # For now we'll return a mock response
        await asyncio.sleep(0.5)  # Simulate API call
        
        return {
            "text": f"THINKING: I'll analyze the temporal sequence of events in this situation.\n\nTHINKING: Looking at causality, event A likely led to event B because of mechanism X.\n\nTHINKING: However, we need to consider alternative causal pathways including factors Y and Z.\n\n### Refined Comprehensive Solution:\nThe temporal analysis reveals that..."
        }
    
    def _extract_causal_chains(self, text: str) -> List[Dict]:
        """Extract causal chains from the response."""
        causal_chains = []
        current_chain = []
        
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith("THINKING:") and "cause" in line.lower() or "led to" in line.lower() or "result" in line.lower():
                if current_chain:
                    causal_chains.append({"steps": current_chain})
                    current_chain = []
                current_chain.append(line.strip()[9:].strip())
            elif current_chain and line:
                current_chain.append(line)
        
        # Add the last chain if exists
        if current_chain:
            causal_chains.append({"steps": current_chain})
        
        return causal_chains


class EnsembleMethod:
    """Base class for ensemble methods that aggregate multiple model outputs."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    async def combine_outputs(self, outputs: List[Dict], query: str) -> Dict:
        """Combine multiple outputs into a single result."""
        raise NotImplementedError("Ensemble methods must implement combine_outputs")
    
    def to_dict(self) -> Dict:
        """Convert ensemble method to a dictionary."""
        return {
            "name": self.name,
            "description": self.description
        }


class WeightedEnsemble(EnsembleMethod):
    """Ensemble method that uses weighted voting based on confidence scores."""
    
    def __init__(self):
        super().__init__(
            name="Weighted Ensemble",
            description="Combines multiple outputs with weights based on confidence scores"
        )
    
    async def combine_outputs(self, outputs: List[Dict], query: str) -> Dict:
        """Combine multiple outputs with weighted voting."""
        if not outputs:
            return {"error": "No outputs to combine"}
        
        if len(outputs) == 1:
            return outputs[0]
        
        # Extract confidence scores and normalize weights
        total_confidence = sum(output.get("confidence", 0.5) for output in outputs)
        weights = [output.get("confidence", 0.5) / total_confidence for output in outputs]
        
        # Combine text outputs with weights
        combined_text = ""
        for output, weight in zip(outputs, weights):
            text = output.get("response", "")
            source = output.get("source", "unknown")
            
            # Add weighted contribution to combined text
            if "### Refined Comprehensive Solution:" in text:
                solution_part = text.split("### Refined Comprehensive Solution:")[1].strip()
                combined_text += f"From {source} (weight {weight:.2f}):\n{solution_part}\n\n"
        
        # Create combined output
        combined_output = {
            "response": f"### Refined Comprehensive Solution:\n{combined_text}",
            "sources": [output.get("source", "unknown") for output in outputs],
            "weights": weights,
            "ensemble_method": self.name,
            "query": query
        }
        
        return combined_output


class MajorityVoteEnsemble(EnsembleMethod):
    """Ensemble method that uses majority voting for categorical outputs."""
    
    def __init__(self):
        super().__init__(
            name="Majority Vote Ensemble",
            description="Combines multiple outputs using majority voting"
        )
    
    async def combine_outputs(self, outputs: List[Dict], query: str) -> Dict:
        """Combine multiple outputs with majority voting."""
        if not outputs:
            return {"error": "No outputs to combine"}
        
        if len(outputs) == 1:
            return outputs[0]
        
        # Extract key findings from each output
        findings = []
        for output in outputs:
            text = output.get("response", "")
            if "### Refined Comprehensive Solution:" in text:
                solution_part = text.split("### Refined Comprehensive Solution:")[1].strip()
                lines = [line.strip() for line in solution_part.split('\n') if line.strip()]
                findings.extend(lines)
        
        # Count occurrences of each finding
        finding_counts = {}
        for finding in findings:
            # Normalize the finding text to handle minor variations
            normalized = self._normalize_text(finding)
            finding_counts[normalized] = finding_counts.get(normalized, 0) + 1
        
        # Sort findings by count (majority vote)
        sorted_findings = sorted(finding_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create combined output with majority findings
        top_findings = [finding for finding, count in sorted_findings[:3]]
        
        combined_output = {
            "response": f"### Refined Comprehensive Solution:\n" + "\n".join(top_findings),
            "finding_counts": finding_counts,
            "ensemble_method": self.name,
            "query": query
        }
        
        return combined_output
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        # Remove punctuation and extra whitespace
        normalized = ''.join(c for c in text if c.isalnum() or c.isspace())
        normalized = ' '.join(normalized.split())
        return normalized.lower()


class DiversityPromotingEnsemble(EnsembleMethod):
    """Ensemble method that promotes diverse perspectives and approaches."""
    
    def __init__(self):
        super().__init__(
            name="Diversity Promoting Ensemble",
            description="Combines multiple outputs to maximize diversity of perspectives"
        )
    
    async def combine_outputs(self, outputs: List[Dict], query: str) -> Dict:
        """Combine multiple outputs to promote diversity."""
        if not outputs:
            return {"error": "No outputs to combine"}
        
        if len(outputs) == 1:
            return outputs[0]
        
        # Extract key perspectives from each output
        perspectives = []
        for output in outputs:
            text = output.get("response", "")
            source = output.get("source", "unknown")
            
            if "### Refined Comprehensive Solution:" in text:
                solution_part = text.split("### Refined Comprehensive Solution:")[1].strip()
                perspectives.append({
                    "source": source,
                    "content": solution_part
                })
        
        # Calculate diversity scores (simplified - could be more sophisticated)
        diversity_scores = self._calculate_diversity(perspectives)
        
        # Select diverse perspectives
        selected_perspectives = self._select_diverse_perspectives(perspectives, diversity_scores)
        
        # Create combined output
        combined_text = "Perspectives from multiple approaches:\n\n"
        for perspective in selected_perspectives:
            combined_text += f"From {perspective['source']}:\n{perspective['content']}\n\n"
        
        combined_output = {
            "response": f"### Refined Comprehensive Solution:\n{combined_text}",
            "perspectives": selected_perspectives,
            "diversity_scores": diversity_scores,
            "ensemble_method": self.name,
            "query": query
        }
        
        return combined_output
    
    def _calculate_diversity(self, perspectives: List[Dict]) -> List[float]:
        """Calculate diversity scores between perspectives."""
        # Simplified implementation - in practice, use NLP techniques for better comparison
        scores = []
        
        for i, p1 in enumerate(perspectives):
            # Calculate average diversity with all other perspectives
            avg_diversity = 0.0
            count = 0
            
            for j, p2 in enumerate(perspectives):
                if i != j:
                    # Simple text difference as diversity measure
                    text1 = p1["content"]
                    text2 = p2["content"]
                    
                    # Calculate Jaccard similarity of words
                    words1 = set(text1.lower().split())
                    words2 = set(text2.lower().split())
                    
                    if not words1 or not words2:
                        similarity = 0.0
                    else:
                        intersection = len(words1.intersection(words2))
                        union = len(words1.union(words2))
                        similarity = intersection / union
                    
                    # Convert similarity to diversity (1 - similarity)
                    diversity = 1.0 - similarity
                    avg_diversity += diversity
                    count += 1
            
            if count > 0:
                avg_diversity /= count
            
            scores.append(avg_diversity)
        
        return scores
    
    def _select_diverse_perspectives(self, perspectives: List[Dict], diversity_scores: List[float]) -> List[Dict]:
        """Select a diverse set of perspectives."""
        # Sort perspectives by diversity score
        sorted_perspectives = [p for _, p in sorted(zip(diversity_scores, perspectives), key=lambda x: x[0], reverse=True)]
        
        # Take top half or at least 2 perspectives
        num_to_select = max(2, len(sorted_perspectives) // 2)
        return sorted_perspectives[:num_to_select]


class APIClient:
    """Enhanced API client with support for various endpoints and advanced features."""
    
    def __init__(self, api_key: str, api_url: str = "https://openrouter.ai/api/v1/chat/completions"):
        self.api_key = api_key
        self.api_url = api_url
        self.session = None
        self.response_cache = {}
        self.rate_limit_remaining = 1000  # Default high value
        self.rate_limit_reset = 0
        self.request_history = []
        self.request_count = 0
        self.success_count = 0
        
        # Performance monitoring
        self.total_latency = 0
        self.max_latency = 0
        self.min_latency = float('inf')
    
    async def initialize(self):
        """Initialize the API client."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the API client session."""
        if self.session is not None:
            await self.session.close()
            self.session = None
    
    async def generate_completion(self, prompt: str, system_prompt: str = None, 
                                 model: str = "deepseek/deepseek-chat-v3", 
                                 temperature: float = 0.2, 
                                 max_tokens: int = 8000,
                                 cache_key: str = None) -> Dict:
        """Generate a completion from the API with enhanced error handling and caching."""
        await self.initialize()
        
        # Check if we're rate limited
        current_time = time.time()
        if self.rate_limit_remaining <= 0 and current_time < self.rate_limit_reset:
            wait_time = self.rate_limit_reset - current_time
            print(f"Rate limited, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        # Use provided cache key or generate one
        if cache_key is None:
            cache_key = self._generate_cache_key(prompt, system_prompt, model, temperature, max_tokens)
        
        # Check cache
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are an advanced reasoning AI assistant. Be thorough yet efficient."
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_params": {
                "provider_order": ["deepseek"]
            }
        }
        
        # Track request
        request_id = self.request_count
        self.request_count += 1
        request_time = datetime.now()
        
        # Record request in history
        request_record = {
            "id": request_id,
            "time": request_time,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        self.request_history.append(request_record)
        
        # Make request with timing
        start_time = time.time()
        
        try:
            async with self.session.post(self.api_url, headers=headers, json=payload, timeout=120) as response:
                # Update rate limit info
                if "x-ratelimit-remaining" in response.headers:
                    self.rate_limit_remaining = int(response.headers["x-ratelimit-remaining"])
                if "x-ratelimit-reset" in response.headers:
                    self.rate_limit_reset = int(response.headers["x-ratelimit-reset"])
                
                # Calculate latency
                latency = time.time() - start_time
                self.total_latency += latency
                self.max_latency = max(self.max_latency, latency)
                self.min_latency = min(self.min_latency, latency)
                
                # Update request record with status
                request_record["status_code"] = response.status
                request_record["latency"] = latency
                
                # Process response
                if response.status == 200:
                    self.success_count += 1
                    request_record["success"] = True
                    
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Prepare result with metadata
                    api_result = {
                        "text": content,
                        "model": model,
                        "request_id": request_id,
                        "latency": latency
                    }
                    
                    # Cache the result
                    self.response_cache[cache_key] = api_result
                    
                    return api_result
                elif response.status == 429:
                    # Rate limit error
                    request_record["success"] = False
                    request_record["error"] = "Rate limit exceeded"
                    
                    error_text = await response.text()
                    print(f"Rate limit error: {error_text}")
                    
                    return {
                        "error": "rate_limit",
                        "message": "Rate limit exceeded. Please try again later.",
                        "request_id": request_id
                    }
                else:
                    # Other errors
                    request_record["success"] = False
                    error_text = await response.text()
                    request_record["error"] = error_text
                    
                    print(f"API error {response.status}: {error_text}")
                    
                    return {
                        "error": "api_error",
                        "status": response.status,
                        "message": error_text,
                        "request_id": request_id
                    }
        except asyncio.TimeoutError:
            request_record["success"] = False
            request_record["error"] = "timeout"
            
            return {
                "error": "timeout",
                "message": "Request timed out. The server might be experiencing high load.",
                "request_id": request_id
            }
        except Exception as e:
            request_record["success"] = False
            request_record["error"] = str(e)
            
            return {
                "error": "client_error",
                "message": str(e),
                "request_id": request_id
            }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the API client."""
        avg_latency = self.total_latency / max(1, self.request_count)
        success_rate = self.success_count / max(1, self.request_count) * 100
        
        return {
            "request_count": self.request_count,
            "success_count": self.success_count,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "min_latency": self.min_latency if self.min_latency != float('inf') else 0,
            "max_latency": self.max_latency
        }
    
    def _generate_cache_key(self, prompt: str, system_prompt: str, model: str, 
                          temperature: float, max_tokens: int) -> str:
        """Generate a cache key for a request."""
        # Combine relevant parameters
        key_data = f"{prompt}_{system_prompt}_{model}_{temperature}_{max_tokens}"
        
        # Create a hash for the cache key
        return hashlib.md5(key_data.encode()).hexdigest()


class PromptTemplate:
    """Dynamic prompt templates for specialized tasks."""
    
    def __init__(self, name: str, template: str, variables: List[str] = None,
                 description: str = "", domain: str = "general"):
        self.name = name
        self.template = template
        self.variables = variables or []
        self.description = description
        self.domain = domain
        self.usage_count = 0
        self.success_rate = 0.0
    
    def fill(self, **kwargs) -> str:
        """Fill in the template with provided variables."""
        filled_template = self.template
        
        for var in self.variables:
            if var in kwargs:
                # Replace placeholder with provided value
                placeholder = f"{{{var}}}"
                filled_template = filled_template.replace(placeholder, str(kwargs[var]))
        
        self.usage_count += 1
        return filled_template
    
    def record_success(self, success: bool):
        """Record success or failure of this template."""
        # Update success rate using weighted average
        self.success_rate = ((self.success_rate * (self.usage_count - 1)) + (1 if success else 0)) / self.usage_count
    
    def to_dict(self) -> Dict:
        """Convert template to a dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "variables": self.variables,
            "description": self.description,
            "domain": self.domain,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate
        }


class PromptLibrary:
    """Library of prompt templates organized by domain and task."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.domain_index: Dict[str, List[str]] = {}  # Map domains to template names
        
        # Initialize with default templates
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize the library with default templates."""
        # General reasoning template
        self.add_template(PromptTemplate(
            name="general_reasoning",
            template="""You are an advanced reasoning AI assistant. Analyze and solve the following problem step-by-step.

Problem: {query}

Think through this carefully:
1. Identify the key components of the problem
2. Consider multiple approaches
3. Apply relevant knowledge and techniques
4. Verify your solution

THINKING: Begin each step of your reasoning process with "THINKING:" and show your work clearly.
""",
            variables=["query"],
            description="General template for reasoning tasks",
            domain="general"
        ))
        
        # Mathematical problem solving
        self.add_template(PromptTemplate(
            name="math_problem_solving",
            template="""You are an expert mathematical problem solver. Solve the following math problem step-by-step.

Problem: {query}

Use these guidelines:
1. Identify the mathematical concepts involved
2. Set up any equations or formulas needed
3. Solve methodically, showing all steps clearly
4. Verify your answer with a different approach if possible
5. Express the final answer in a clear format

THINKING: Begin each step of your mathematical reasoning with "THINKING:" and show all calculations.
""",
            variables=["query"],
            description="Template for mathematical problem solving",
            domain="mathematics"
        ))
        
        # Ethical reasoning
        self.add_template(PromptTemplate(
            name="ethical_analysis",
            template="""You are an expert in ethical reasoning. Analyze the following ethical question or dilemma from multiple perspectives.

Ethical question: {query}

Use these frameworks:
1. Consequentialist/Utilitarian: Focus on outcomes and the greatest good
2. Deontological: Focus on duties, rights, and universal principles
3. Virtue Ethics: Focus on character and virtues
4. Care Ethics: Focus on relationships and context
5. Justice: Focus on fairness and equitable distribution

For each framework:
- Identify key considerations
- Analyze relevant principles or values
- Consider stakeholders affected
- Evaluate potential actions

THINKING: Begin each step of your ethical analysis with "THINKING:" and explore multiple perspectives.
""",
            variables=["query"],
            description="Template for ethical analysis",
            domain="ethics"
        ))
        
        # Scientific explanation
        self.add_template(PromptTemplate(
            name="scientific_explanation",
            template="""You are an expert scientific explainer. Provide a clear, accurate scientific explanation for the following question.

Question: {query}

In your explanation:
1. Start with foundational concepts
2. Build logically to more complex ideas
3. Use analogies where helpful
4. Address common misconceptions
5. Cite scientific principles accurately
6. Explain uncertainties where they exist

THINKING: Begin each step of your scientific explanation with "THINKING:" and ensure accuracy.
""",
            variables=["query"],
            description="Template for scientific explanations",
            domain="science"
        ))
        
        # Creative writing
        self.add_template(PromptTemplate(
            name="creative_writing",
            template="""You are an expert creative writer. Create an original piece based on the following prompt.

Creative prompt: {query}

As you write:
1. Develop vivid, engaging characters and/or settings
2. Use sensory details and evocative language
3. Create a coherent, satisfying narrative structure
4. Employ appropriate literary techniques
5. Maintain a consistent tone and style
6. Evoke genuine emotional response

Genre/Style preferences: {style}
Length: {length}

THINKING: Begin your creative process with "THINKING:" to show how you develop your ideas.
""",
            variables=["query", "style", "length"],
            description="Template for creative writing tasks",
            domain="creativity"
        ))
        
        # Code generation
        self.add_template(PromptTemplate(
            name="code_generation",
            template="""You are an expert programmer. Write clean, efficient, well-documented code for the following task.

Programming task: {query}

Language: {language}
Requirements:
{requirements}

In your code:
1. Use best practices for the specified language
2. Consider edge cases and error handling
3. Optimize for readability and maintainability
4. Add helpful comments
5. Include example usage if applicable

THINKING: Begin each step of your coding process with "THINKING:" to show your approach.
""",
            variables=["query", "language", "requirements"],
            description="Template for code generation",
            domain="programming"
        ))
        
        # Temporal reasoning
        self.add_template(PromptTemplate(
            name="temporal_analysis",
            template="""You are an expert in temporal and causal reasoning. Analyze the following scenario with attention to time-based relationships and cause-effect patterns.

Scenario: {query}

In your analysis:
1. Identify key events and their temporal sequence
2. Distinguish between correlation and causation
3. Analyze causal mechanisms and pathways
4. Consider counterfactual scenarios
5. Evaluate the strength of causal evidence
6. Address possible alternative explanations

THINKING: Begin each step of your temporal analysis with "THINKING:" and clearly articulate causal relationships.
""",
            variables=["query"],
            description="Template for temporal and causal analysis",
            domain="temporal"
        ))
        
        # Decision analysis
        self.add_template(PromptTemplate(
            name="decision_analysis",
            template="""You are an expert decision analyst. Evaluate the following decision situation and provide structured analysis.

Decision situation: {query}

Options to consider:
{options}

Decision criteria:
{criteria}

In your analysis:
1. Evaluate each option against all criteria
2. Consider uncertainty and risk factors
3. Identify trade-offs between different options
4. Apply decision-making frameworks as appropriate
5. Recommend a course of action with justification
6. Suggest risk mitigation strategies

THINKING: Begin each step of your decision analysis with "THINKING:" and show your evaluation process.
""",
            variables=["query", "options", "criteria"],
            description="Template for decision analysis",
            domain="decision_making"
        ))
    
    def add_template(self, template: PromptTemplate):
        """Add a template to the library."""
        self.templates[template.name] = template
        
        # Update domain index
        if template.domain not in self.domain_index:
            self.domain_index[template.domain] = []
        self.domain_index[template.domain].append(template.name)
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def get_templates_by_domain(self, domain: str) -> List[PromptTemplate]:
        """Get all templates for a specific domain."""
        template_names = self.domain_index.get(domain, [])
        return [self.templates[name] for name in template_names if name in self.templates]
    
    def find_best_template(self, query: str, domain: str = None) -> PromptTemplate:
        """Find the best template for a given query and domain."""
        candidates = []
        
        # If domain is specified, limit to templates in that domain
        if domain:
            candidates = self.get_templates_by_domain(domain)
        else:
            # Try to determine domain from query
            domain = self._infer_domain(query)
            candidates = self.get_templates_by_domain(domain)
            
            # If no domain-specific templates found, use general templates
            if not candidates:
                candidates = self.get_templates_by_domain("general")
        
        # If still no candidates, use the default general reasoning template
        if not candidates:
            return self.get_template("general_reasoning")
        
        # Select best template based on success rate and usage count
        # This is a simple heuristic - could be more sophisticated
        best_template = max(candidates, key=lambda t: (t.success_rate * 0.8) + (min(t.usage_count, 100) / 100 * 0.2))
        
        return best_template
    
    def _infer_domain(self, query: str) -> str:
        """Infer the domain of a query using keywords and patterns."""
        query_lower = query.lower()
        
        # Check for mathematical content
        math_keywords = {"calculate", "solve", "equation", "math", "formula", "computation"}
        if any(keyword in query_lower for keyword in math_keywords) or any(char in query_lower for char in "+-*/^="):
            return "mathematics"
        
        # Check for ethical content
        ethics_keywords = {"ethical", "moral", "right", "wrong", "should", "ought", "dilemma"}
        if any(keyword in query_lower for keyword in ethics_keywords):
            return "ethics"
        
        # Check for scientific content
        science_keywords = {"science", "scientific", "theory", "experiment", "evidence", "research"}
        if any(keyword in query_lower for keyword in science_keywords):
            return "science"
        
        # Check for creative content
        creative_keywords = {"creative", "story", "write", "poem", "novel", "imagine"}
        if any(keyword in query_lower for keyword in creative_keywords):
            return "creativity"
        
        # Check for programming content
        code_keywords = {"code", "program", "function", "algorithm", "programming", "software"}
        if any(keyword in query_lower for keyword in code_keywords):
            return "programming"
        
        # Default to general domain
        return "general"
    
    def to_dict(self) -> Dict:
        """Convert library to a dictionary."""
        return {
            "templates": {name: template.to_dict() for name, template in self.templates.items()},
            "domain_index": self.domain_index
        }


class MetaLearningSystem:
    """System for tracking and learning from past reasoning attempts."""
    
    def __init__(self):
        self.reasoning_history = []
        self.strategy_performance = {}
        self.domain_performance = {}
        self.template_performance = {}
        self.module_performance = {}
        self.improvement_suggestions = []
        
        # For A/B testing of strategies
        self.ab_tests = {}
        self.current_ab_test = None
    
    def record_reasoning_attempt(self, record: Dict):
        """Record a reasoning attempt with relevant metadata."""
        # Add timestamp if not present
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()
        
        self.reasoning_history.append(record)
        
        # Update performance tracking
        self._update_strategy_performance(record)
        self._update_domain_performance(record)
        self._update_template_performance(record)
        self._update_module_performance(record)
        
        # Update A/B test if applicable
        if self.current_ab_test and "ab_test_variant" in record:
            self._update_ab_test(record)
        
        # Generate improvement suggestions periodically
        if len(self.reasoning_history) % 10 == 0:
            self._generate_improvement_suggestions()
    
    def get_best_strategy(self, domain: str, query_type: str = None) -> Dict:
        """Get the best strategy for a given domain and query type."""
        # Get all strategies for this domain
        domain_key = f"domain:{domain}"
        if query_type:
            domain_key += f":type:{query_type}"
        
        strategies = self.strategy_performance.get(domain_key, {})
        if not strategies:
            # Fall back to general domain
            strategies = self.strategy_performance.get("domain:general", {})
        
        if not strategies:
            # Return default strategy if no performance data
            return {
                "module": "general",
                "template": "general_reasoning",
                "ensemble": "WeightedEnsemble",
                "confidence": 0.5
            }
        
        # Find strategy with highest success rate
        best_strategy_key = max(strategies, key=lambda k: strategies[k]["success_rate"])
        
        # Parse strategy key
        parts = best_strategy_key.split(":")
        module = parts[1] if len(parts) > 1 else "general"
        template = parts[3] if len(parts) > 3 else "general_reasoning"
        ensemble = parts[5] if len(parts) > 5 else "WeightedEnsemble"
        
        return {
            "module": module,
            "template": template,
            "ensemble": ensemble,
            "confidence": strategies[best_strategy_key]["success_rate"],
            "uses": strategies[best_strategy_key]["uses"]
        }
    
    def start_ab_test(self, name: str, variants: List[Dict], success_criteria: Callable[[Dict], bool]):
        """Start an A/B test between different strategies."""
        if name in self.ab_tests:
            # Update existing test
            self.ab_tests[name]["variants"] = variants
            self.ab_tests[name]["success_criteria"] = success_criteria
            self.ab_tests[name]["active"] = True
        else:
            # Create new test
            self.ab_tests[name] = {
                "variants": variants,
                "results": {i: {"uses": 0, "successes": 0} for i in range(len(variants))},
                "success_criteria": success_criteria,
                "active": True,
                "start_time": datetime.now().isoformat()
            }
        
        self.current_ab_test = name
    
    def stop_ab_test(self) -> Dict:
        """Stop the current A/B test and return results."""
        if not self.current_ab_test:
            return {"error": "No active A/B test"}
        
        # Mark test as inactive
        test = self.ab_tests[self.current_ab_test]
        test["active"] = False
        test["end_time"] = datetime.now().isoformat()
        
        # Calculate success rates
        for variant_id, result in test["results"].items():
            if result["uses"] > 0:
                result["success_rate"] = result["successes"] / result["uses"]
            else:
                result["success_rate"] = 0.0
        
        # Determine winner
        winner_id = max(test["results"].keys(), key=lambda k: test["results"][k].get("success_rate", 0))
        test["winner"] = winner_id
        test["winner_variant"] = test["variants"][int(winner_id)]
        
        # Reset current test
        self.current_ab_test = None
        
        return test
    
    def get_improvement_suggestions(self) -> List[Dict]:
        """Get suggestions for improving the reasoning system."""
        return self.improvement_suggestions
    
    def _update_strategy_performance(self, record: Dict):
        """Update performance tracking for strategies."""
        if "strategy" not in record or "success" not in record:
            return
        
        strategy = record["strategy"]
        success = record["success"]
        domain = record.get("domain", "general")
        query_type = record.get("query_type")
        
        # Create strategy key
        strategy_key = f"module:{strategy.get('module', 'general')}:template:{strategy.get('template', 'general_reasoning')}:ensemble:{strategy.get('ensemble', 'WeightedEnsemble')}"
        
        # Domain key
        domain_key = f"domain:{domain}"
        if query_type:
            domain_key += f":type:{query_type}"
        
        # Initialize if needed
        if domain_key not in self.strategy_performance:
            self.strategy_performance[domain_key] = {}
        
        if strategy_key not in self.strategy_performance[domain_key]:
            self.strategy_performance[domain_key][strategy_key] = {
                "uses": 0,
                "successes": 0,
                "success_rate": 0.0,
                "avg_latency": 0.0,
                "total_latency": 0.0
            }
        
        # Update stats
        stats = self.strategy_performance[domain_key][strategy_key]
        stats["uses"] += 1
        if success:
            stats["successes"] += 1
        stats["success_rate"] = stats["successes"] / stats["uses"]
        
        # Update latency if available
        if "latency" in record:
            latency = record["latency"]
            stats["total_latency"] += latency
            stats["avg_latency"] = stats["total_latency"] / stats["uses"]
    
    def _update_domain_performance(self, record: Dict):
        """Update performance tracking for domains."""
        if "domain" not in record or "success" not in record:
            return
        
        domain = record["domain"]
        success = record["success"]
        
        # Initialize if needed
        if domain not in self.domain_performance:
            self.domain_performance[domain] = {
                "uses": 0,
                "successes": 0,
                "success_rate": 0.0
            }
        
        # Update stats
        stats = self.domain_performance[domain]
        stats["uses"] += 1
        if success:
            stats["successes"] += 1
        stats["success_rate"] = stats["successes"] / stats["uses"]
    
    def _update_template_performance(self, record: Dict):
        """Update performance tracking for templates."""
        if "template" not in record or "success" not in record:
            return
        
        template = record["template"]
        success = record["success"]
        
        # Initialize if needed
        if template not in self.template_performance:
            self.template_performance[template] = {
                "uses": 0,
                "successes": 0,
                "success_rate": 0.0
            }
        
        # Update stats
        stats = self.template_performance[template]
        stats["uses"] += 1
        if success:
            stats["successes"] += 1
        stats["success_rate"] = stats["successes"] / stats["uses"]
    
    def _update_module_performance(self, record: Dict):
        """Update performance tracking for specialized modules."""
        if "module" not in record or "success" not in record:
            return
        
        module = record["module"]
        success = record["success"]
        
        # Initialize if needed
        if module not in self.module_performance:
            self.module_performance[module] = {
                "uses": 0,
                "successes": 0,
                "success_rate": 0.0
            }
        
        # Update stats
        stats = self.module_performance[module]
        stats["uses"] += 1
        if success:
            stats["successes"] += 1
        stats["success_rate"] = stats["successes"] / stats["uses"]
    
    def _update_ab_test(self, record: Dict):
        """Update the current A/B test with results."""
        if not self.current_ab_test:
            return
        
        variant_id = record.get("ab_test_variant")
        if variant_id is None:
            return
        
        test = self.ab_tests[self.current_ab_test]
        if variant_id not in test["results"]:
            return
        
        # Update usage count
        test["results"][variant_id]["uses"] += 1
        
        # Check success using the test's criteria
        if test["success_criteria"](record):
            test["results"][variant_id]["successes"] += 1
    
    def _generate_improvement_suggestions(self):
        """Generate suggestions for improving the reasoning system."""
        suggestions = []
        
        # Check if any domain has low success rate
        for domain, stats in self.domain_performance.items():
            if stats["uses"] >= 5 and stats["success_rate"] < 0.5:
                suggestions.append({
                    "type": "domain_improvement",
                    "domain": domain,
                    "current_success_rate": stats["success_rate"],
                    "suggestion": f"Consider enhancing capabilities for the '{domain}' domain",
                    "priority": "high" if stats["success_rate"] < 0.3 else "medium"
                })
        
        # Check if any template has low success rate
        for template, stats in self.template_performance.items():
            if stats["uses"] >= 5 and stats["success_rate"] < 0.5:
                suggestions.append({
                    "type": "template_improvement",
                    "template": template,
                    "current_success_rate": stats["success_rate"],
                    "suggestion": f"Revise the '{template}' prompt template",
                    "priority": "high" if stats["success_rate"] < 0.3 else "medium"
                })
        
        # Check if any module has low success rate
        for module, stats in self.module_performance.items():
            if stats["uses"] >= 5 and stats["success_rate"] < 0.5:
                suggestions.append({
                    "type": "module_improvement",
                    "module": module,
                    "current_success_rate": stats["success_rate"],
                    "suggestion": f"Improve the '{module}' specialized module",
                    "priority": "high" if stats["success_rate"] < 0.3 else "medium"
                })
        
        # Update suggestions list (keep only top 5 by priority)
        self.improvement_suggestions = sorted(suggestions, key=lambda s: 0 if s["priority"] == "high" else 1)[:5]
    
    def to_dict(self) -> Dict:
        """Convert meta-learning system to a dictionary."""
        return {
            "reasoning_history_count": len(self.reasoning_history),
            "strategy_performance": self.strategy_performance,
            "domain_performance": self.domain_performance,
            "template_performance": self.template_performance,
            "module_performance": self.module_performance,
            "improvement_suggestions": self.improvement_suggestions,
            "ab_tests": self.ab_tests
        }


class NeoCortex:
    """Enhanced NeoCortex cognitive architecture with advanced capabilities."""
    
    def __init__(self, model_name: str = "deepseek/deepseek-chat-v3", temperature: float = 0.2):
        """Initialize NeoCortex with specific model settings."""
        self.model_name = model_name
        self.temperature = temperature
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        # Hardcoded API key
        self.api_key = "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"
        self.cognitive_graph = CognitiveGraph()
        
        # Initialize knowledge store
        self.knowledge_store = {}
        
        # Initialize API client
        self.api_client = APIClient(self.api_key, self.api_url)
        
        # Initialize specialized modules
        self.modules = {
            "mathematics": MathematicalModule(),
            "ethics": EthicalModule(),
            "logic": LogicalModule(),
            "creativity": CreativeModule(),
            "temporal": TemporalReasoningModule()
        }
        
        # Initialize ensemble methods
        self.ensemble_methods = {
            "weighted": WeightedEnsemble(),
            "majority_vote": MajorityVoteEnsemble(),
            "diversity": DiversityPromotingEnsemble()
        }
        
        # Initialize prompt library
        self.prompt_library = PromptLibrary()
        
        # Initialize meta-learning system
        self.meta_learning = MetaLearningSystem()
        
        # Module states
        self.module_states = {
            "concept_network": True,
            "self_regulation": True,
            "spatial_reasoning": True,
            "causal_reasoning": True,
            "counterfactual": True,
            "metacognition": True
        }
        
        # Performance settings
        self.max_tokens = 8000  # Substantially increased token generation limit
        self.concurrent_requests = True  # Enable parallel processing
        self.reasoning_depth = "balanced"  # Options: "minimal", "balanced", "thorough"
        
        # This will be assigned by the ReasoningThread
        self.thought_generated = None
        
        # Interactive learning settings
        self.interactive_mode = False
        self.clarification_threshold = 0.5  # Threshold for asking clarifying questions
        self.feedback_incorporation = True  # Whether to incorporate user feedback
        
        # Initialize session
        self.session = requests.Session()
        
        # Thread pool for parallel requests
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    async def generate_response(self, prompt: str, system_prompt: str = None, emit_thoughts: bool = False) -> str:
        """Generate a response using the API with enhanced error handling and features."""
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = """You are an advanced reasoning AI assistant. Be thorough yet efficient.
            
            IMPORTANT INSTRUCTION FOR REASONING: For every step of your analysis process,
            explicitly share your detailed thought process by starting lines with "THINKING: " followed by the specific
            reasoning step you're taking. Include your internal deliberations, considerations of different angles,
            evaluation of evidence, and how you arrive at conclusions.
            
            After each reasoning step, begin your next thought with a new "THINKING: " line. Be verbose about your reasoning.
            Don't hide any intermediate steps - share your complete analysis process with fine-grained detail.
            
            When providing solutions, especially lengthy ones, always begin your solution section with
            "### Refined Comprehensive Solution:" and make sure you complete all code blocks and sections.
            Never truncate or abbreviate your responses. Structure your answers with clear section markers.
            """
        
        # Call the API
        result = await self.api_client.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Check for error
        if "error" in result:
            return f"Error: {result.get('message', 'Unknown error')}"
        
        # Extract the response text
        response_text = result.get("text", "")
        
        # Check for truncated response and fix if needed
        if self._is_truncated(response_text):
            completion = await self._generate_completion(response_text, prompt)
            if completion:
                response_text = response_text + "\n\n" + completion
        
        # Extract thinking processes for realtime display
        if emit_thoughts and hasattr(self, 'thought_generated') and self.thought_generated:
            # More thorough extraction of thinking processes
            current_thought = ""
            thoughts = []
            lines = response_text.split('\n')
            
            for line in lines:
                if line.strip().startswith("THINKING:"):
                    # If we already have a thought in progress, add it to thoughts
                    if current_thought:
                        thoughts.append(current_thought.strip())
                        
                    # Start a new thought
                    current_thought = line.strip()[9:].strip()  # Remove "THINKING:" prefix
                elif line.strip().startswith("Thinking:"):
                    # Alternative format
                    if current_thought:
                        thoughts.append(current_thought.strip())
                    current_thought = line.strip()[9:].strip()
                elif current_thought and line.strip():
                    # Continue current thought
                    current_thought += " " + line.strip()
            
            # Add the last thought if there is one
            if current_thought:
                thoughts.append(current_thought.strip())
                
            # Emit all collected thoughts
            for thought in thoughts:
                self.thought_generated.emit(thought)
        
        return response_text
    
    def _is_truncated(self, content: str) -> bool:
        """Check if a response appears to be truncated."""
        # Check for unmatched code blocks
        open_code_blocks = content.count("```") % 2 != 0
        
        # Check for obvious truncation markers
        truncation_markers = ["...", "[truncated]", "[cut off]", "to be continued", "end of section", "incomplete"]
        has_truncation_marker = any(marker in content.lower() for marker in truncation_markers)
        
        # Check for unfinished sentences
        last_char = content.strip()[-1] if content.strip() else ""
        unfinished_sentence = last_char not in ['.', '!', '?', ')', '}', ']', '"', "'"]
        
        # Check for abrupt ending in code or math
        abrupt_ending = content.strip().endswith("```") or content.strip().endswith("$")
        
        return open_code_blocks or has_truncation_marker or unfinished_sentence or abrupt_ending
    
    async def _generate_completion(self, truncated_content: str, original_prompt: str) -> str:
        """Generate a completion for a truncated response."""
        completion_prompt = f"""
        The following response to a query appears to be truncated or cut off:
        
        ---
        {truncated_content}
        ---
        
        Original query: {original_prompt}
        
        Please provide a concise completion that finishes any incomplete sections,
        closes any open code blocks, and provides any missing conclusions.
        Focus only on completing what's missing, not repeating what's already provided.
        """
        
        system_prompt = "You are an assistant that concisely completes truncated responses."
        
        # Use fewer tokens for the completion
        result = await self.api_client.generate_completion(
            prompt=completion_prompt,
            system_prompt=system_prompt,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=min(4000, self.max_tokens // 2)
        )
        
        # Check for error
        if "error" in result:
            return "\n\n[Note: The response appears to be truncated, but a completion couldn't be generated.]"
        
        # Extract the completion
        completion = result.get("text", "")
        
        return f"\n\n[Note: The previous response appeared to be truncated. Here's the completion:]\n\n{completion}"
    
    async def select_specialized_modules(self, query: str) -> List[Tuple[SpecializedModule, float]]:
        """Select appropriate specialized modules for a query with confidence scores."""
        candidates = []
        
        # Evaluate each module's confidence for this query
        for domain, module in self.modules.items():
            if module.active:
                confidence = module.can_handle(query)
                if confidence > 0.0:
                    candidates.append((module, confidence))
        
        # Sort by confidence score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Always include at least one module
        if not candidates:
            # Use mathematics module as a fallback (arbitrary choice)
            candidates.append((self.modules["mathematics"], 0.1))
        
        return candidates
    
    async def get_ensemble_method(self, query: str, modules: List[Tuple[SpecializedModule, float]]) -> EnsembleMethod:
        """Select the appropriate ensemble method for the query and modules."""
        # For now, use a simple heuristic
        # In the future, this could use meta-learning
        
        if len(modules) <= 1:
            # No need for ensemble with just one module
            return self.ensemble_methods["weighted"]
        
        # Check if we need diverse perspectives
        if "different perspectives" in query.lower() or "various approaches" in query.lower():
            return self.ensemble_methods["diversity"]
        
        # Check if it's a categorical question
        if any(word in query.lower() for word in ["categorize", "classify", "which", "what type"]):
            return self.ensemble_methods["majority_vote"]
        
        # Default to weighted ensemble
        return self.ensemble_methods["weighted"]
    
    async def get_prompt_template(self, query: str, module: SpecializedModule) -> PromptTemplate:
        """Get the appropriate prompt template for a query and module."""
        # First, try meta-learning to find the best template
        best_strategy = self.meta_learning.get_best_strategy(module.domain)
        template_name = best_strategy.get("template")
        
        if template_name:
            template = self.prompt_library.get_template(template_name)
            if template:
                return template
        
        # Fallback: ask the prompt library to find the best template for this domain
        return self.prompt_library.find_best_template(query, module.domain)
    
    async def should_ask_clarification(self, query: str) -> bool:
        """Determine if clarification should be asked for the query."""
        if not self.interactive_mode:
            return False
        
        # Analyze query for ambiguity
        ambiguity_score = await self._analyze_ambiguity(query)
        
        return ambiguity_score > self.clarification_threshold
    
    async def _analyze_ambiguity(self, query: str) -> float:
        """Analyze the ambiguity level of a query."""
        # Simple heuristic for now - could be more sophisticated
        
        # Check for obvious ambiguity markers
        ambiguity_markers = ["unclear", "ambiguous", "vague", "confusing", "not sure"]
        has_ambiguity_marker = any(marker in query.lower() for marker in ambiguity_markers)
        
        # Check for missing context
        context_markers = ["it", "this", "that", "they", "them", "these", "those"]
        has_missing_context = any(marker + " " in query.lower() for marker in context_markers)
        
        # Check for very short queries
        is_short_query = len(query.split()) < 5
        
        # Check for multiple possible interpretations
        has_multiple_questions = query.count("?") > 1
        
        # Calculate ambiguity score
        ambiguity_score = 0.0
        if has_ambiguity_marker:
            ambiguity_score += 0.3
        if has_missing_context:
            ambiguity_score += 0.2
        if is_short_query:
            ambiguity_score += 0.2
        if has_multiple_questions:
            ambiguity_score += 0.2
        
        return min(ambiguity_score, 1.0)
    
    async def generate_clarification_question(self, query: str) -> str:
        """Generate a clarification question for an ambiguous query."""
        prompt = f"""
        The following query is ambiguous or lacks necessary context:
        
        {query}
        
        Generate a concise, specific clarification question that would help resolve the ambiguity.
        Focus on the most important missing information. Phrase the question in a natural, conversational way.
        """
        
        system_prompt = "You are an assistant that helps clarify ambiguous queries by asking precise questions."
        
        result = await self.api_client.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model_name,
            temperature=0.3,  # Lower temperature for more predictable output
            max_tokens=100  # Short response for the clarification question
        )
        
        if "error" in result:
            return "Could you please clarify your request?"
        
        return result.get("text", "Could you please provide more context?")
    
    async def incorporate_feedback(self, query: str, feedback: str) -> str:
        """Incorporate user feedback into the query understanding."""
        prompt = f"""
        Original query: {query}
        
        User feedback: {feedback}
        
        Based on the original query and the user's feedback, provide an improved,
        clarified version of the query that incorporates the additional context or corrections.
        """
        
        system_prompt = "You are an assistant that reformulates queries based on user feedback."
        
        result = await self.api_client.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model_name,
            temperature=0.2,
            max_tokens=200
        )
        
        if "error" in result:
            return query  # Fall back to original query
        
        improved_query = result.get("text", query)
        
        # Record the feedback in meta-learning for future improvement
        self.meta_learning.record_reasoning_attempt({
            "type": "feedback_incorporation",
            "original_query": query,
            "feedback": feedback,
            "improved_query": improved_query,
            "timestamp": datetime.now().isoformat()
        })
        
        return improved_query
    
    async def solve(self, query: str, show_work: bool = True, interactive: bool = False) -> Dict:
        """Solve a problem using the enhanced cognitive architecture."""
        # Set interactive mode
        self.interactive_mode = interactive
        
        # Reset cognitive graph for new problem
        self.cognitive_graph = CognitiveGraph()
        
        # Create problem node
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        # Check if clarification is needed
        if interactive and await self.should_ask_clarification(query):
            clarification_question = await self.generate_clarification_question(query)
            
            # Create clarification node
            clarification_node = CognitiveNode(
                node_id="clarification_0",
                node_type="clarification",
                content=clarification_question,
                parent_id="problem_0"
            )
            self.cognitive_graph.add_node(clarification_node)
            
            return {
                "needs_clarification": True,
                "clarification_question": clarification_question,
                "cognitive_graph": self.cognitive_graph.to_dict()
            }
        
        # Select specialized modules
        module_candidates = await self.select_specialized_modules(query)
        
        # Create module selection node
        module_selection_node = CognitiveNode(
            node_id="module_selection_0",
            node_type="module_selection",
            content=f"Selected modules: {', '.join(m[0].name for m in module_candidates)}",
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(module_selection_node)
        
        # Process with each selected module
        module_outputs = []
        
        for module, confidence in module_candidates:
            # Get appropriate prompt template
            template = await self.get_prompt_template(query, module)
            
            # Fill template
            filled_prompt = template.fill(query=query)
            
            # Process with module
            try:
                module_result = await module.process(query, {"prompt": filled_prompt})
                
                # Create module node
                module_node = CognitiveNode(
                    node_id=f"module_{module.domain}",
                    node_type="module_processing",
                    content=module_result.get("response", ""),
                    parent_id="module_selection_0",
                    domain=module.domain,
                    confidence=confidence
                )
                self.cognitive_graph.add_node(module_node)
                
                # Add to outputs with source and confidence
                module_outputs.append({
                    "response": module_result.get("response", ""),
                    "source": module.name,
                    "confidence": confidence,
                    "domain": module.domain,
                    "module_result": module_result
                })
                
                # Record performance for meta-learning
                success = "error" not in module_result
                module.record_performance(success)
                template.record_success(success)
                
            except Exception as e:
                print(f"Error in module {module.name}: {str(e)}")
                # Record failure
                module.record_performance(False)
                template.record_success(False)
        
        # Select ensemble method
        ensemble_method = await self.get_ensemble_method(query, module_candidates)
        
        # Create ensemble node
        ensemble_node = CognitiveNode(
            node_id="ensemble_0",
            node_type="ensemble",
            content=f"Ensemble method: {ensemble_method.name}",
            parent_id="module_selection_0"
        )
        self.cognitive_graph.add_node(ensemble_node)
        
        # Combine outputs with ensemble method
        combined_output = await ensemble_method.combine_outputs(module_outputs, query)
        
        # Create solution node
        solution_node = CognitiveNode(
            node_id="solution_0",
            node_type="solution",
            content=combined_output.get("response", ""),
            parent_id="ensemble_0"
        )
        self.cognitive_graph.add_node(solution_node)
        
        # Meta-learning: record the reasoning attempt
        strategy = {
            "module": module_candidates[0][0].name if module_candidates else "none",
            "template": template.name if 'template' in locals() else "none",
            "ensemble": ensemble_method.name
        }
        
        self.meta_learning.record_reasoning_attempt({
            "query": query,
            "strategy": strategy,
            "domain": module_candidates[0][0].domain if module_candidates else "general",
            "success": True,  # Assuming success for now - could be updated with feedback
            "timestamp": datetime.now().isoformat()
        })
        
        # Construct the result
        result = {
            "final_answer": combined_output.get("response", ""),
            "cognitive_graph": self.cognitive_graph.to_dict(),
            "modules_used": [m[0].name for m in module_candidates],
            "ensemble_method": ensemble_method.name
        }
        
        # Include detailed reasoning process if requested
        if show_work:
            result["reasoning_process"] = {
                "module_selection": {
                    "candidates": [{"name": m[0].name, "confidence": m[1]} for m in module_candidates],
                    "ensemble_method": ensemble_method.name
                },
                "module_outputs": module_outputs,
                "ensemble_result": combined_output
            }
        
        return result
    
    async def process_feedback(self, query: str, feedback: str, previous_result: Dict = None) -> Dict:
        """Process user feedback to improve the reasoning result."""
        if not self.feedback_incorporation:
            return {"error": "Feedback incorporation is disabled"}
        
        # Improve query with feedback
        improved_query = await self.incorporate_feedback(query, feedback)
        
        # Solve the improved query
        result = await self.solve(improved_query, show_work=True, interactive=self.interactive_mode)
        
        # Record the feedback interaction
        self.meta_learning.record_reasoning_attempt({
            "type": "feedback_improvement",
            "original_query": query,
            "feedback": feedback,
            "improved_query": improved_query,
            "success": True,  # Assuming success for now
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def _analyze_problem(self, query: str) -> Dict:
        """Analyze the problem structure."""
        prompt = f"""
        Analyze the following problem:
        
        {query}
        
        Break it down into its core components and subproblems. Identify key concepts, constraints, and goals.
        """
        
        response_text = self._generate_response(prompt, emit_thoughts=True)
        
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
        
        response_text = self._generate_response(prompt, emit_thoughts=True)
        
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
            
            response_text = self._generate_response(prompt, emit_thoughts=True)
            
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
        
        response_text = self._generate_response(prompt, emit_thoughts=True)
        
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
        
        Provide a comprehensive solution. Begin your solution with "### Refined Comprehensive Solution:" 
        and ensure all code blocks and sections are complete. Do not truncate or abbreviate your response.
        """
        
        response_text = self._generate_response(prompt, emit_thoughts=True)
        
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
        
        response_text = self._generate_response(prompt, emit_thoughts=True)
        
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
        Begin your refined solution with "### Refined Comprehensive Solution:" 
        and ensure all code blocks and sections are complete.
        """
        
        response_text = self._generate_response(prompt, emit_thoughts=True)
        
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
        
        response_text = self._generate_response(prompt, emit_thoughts=True)
        
        # Create metacognitive reflection node in the cognitive graph
        reflection_node = CognitiveNode(
            node_id="reflection_0",
            node_type="metacognitive_reflection",
            content=response_text,
            parent_id="final_answer_0"
        )
        self.cognitive_graph.add_node(reflection_node)
        
        return response_text
    
    def _is_simple_query(self, query: str) -> bool:
        """Check if this is a simple query that doesn't need the full reasoning pipeline."""
        # Simple implementation - check if query is very short
        return len(query.split()) < 10
        
    def _fast_response(self, query: str, show_work: bool = True) -> Dict:
        """Generate a fast response for simple queries."""
        prompt = f"Provide a direct answer to this question: {query}"
        response = self._generate_response(prompt, emit_thoughts=True)
        
        # Create minimal cognitive graph for UI
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        answer_node = CognitiveNode(
            node_id="final_answer_0",
            node_type="final_answer",
            content=response,
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(answer_node)
        
        result = {
            "final_answer": response,
            "cognitive_graph": self.cognitive_graph.to_dict()
        }
        
        if show_work:
            result["reasoning_process"] = {
                "decomposition": {"full_decomposition": "Simple query, no decomposition needed."},
                "perspectives": [],
                "evidence": {},
                "integration": {"full_integration": "Simple query, no integration needed."},
                "solution": {"full_solution": response},
                "verification": {"full_verification": "Simple query, no verification needed."},
                "reflection": "Simple query processed with fast response."
            }
        
        return result

    def _solve_parallel(self, query: str, show_work: bool = True) -> Dict:
        """Solve a problem using parallel API calls for faster processing."""
        # This is a placeholder - actual implementation would use ThreadPoolExecutor
        # For now, just fall back to the standard approach
        return self._minimal_reasoning(query, show_work)
        
    def _minimal_reasoning(self, query: str, show_work: bool = True) -> Dict:
        """Generate a minimalistic reasoning approach with fewer API calls."""
        # Single API call for both analysis and solution
        prompt = f"""
        Analyze and solve the following problem:
        
        {query}
        
        Provide a direct, comprehensive solution.
        """
        
        response = self._generate_response(prompt, emit_thoughts=True)
        
        # Create minimal cognitive graph for UI
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        answer_node = CognitiveNode(
            node_id="final_answer_0",
            node_type="final_answer",
            content=response,
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(answer_node)
        
        result = {
            "final_answer": response,
            "cognitive_graph": self.cognitive_graph.to_dict()
        }
        
        if show_work:
            result["reasoning_process"] = {
                "decomposition": {"full_decomposition": "Simplified analysis for quick response."},
                "perspectives": [{"name": "Direct Approach", "description": "Direct problem-solving approach"}],
                "evidence": {},
                "integration": {"full_integration": "Simplified integration for quick response."},
                "solution": {"full_solution": response},
                "verification": {"full_verification": "Verification skipped for performance."},
                "reflection": "Optimized reasoning with minimal API calls."
            }
        
        return result
    
    # Compatibility method for old code that doesn't use async
    def _generate_response(self, prompt: str, emit_thoughts: bool = False) -> str:
        """Non-async version of generate_response for backward compatibility."""
        # Create event loop if not existing
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async method
        return loop.run_until_complete(self.generate_response(prompt, None, emit_thoughts))

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
        "metacognitive_reflection": "#BD93F9",
        "module_selection": "#80FFEA",
        "module_processing": "#9580FF",
        "ensemble": "#FFCA80",
        "clarification": "#FF80BF"
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
            "Fast Mode uses 4000 tokens instead of 8000 and reduces API calls for quicker responses. "
            "Turn off for more thorough analysis on complex problems that require extensive token generation."
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
    thought_generated = pyqtSignal(str)  # Signal for realtime reasoning thoughts
    clarification_needed = pyqtSignal(str)  # Signal for interactive clarification
    
    def __init__(self, neocortex, query, show_work=True, interactive=False):
        super().__init__()
        self.neocortex = neocortex
        self.query = query
        self.show_work = show_work
        self.interactive = interactive
        self.clarification_response = None
        self.clarification_event = threading.Event()
    
    def set_clarification_response(self, response):
        """Set the clarification response from the user."""
        self.clarification_response = response
        self.clarification_event.set()
    
    def run(self):
        try:
            # Create event loop for async
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async solve method
            result = loop.run_until_complete(self._async_run())
            
            # Emit the result
            self.progress_update.emit("Processing complete!", 100)
            self.result_ready.emit(result)
            
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Error: {str(e)}\n{traceback.format_exc()}")
    
    async def _async_run(self):
        """Async version of the run method."""
        # Emit progress updates for each major step
        steps = [
            "Analyzing problem structure...",
            "Selecting specialized modules...",
            "Processing with modules...",
            "Combining results with ensemble method...",
            "Generating final answer...",
            "Performing metacognitive reflection..."
        ]
        
        # Report initial progress
        self.progress_update.emit(steps[0], 0)
        
        # Attach thought_generated signal
        self.neocortex.thought_generated = self.thought_generated
        
        # Solve the problem
        result = await self.neocortex.solve(self.query, self.show_work, self.interactive)
        
        # Check if clarification is needed
        if result.get("needs_clarification", False):
            clarification_question = result.get("clarification_question", "Could you please provide more details?")
            
            # Emit signal to request clarification
            self.clarification_needed.emit(clarification_question)
            
            # Wait for clarification response
            self.clarification_event.wait()
            
            # Use the clarification to improve the query
            improved_query = await self.neocortex.incorporate_feedback(self.query, self.clarification_response)
            
            # Solve with improved query
            self.progress_update.emit("Processing with clarified information...", 20)
            result = await self.neocortex.solve(improved_query, self.show_work, False)  # No more clarification needed
        
        # Normal progress updates
        progress_step = 100 / len(steps)
        for i, step in enumerate(steps):
            progress = min(int((i + 1) * progress_step), 99)
            self.progress_update.emit(step, progress)
        
        return result

class ModuleToggleWidget(QWidget):
    """Widget for toggling NeoCortex modules on and off."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Add title
        title = QLabel("Specialized Reasoning Modules")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        
        # Add description
        description = QLabel("Enable or disable specialized reasoning modules:")
        layout.addWidget(description)
        
        self.module_checkboxes = {}
        
        modules = [
            ("mathematics", "Mathematical Reasoning", True),
            ("ethics", "Ethical Reasoning", True),
            ("logic", "Logical Reasoning", True),
            ("creativity", "Creative Thinking", True),
            ("temporal", "Temporal Reasoning", True),
            ("concept_network", "Concept Network", True),
            ("counterfactual", "Counterfactual Reasoning", True),
            ("metacognition", "Metacognitive Reflection", True)
        ]
        
        for module_id, module_name, default_state in modules:
            checkbox = QCheckBox(module_name)
            checkbox.setChecked(default_state)
            checkbox.stateChanged.connect(self.on_module_toggled)
            layout.addWidget(checkbox)
            self.module_checkboxes[module_id] = checkbox
        
        # Add section for ensemble methods
        ensemble_group = QGroupBox("Ensemble Methods")
        ensemble_layout = QVBoxLayout()
        
        self.ensemble_radios = {}
        
        ensembles = [
            ("weighted", "Weighted Ensemble (Default)", True),
            ("majority_vote", "Majority Vote Ensemble", False),
            ("diversity", "Diversity Promoting Ensemble", False)
        ]
        
        for ensemble_id, ensemble_name, default_state in ensembles:
            radio = QRadioButton(ensemble_name)
            radio.setChecked(default_state)
            ensemble_layout.addWidget(radio)
            self.ensemble_radios[ensemble_id] = radio
        
        ensemble_group.setLayout(ensemble_layout)
        layout.addWidget(ensemble_group)
        
        # Add interactive mode checkbox
        self.interactive_mode = QCheckBox("Interactive Clarification Mode")
        self.interactive_mode.setChecked(False)
        self.interactive_mode.setToolTip("When enabled, NeoCortex will ask for clarification if a query is ambiguous")
        layout.addWidget(self.interactive_mode)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def on_module_toggled(self):
        # This would connect to NeoCortex to enable/disable modules
        pass
    
    def get_active_modules(self):
        return {module_id: checkbox.isChecked() 
                for module_id, checkbox in self.module_checkboxes.items()}
    
    def get_selected_ensemble(self):
        for ensemble_id, radio in self.ensemble_radios.items():
            if radio.isChecked():
                return ensemble_id
        return "weighted"  # Default
    
    def is_interactive_mode(self):
        return self.interactive_mode.isChecked()

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
        
        # Token limit
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("Max Tokens:"))
        self.token_slider = QSlider(Qt.Horizontal)
        self.token_slider.setRange(1000, 16000)  # Significantly increased range
        self.token_slider.setValue(8000)  # Default 8000
        self.token_slider.setTickPosition(QSlider.TicksBelow)
        self.token_slider.setTickInterval(2000)
        self.token_value = QLabel("8000")
        self.token_slider.valueChanged.connect(self.update_token_label)
        token_layout.addWidget(self.token_slider)
        token_layout.addWidget(self.token_value)
        params_layout.addLayout(token_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Performance settings
        performance_group = QGroupBox("Performance Settings")
        performance_layout = QVBoxLayout()
        
        # Reasoning depth
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Reasoning Depth:"))
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["Minimal", "Balanced", "Thorough"])
        self.depth_combo.setCurrentText("Balanced")
        depth_layout.addWidget(self.depth_combo)
        performance_layout.addLayout(depth_layout)
        
        # Concurrent requests
        self.concurrent_check = QCheckBox("Enable Concurrent API Requests")
        self.concurrent_check.setChecked(True)
        performance_layout.addWidget(self.concurrent_check)
        
        # Response caching
        self.caching_check = QCheckBox("Enable Response Caching")
        self.caching_check.setChecked(True)
        performance_layout.addWidget(self.caching_check)
        
        performance_group.setLayout(performance_layout)
        layout.addWidget(performance_group)
        
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
        
    def update_token_label(self, value):
        self.token_value.setText(f"{value}")
    
    def apply_settings(self):
        # Use the hardcoded API key
        api_key = "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"
        os.environ["OPENROUTER_API_KEY"] = api_key
        
        settings = {
            "model": self.model_combo.currentText(),
            "temperature": float(self.temp_value.text()),
            "top_p": float(self.topp_value.text()),
            "max_tokens": int(self.token_value.text()),
            "reasoning_depth": self.depth_combo.currentText().lower(),
            "concurrent_requests": self.concurrent_check.isChecked(),
            "response_caching": self.caching_check.isChecked()
        }
        self.settings_changed.emit(settings)

class MetaLearningWidget(QWidget):
    """Widget for displaying meta-learning statistics and insights."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Meta-learning statistics
        stats_group = QGroupBox("Performance Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Improvement suggestions
        suggestions_group = QGroupBox("Improvement Suggestions")
        suggestions_layout = QVBoxLayout()
        
        self.suggestions_list = QListWidget()
        suggestions_layout.addWidget(self.suggestions_list)
        
        suggestions_group.setLayout(suggestions_layout)
        layout.addWidget(suggestions_group)
        
        # A/B testing controls
        ab_group = QGroupBox("A/B Testing")
        ab_layout = QVBoxLayout()
        
        ab_controls = QHBoxLayout()
        self.start_ab_btn = QPushButton("Start New A/B Test")
        self.start_ab_btn.clicked.connect(self.start_ab_test)
        self.stop_ab_btn = QPushButton("Stop Current Test")
        self.stop_ab_btn.clicked.connect(self.stop_ab_test)
        ab_controls.addWidget(self.start_ab_btn)
        ab_controls.addWidget(self.stop_ab_btn)
        ab_layout.addLayout(ab_controls)
        
        self.ab_results = QTextEdit()
        self.ab_results.setReadOnly(True)
        ab_layout.addWidget(self.ab_results)
        
        ab_group.setLayout(ab_layout)
        layout.addWidget(ab_group)
        
        self.setLayout(layout)
    
    def update_stats(self, meta_learning_system):
        """Update the statistics display."""
        stats = meta_learning_system.to_dict()
        
        # Format stats as HTML
        html = "<h3>System Performance</h3>"
        html += f"<p>Total reasoning attempts: {stats.get('reasoning_history_count', 0)}</p>"
        
        # Domain performance
        html += "<h3>Domain Performance</h3>"
        html += "<table border='1' cellspacing='0' cellpadding='3' style='border-collapse: collapse;'>"
        html += "<tr><th>Domain</th><th>Uses</th><th>Success Rate</th></tr>"
        
        for domain, domain_stats in stats.get("domain_performance", {}).items():
            uses = domain_stats.get("uses", 0)
            success_rate = domain_stats.get("success_rate", 0.0) * 100
            html += f"<tr><td>{domain}</td><td>{uses}</td><td>{success_rate:.1f}%</td></tr>"
        
        html += "</table>"
        
        # Module performance
        html += "<h3>Module Performance</h3>"
        html += "<table border='1' cellspacing='0' cellpadding='3' style='border-collapse: collapse;'>"
        html += "<tr><th>Module</th><th>Uses</th><th>Success Rate</th></tr>"
        
        for module, module_stats in stats.get("module_performance", {}).items():
            uses = module_stats.get("uses", 0)
            success_rate = module_stats.get("success_rate", 0.0) * 100
            html += f"<tr><td>{module}</td><td>{uses}</td><td>{success_rate:.1f}%</td></tr>"
        
        html += "</table>"
        
        self.stats_text.setHtml(html)
        
        # Update suggestions list
        self.suggestions_list.clear()
        for suggestion in stats.get("improvement_suggestions", []):
            item = QListWidgetItem()
            item.setText(f"[{suggestion.get('priority', 'medium')}] {suggestion.get('suggestion', '')}")
            if suggestion.get("priority") == "high":
                item.setBackground(QColor(255, 200, 200))
            self.suggestions_list.addItem(item)
    
    def update_ab_results(self, ab_tests):
        """Update the A/B testing results display."""
        html = "<h3>Active A/B Tests</h3>"
        
        active_tests = {name: test for name, test in ab_tests.items() if test.get("active", False)}
        completed_tests = {name: test for name, test in ab_tests.items() if not test.get("active", True)}
        
        if active_tests:
            for name, test in active_tests.items():
                html += f"<h4>Test: {name}</h4>"
                html += "<table border='1' cellspacing='0' cellpadding='3' style='border-collapse: collapse;'>"
                html += "<tr><th>Variant</th><th>Uses</th><th>Successes</th><th>Success Rate</th></tr>"
                
                for variant_id, result in test.get("results", {}).items():
                    uses = result.get("uses", 0)
                    successes = result.get("successes", 0)
                    success_rate = (successes / max(1, uses)) * 100
                    html += f"<tr><td>Variant {variant_id}</td><td>{uses}</td><td>{successes}</td><td>{success_rate:.1f}%</td></tr>"
                
                html += "</table>"
        else:
            html += "<p>No active tests</p>"
        
        html += "<h3>Completed A/B Tests</h3>"
        
        if completed_tests:
            for name, test in completed_tests.items():
                html += f"<h4>Test: {name}</h4>"
                html += f"<p>Winner: Variant {test.get('winner', 'unknown')}</p>"
                html += "<table border='1' cellspacing='0' cellpadding='3' style='border-collapse: collapse;'>"
                html += "<tr><th>Variant</th><th>Uses</th><th>Success Rate</th></tr>"
                
                for variant_id, result in test.get("results", {}).items():
                    uses = result.get("uses", 0)
                    success_rate = result.get("success_rate", 0.0) * 100
                    html += f"<tr><td>Variant {variant_id}</td><td>{uses}</td><td>{success_rate:.1f}%</td></tr>"
                
                html += "</table>"
        else:
            html += "<p>No completed tests</p>"
        
        self.ab_results.setHtml(html)
    
    def start_ab_test(self):
        """Open dialog to configure a new A/B test."""
        pass  # Implementation would open a dialog to configure the test
    
    def stop_ab_test(self):
        """Stop the current A/B test."""
        pass  # Implementation would stop the current test

class RealtimeReasoningWidget(QWidget):
    """Widget for displaying realtime reasoning thoughts from the model."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.thought_count = 0
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title and instruction
        title = QLabel("Real-time Reasoning Process")
        title.setStyleSheet(f"font-weight: bold; color: {COLOR_SCHEME['accent']}")
        layout.addWidget(title)
        
        instructions = QLabel("Watch as the model reasons through the problem step by step:")
        layout.addWidget(instructions)
        
        # Realtime reasoning display with improved formatting
        self.thought_display = QTextEdit()
        self.thought_display.setReadOnly(True)
        self.thought_display.setPlaceholderText("Model reasoning steps will appear here in real-time...")
        self.thought_display.setStyleSheet(f"""
            QTextEdit {{
                line-height: 1.5;
                padding: 10px;
                background-color: {COLOR_SCHEME['secondary_bg']};
                border: 1px solid {COLOR_SCHEME['accent']};
            }}
        """)
        layout.addWidget(self.thought_display)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Auto-scroll checkbox
        self.auto_scroll = QCheckBox("Auto-scroll")
        self.auto_scroll.setChecked(True)
        controls_layout.addWidget(self.auto_scroll)
        
        # Word wrap checkbox
        self.word_wrap = QCheckBox("Word wrap")
        self.word_wrap.setChecked(True)
        self.word_wrap.stateChanged.connect(self.toggle_word_wrap)
        controls_layout.addWidget(self.word_wrap)
        
        # Copy to clipboard button
        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        controls_layout.addWidget(self.copy_button)
        
        layout.addLayout(controls_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # Set default word wrap
        self.toggle_word_wrap(Qt.Checked)
    
    def add_thought(self, thought):
        """Add a new thought to the display with improved formatting."""
        if not thought.strip():
            return
            
        self.thought_count += 1
        current_text = self.thought_display.toHtml()
        
        # Format the new thought with step number and styling
        formatted_thought = f"""
        <div style="margin-bottom: 12px;">
            <span style="color: {COLOR_SCHEME['accent']}; font-weight: bold;">Step {self.thought_count}:</span> 
            <span style="color: {COLOR_SCHEME['text']};">{thought}</span>
        </div>
        """
        
        # Combine with existing content
        if "<body" in current_text and "</body>" in current_text:
            # If already HTML, insert before closing body
            new_text = current_text.replace("</body>", formatted_thought + "</body>")
        elif current_text.strip():
            # If text but not HTML
            new_text = f"<html><body>{current_text}{formatted_thought}</body></html>"
        else:
            # If empty
            new_text = f"<html><body>{formatted_thought}</body></html>"
        
        # Update the status label
        self.status_label.setText(f"Captured {self.thought_count} reasoning steps")
        
        # Limit display size for performance with very large content
        if len(new_text) > 200000:
            truncated_notice = "<div style='color:red;'>Earlier thoughts truncated for performance...</div>"
            # Find a suitable div to truncate after
            truncate_pos = new_text.find("<div", 50000)
            if truncate_pos > 0:
                new_text = new_text[:truncate_pos] + truncated_notice + new_text[truncate_pos:]
        
        self.thought_display.setHtml(new_text)
        
        # Auto-scroll to bottom if enabled
        if self.auto_scroll.isChecked():
            scrollbar = self.thought_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def toggle_word_wrap(self, state):
        """Toggle word wrap in the thought display."""
        if state == Qt.Checked:
            self.thought_display.setLineWrapMode(QTextEdit.WidgetWidth)
        else:
            self.thought_display.setLineWrapMode(QTextEdit.NoWrap)
    
    def clear(self):
        """Clear the thought display and reset counter."""
        self.thought_display.clear()
        self.thought_count = 0
        self.status_label.setText("")
    
    def copy_to_clipboard(self):
        """Copy the thoughts to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.thought_display.toPlainText())
        
        # Show a brief status message
        self.copy_button.setText("Copied!")
        self.status_label.setText("Content copied to clipboard")
        QTimer.singleShot(2000, lambda: self.copy_button.setText("Copy to Clipboard"))

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
        
        # Module Outputs tab
        self.modules_widget = QTabWidget()
        self.tab_widget.addTab(self.modules_widget, "Module Outputs")
        
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
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self.copy_current_tab)
        self.feedback_btn = QPushButton("Provide Feedback")
        self.feedback_btn.clicked.connect(self.provide_feedback)
        export_layout.addWidget(self.export_answer_btn)
        export_layout.addWidget(self.export_full_btn)
        export_layout.addWidget(self.copy_btn)
        export_layout.addWidget(self.feedback_btn)
        layout.addLayout(export_layout)
        
        self.setLayout(layout)
    
    def display_result(self, result):
        """Display the reasoning result in the appropriate widgets."""
        # Display final answer
        final_answer = result.get("final_answer", "No final answer available.")
        self.final_answer_widget.setHtml(f"<h2>Final Answer</h2><p>{final_answer}</p>")
        
        # Clear module widgets from previous runs
        while self.modules_widget.count() > 0:
            self.modules_widget.removeTab(0)
        
        # Display module outputs if available
        if "reasoning_process" in result and "module_outputs" in result["reasoning_process"]:
            module_outputs = result["reasoning_process"]["module_outputs"]
            
            for i, output in enumerate(module_outputs):
                module_name = output.get("source", f"Module {i+1}")
                module_widget = QTextEdit()
                module_widget.setReadOnly(True)
                
                # Format module output
                response = output.get("response", "")
                confidence = output.get("confidence", 0.0) * 100
                domain = output.get("domain", "unknown")
                
                module_html = f"<h2>{module_name} (Confidence: {confidence:.1f}%)</h2>"
                module_html += f"<p><strong>Domain:</strong> {domain}</p>"
                module_html += f"<h3>Output:</h3><p>{response}</p>"
                
                # Add additional details if available
                module_result = output.get("module_result", {})
                if "steps" in module_result:
                    module_html += "<h3>Reasoning Steps:</h3><ol>"
                    for step in module_result["steps"]:
                        module_html += f"<li>{step}</li>"
                    module_html += "</ol>"
                
                if "perspectives" in module_result:
                    module_html += "<h3>Perspectives:</h3><ul>"
                    for perspective in module_result["perspectives"]:
                        module_html += f"<li><strong>{perspective.get('framework', '')}</strong>: {perspective.get('content', '')}</li>"
                    module_html += "</ul>"
                
                if "causal_chains" in module_result:
                    module_html += "<h3>Causal Chains:</h3><ul>"
                    for chain in module_result["causal_chains"]:
                        module_html += "<li><ol>"
                        for step in chain.get("steps", []):
                            module_html += f"<li>{step}</li>"
                        module_html += "</ol></li>"
                    module_html += "</ul>"
                
                module_widget.setHtml(module_html)
                self.modules_widget.addTab(module_widget, module_name)
            
            # Display ensemble result if available
            if "ensemble_result" in result["reasoning_process"]:
                ensemble_result = result["reasoning_process"]["ensemble_result"]
                ensemble_widget = QTextEdit()
                ensemble_widget.setReadOnly(True)
                
                ensemble_method = ensemble_result.get("ensemble_method", "Unknown")
                sources = ensemble_result.get("sources", [])
                weights = ensemble_result.get("weights", [])
                
                ensemble_html = f"<h2>Ensemble Method: {ensemble_method}</h2>"
                
                if sources and weights and len(sources) == len(weights):
                    ensemble_html += "<h3>Source Weights:</h3><ul>"
                    for source, weight in zip(sources, weights):
                        ensemble_html += f"<li>{source}: {weight:.2f}</li>"
                    ensemble_html += "</ul>"
                
                ensemble_widget.setHtml(ensemble_html)
                self.modules_widget.addTab(ensemble_widget, "Ensemble")
        
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
            with open(file_path, 'w', encoding='utf-8') as f:
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
            with open(file_path, 'w', encoding='utf-8') as f:
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
    
    def copy_current_tab(self):
        """Copy the current tab's content to clipboard."""
        clipboard = QApplication.clipboard()
        
        # Determine which tab is active
        current_tab_idx = self.tab_widget.currentIndex()
        tab_name = self.tab_widget.tabText(current_tab_idx)
        
        if tab_name == "Final Answer":
            content = self.final_answer_widget.toPlainText()
        elif tab_name == "Detailed Reasoning":
            # For the reasoning tab, determine which sub-tab is active
            current_subtab_idx = self.reasoning_widget.currentIndex()
            subtab_name = self.reasoning_widget.tabText(current_subtab_idx)
            
            if subtab_name == "Problem Decomposition":
                content = self.decomposition_widget.toPlainText()
            elif subtab_name == "Perspectives":
                content = self.perspectives_widget.toPlainText()
            elif subtab_name == "Evidence":
                content = self.evidence_widget.toPlainText()
            elif subtab_name == "Solution":
                content = self.solution_widget.toPlainText()
            elif subtab_name == "Verification":
                content = self.verification_widget.toPlainText()
            elif subtab_name == "Metacognitive Reflection":
                content = self.reflection_widget.toPlainText()
            else:
                content = "No content to copy from this tab."
        elif tab_name == "Module Outputs":
            # For the modules tab, determine which module is active
            current_module_idx = self.modules_widget.currentIndex()
            if current_module_idx >= 0:
                module_widget = self.modules_widget.widget(current_module_idx)
                content = module_widget.toPlainText()
            else:
                content = "No content to copy from this tab."
        elif tab_name == "Cognitive Graph":
            content = "Graph visualization cannot be copied as text."
        else:
            content = "No content to copy from this tab."
        
        clipboard.setText(content)
        self.copy_btn.setText("Copied!")
        QTimer.singleShot(2000, lambda: self.copy_btn.setText("Copy to Clipboard"))
    
    def provide_feedback(self):
        """Provide feedback to improve the reasoning."""
        # Implementation would collect feedback and send it to NeoCortex
        # This would open a dialog to collect feedback
        pass

class CognitiveGraphVisualizer(QWidget):
    """Enhanced widget for visualizing the cognitive graph."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.graph_data = None
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Controls for graph visualization
        controls_layout = QHBoxLayout()
        
        # Layout selection
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Spring", "Circular", "Hierarchical", "Spectral"])
        self.layout_combo.currentTextChanged.connect(self.redraw_graph)
        controls_layout.addWidget(QLabel("Layout:"))
        controls_layout.addWidget(self.layout_combo)
        
        # Node size control
        self.node_size_slider = QSlider(Qt.Horizontal)
        self.node_size_slider.setRange(10, 50)
        self.node_size_slider.setValue(30)
        self.node_size_slider.valueChanged.connect(self.redraw_graph)
        controls_layout.addWidget(QLabel("Node Size:"))
        controls_layout.addWidget(self.node_size_slider)
        
        # Node filter
        self.node_filter = QComboBox()
        self.node_filter.addItem("All Node Types")
        self.node_filter.currentTextChanged.connect(self.redraw_graph)
        controls_layout.addWidget(QLabel("Filter:"))
        controls_layout.addWidget(self.node_filter)
        
        layout.addLayout(controls_layout)
        
        # Toolbar for graph interactions
        toolbar_layout = QHBoxLayout()
        
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        toolbar_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        toolbar_layout.addWidget(self.zoom_out_btn)
        
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        toolbar_layout.addWidget(self.reset_view_btn)
        
        self.export_graph_btn = QPushButton("Export Graph")
        self.export_graph_btn.clicked.connect(self.export_graph)
        toolbar_layout.addWidget(self.export_graph_btn)
        
        layout.addLayout(toolbar_layout)
        
        # Matplotlib canvas for graph visualization
        self.figure = plt.figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Node details display
        self.node_details = QTextEdit()
        self.node_details.setReadOnly(True)
        self.node_details.setMaximumHeight(150)
        self.node_details.setPlaceholderText("Select a node to view details...")
        layout.addWidget(self.node_details)
        
        self.setLayout(layout)
        
        # Initialize variables for zoom
        self.zoom_level = 1.0
    
    def set_graph_data(self, graph_data):
        """Set the cognitive graph data and visualize it."""
        self.graph_data = graph_data
        
        # Update node filter with available node types
        self.node_filter.clear()
        self.node_filter.addItem("All Node Types")
        
        if graph_data and "nodes" in graph_data:
            node_types = set()
            for node in graph_data["nodes"].values():
                if "node_type" in node:
                    node_types.add(node["node_type"])
            
            for node_type in sorted(node_types):
                self.node_filter.addItem(node_type)
        
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
        
        # Apply filter if needed
        filter_type = self.node_filter.currentText()
        if filter_type != "All Node Types":
            filtered_nodes = {node_id: node for node_id, node in nodes.items() 
                             if node.get("node_type") == filter_type}
        else:
            filtered_nodes = nodes
        
        for node_id, node in filtered_nodes.items():
            G.add_node(node_id, 
                      node_type=node.get("node_type", "unknown"),
                      content=node.get("content", "")[:50],  # Truncate content for display
                      domain=node.get("domain", "general"),
                      confidence=node.get("confidence", 0.8))
        
        # Add edges based on parent-child relationships
        for node_id, node in filtered_nodes.items():
            parent_id = node.get("parent_id")
            if parent_id and parent_id in filtered_nodes:
                G.add_edge(parent_id, node_id)
            
            # Add edges based on relationships
            relationships = node.get("relationships", {})
            for rel_type, targets in relationships.items():
                for target in targets:
                    if target in filtered_nodes:
                        G.add_edge(node_id, target, relationship=rel_type)
        
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
        
        # Add title and remove axis
        ax.set_title("Cognitive Process Graph")
        ax.axis('off')
        
        # Apply zoom
        ax.set_xlim(ax.get_xlim()[0] / self.zoom_level, ax.get_xlim()[1] / self.zoom_level)
        ax.set_ylim(ax.get_ylim()[0] / self.zoom_level, ax.get_ylim()[1] / self.zoom_level)
        
        # Save the graph data for later interaction
        self.G = G
        self.pos = pos
        
        # Connect event for node selection
        self.canvas.mpl_connect('button_press_event', self.on_node_click)
        
        self.canvas.draw()
    
    def on_node_click(self, event):
        """Handle node click to display details."""
        if not hasattr(self, 'G') or not hasattr(self, 'pos'):
            return
        
        # Convert click to graph coordinates
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # Find the closest node
        node_id = self._find_closest_node(x, y)
        if node_id is None:
            return
        
        # Display node details
        node_data = self.graph_data["nodes"].get(node_id, {})
        
        details = f"<h3>Node: {node_id}</h3>"
        details += f"<p><strong>Type:</strong> {node_data.get('node_type', '')}</p>"
        details += f"<p><strong>Domain:</strong> {node_data.get('domain', 'general')}</p>"
        
        if 'confidence' in node_data:
            details += f"<p><strong>Confidence:</strong> {node_data['confidence'] * 100:.1f}%</p>"
        
        content = node_data.get('content', '')
        if len(content) > 500:
            content = content[:500] + "..."
        details += f"<p><strong>Content:</strong> {content}</p>"
        
        self.node_details.setHtml(details)
    
    def _find_closest_node(self, x, y, threshold=0.1):
        """Find the closest node to the given coordinates."""
        min_dist = float('inf')
        closest_node = None
        
        for node, (nx, ny) in self.pos.items():
            dist = ((nx - x) ** 2 + (ny - y) ** 2) ** 0.5
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_node = node
        
        return closest_node
    
    def zoom_in(self):
        """Zoom in on the graph."""
        self.zoom_level *= 1.2
        self.redraw_graph()
    
    def zoom_out(self):
        """Zoom out of the graph."""
        self.zoom_level /= 1.2
        self.redraw_graph()
    
    def reset_view(self):
        """Reset the graph view."""
        self.zoom_level = 1.0
        self.redraw_graph()
    
    def export_graph(self):
        """Export the graph as an image."""
        if not hasattr(self, 'figure'):
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Graph", "", "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful",
                                   f"Graph exported successfully to {file_path}")
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
        
        # Add search field
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search history...")
        self.search_field.textChanged.connect(self.filter_history)
        search_layout.addWidget(self.search_field)
        layout.addLayout(search_layout)
        
        # Add history list
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_item_selected)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.list_widget)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.export_button = QPushButton("Export History")
        self.export_button.clicked.connect(self.export_history)
        buttons_layout.addWidget(self.export_button)
        
        self.clear_button = QPushButton("Clear History")
        self.clear_button.clicked.connect(self.clear_history)
        buttons_layout.addWidget(self.clear_button)
        
        layout.addLayout(buttons_layout)
        
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
        self._add_list_item(item_data, len(self.history) - 1)
    
    def _add_list_item(self, item_data, index):
        """Add an item to the list widget."""
        query = item_data["query"]
        timestamp = item_data["timestamp"]
        
        # Format display text
        if len(query) > 50:
            display_text = f"{timestamp}: {query[:50]}..."
        else:
            display_text = f"{timestamp}: {query}"
        
        list_item = QListWidgetItem(display_text)
        list_item.setData(Qt.UserRole, index)  # Store index
        self.list_widget.addItem(list_item)
    
    def on_item_selected(self, item):
        """Handle selection of a history item."""
        index = item.data(Qt.UserRole)
        self.item_selected.emit(self.history[index])
    
    def filter_history(self, text):
        """Filter history items based on search text."""
        self.list_widget.clear()
        
        # If no search text, show all items
        if not text.strip():
            for i, item in enumerate(self.history):
                self._add_list_item(item, i)
            return
        
        # Otherwise, filter based on query content
        text = text.lower()
        for i, item in enumerate(self.history):
            if text in item["query"].lower():
                self._add_list_item(item, i)
    
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
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2)
            
            QMessageBox.information(self, "Export Successful", 
                                   f"History exported successfully to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export: {str(e)}")
    
    def clear_history(self):
        """Clear the history."""
        reply = QMessageBox.question(
            self, "Clear History",
            "Are you sure you want to clear all history items?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.history = []
            self.list_widget.clear()
    
    def show_context_menu(self, position):
        """Show context menu for history items."""
        item = self.list_widget.itemAt(position)
        if not item:
            return
        
        menu = QMenu()
        view_action = menu.addAction("View")
        delete_action = menu.addAction("Delete")
        
        action = menu.exec_(self.list_widget.mapToGlobal(position))
        
        if action == view_action:
            self.on_item_selected(item)
        elif action == delete_action:
            self._delete_item(item)
    
    def _delete_item(self, item):
        """Delete a history item."""
        index = item.data(Qt.UserRole)
        
        # Remove from history list
        del self.history[index]
        
        # Refresh list widget
        self.list_widget.clear()
        for i, item_data in enumerate(self.history):
            self._add_list_item(item_data, i)

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
        
        # History views
        self.history_check = QCheckBox("Show Activity History")
        self.history_check.stateChanged.connect(self.update_view)
        layout.addWidget(self.history_check)
        
        self.setLayout(layout)
        
        # Initialize activity history
        self.activity_history = []
    
    def update_activity(self, module_activity):
        """Update the module activity visualization."""
        self.module_activity = module_activity
        
        # Add to history
        self.activity_history.append(module_activity.copy())
        if len(self.activity_history) > 20:
            self.activity_history.pop(0)
        
        self.redraw()
    
    def update_view(self):
        """Update the view based on history checkbox."""
        self.redraw()
    
    def redraw(self):
        """Redraw the module activity visualization."""
        if not self.module_activity:
            return
        
        self.figure.clear()
        
        if self.history_check.isChecked() and len(self.activity_history) > 1:
            # Draw activity history
            ax = self.figure.add_subplot(111)
            
            # Get modules from current activity
            modules = list(self.module_activity.keys())
            
            # Create arrays for time series
            x = list(range(len(self.activity_history)))
            data = {module: [] for module in modules}
            
            # Collect data points
            for hist_point in self.activity_history:
                for module in modules:
                    data[module].append(hist_point.get(module, 0.0))
            
            # Plot each module as a line
            for module in modules:
                ax.plot(x, data[module], label=module)
            
            ax.set_xlabel("Time")
            ax.set_ylabel("Activity Level")
            ax.set_title("Module Activity Over Time")
            ax.legend(loc='best')
            ax.set_ylim([0, 1])
        else:
            # Draw current activity as bar chart
            ax = self.figure.add_subplot(111)
            
            modules = list(self.module_activity.keys())
            activity = list(self.module_activity.values())
            
            ax.barh(modules, activity, color=COLOR_SCHEME["accent"])
            ax.set_xlabel("Activity Level")
            ax.set_title("Current Module Activity")
            ax.set_xlim([0, 1])
        
        self.canvas.draw()

class NeoCortexGUI(QMainWindow):
    """Enhanced GUI application for NeoCortex."""
    
    def __init__(self):
        super().__init__()
        self.neocortex = NeoCortex()
        self.current_result = None
        self.reasoning_thread = None
        self.init_ui()
        self.apply_dark_theme()
        
        # Initialize meta-learning statistics update timer
        self.meta_learning_timer = QTimer()
        self.meta_learning_timer.timeout.connect(self.update_meta_learning_stats)
        self.meta_learning_timer.start(60000)  # Update every minute
    
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
        left_tabs.addTab(self.module_toggle_widget, "Specialized Modules")
        
        # Settings tab
        self.model_settings_widget = ModelSettingsWidget()
        self.model_settings_widget.settings_changed.connect(self.update_model_settings)
        left_tabs.addTab(self.model_settings_widget, "Settings")
        
        # History tab
        self.history_widget = HistoryWidget()
        self.history_widget.item_selected.connect(self.load_history_item)
        left_tabs.addTab(self.history_widget, "History")
        
        # Meta-learning tab
        self.meta_learning_widget = MetaLearningWidget()
        left_tabs.addTab(self.meta_learning_widget, "Meta-Learning")
        
        # Module activity visualization
        self.module_activity_widget = ModuleActivityWidget()
        left_tabs.addTab(self.module_activity_widget, "Activity")
        
        # Realtime reasoning visualization
        self.realtime_reasoning_widget = RealtimeReasoningWidget()
        left_tabs.addTab(self.realtime_reasoning_widget, "Realtime Thoughts")
        
        left_layout.addWidget(left_tabs)
        left_widget.setLayout(left_layout)
        
        # Right side - Results display
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
        
        # Dialog for clarification questions
        self.clarification_dialog = None
    
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
        self.neocortex.max_tokens = 4000 if use_fast_mode else 8000  # Significantly increased token limits
        
        # Apply module settings
        active_modules = self.module_toggle_widget.get_active_modules()
        for module_id, active in active_modules.items():
            if module_id in self.neocortex.module_states:
                self.neocortex.module_states[module_id] = active
            if module_id in self.neocortex.modules:
                self.neocortex.modules[module_id].active = active
        
        # Apply ensemble method selection
        selected_ensemble = self.module_toggle_widget.get_selected_ensemble()
        self.neocortex.default_ensemble = selected_ensemble
        
        # Enable/disable interactive mode
        interactive_mode = self.module_toggle_widget.is_interactive_mode()
        
        # Start reasoning thread
        show_work = self.show_work_check.isChecked()
        
        # Use optimized flow if in fast mode
        self.reasoning_thread = ReasoningThread(self.neocortex, query, show_work, interactive_mode)
        self.reasoning_thread.progress_update.connect(self.update_progress)
        self.reasoning_thread.result_ready.connect(self.handle_result)
        self.reasoning_thread.error_occurred.connect(self.handle_error)
        self.reasoning_thread.thought_generated.connect(self.realtime_reasoning_widget.add_thought)
        self.reasoning_thread.clarification_needed.connect(self.handle_clarification)
        
        # Clear previous thoughts
        self.realtime_reasoning_widget.clear()
        
        # Switch to the realtime thoughts tab to show thoughts as they happen
        left_tabs = self.realtime_reasoning_widget.parent()
        if isinstance(left_tabs, QTabWidget):
            for i in range(left_tabs.count()):
                if left_tabs.tabText(i) == "Realtime Thoughts":
                    left_tabs.setCurrentIndex(i)
                    break
                    
        self.reasoning_thread.start()
        
        # Simulate module activity for UI demonstration
        self.simulate_module_activity()
    
    def handle_clarification(self, question):
        """Handle a clarification request from the reasoning thread."""
        # Create a dialog to get clarification from the user
        if self.clarification_dialog is not None:
            self.clarification_dialog.close()
        
        self.clarification_dialog = QDialog(self)
        self.clarification_dialog.setWindowTitle("Clarification Needed")
        self.clarification_dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Display the clarification question
        layout.addWidget(QLabel("NeoCortex needs additional information:"))
        question_label = QLabel(question)
        question_label.setWordWrap(True)
        layout.addWidget(question_label)
        
        # Input field for the user's response
        layout.addWidget(QLabel("Your clarification:"))
        response_edit = QTextEdit()
        response_edit.setPlaceholderText("Type your clarification here...")
        layout.addWidget(response_edit)
        
        # Buttons
        button_layout = QHBoxLayout()
        submit_btn = QPushButton("Submit")
        cancel_btn = QPushButton("Cancel")
        
        submit_btn.clicked.connect(lambda: self._submit_clarification(response_edit.toPlainText()))
        cancel_btn.clicked.connect(lambda: self._cancel_clarification())
        
        button_layout.addWidget(submit_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.clarification_dialog.setLayout(layout)
        self.clarification_dialog.exec_()
    
    def _submit_clarification(self, response):
        """Submit the clarification response."""
        if self.reasoning_thread:
            self.reasoning_thread.set_clarification_response(response)
        
        if self.clarification_dialog:
            self.clarification_dialog.accept()
            self.clarification_dialog = None
    
    def _cancel_clarification(self):
        """Cancel the clarification request."""
        if self.reasoning_thread:
            self.reasoning_thread.set_clarification_response("No additional information provided.")
        
        if self.clarification_dialog:
            self.clarification_dialog.reject()
            self.clarification_dialog = None
    
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
        
        # Update meta-learning statistics
        self.update_meta_learning_stats()
        
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
            
            # Apply additional performance settings with significantly increased token limit
            neocortex.max_tokens = settings.get("max_tokens", 8000)
            
            # Apply other settings
            neocortex.reasoning_depth = settings.get("reasoning_depth", "balanced")
            neocortex.concurrent_requests = settings.get("concurrent_requests", True)
            
            # Transfer modules and other components from old instance
            neocortex.modules = self.neocortex.modules
            neocortex.ensemble_methods = self.neocortex.ensemble_methods
            neocortex.prompt_library = self.neocortex.prompt_library
            neocortex.meta_learning = self.neocortex.meta_learning
            
            # Update API client settings
            if hasattr(neocortex.api_client, 'response_cache') and not settings.get("response_caching", True):
                neocortex.api_client.response_cache = {}  # Clear cache if disabled
            
            self.neocortex = neocortex
            self.status_bar.showMessage(f"Model settings updated: {settings['model']}, temp={settings['temperature']}, tokens={settings.get('max_tokens', 8000)}")
        except Exception as e:
            QMessageBox.critical(self, "Settings Error", f"Failed to update settings: {str(e)}")
    
    def update_meta_learning_stats(self):
        """Update the meta-learning statistics display."""
        if hasattr(self, 'meta_learning_widget') and hasattr(self.neocortex, 'meta_learning'):
            self.meta_learning_widget.update_stats(self.neocortex.meta_learning)
            
            # Also update AB test results if available
            if hasattr(self.neocortex.meta_learning, 'ab_tests'):
                self.meta_learning_widget.update_ab_results(self.neocortex.meta_learning.ab_tests)
    
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
            self.realtime_reasoning_widget.clear()
    
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
                "timestamp": datetime.now().isoformat(),
                "meta": {
                    "model": self.neocortex.model_name,
                    "temperature": self.neocortex.temperature,
                    "max_tokens": self.neocortex.max_tokens,
                    "version": "3.0.0"  # Add version for compatibility checks
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
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
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Check version compatibility if available
            if "meta" in session_data and "version" in session_data["meta"]:
                version = session_data["meta"]["version"]
                if not version.startswith("3."):
                    QMessageBox.warning(self, "Version Mismatch", 
                                      f"The session was created with version {version}, which may not be fully compatible with the current version.")
            
            self.query_edit.setPlainText(session_data["query"])
            self.result_display_widget.display_result(session_data["result"])
            self.current_result = session_data["result"]
            
            # If available, update model settings to match the session
            if "meta" in session_data:
                if "model" in session_data["meta"]:
                    self.model_settings_widget.model_combo.setCurrentText(session_data["meta"]["model"])
                if "temperature" in session_data["meta"]:
                    temp = int(session_data["meta"]["temperature"] * 100)
                    self.model_settings_widget.temp_slider.setValue(temp)
                if "max_tokens" in session_data["meta"]:
                    self.model_settings_widget.token_slider.setValue(session_data["meta"]["max_tokens"])
            
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
        
        <h2>Key Features</h2>
        <ul>
            <li><strong>Specialized Reasoning Modules:</strong> Domain-specific modules for mathematics, ethics, logic, temporal reasoning, and more</li>
            <li><strong>Ensemble Methods:</strong> Multiple approaches to aggregate and synthesize outputs from different reasoning modules</li>
            <li><strong>Meta-Learning System:</strong> Tracks performance and adapts strategies based on success rates</li>
            <li><strong>Interactive Clarification:</strong> Can request additional information when queries are ambiguous</li>
            <li><strong>Advanced Prompt Engineering:</strong> Domain-specific prompt templates optimized for different problem types</li>
            <li><strong>Real-time Reasoning:</strong> Visualize the model's thinking process as it happens</li>
            <li><strong>Enhanced Cognitive Graph:</strong> Interactive visualization of the reasoning process</li>
            <li><strong>High Token Capacity:</strong> Generate up to 16,000 tokens for comprehensive solutions</li>
        </ul>
        
        <h2>Using the Interface</h2>
        <ol>
            <li><strong>Problem Input:</strong> Enter your question or problem in the text area at the top.</li>
            <li><strong>Specialized Modules:</strong> Enable or disable specific reasoning modules in the Specialized Modules tab.</li>
            <li><strong>Ensemble Methods:</strong> Select the approach for combining outputs from different modules.</li>
            <li><strong>Interactive Mode:</strong> Enable to allow NeoCortex to ask for clarification when needed.</li>
            <li><strong>Process:</strong> Click "Process with NeoCortex" to start the reasoning process.</li>
            <li><strong>Realtime Thoughts:</strong> Watch the model's reasoning unfold in real-time in the "Realtime Thoughts" tab.</li>
            <li><strong>Results:</strong> View the final answer, detailed reasoning, and module outputs in the tabs on the right.</li>
            <li><strong>Cognitive Graph:</strong> Explore the reasoning structure visually in the Cognitive Graph tab.</li>
            <li><strong>Meta-Learning:</strong> Review system performance statistics and improvement suggestions.</li>
        </ol>
        
        <h2>Tips for Best Results</h2>
        <ul>
            <li>For mathematical problems, ensure the Mathematics module is enabled</li>
            <li>For ethical questions, ensure the Ethics module is enabled</li>
            <li>For complex logical puzzles, use the Logic module</li>
            <li>For creative tasks, enable the Creativity module</li>
            <li>For time-based or causal reasoning, use the Temporal module</li>
            <li>Enable Interactive Mode for ambiguous queries to get clarification</li>
            <li>Use Diversity Promoting ensemble for problems that benefit from multiple perspectives</li>
            <li>Disable Fast Mode for more thorough analysis on complex problems</li>
            <li>Check the Meta-Learning tab for performance insights and improvement suggestions</li>
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
            "Metacognition": 0.0,
            "Mathematical": 0.0,
            "Ethical": 0.0,
            "Logical": 0.0,
            "Creative": 0.0,
            "Temporal": 0.0
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
                "Metacognition": 0.8,
                "Mathematical": 0.5,
                "Ethical": 0.7,
                "Logical": 0.9,
                "Creative": 0.4,
                "Temporal": 0.6
            }
            self.module_activity_widget.update_activity(self.simulated_activity)
            return
        
        # Simulate changing activity levels
        progress = self.progress_bar.value() / 100.0
        
        # Different modules become active at different stages
        if progress < 0.2:
            # Initial analysis and module selection phase
            self.simulated_activity["Concept Network"] = min(0.9, self.simulated_activity["Concept Network"] + 0.1)
            self.simulated_activity["Mathematical"] = min(0.5, self.simulated_activity["Mathematical"] + 0.1)
            self.simulated_activity["Logical"] = min(0.5, self.simulated_activity["Logical"] + 0.1)
            self.simulated_activity["Ethical"] = min(0.3, self.simulated_activity["Ethical"] + 0.05)
        elif progress < 0.4:
            # Module processing phase
            self.simulated_activity["Mathematical"] = min(0.9, self.simulated_activity["Mathematical"] + 0.1)
            self.simulated_activity["Logical"] = min(0.9, self.simulated_activity["Logical"] + 0.1)
            self.simulated_activity["Causal Reasoning"] = min(0.8, self.simulated_activity["Causal Reasoning"] + 0.1)
            self.simulated_activity["Temporal"] = min(0.6, self.simulated_activity["Temporal"] + 0.1)
        elif progress < 0.6:
            # Ensemble and integration phase
            self.simulated_activity["Creative"] = min(0.7, self.simulated_activity["Creative"] + 0.1)
            self.simulated_activity["Self-Regulation"] = min(0.8, self.simulated_activity["Self-Regulation"] + 0.1)
            self.simulated_activity["Ethical"] = min(0.7, self.simulated_activity["Ethical"] + 0.1)
        elif progress < 0.8:
            # Solution verification phase
            self.simulated_activity["Self-Regulation"] = min(0.9, self.simulated_activity["Self-Regulation"] + 0.05)
            self.simulated_activity["Logical"] = min(0.9, self.simulated_activity["Logical"] + 0.05)
            self.simulated_activity["Counterfactual"] = min(0.8, self.simulated_activity["Counterfactual"] + 0.1)
        else:
            # Metacognitive and reflection phase
            self.simulated_activity["Metacognition"] = min(0.9, self.simulated_activity["Metacognition"] + 0.1)
            self.simulated_activity["Concept Network"] = min(0.9, self.simulated_activity["Concept Network"] + 0.05)
        
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
