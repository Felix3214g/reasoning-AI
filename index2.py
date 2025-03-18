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
# Base Knowledge Representation Classes
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

# ------------------------------------------------------------------------------------------
# DYNAMIC PROMPT ENGINEERING CLASSES
# ------------------------------------------------------------------------------------------

class PromptTemplate:
    """Class to represent a dynamic prompt template."""
    
    def __init__(self, template, variables=None, name=None, description=None):
        """
        Initialize a prompt template.
        
        Args:
            template (str): The template string with placeholders
            variables (list, optional): List of required variables
            name (str, optional): Name of the template
            description (str, optional): Description of the template's purpose
        """
        self.template = template
        self.variables = variables or []
        self.name = name or "Unnamed Template"
        self.description = description or "No description"
    
    def format(self, **kwargs):
        """
        Format the template with provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template
            
        Returns:
            str: The formatted prompt
        """
        # Check for missing required variables
        missing = [var for var in self.variables if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {', '.join(missing)}")
        
        # Format the template
        return self.template.format(**kwargs)
    
    def to_dict(self):
        """Convert the template to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "variables": self.variables
        }

class PromptLibrary:
    """Library of prompt templates for different reasoning tasks."""
    
    def __init__(self):
        """Initialize the prompt library with default templates."""
        self.templates = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize the default set of prompt templates."""
        # Problem Analysis Templates
        self.add_template(
            "problem_analysis_standard",
            PromptTemplate(
                template="""
                Analyze the following problem:
                
                {query}
                
                Break it down into its core components and subproblems. Identify key concepts, constraints, and goals.
                """,
                variables=["query"],
                name="Standard Problem Analysis",
                description="General template for breaking down problems"
            )
        )
        
        self.add_template(
            "problem_analysis_detailed",
            PromptTemplate(
                template="""
                Perform a detailed analysis of the following problem:
                
                {query}
                
                1. Identify the domain(s) of knowledge this problem relates to.
                2. Break down the problem into its atomic components.
                3. Identify the key variables, parameters, and constraints.
                4. Determine the explicit and implicit goals.
                5. Identify any potential obstacles or challenges.
                6. Determine what approaches or methodologies might be applicable.
                
                Provide a structured analysis with clear categories.
                """,
                variables=["query"],
                name="Detailed Problem Analysis",
                description="Comprehensive template for complex problem decomposition"
            )
        )
        
        # Multiple Perspectives Templates
        self.add_template(
            "perspectives_standard",
            PromptTemplate(
                template="""
                Consider the following problem from multiple perspectives:
                
                {query}
                
                Provide at least 3 different perspectives or approaches to this problem.
                """,
                variables=["query"],
                name="Standard Multiple Perspectives",
                description="General template for generating multiple perspectives"
            )
        )
        
        self.add_template(
            "perspectives_contrasting",
            PromptTemplate(
                template="""
                Consider the following problem from contrasting theoretical frameworks:
                
                {query}
                
                Analyze this problem from the following contrasting perspectives:
                
                1. A conventional/traditional perspective
                2. A critical/alternative perspective
                3. A synthesis or middle-ground perspective
                
                For each perspective:
                - Explain the key assumptions and values
                - Outline how this perspective would approach the problem
                - Identify unique insights this perspective offers
                - Note potential blind spots or limitations
                
                Present these perspectives as distinct approaches that highlight different aspects of the problem.
                """,
                variables=["query"],
                name="Contrasting Perspectives",
                description="Template for generating explicitly contrasting perspectives"
            )
        )
        
        # Evidence Gathering Templates
        self.add_template(
            "evidence_standard",
            PromptTemplate(
                template="""
                Analyze the following aspect of the problem:
                
                {subproblem_desc}
                
                Provide factual evidence, logical analysis, and any computational evidence.
                """,
                variables=["subproblem_desc"],
                name="Standard Evidence Gathering",
                description="General template for gathering evidence"
            )
        )
        
        # Integration Templates
        self.add_template(
            "integration_standard",
            PromptTemplate(
                template="""
                Integrate the following perspectives and evidence to form a complete understanding:
                
                Problem: {query}
                
                Perspectives: {perspectives}
                
                Evidence: {evidence}
                
                Provide a coherent integration of these elements.
                """,
                variables=["query", "perspectives", "evidence"],
                name="Standard Integration",
                description="General template for integrating perspectives and evidence"
            )
        )
        
        # Solution Generation Templates
        self.add_template(
            "solution_standard",
            PromptTemplate(
                template="""
                Generate a solution for the following problem based on the integrated understanding:
                
                Problem: {query}
                
                Integrated understanding: {integration}
                
                Provide a comprehensive solution.
                """,
                variables=["query", "integration"],
                name="Standard Solution Generation",
                description="General template for generating solutions"
            )
        )
        
        # Verification Templates
        self.add_template(
            "verification_standard",
            PromptTemplate(
                template="""
                Critically evaluate the following solution to this problem:
                
                Problem: {query}
                
                Solution: {solution}
                
                Identify any weaknesses, errors, or limitations, and suggest improvements.
                """,
                variables=["query", "solution"],
                name="Standard Verification",
                description="General template for critical evaluation"
            )
        )
        
        # Final Answer Templates
        self.add_template(
            "final_answer_standard",
            PromptTemplate(
                template="""
                Refine the following solution based on the verification:
                
                Problem: {query}
                
                Solution: {solution}
                
                Verification: {verification}
                
                Provide a refined, final answer that addresses any issues identified in verification.
                """,
                variables=["query", "solution", "verification"],
                name="Standard Final Answer",
                description="General template for final answer refinement"
            )
        )
        
        # Metacognitive Reflection Templates
        self.add_template(
            "reflection_standard",
            PromptTemplate(
                template="""
                Reflect on the reasoning process used to solve the following problem:
                
                Problem: {query}
                
                Consider the effectiveness of the reasoning strategy, potential biases, 
                alternative approaches, and lessons learned for future problems.
                """,
                variables=["query"],
                name="Standard Metacognitive Reflection",
                description="General template for metacognitive reflection"
            )
        )
    
    def add_template(self, template_id, template):
        """
        Add a template to the library.
        
        Args:
            template_id (str): Unique identifier for the template
            template (PromptTemplate): The template to add
        """
        self.templates[template_id] = template
    
    def get_template(self, template_id):
        """
        Get a template by its ID.
        
        Args:
            template_id (str): The template identifier
            
        Returns:
            PromptTemplate: The requested template
            
        Raises:
            KeyError: If template_id is not in the library
        """
        if template_id not in self.templates:
            raise KeyError(f"Template '{template_id}' not found in library")
        
        return self.templates[template_id]
    
    def get_template_ids(self, category=None):
        """
        Get all template IDs, optionally filtered by category.
        
        Args:
            category (str, optional): Category to filter by (e.g., 'solution', 'verification')
            
        Returns:
            list: List of template IDs
        """
        if category:
            return [tid for tid in self.templates if tid.startswith(f"{category}_")]
        
        return list(self.templates.keys())
    
    def export_templates(self):
        """
        Export all templates as a dictionary.
        
        Returns:
            dict: Dictionary of templates
        """
        return {tid: template.to_dict() for tid, template in self.templates.items()}
    
    def import_templates(self, templates_dict):
        """
        Import templates from a dictionary.
        
        Args:
            templates_dict (dict): Dictionary of template data
        """
        for template_id, template_data in templates_dict.items():
            self.templates[template_id] = PromptTemplate(
                template=template_data["template"],
                variables=template_data["variables"],
                name=template_data["name"],
                description=template_data["description"]
            )

class DynamicPromptManager:
    """Manager for dynamically selecting and adapting prompts based on context."""
    
    def __init__(self, neocortex):
        """
        Initialize the dynamic prompt manager.
        
        Args:
            neocortex: Reference to the NeoCortex instance
        """
        self.neocortex = neocortex
        self.prompt_library = PromptLibrary()
        self.prompt_selection_history = []
        self.query_templates_cache = {}
    
    def select_template(self, category, query, context=None):
        """
        Select the most appropriate template for a given category and query.
        
        Args:
            category (str): Template category (e.g., 'problem_analysis', 'solution')
            query (str): The query being processed
            context (dict, optional): Additional context for template selection
            
        Returns:
            str: ID of the selected template
        """
        context = context or {}
        
        # Simplified template selection based on reasoning depth
        reasoning_depth = self.neocortex.reasoning_depth
        
        # Cache key for efficiency
        cache_key = f"{category}_{reasoning_depth}_{hash(query)}"
        if cache_key in self.query_templates_cache:
            return self.query_templates_cache[cache_key]
        
        # Get available templates for this category
        available_templates = self.prompt_library.get_template_ids(category)
        
        if not available_templates:
            raise ValueError(f"No templates available for category '{category}'")
        
        # Select based on reasoning depth
        if reasoning_depth == "thorough":
            # Prefer detailed/comprehensive templates
            for template_id in available_templates:
                if any(suffix in template_id for suffix in ["detailed", "comprehensive", "rigorous", "structured"]):
                    self.query_templates_cache[cache_key] = template_id
                    return template_id
        
        elif reasoning_depth == "minimal":
            # Prefer standard/simple templates
            for template_id in available_templates:
                if "standard" in template_id:
                    self.query_templates_cache[cache_key] = template_id
                    return template_id
        
        # Default: choose the first available template
        template_id = available_templates[0]
        self.query_templates_cache[cache_key] = template_id
        return template_id
    
    def format_prompt(self, category, query, **kwargs):
        """
        Format a prompt for a given category and query.
        
        Args:
            category (str): Template category
            query (str): The query being processed
            **kwargs: Additional variables for template formatting
            
        Returns:
            str: The formatted prompt
        """
        # Always include the query in kwargs
        kwargs["query"] = query
        
        # Select the appropriate template
        template_id = self.select_template(category, query, kwargs.get("context"))
        
        # Get the template
        template = self.prompt_library.get_template(template_id)
        
        # Record the selection for analysis
        self.prompt_selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "template_id": template_id,
            "query_hash": hash(query)
        })
        
        # Format the template
        try:
            return template.format(**kwargs)
        except KeyError as e:
            # Fallback to standard template if missing variables
            print(f"Error formatting template {template_id}: {e}")
            
            # Try to get a standard template
            standard_id = f"{category}_standard"
            if standard_id in self.prompt_library.templates:
                template = self.prompt_library.get_template(standard_id)
                return template.format(**kwargs)
            else:
                raise ValueError(f"Failed to format prompt for {category} and no standard fallback available")
    
    def analyze_template_effectiveness(self):
        """
        Analyze the effectiveness of different templates based on usage history.
        
        Returns:
            dict: Analysis of template effectiveness
        """
        # This would be enhanced with feedback from meta-learning
        template_usage = {}
        
        for entry in self.prompt_selection_history:
            template_id = entry["template_id"]
            if template_id not in template_usage:
                template_usage[template_id] = 0
            
            template_usage[template_id] += 1
        
        return {
            "template_usage": template_usage,
            "total_selections": len(self.prompt_selection_history)
        }

# ------------------------------------------------------------------------------------------
# SPECIALIZED REASONING MODULES
# ------------------------------------------------------------------------------------------

class ReasoningModule:
    """Base class for specialized reasoning modules."""
    
    def __init__(self, neocortex):
        self.neocortex = neocortex
        self.name = "Base Module"
        self.description = "Base reasoning module"
        self.active = True
    
    def is_applicable(self, query):
        """Determine if this module is applicable to the given query."""
        return False
    
    def process(self, query, context=None):
        """Process the query using this module's specialized reasoning."""
        raise NotImplementedError("Each reasoning module must implement process()")
    
    def get_prompt_template(self):
        """Get the specialized prompt template for this module."""
        return ""
    
    def is_active(self):
        """Check if this module is active."""
        return self.active

    def activate(self):
        """Activate this module."""
        self.active = True

    def deactivate(self):
        """Deactivate this module."""
        self.active = False

class MathReasoningModule(ReasoningModule):
    """Module specialized for mathematical reasoning and problem-solving."""
    
    def __init__(self, neocortex):
        super().__init__(neocortex)
        self.name = "Mathematical Reasoning"
        self.description = "Specialized for mathematical problems, equations, and proofs"
        
        # Math-specific keywords for detection
        self.math_keywords = [
            "calculate", "compute", "solve", "equation", "formula", "theorem",
            "proof", "mathematical", "algebra", "geometry", "calculus",
            "arithmetic", "trigonometry", "probability", "statistics"
        ]
        
        # Math symbols for detection
        self.math_symbols = [
            "+", "-", "*", "/", "=", "<", ">", "≤", "≥", "≠", "≈",
            "∑", "∏", "∫", "∂", "√", "∞", "π", "θ", "°", "^"
        ]
    
    def is_applicable(self, query):
        """Check if the query appears to be a mathematical problem."""
        # Check for presence of math keywords
        if any(keyword in query.lower() for keyword in self.math_keywords):
            return True
        
        # Check for presence of math symbols
        if any(symbol in query for symbol in self.math_symbols):
            return True
        
        # Check for presence of numbers with operators
        import re
        if re.search(r'\d+\s*[\+\-\*/\^]\s*\d+', query):
            return True
            
        return False
    
    def process(self, query, context=None):
        """Process a mathematical query with specialized reasoning."""
        prompt = self.get_prompt_template().format(query=query)
        
        try:
            response = self.neocortex._generate_response(prompt)
            
            # Add a math reasoning node to the cognitive graph
            math_node = CognitiveNode(
                node_id=f"math_reasoning_{int(time.time())}",
                node_type="specialized_reasoning",
                content=f"Mathematical reasoning: {response[:100]}...",
                parent_id=context.get("parent_id") if context else None
            )
            self.neocortex.cognitive_graph.add_node(math_node)
            
            return {
                "module": self.name,
                "result": response,
                "confidence": 0.9 if self.is_high_confidence(query) else 0.7,
                "node_id": math_node.node_id
            }
        except Exception as e:
            print(f"Error in mathematical reasoning module: {str(e)}")
            return {
                "module": self.name,
                "result": f"Error in mathematical processing: {str(e)}",
                "confidence": 0.0,
                "node_id": None
            }
    
    def is_high_confidence(self, query):
        """Determine if this is a high-confidence math problem."""
        # More symbolic math content indicates higher confidence
        symbol_count = sum(query.count(symbol) for symbol in self.math_symbols)
        return symbol_count >= 3
    
    def get_prompt_template(self):
        """Get the specialized math reasoning prompt template."""
        return """
        You are a mathematical reasoning expert. Solve the following problem step-by-step:
        
        {query}
        
        First, identify the type of mathematical problem.
        Second, determine the appropriate approach or formula.
        Third, work through the solution showing each step clearly.
        Finally, verify your answer by checking it against the original problem.
        
        Ensure all steps are explicitly shown and explained.
        """

class EthicalReasoningModule(ReasoningModule):
    """Module specialized for ethical reasoning and analysis."""
    
    def __init__(self, neocortex):
        super().__init__(neocortex)
        self.name = "Ethical Reasoning"
        self.description = "Specialized for ethical dilemmas and moral questions"
        
        # Ethics-specific keywords for detection
        self.ethics_keywords = [
            "ethical", "moral", "right", "wrong", "good", "bad", "justice",
            "fairness", "rights", "responsibility", "duty", "virtue", "vice",
            "consequentialism", "deontology", "utilitarianism", "ought",
            "should", "permissible", "forbidden", "allowed", "dilemma"
        ]
    
    def is_applicable(self, query):
        """Check if the query appears to be an ethical question."""
        query_lower = query.lower()
        # Check for presence of ethics keywords
        if any(keyword in query_lower for keyword in self.ethics_keywords):
            return True
        
        # Check for ethical question patterns
        if "is it ethical" in query_lower or "is it moral" in query_lower:
            return True
        if "should i" in query_lower and any(term in query_lower for term in ["right", "wrong", "moral", "ethical"]):
            return True
            
        return False
    
    def process(self, query, context=None):
        """Process an ethical query with specialized reasoning."""
        prompt = self.get_prompt_template().format(query=query)
        
        try:
            response = self.neocortex._generate_response(prompt)
            
            # Add an ethics reasoning node to the cognitive graph
            ethics_node = CognitiveNode(
                node_id=f"ethics_reasoning_{int(time.time())}",
                node_type="specialized_reasoning",
                content=f"Ethical reasoning: {response[:100]}...",
                parent_id=context.get("parent_id") if context else None
            )
            self.neocortex.cognitive_graph.add_node(ethics_node)
            
            return {
                "module": self.name,
                "result": response,
                "confidence": 0.85,
                "node_id": ethics_node.node_id
            }
        except Exception as e:
            print(f"Error in ethical reasoning module: {str(e)}")
            return {
                "module": self.name,
                "result": f"Error in ethical analysis: {str(e)}",
                "confidence": 0.0,
                "node_id": None
            }
    
    def get_prompt_template(self):
        """Get the specialized ethical reasoning prompt template."""
        return """
        You are an ethical reasoning expert. Analyze the following ethical question or dilemma:
        
        {query}
        
        Approach this from multiple ethical frameworks:
        1. Consequentialism/Utilitarianism: Analyze based on outcomes and greatest good
        2. Deontology/Kant: Analyze based on duties, rights, and universal principles
        3. Virtue Ethics: Analyze based on character and virtues
        4. Care Ethics: Analyze based on relationships and responsibilities
        
        For each framework:
        - Explain key considerations
        - Analyze potential positions
        - Identify strengths and weaknesses of each position
        
        Finally, provide a balanced perspective that acknowledges complexity and context.
        """

class ScientificReasoningModule(ReasoningModule):
    """Module specialized for scientific reasoning and analysis."""
    
    def __init__(self, neocortex):
        super().__init__(neocortex)
        self.name = "Scientific Reasoning"
        self.description = "Specialized for scientific questions and analysis"
        
        # Science-specific keywords for detection
        self.science_keywords = [
            "scientific", "science", "experiment", "hypothesis", "theory",
            "evidence", "data", "observation", "empirical", "research",
            "study", "physics", "chemistry", "biology", "neuroscience",
            "psychology", "sociology", "geology", "astronomy", "experiment",
            "methodology", "laboratory", "measurement", "variables"
        ]
    
    def is_applicable(self, query):
        """Check if the query appears to be a scientific question."""
        query_lower = query.lower()
        # Check for presence of science keywords
        if any(keyword in query_lower for keyword in self.science_keywords):
            return True
        
        # Check for scientific question patterns
        if query_lower.startswith("why does") or query_lower.startswith("how does"):
            return True
        if "explain" in query_lower and any(term in query_lower for term in ["phenomenon", "effect", "process"]):
            return True
            
        return False
    
    def process(self, query, context=None):
        """Process a scientific query with specialized reasoning."""
        prompt = self.get_prompt_template().format(query=query)
        
        try:
            response = self.neocortex._generate_response(prompt)
            
            # Add a science reasoning node to the cognitive graph
            science_node = CognitiveNode(
                node_id=f"science_reasoning_{int(time.time())}",
                node_type="specialized_reasoning",
                content=f"Scientific reasoning: {response[:100]}...",
                parent_id=context.get("parent_id") if context else None
            )
            self.neocortex.cognitive_graph.add_node(science_node)
            
            return {
                "module": self.name,
                "result": response,
                "confidence": 0.85,
                "node_id": science_node.node_id
            }
        except Exception as e:
            print(f"Error in scientific reasoning module: {str(e)}")
            return {
                "module": self.name,
                "result": f"Error in scientific analysis: {str(e)}",
                "confidence": 0.0,
                "node_id": None
            }
    
    def get_prompt_template(self):
        """Get the specialized scientific reasoning prompt template."""
        return """
        You are a scientific reasoning expert. Analyze the following scientific question:
        
        {query}
        
        Apply the scientific method:
        1. Define the question precisely
        2. Gather background information and relevant theories
        3. Form hypotheses that could explain the phenomenon
        4. Consider what evidence would support or refute these hypotheses
        5. Analyze existing research and evidence
        6. Draw tentative conclusions based on the best available evidence
        7. Identify limitations and areas for further investigation
        
        Ensure you're distinguishing between:
        - Established scientific consensus
        - Emerging research and findings
        - Areas of ongoing debate
        - Speculation or hypotheses
        
        Use scientific terminology precisely and explain complex concepts clearly.
        """

# ------------------------------------------------------------------------------------------
# ADVANCED REASONING TYPES
# ------------------------------------------------------------------------------------------

class TemporalReasoning:
    """Module for reasoning about time, sequences, causality over time, and changes."""
    
    def __init__(self, neocortex):
        self.neocortex = neocortex
        self.name = "Temporal Reasoning"
    
    def is_applicable(self, query):
        """Determine if temporal reasoning is applicable to this query."""
        # Check for temporal keywords and patterns
        temporal_indicators = [
            "timeline", "sequence", "before", "after", "during", "while",
            "since", "until", "when", "history", "evolution", "development",
            "stages", "steps", "process", "change over time", "trend",
            "future", "past", "present", "predict", "forecast", "retrospective"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in temporal_indicators)
    
    def analyze_temporal_patterns(self, query, context=None):
        """Analyze temporal patterns in the query."""
        prompt = f"""
        Analyze the temporal aspects of the following:
        
        {query}
        
        Identify:
        1. All temporal entities (events, periods, moments)
        2. Their relative ordering and relationships
        3. Any causal relationships across time
        4. Temporal patterns or cycles
        5. Rate of change or development
        
        Organize these into a coherent timeline or temporal framework.
        """
        
        response = self.neocortex._generate_response(prompt)
        
        # Create temporal reasoning node in cognitive graph
        node_id = f"temporal_reasoning_{int(time.time())}"
        temporal_node = CognitiveNode(
            node_id=node_id,
            node_type="temporal_reasoning",
            content=f"Temporal analysis: {response[:100]}...",
            parent_id=context.get("parent_id") if context else None
        )
        self.neocortex.cognitive_graph.add_node(temporal_node)
        
        return {
            "analysis": response,
            "node_id": node_id
        }

class CounterfactualReasoning:
    """Module for reasoning about hypothetical scenarios and alternatives."""
    
    def __init__(self, neocortex):
        self.neocortex = neocortex
        self.name = "Counterfactual Reasoning"
    
    def is_applicable(self, query):
        """Determine if counterfactual reasoning is applicable to this query."""
        # Check for counterfactual indicators
        counterfactual_indicators = [
            "what if", "if only", "had", "would have", "could have", "might have",
            "imagine if", "suppose", "alternative", "scenario", "possibility",
            "hypothetical", "counterfactual", "different outcome", "instead of"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in counterfactual_indicators)
    
    def analyze_counterfactual(self, query, context=None):
        """Analyze counterfactual scenarios for a query."""
        # First, identify key facts that could be changed
        facts_prompt = f"""
        For the following situation:
        
        {query}
        
        Identify 3 key facts or conditions that, if changed, would lead to significantly different outcomes.
        List each one briefly with a short explanation of why it's pivotal.
        """
        
        facts_response = self.neocortex._generate_response(facts_prompt)
        
        # Generate counterfactual for the most significant fact
        counterfactual_prompt = f"""
        Consider the following counterfactual scenario:
        
        Original situation: {query}
        
        What if these aspects were different: {facts_response}
        
        Generate three counterfactual scenarios with different outcomes:
        1. A modest change with subtle differences
        2. A significant change with moderate differences
        3. A dramatic change with profound differences
        
        For each scenario:
        - Describe the altered initial conditions
        - Trace the causal chain of how events would unfold differently
        - Explain the resulting alternative outcome
        - Identify key insights this counterfactual reveals about the actual situation
        """
        
        response = self.neocortex._generate_response(counterfactual_prompt)
        
        # Create counterfactual reasoning node in cognitive graph
        node_id = f"counterfactual_reasoning_{int(time.time())}"
        counterfactual_node = CognitiveNode(
            node_id=node_id,
            node_type="counterfactual_reasoning",
            content=f"Counterfactual analysis: {response[:100]}...",
            parent_id=context.get("parent_id") if context else None
        )
        self.neocortex.cognitive_graph.add_node(counterfactual_node)
        
        return {
            "key_facts": facts_response,
            "counterfactuals": response,
            "node_id": node_id
        }

class VisualSpatialReasoning:
    """Module for reasoning about visual and spatial concepts."""
    
    def __init__(self, neocortex):
        self.neocortex = neocortex
        self.name = "Visual-Spatial Reasoning"
    
    def is_applicable(self, query):
        """Determine if visual-spatial reasoning is applicable to this query."""
        # Check for visual-spatial indicators
        spatial_indicators = [
            "visual", "spatial", "diagram", "layout", "arrangement", "position",
            "orientation", "direction", "distance", "shape", "size", "pattern",
            "map", "graph", "chart", "visualization", "image", "picture",
            "perspective", "view", "angle", "rotation", "transformation"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in spatial_indicators)
    
    def describe_spatial_relationships(self, query, context=None):
        """Describe spatial relationships in a query."""
        prompt = f"""
        Analyze the visual-spatial aspects of the following:
        
        {query}
        
        Describe:
        1. The key spatial elements and their properties
        2. Spatial relationships between elements
        3. Any spatial patterns or structures
        4. How spatial arrangement contributes to meaning or function
        5. A clear mental model that would help visualize this
        
        If appropriate, describe how this could be visualized in a diagram.
        """
        
        response = self.neocortex._generate_response(prompt)
        
        # Create spatial reasoning node in cognitive graph
        node_id = f"spatial_reasoning_{int(time.time())}"
        spatial_node = CognitiveNode(
            node_id=node_id,
            node_type="spatial_reasoning",
            content=f"Spatial analysis: {response[:100]}...",
            parent_id=context.get("parent_id") if context else None
        )
        self.neocortex.cognitive_graph.add_node(spatial_node)
        
        return {
            "analysis": response,
            "node_id": node_id
        }

class SystemsReasoning:
    """Module for reasoning about complex systems and their interactions."""
    
    def __init__(self, neocortex):
        self.neocortex = neocortex
        self.name = "Systems Reasoning"
    
    def is_applicable(self, query):
        """Determine if systems reasoning is applicable to this query."""
        # Check for systems-related indicators
        systems_indicators = [
            "system", "network", "complex", "interaction", "feedback", "loop",
            "emergence", "component", "structure", "function", "dynamics",
            "equilibrium", "stability", "change", "adaptation", "resilience",
            "organization", "process", "flow", "cycle", "holistic", "ecosystem"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in systems_indicators)
    
    def analyze_system(self, query, context=None):
        """Analyze a system described in the query."""
        prompt = f"""
        Conduct a systems analysis of the following:
        
        {query}
        
        Analyze these aspects of the system:
        
        1. Components: Identify key elements and their properties
        2. Structure: How components are organized and connected
        3. Processes: What flows, changes, or activities occur within the system
        4. Boundaries: What defines the system and distinguishes it from environment
        5. Inputs/Outputs: What enters and exits the system
        6. Feedback loops: How the system regulates and adapts itself
        7. Emergent properties: What characteristics arise from the system as a whole
        8. System dynamics: How the system changes over time
        
        Create a coherent systems-level understanding that captures complexity.
        """
        
        response = self.neocortex._generate_response(prompt)
        
        # Create systems reasoning node in cognitive graph
        node_id = f"systems_reasoning_{int(time.time())}"
        systems_node = CognitiveNode(
            node_id=node_id,
            node_type="systems_reasoning",
            content=f"Systems analysis: {response[:100]}...",
            parent_id=context.get("parent_id") if context else None
        )
        self.neocortex.cognitive_graph.add_node(systems_node)
        
        return {
            "analysis": response,
            "node_id": node_id
        }

# ------------------------------------------------------------------------------------------
# META-LEARNING CLASSES
# ------------------------------------------------------------------------------------------

class ReasoningHistory:
    """Class for storing and analyzing reasoning history."""
    
    def __init__(self, max_history=100):
        """Initialize the reasoning history tracker."""
        self.history = []
        self.max_history = max_history
        self.success_patterns = {}
        self.failure_patterns = {}
        self.strategy_effectiveness = {}
    
    def add_entry(self, query, reasoning_process, final_answer, feedback=None, success=None):
        """
        Add a new entry to the reasoning history.
        
        Args:
            query (str): The original query
            reasoning_process (dict): The full reasoning process data
            final_answer (str): The final answer provided
            feedback (str, optional): User feedback if available
            success (bool, optional): Whether the reasoning was successful
        """
        timestamp = datetime.now().isoformat()
        
        # Create a simplified representation of the reasoning process
        reasoning_summary = {
            "steps": [],
            "modules_used": [],
            "strategy": reasoning_process.get("strategy", "standard")
        }
        
        # Extract steps and modules from reasoning process
        if "reasoning_process" in reasoning_process:
            rp = reasoning_process["reasoning_process"]
            
            if "decomposition" in rp:
                reasoning_summary["steps"].append("decomposition")
            
            if "perspectives" in rp:
                reasoning_summary["steps"].append("perspectives")
                reasoning_summary["perspective_count"] = len(rp["perspectives"])
            
            if "evidence" in rp:
                reasoning_summary["steps"].append("evidence")
                reasoning_summary["evidence_count"] = len(rp["evidence"])
            
            if "specialized_modules" in rp:
                for module in rp["specialized_modules"]:
                    reasoning_summary["modules_used"].append(module["name"])
        
        # Create the history entry
        entry = {
            "id": len(self.history),
            "timestamp": timestamp,
            "query": query,
            "query_type": self._classify_query_type(query),
            "query_complexity": self._estimate_complexity(query),
            "reasoning_summary": reasoning_summary,
            "final_answer": final_answer,
            "answer_length": len(final_answer),
            "feedback": feedback,
            "success": success
        }
        
        # Add to history, maintaining max size
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Update patterns if success is known
        if success is not None:
            self._update_patterns(entry, success)
    
    def _classify_query_type(self, query):
        """Classify the type of query based on content analysis."""
        query_lower = query.lower()
        
        # Simple classification based on keywords and patterns
        if any(kw in query_lower for kw in ["math", "calculate", "compute", "solve"]):
            return "mathematical"
        elif any(kw in query_lower for kw in ["ethical", "moral", "right", "wrong"]):
            return "ethical"
        elif any(kw in query_lower for kw in ["science", "scientific", "experiment", "theory"]):
            return "scientific"
        elif any(kw in query_lower for kw in ["how", "explain", "describe", "what is"]):
            return "informational"
        elif any(kw in query_lower for kw in ["compare", "contrast", "difference", "similarity"]):
            return "comparative"
        elif any(kw in query_lower for kw in ["why", "reason", "cause", "effect"]):
            return "causal"
        else:
            return "general"
    
    def _estimate_complexity(self, query):
        """Estimate the complexity of a query on a scale of 1-10."""
        # Simplistic complexity estimation based on length and structure
        complexity = 1
        
        # Length-based complexity
        words = query.split()
        if len(words) > 50:
            complexity += 3
        elif len(words) > 20:
            complexity += 2
        elif len(words) > 10:
            complexity += 1
        
        # Structure-based complexity
        if "?" in query:
            question_count = query.count("?")
            complexity += min(question_count, 3)
        
        # Keyword-based complexity
        complexity_keywords = ["complex", "detailed", "thorough", "comprehensive", 
                              "analyze", "synthesize", "evaluate", "compare"]
        for keyword in complexity_keywords:
            if keyword in query.lower():
                complexity += 1
        
        return min(10, complexity)
    
    def _update_patterns(self, entry, success):
        """Update success and failure patterns based on a new entry."""
        query_type = entry["query_type"]
        strategy = entry["reasoning_summary"]["strategy"]
        modules_used = entry["reasoning_summary"].get("modules_used", [])
        
        # Update strategy effectiveness
        key = f"{query_type}_{strategy}"
        if key not in self.strategy_effectiveness:
            self.strategy_effectiveness[key] = {"success": 0, "failure": 0, "total": 0}
        
        self.strategy_effectiveness[key]["total"] += 1
        if success:
            self.strategy_effectiveness[key]["success"] += 1
        else:
            self.strategy_effectiveness[key]["failure"] += 1
        
        # Update module effectiveness if modules were used
        for module in modules_used:
            module_key = f"{query_type}_{module}"
            if module_key not in self.strategy_effectiveness:
                self.strategy_effectiveness[module_key] = {"success": 0, "failure": 0, "total": 0}
            
            self.strategy_effectiveness[module_key]["total"] += 1
            if success:
                self.strategy_effectiveness[module_key]["success"] += 1
            else:
                self.strategy_effectiveness[module_key]["failure"] += 1
    
    def get_recommended_strategy(self, query):
        """Get the recommended reasoning strategy for a given query."""
        query_type = self._classify_query_type(query)
        complexity = self._estimate_complexity(query)
        
        # Get all strategies used for this query type
        relevant_strategies = {}
        for key, stats in self.strategy_effectiveness.items():
            if key.startswith(f"{query_type}_"):
                strategy = key.split("_")[1]
                if stats["total"] > 0:
                    success_rate = stats["success"] / stats["total"]
                    relevant_strategies[strategy] = (success_rate, stats["total"])
        
        # If we have relevant strategies with enough data, use the best one
        if relevant_strategies:
            # Sort by success rate, but only consider strategies with enough data
            min_samples = 3
            good_strategies = [(s, rate, count) for s, (rate, count) in relevant_strategies.items() if count >= min_samples]
            
            if good_strategies:
                # Sort by success rate
                good_strategies.sort(key=lambda x: x[1], reverse=True)
                return good_strategies[0][0]
        
        # Default strategies based on complexity if no history
        if complexity >= 7:
            return "thorough"
        elif complexity >= 4:
            return "balanced"
        else:
            return "minimal"
    
    def get_recommended_modules(self, query):
        """Get recommended modules for a given query based on past performance."""
        query_type = self._classify_query_type(query)
        
        # Get module effectiveness for this query type
        module_effectiveness = {}
        for key, stats in self.strategy_effectiveness.items():
            if key.startswith(f"{query_type}_") and stats["total"] >= 3:
                parts = key.split("_")
                if len(parts) > 1:
                    module = parts[1]
                    success_rate = stats["success"] / stats["total"]
                    module_effectiveness[module] = success_rate
        
        # Return modules with success rate above threshold
        threshold = 0.6
        recommended_modules = [module for module, rate in module_effectiveness.items() if rate >= threshold]
        
        return recommended_modules
    
    def get_performance_summary(self):
        """Get a summary of reasoning performance across different query types."""
        summary = {
            "overall": {"success": 0, "failure": 0, "total": 0},
            "by_query_type": {},
            "by_strategy": {},
            "by_module": {}
        }
        
        # Count entries with success feedback
        valid_entries = [entry for entry in self.history if entry.get("success") is not None]
        
        for entry in valid_entries:
            # Update overall stats
            summary["overall"]["total"] += 1
            if entry["success"]:
                summary["overall"]["success"] += 1
            else:
                summary["overall"]["failure"] += 1
            
            # Update query type stats
            query_type = entry["query_type"]
            if query_type not in summary["by_query_type"]:
                summary["by_query_type"][query_type] = {"success": 0, "failure": 0, "total": 0}
            
            summary["by_query_type"][query_type]["total"] += 1
            if entry["success"]:
                summary["by_query_type"][query_type]["success"] += 1
            else:
                summary["by_query_type"][query_type]["failure"] += 1
            
            # Update strategy stats
            strategy = entry["reasoning_summary"]["strategy"]
            if strategy not in summary["by_strategy"]:
                summary["by_strategy"][strategy] = {"success": 0, "failure": 0, "total": 0}
            
            summary["by_strategy"][strategy]["total"] += 1
            if entry["success"]:
                summary["by_strategy"][strategy]["success"] += 1
            else:
                summary["by_strategy"][strategy]["failure"] += 1
            
            # Update module stats
            for module in entry["reasoning_summary"].get("modules_used", []):
                if module not in summary["by_module"]:
                    summary["by_module"][module] = {"success": 0, "failure": 0, "total": 0}
                
                summary["by_module"][module]["total"] += 1
                if entry["success"]:
                    summary["by_module"][module]["success"] += 1
                else:
                    summary["by_module"][module]["failure"] += 1
        
        # Calculate success rates
        if summary["overall"]["total"] > 0:
            summary["overall"]["success_rate"] = summary["overall"]["success"] / summary["overall"]["total"]
        
        for category in ["by_query_type", "by_strategy", "by_module"]:
            for key, stats in summary[category].items():
                if stats["total"] > 0:
                    stats["success_rate"] = stats["success"] / stats["total"]
        
        return summary

# ------------------------------------------------------------------------------------------
# INTERACTIVE LEARNING CLASSES
# ------------------------------------------------------------------------------------------

class AmbiguityDetector:
    """Detects ambiguities in queries that may require clarification."""
    
    def __init__(self):
        # Ambiguity indicators - words and phrases that suggest ambiguity
        self.ambiguity_indicators = [
            "it", "they", "them", "those", "these", "that", "this",
            "something", "someone", "somewhere", "anywhere", "anything",
            "maybe", "perhaps", "possibly", "either", "or", "alternatively",
            "unclear", "ambiguous", "vague", "uncertain"
        ]
        
        # Question starters that often lead to broad queries
        self.broad_question_starters = [
            "what is", "how do", "why is", "can you", "could you", 
            "would you", "explain", "describe", "tell me about"
        ]
        
        # Context-dependent terms
        self.context_dependent_terms = [
            "here", "there", "now", "then", "today", "tomorrow", "yesterday",
            "previous", "next", "before", "after", "local", "current"
        ]
    
    def detect_ambiguities(self, query):
        """
        Detect potential ambiguities in a query.
        
        Args:
            query (str): The query to analyze
            
        Returns:
            dict: Information about detected ambiguities
        """
        ambiguities = []
        query_lower = query.lower()
        
        # Check for pronoun references without clear antecedents
        for indicator in self.ambiguity_indicators:
            if f" {indicator} " in f" {query_lower} ":
                ambiguities.append({
                    "type": "reference_ambiguity",
                    "term": indicator,
                    "description": f"The term '{indicator}' may refer to multiple things."
                })
        
        # Check for very short queries (likely too vague)
        if len(query.split()) < 4:
            ambiguities.append({
                "type": "brevity",
                "term": query,
                "description": "The query is very brief and may lack necessary context."
            })
        
        # Check for very broad questions
        for starter in self.broad_question_starters:
            if query_lower.startswith(starter):
                words = query.split()
                if len(words) < 6:  # Short broad questions are often ambiguous
                    ambiguities.append({
                        "type": "broad_question",
                        "term": starter,
                        "description": f"The question may be too broad with '{starter}'."
                    })
        
        # Check for context-dependent terms
        for term in self.context_dependent_terms:
            if f" {term} " in f" {query_lower} ":
                ambiguities.append({
                    "type": "context_dependency",
                    "term": term,
                    "description": f"The term '{term}' depends on context that may be missing."
                })
        
        # Limit to most relevant ambiguities
        unique_ambiguities = []
        ambiguity_types = set()
        for ambiguity in ambiguities:
            if ambiguity["type"] not in ambiguity_types:
                unique_ambiguities.append(ambiguity)
                ambiguity_types.add(ambiguity["type"])
        
        return {
            "has_ambiguities": len(unique_ambiguities) > 0,
            "ambiguities": unique_ambiguities[:3],  # Limit to top 3 most relevant
            "ambiguity_score": min(1.0, len(unique_ambiguities) / 3)  # Scale 0-1
        }

class ClarificationGenerator:
    """Generates clarification questions for ambiguous queries."""
    
    def __init__(self, neocortex):
        self.neocortex = neocortex
    
    def generate_clarifications(self, query, ambiguities):
        """
        Generate clarification questions based on detected ambiguities.
        
        Args:
            query (str): The original query
            ambiguities (dict): Ambiguity information from AmbiguityDetector
            
        Returns:
            list: List of clarification questions
        """
        if not ambiguities["has_ambiguities"]:
            return []
        
        questions = []
        
        # Generate targeted questions based on ambiguity types
        for ambiguity in ambiguities["ambiguities"]:
            ambiguity_type = ambiguity["type"]
            term = ambiguity["term"]
            
            if ambiguity_type == "reference_ambiguity":
                questions.append(f"What specifically does '{term}' refer to in your question?")
            
            elif ambiguity_type == "brevity":
                questions.append("Could you provide more details about what you're looking for?")
            
            elif ambiguity_type == "broad_question":
                questions.append(f"Your question about '{term}' is quite broad. Could you specify a particular aspect you're interested in?")
            
            elif ambiguity_type == "context_dependency":
                questions.append(f"When you mention '{term}', what specific context or timeframe are you referring to?")
        
        # If we have simple clarifications, just use those
        if questions:
            return questions
        
        # For complex cases, use the LLM to generate better clarification questions
        if ambiguities["ambiguity_score"] > 0.5:
            prompt = f"""
            The following query may be ambiguous or unclear:
            
            "{query}"
            
            Generate 1-2 concise clarification questions that would help resolve the ambiguity.
            Each question should be specific and directly address a potential source of confusion.
            Format the questions as a numbered list.
            """
            
            try:
                response = self.neocortex._generate_response(prompt)
                
                # Extract questions from the response (assuming numbered list format)
                import re
                llm_questions = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', response, re.DOTALL)
                
                # Clean up questions
                llm_questions = [q.strip() for q in llm_questions if q.strip()]
                
                if llm_questions:
                    return llm_questions
            except Exception as e:
                print(f"Error generating clarification questions: {str(e)}")
        
        # Fallback generic question
        return ["Could you please clarify or provide more details about your question?"]

# ------------------------------------------------------------------------------------------
# MAIN NEOCORTEX CLASS
# ------------------------------------------------------------------------------------------

class NeoCortex:
    """Enhanced cognitive architecture with integrated reasoning capabilities."""
    
    def __init__(self, model_name="deepseek/deepseek-chat-v3", temperature=0.2):
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
        
        # Initialize enhanced components
        self._initialize_reasoning_modules()
        self._initialize_meta_learning()
        self._initialize_interactive_learning()
        self._initialize_dynamic_prompts()
        self._initialize_advanced_reasoning()
        
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
        
        # Ensemble methods settings
        self.ensemble_enabled = True
        self.ensemble_size = 3  # Number of ensemble prompts to generate
    
    def _initialize_reasoning_modules(self):
        """Initialize the specialized reasoning modules."""
        self.reasoning_modules = [
            MathReasoningModule(self),
            EthicalReasoningModule(self),
            ScientificReasoningModule(self)
            # More modules can be added here
        ]
    
    def _initialize_meta_learning(self):
        """Initialize the meta-learning system."""
        self.reasoning_history = ReasoningHistory()
        self.meta_learning_enabled = True
    
    def _initialize_interactive_learning(self):
        """Initialize the interactive learning components."""
        self.ambiguity_detector = AmbiguityDetector()
        self.clarification_generator = ClarificationGenerator(self)
        self.interaction_history = []
        self.interaction_enabled = True
    
    def _initialize_dynamic_prompts(self):
        """Initialize the dynamic prompt system."""
        self.prompt_manager = DynamicPromptManager(self)
    
    def _initialize_advanced_reasoning(self):
        """Initialize advanced reasoning modules."""
        self.advanced_reasoning_modules = {
            "temporal": TemporalReasoning(self),
            "counterfactual": CounterfactualReasoning(self),
            "ethical": EthicalReasoningModule(self),
            "visual_spatial": VisualSpatialReasoning(self),
            "systems": SystemsReasoning(self)
        }
        
        # Default enabled modules
        self.enabled_advanced_modules = ["temporal", "counterfactual", "ethical"]
    
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
    
    # ------------------------------------------------------------------------------------------
    # ENSEMBLE METHODS
    # ------------------------------------------------------------------------------------------
    
    def _ensemble_generate(self, prompt_templates, query, **kwargs):
        """
        Generate multiple responses using different prompt templates and aggregate results.
        
        Args:
            prompt_templates (list): List of different prompt templates to use
            query (str): The query/problem to process
            **kwargs: Additional parameters to include in the prompt
        
        Returns:
            dict: Aggregated response containing the consensus and all individual responses
        """
        responses = []
        futures = []
        
        # Generate responses in parallel if enabled
        if self.concurrent_requests:
            for template in prompt_templates:
                formatted_prompt = template.format(query=query, **kwargs)
                future = self.executor.submit(self._generate_response, formatted_prompt)
                futures.append(future)
            
            # Collect responses as they complete
            for future in as_completed(futures):
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    print(f"Error in ensemble generation: {str(e)}")
                    # Add a placeholder to maintain alignment
                    responses.append(f"Error: {str(e)}")
        else:
            # Sequential generation
            for template in prompt_templates:
                formatted_prompt = template.format(query=query, **kwargs)
                try:
                    response = self._generate_response(formatted_prompt)
                    responses.append(response)
                except Exception as e:
                    print(f"Error in ensemble generation: {str(e)}")
                    responses.append(f"Error: {str(e)}")
        
        # Create aggregate response
        return {
            "individual_responses": responses,
            "consensus": self._aggregate_responses(responses),
            "confidence": self._calculate_confidence(responses)
        }
    
    def _aggregate_responses(self, responses):
        """
        Aggregate multiple responses into a consensus answer.
        
        Args:
            responses (list): List of response strings to aggregate
        
        Returns:
            str: Aggregated consensus response
        """
        # Remove any error messages
        valid_responses = [r for r in responses if not r.startswith("Error:")]
        
        if not valid_responses:
            return "Unable to generate a consensus due to errors in all responses."
        
        # If only one valid response, return it
        if len(valid_responses) == 1:
            return valid_responses[0]
        
        # Use meta-prompt to synthesize responses
        synthesis_prompt = f"""
        You are given multiple expert responses to the same problem. Your task is to create a 
        synthesis that combines the best elements of each response, resolves any contradictions,
        and creates a coherent, comprehensive answer.
        
        Here are the responses:
        
        {json.dumps(valid_responses, indent=2)}
        
        Please provide a synthesized response that represents the best consensus.
        """
        
        try:
            consensus = self._generate_response(synthesis_prompt)
            return consensus
        except Exception as e:
            # Fall back to simpler method if API call fails
            print(f"Error in response aggregation: {str(e)}")
            
            # Basic fallback aggregation - join the first paragraph from each response
            first_paragraphs = [r.split('\n\n')[0] for r in valid_responses]
            return "\n\n".join(first_paragraphs)
    
    def _calculate_confidence(self, responses):
        """
        Calculate confidence score based on agreement between responses.
        
        Args:
            responses (list): List of response strings
        
        Returns:
            float: Confidence score between 0 and 1
        """
        # Remove any error messages
        valid_responses = [r for r in responses if not r.startswith("Error:")]
        
        if not valid_responses:
            return 0.0
        
        if len(valid_responses) == 1:
            return 0.7  # Default confidence for single response
        
        # Simple similarity-based confidence
        # Higher similarity between responses = higher confidence
        similarity_sum = 0
        comparison_count = 0
        
        for i in range(len(valid_responses)):
            for j in range(i+1, len(valid_responses)):
                similarity = self._calculate_text_similarity(valid_responses[i], valid_responses[j])
                similarity_sum += similarity
                comparison_count += 1
        
        # Average similarity as confidence (with scaling)
        if comparison_count > 0:
            avg_similarity = similarity_sum / comparison_count
            # Scale similarity to confidence (0.5-1.0 range)
            confidence = 0.5 + (avg_similarity * 0.5)
            return min(1.0, max(0.5, confidence))
        else:
            return 0.7  # Default confidence
    
    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate a simple similarity score between two text strings.
        
        Args:
            text1 (str): First text string
            text2 (str): Second text string
        
        Returns:
            float: Similarity score between 0 and 1
        """
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0
        
        return intersection / union
    
    # ------------------------------------------------------------------------------------------
    # CORE REASONING STEPS (ENHANCED)
    # ------------------------------------------------------------------------------------------
    
    def _is_simple_query(self, query):
        """
        Check if this is a simple query that doesn't need the full reasoning pipeline.
        
        Args:
            query (str): The query to check
            
        Returns:
            bool: True if this is a simple query
        """
        # Check query length - very short queries are often simple
        if len(query.split()) < 8:
            return True
        
        # Check for simple question patterns
        simple_patterns = [
            "what is", "who is", "when was", "where is", 
            "define", "explain", "describe", "tell me about"
        ]
        
        query_lower = query.lower()
        if any(query_lower.startswith(pattern) for pattern in simple_patterns):
            # But make sure it doesn't contain complexity indicators
            complexity_indicators = [
                "analyze", "compare", "contrast", "evaluate", "explain why",
                "how does", "what if", "implications", "consequences",
                "relationship between", "difference between"
            ]
            
            if not any(indicator in query_lower for indicator in complexity_indicators):
                return True
        
        return False
    
    def _analyze_problem(self, query):
        """Analyze the problem structure."""
        if self.ensemble_enabled:
            return self._analyze_problem_ensemble(query)
        
        # Using dynamic prompts for better tailored analysis
        prompt = self.prompt_manager.format_prompt("problem_analysis", query)
        
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
    
    def _analyze_problem_ensemble(self, query):
        """Enhanced problem analysis using ensemble methods."""
        prompt_templates = [
            # Template 1: Standard analytical decomposition
            """
            Analyze the following problem:
            
            {query}
            
            Break it down into its core components and subproblems. Identify key concepts, constraints, and goals.
            """,
            
            # Template 2: First-principles approach
            """
            Examine the following problem from first principles:
            
            {query}
            
            What are the fundamental elements of this problem? Identify the essential variables, constraints, 
            and objectives. Break it down into atomic components that can be solved independently.
            """,
            
            # Template 3: Domain-specific approach
            """
            Consider the following problem from the perspective of domain expertise:
            
            {query}
            
            What specialized knowledge domains are relevant to this problem? Identify the key subproblems
            within each relevant domain. Structure the problem hierarchically.
            """
        ]
        
        # Generate ensemble responses
        ensemble_result = self._ensemble_generate(prompt_templates, query)
        
        # Create a problem node in the cognitive graph
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        # Parse response to identify subproblems
        consensus = ensemble_result["consensus"]
        confidence = ensemble_result["confidence"]
        
        # Enhanced structure for tracking ensemble data
        decomposition = {
            "core_components": consensus,
            "subproblems": [
                {"id": "sub_1", "description": "First aspect of the problem"},
                {"id": "sub_2", "description": "Second aspect of the problem"}
            ],
            "full_decomposition": consensus,
            "ensemble_data": {
                "responses": ensemble_result["individual_responses"],
                "confidence": confidence
            }
        }
        
        # Add subproblem nodes with confidence data
        for subproblem in decomposition["subproblems"]:
            subproblem_node = CognitiveNode(
                node_id=subproblem["id"],
                node_type="subproblem",
                content=subproblem["description"],
                parent_id="problem_0"
            )
            self.cognitive_graph.add_node(subproblem_node)
        
        return decomposition
    
    def _generate_perspectives(self, query, decomposition):
        """Generate multiple perspectives on the problem."""
        # Use dynamic prompt selection for better perspectives
        prompt = self.prompt_manager.format_prompt("perspectives", query)
        
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
    
    def _gather_evidence(self, query, decomposition):
        """Gather evidence for each subproblem."""
        # Check for specialized modules that could provide evidence
        applicable_modules = self._detect_applicable_modules(query)
        specialized_evidence = None
        
        if applicable_modules:
            # Get specialized reasoning results for evidence
            context = {"parent_id": "problem_0"}
            specialized_evidence = self._apply_specialized_reasoning(query, context)
        
        # Use standard evidence gathering for each subproblem
        evidence = {}
        
        for subproblem in decomposition["subproblems"]:
            subproblem_id = subproblem["id"]
            subproblem_desc = subproblem["description"]
            
            # Use dynamic prompts for evidence gathering
            prompt = self.prompt_manager.format_prompt("evidence", query, subproblem_desc=subproblem_desc)
            
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
        
        # Combine with specialized evidence if available
        if specialized_evidence:
            for module_name, result in specialized_evidence["all_results"]:
                # Create a specialized evidence node
                spec_node_id = f"specialized_evidence_{module_name}"
                spec_evidence_node = CognitiveNode(
                    node_id=spec_node_id,
                    node_type="specialized_evidence",
                    content=f"Evidence from {module_name}: {result['result'][:100]}...",
                    parent_id="problem_0"
                )
                self.cognitive_graph.add_node(spec_evidence_node)
                
                # Add to evidence collection
                evidence[spec_node_id] = {
                    "module": module_name,
                    "synthesis": result["result"],
                    "confidence": result["confidence"]
                }
        
        return evidence
    
    def _detect_applicable_modules(self, query):
        """Detect which specialized modules are applicable to the query."""
        applicable_modules = []
        
        for module in self.reasoning_modules:
            if module.is_active() and module.is_applicable(query):
                applicable_modules.append(module)
        
        return applicable_modules
    
    def _apply_specialized_reasoning(self, query, context=None):
        """Apply specialized reasoning modules to the query if applicable."""
        applicable_modules = self._detect_applicable_modules(query)
        
        if not applicable_modules:
            return None  # No specialized reasoning applicable
        
        results = []
        
        # Process with each applicable module
        for module in applicable_modules:
            result = module.process(query, context)
            results.append(result)
        
        # Sort results by confidence
        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Create a specialized reasoning node in the cognitive graph
        reasoning_node = CognitiveNode(
            node_id=f"specialized_reasoning_{int(time.time())}",
            node_type="module_reasoning",
            content=f"Specialized reasoning using {', '.join(r['module'] for r in results)}",
            parent_id=context.get("parent_id") if context else None
        )
        self.cognitive_graph.add_node(reasoning_node)
        
        # Return the highest confidence result for now, but keep all for potential ensemble
        return {
            "primary_result": results[0],
            "all_results": results,
            "node_id": reasoning_node.node_id
        }
    
    def _detect_advanced_reasoning_needs(self, query):
        """
        Detect which advanced reasoning modules would be helpful for a query.
        
        Args:
            query (str): The query to analyze
            
        Returns:
            list: Names of applicable advanced reasoning modules
        """
        applicable_modules = []
        
        # Check each module
        for name, module in self.advanced_reasoning_modules.items():
            if name in self.enabled_advanced_modules and module.is_applicable(query):
                applicable_modules.append(name)
        
        return applicable_modules
    
    def _apply_advanced_reasoning(self, query, applicable_modules, context=None):
        """
        Apply advanced reasoning modules to enrich understanding.
        
        Args:
            query (str): The query to process
            applicable_modules (list): List of applicable module names
            context (dict, optional): Additional context
            
        Returns:
            dict: Results from advanced reasoning
        """
        results = {}
        
        # Create advanced reasoning container node
        container_id = f"advanced_reasoning_{int(time.time())}"
        container_node = CognitiveNode(
            node_id=container_id,
            node_type="advanced_reasoning_container",
            content=f"Advanced reasoning using {', '.join(applicable_modules)}",
            parent_id=context.get("parent_id") if context else None
        )
        self.cognitive_graph.add_node(container_node)
        
        # Update context with container node
        if context is None:
            context = {}
        context["parent_id"] = container_id
        
        # Apply each applicable module
        for module_name in applicable_modules:
            module = self.advanced_reasoning_modules.get(module_name)
            if not module:
                continue
            
            try:
                # Call the appropriate analysis method based on module type
                if module_name == "temporal":
                    result = module.analyze_temporal_patterns(query, context)
                elif module_name == "counterfactual":
                    result = module.analyze_counterfactual(query, context)
                elif module_name == "ethical":
                    result = module.analyze_ethical_dimensions(query, context)
                elif module_name == "visual_spatial":
                    result = module.describe_spatial_relationships(query, context)
                elif module_name == "systems":
                    result = module.analyze_system(query, context)
                else:
                    continue
                
                results[module_name] = result
            except Exception as e:
                print(f"Error in {module_name} reasoning: {str(e)}")
                results[module_name] = {"error": str(e)}
        
        return {
            "results": results,
            "container_node_id": container_id
        }
    
    def _integrate_perspectives_evidence(self, query, perspectives, evidence):
        """Integrate perspectives and evidence into a coherent understanding."""
        # Use dynamic prompts for integration
        prompt = self.prompt_manager.format_prompt(
            "integration", 
            query, 
            perspectives=json.dumps(perspectives),
            evidence=json.dumps(evidence)
        )
        
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
    
    def _generate_solution(self, query, integration):
        """Generate a solution based on the integrated understanding."""
        # Use dynamic prompts for solution generation
        prompt = self.prompt_manager.format_prompt(
            "solution", 
            query, 
            integration=integration["full_integration"]
        )
        
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
    
    def _verify_solution(self, query, solution):
        """Verify the solution through critical analysis."""
        # Use dynamic prompts for verification
        prompt = self.prompt_manager.format_prompt(
            "verification", 
            query, 
            solution=solution["full_solution"]
        )
        
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
    
    def _generate_final_answer(self, query, solution, verification):
        """Generate a final answer based on the solution and verification."""
        # Use dynamic prompts for final answer
        prompt = self.prompt_manager.format_prompt(
            "final_answer", 
            query, 
            solution=solution["full_solution"],
            verification=verification["full_verification"]
        )
        
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
    
    def _metacognitive_reflection(self, query, reasoning_process):
        """Perform metacognitive reflection on the reasoning process."""
        # Use dynamic prompts for metacognitive reflection
        prompt = self.prompt_manager.format_prompt("reflection", query)
        
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
    
    # ------------------------------------------------------------------------------------------
    # META-LEARNING METHODS
    # ------------------------------------------------------------------------------------------
    
    def _adapt_strategy_based_on_history(self, query):
        """
        Adapt the reasoning strategy based on historical performance.
        
        Args:
            query (str): The query to process
        
        Returns:
            dict: Strategy parameters to use
        """
        if not self.meta_learning_enabled or not hasattr(self, 'reasoning_history'):
            # Default strategy if meta-learning is disabled
            return {
                "reasoning_depth": self.reasoning_depth,
                "modules_to_use": []
            }
        
        # Get recommended strategy and modules
        recommended_strategy = self.reasoning_history.get_recommended_strategy(query)
        recommended_modules = self.reasoning_history.get_recommended_modules(query)
        
        return {
            "reasoning_depth": recommended_strategy,
            "modules_to_use": recommended_modules
        }
    
    def _update_meta_learning(self, query, result, feedback=None, success=None):
        """
        Update meta-learning with the results of a reasoning process.
        
        Args:
            query (str): The original query
            result (dict): The full result of reasoning
            feedback (str, optional): User feedback if available
            success (bool, optional): Whether the reasoning was successful
        """
        if not self.meta_learning_enabled or not hasattr(self, 'reasoning_history'):
            return
        
        # Extract the final answer
        final_answer = result.get("final_answer", "")
        
        # Add to reasoning history
        self.reasoning_history.add_entry(
            query=query,
            reasoning_process=result,
            final_answer=final_answer,
            feedback=feedback,
            success=success
        )
    
    # ------------------------------------------------------------------------------------------
    # INTERACTIVE LEARNING METHODS
    # ------------------------------------------------------------------------------------------
    
    def check_for_ambiguities(self, query):
        """
        Check if a query contains ambiguities that require clarification.
        
        Args:
            query (str): The query to check
            
        Returns:
            dict: Information about ambiguities and clarification questions
        """
        if not self.interaction_enabled:
            return {"requires_clarification": False}
        
        # Detect ambiguities
        ambiguities = self.ambiguity_detector.detect_ambiguities(query)
        
        # Only request clarification if ambiguity score is significant
        if ambiguities["ambiguity_score"] > 0.4:
            # Generate clarification questions
            clarification_questions = self.clarification_generator.generate_clarifications(query, ambiguities)
            
            return {
                "requires_clarification": True,
                "ambiguities": ambiguities["ambiguities"],
                "clarification_questions": clarification_questions
            }
        
        return {"requires_clarification": False}
    
    def resolve_with_clarification(self, query, clarification):
        """
        Resolve a query using clarification information.
        
        Args:
            query (str): The original query
            clarification (str): The clarification provided by the user
            
        Returns:
            dict: The enhanced query and context
        """
        # Add to interaction history
        self.interaction_history.append({
            "original_query": query,
            "clarification": clarification,
            "timestamp": datetime.now().isoformat()
        })
        
        # Enhance the query with the clarification
        enhanced_query = f"{query}\n\nAdditional clarification: {clarification}"
        
        # Create context information for reasoning
        context = {
            "original_query": query,
            "clarification": clarification,
            "has_clarification": True
        }
        
        return {
            "enhanced_query": enhanced_query,
            "context": context
        }
    
    # ------------------------------------------------------------------------------------------
    # MAIN SOLVING METHODS
    # ------------------------------------------------------------------------------------------
    
    def _fast_response(self, query, show_work):
        """
        Generate a fast response for simple queries.
        
        Args:
            query (str): The simple query
            show_work (bool): Whether to include reasoning steps
            
        Returns:
            dict: Simplified result
        """
        # Use a direct prompt for simple queries
        response = self._generate_response(f"Answer concisely: {query}")
        
        # Create simplified reasoning graph
        self.cognitive_graph = CognitiveGraph()
        
        # Create problem node
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        # Create direct answer node
        answer_node = CognitiveNode(
            node_id="direct_answer_0",
            node_type="final_answer",
            content=response,
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(answer_node)
        
        # Construct simplified result
        result = {
            "final_answer": response,
            "cognitive_graph": self.cognitive_graph.to_dict(),
            "fast_mode": True
        }
        
        if show_work:
            # Include minimal reasoning process
            result["reasoning_process"] = {
                "fast_mode": True,
                "direct_response": response
            }
        
        return result
    
    def _minimal_reasoning(self, query, show_work):
        """
        Perform minimal reasoning for simpler queries.
        
        Args:
            query (str): The query to solve
            show_work (bool): Whether to show detailed reasoning
            
        Returns:
            dict: The reasoning result
        """
        # Combined prompt for problem analysis and solution
        prompt = f"""
        Analyze and solve the following problem concisely:
        
        {query}
        
        First, briefly identify the core aspects of this problem.
        Then, provide a direct and efficient solution.
        """
        
        response = self._generate_response(prompt)
        
        # Split response into analysis and solution
        parts = response.split("\n\n", 1)
        analysis = parts[0]
        solution = parts[1] if len(parts) > 1 else response
        
        # Create simplified cognitive graph
        self.cognitive_graph = CognitiveGraph()
        
        # Create problem node
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        # Create analysis node
        analysis_node = CognitiveNode(
            node_id="analysis_0",
            node_type="analysis",
            content=analysis,
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(analysis_node)
        
        # Create solution node
        solution_node = CognitiveNode(
            node_id="solution_0",
            node_type="solution",
            content=solution,
            parent_id="analysis_0"
        )
        self.cognitive_graph.add_node(solution_node)
        
        # Construct result
        result = {
            "final_answer": solution,
            "cognitive_graph": self.cognitive_graph.to_dict(),
            "minimal_reasoning": True
        }
        
        if show_work:
            # Include minimal reasoning process
            result["reasoning_process"] = {
                "minimal_reasoning": True,
                "analysis": analysis,
                "solution": solution
            }
        
        return result
    
    def _solve_parallel(self, query, show_work):
        """
        Solve using parallel processing for faster results.
        
        Args:
            query (str): The query to solve
            show_work (bool): Whether to show detailed reasoning
            
        Returns:
            dict: The reasoning result
        """
        # Reset cognitive graph
        self.cognitive_graph = CognitiveGraph()def _solve_parallel(self, query, show_work):
        """
        Solve using parallel processing for faster results.
        
        Args:
            query (str): The query to solve
            show_work (bool): Whether to show detailed reasoning
            
        Returns:
            dict: The reasoning result
        """
        # Reset cognitive graph
        self.cognitive_graph = CognitiveGraph()
        
        # Create problem node
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        # Launch parallel tasks
        futures = {
            "decomposition": self.executor.submit(self._analyze_problem, query),
            "specialized": self.executor.submit(self._apply_specialized_reasoning, query, {"parent_id": "problem_0"})
        }
        
        # Collect results
        decomposition = futures["decomposition"].result()
        specialized_result = futures["specialized"].result()
        
        # Generate perspectives in parallel
        perspectives_future = self.executor.submit(self._generate_perspectives, query, decomposition)
        evidence_future = self.executor.submit(self._gather_evidence, query, decomposition)
        
        perspectives = perspectives_future.result()
        evidence = evidence_future.result()
        
        # Integrate perspectives and evidence
        integration = self._integrate_perspectives_evidence(query, perspectives, evidence)
        
        # Check for advanced reasoning needs
        applicable_adv_modules = self._detect_advanced_reasoning_needs(query)
        advanced_result = None
        if applicable_adv_modules:
            advanced_result = self._apply_advanced_reasoning(query, applicable_adv_modules, {"parent_id": "problem_0"})
            
            # If we have advanced reasoning, enhance the integration
            if advanced_result:
                # Create a prompt that incorporates advanced reasoning results
                advanced_insights = ""
                for module_name, result in advanced_result["results"].items():
                    if "analysis" in result:
                        advanced_insights += f"\n\n{module_name.capitalize()} Analysis:\n{result['analysis']}"
                
                # Re-integrate with advanced insights
                enhanced_integration_prompt = f"""
                Integrate all perspectives, evidence, and advanced analyses into a comprehensive understanding:
                
                Problem: {query}
                
                Standard Analysis: {integration["full_integration"]}
                
                Advanced Analysis: {advanced_insights}
                
                Create a unified understanding that incorporates all these dimensions.
                """
                
                enhanced_response = self._generate_response(enhanced_integration_prompt)
                
                # Update integration
                integration["full_integration"] = enhanced_response
                integration["has_advanced_reasoning"] = True
                
                # Create enhanced integration node
                enhanced_node = CognitiveNode(
                    node_id="enhanced_integration_0",
                    node_type="enhanced_integration",
                    content=enhanced_response,
                    parent_id="integration_0"
                )
                self.cognitive_graph.add_node(enhanced_node)
        
        # Generate solution and verification in parallel
        solution_future = self.executor.submit(self._generate_solution, query, integration)
        
        # Get solution
        solution = solution_future.result()
        
        # Generate verification and final answer sequentially (they depend on previous steps)
        verification = self._verify_solution(query, solution)
        final_answer = self._generate_final_answer(query, solution, verification)
        
        # Skip full metacognitive reflection for speed
        reflection = "Metacognitive reflection skipped for performance optimization."
        
        # Construct the result
        result = {
            "final_answer": final_answer,
            "cognitive_graph": self.cognitive_graph.to_dict()
        }
        
        # Include reasoning process if requested
        if show_work:
            # Add specialized reasoning results if available
            specialized_info = None
            if specialized_result:
                specialized_info = {
                    "primary_module": specialized_result["primary_result"]["module"],
                    "confidence": specialized_result["primary_result"]["confidence"],
                    "result": specialized_result["primary_result"]["result"]
                }
            
            # Add advanced reasoning results if available
            advanced_info = None
            if advanced_result:
                advanced_info = {
                    "modules_used": list(advanced_result["results"].keys()),
                    "container_id": advanced_result["container_node_id"]
                }
            
            result["reasoning_process"] = {
                "decomposition": decomposition,
                "perspectives": perspectives,
                "evidence": evidence,
                "specialized_reasoning": specialized_info,
                "advanced_reasoning": advanced_info,
                "integration": integration,
                "solution": solution,
                "verification": verification,
                "reflection": reflection
            }
        
        return result
    
    def solve_with_meta_learning(self, query, show_work=True, feedback=None, success=None):
        """
        Solve a problem using the cognitive architecture with meta-learning.
        
        Args:
            query (str): The query to solve
            show_work (bool): Whether to show detailed reasoning
            feedback (str, optional): Feedback on previous reasoning attempt
            success (bool, optional): Whether previous reasoning was successful
        
        Returns:
            dict: The reasoning result
        """
        # If feedback is provided for a previous query, update meta-learning
        if feedback is not None and hasattr(self, 'last_result'):
            self._update_meta_learning(
                query=self.last_query,
                result=self.last_result,
                feedback=feedback,
                success=success
            )
        
        # Adapt strategy based on history
        strategy = self._adapt_strategy_based_on_history(query)
        
        # Apply the adapted strategy
        original_depth = self.reasoning_depth
        self.reasoning_depth = strategy["reasoning_depth"]
        
        # Activate recommended modules
        if hasattr(self, 'reasoning_modules'):
            for module in self.reasoning_modules:
                if module.name in strategy["modules_to_use"]:
                    module.activate()
                else:
                    module.deactivate()
        
        # Solve using the adapted strategy
        result = self.solve(query, show_work)
        
        # Add meta-learning info to result
        result["meta_learning"] = {
            "adapted_strategy": strategy,
            "original_depth": original_depth
        }
        
        # Restore original settings
        self.reasoning_depth = original_depth
        
        # Store for potential feedback
        self.last_query = query
        self.last_result = result
        
        return result
    
    def solve_interactive(self, query, show_work=True):
        """
        Solve a problem interactively, requesting clarification if needed.
        
        Args:
            query (str): The query to solve
            show_work (bool): Whether to show detailed reasoning
            
        Returns:
            dict: Either final result or clarification request
        """
        # Check for ambiguities
        ambiguity_check = self.check_for_ambiguities(query)
        
        if ambiguity_check["requires_clarification"]:
            # Return clarification request
            return {
                "requires_clarification": True,
                "clarification_questions": ambiguity_check["clarification_questions"],
                "ambiguities": ambiguity_check["ambiguities"]
            }
        
        # No clarification needed, proceed with normal solving
        return self.solve(query, show_work)
    
    def continue_with_clarification(self, original_query, clarification, show_work=True):
        """
        Continue solving after receiving clarification.
        
        Args:
            original_query (str): The original query
            clarification (str): The clarification provided by the user
            show_work (bool): Whether to show detailed reasoning
            
        Returns:
            dict: The reasoning result
        """
        # Resolve with clarification
        resolution = self.resolve_with_clarification(original_query, clarification)
        
        # Create a clarification node in the cognitive graph
        clarification_node = CognitiveNode(
            node_id=f"clarification_{int(time.time())}",
            node_type="clarification",
            content=f"Original query: {original_query}\nClarification: {clarification}",
            parent_id=None  # Will be set as the first node
        )
        self.cognitive_graph.add_node(clarification_node)
        
        # Solve with enhanced query
        result = self.solve(resolution["enhanced_query"], show_work)
        
        # Add reference to the clarification process
        if "reasoning_process" in result:
            result["reasoning_process"]["clarification"] = {
                "original_query": original_query,
                "clarification": clarification,
                "node_id": clarification_node.node_id
            }
        
        # Add interactive flag
        result["interactive_resolution"] = True
        
        return result
    
    def solve_with_advanced_reasoning(self, query, show_work=True):
        """
        Solve a problem using enhanced cognitive architecture with advanced reasoning.
        
        Args:
            query (str): The query to solve
            show_work (bool): Whether to include detailed reasoning
            
        Returns:
            dict: The reasoning result
        """
        # Reset cognitive graph for new problem
        self.cognitive_graph = CognitiveGraph()
        
        # Create problem node
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        # Detect applicable advanced reasoning modules
        applicable_modules = self._detect_advanced_reasoning_needs(query)
        
        # Standard reasoning process
        decomposition = self._analyze_problem(query)
        perspectives = self._generate_perspectives(query, decomposition)
        evidence = self._gather_evidence(query, decomposition)
        integration = self._integrate_perspectives_evidence(query, perspectives, evidence)
        
        # Apply advanced reasoning if applicable modules found
        advanced_reasoning_results = None
        if applicable_modules:
            context = {"parent_id": "problem_0"}
            advanced_reasoning_results = self._apply_advanced_reasoning(
                query, applicable_modules, context
            )
            
            # Enhance integration with advanced reasoning insights
            if advanced_reasoning_results:
                # Create a prompt that incorporates advanced reasoning results
                advanced_insights = ""
                for module_name, result in advanced_reasoning_results["results"].items():
                    if "analysis" in result:
                        advanced_insights += f"\n\n{module_name.capitalize()} Analysis:\n{result['analysis']}"
                
                # Re-integrate with advanced insights
                enhanced_integration_prompt = f"""
                Integrate all perspectives, evidence, and advanced analyses into a comprehensive understanding:
                
                Problem: {query}
                
                Standard Analysis: {integration["full_integration"]}
                
                Advanced Analysis: {advanced_insights}
                
                Create a unified understanding that incorporates all these dimensions.
                """
                
                enhanced_response = self._generate_response(enhanced_integration_prompt)
                
                # Update integration
                integration["full_integration"] = enhanced_response
                integration["has_advanced_reasoning"] = True
                
                # Create enhanced integration node
                enhanced_node = CognitiveNode(
                    node_id="enhanced_integration_0",
                    node_type="enhanced_integration",
                    content=enhanced_response,
                    parent_id="integration_0"
                )
                self.cognitive_graph.add_node(enhanced_node)
        
        # Continue with solution generation and verification
        solution = self._generate_solution(query, integration)
        verification = self._verify_solution(query, solution)
        final_answer = self._generate_final_answer(query, solution, verification)
        
        # Metacognitive reflection
        reflection = self._metacognitive_reflection(query, {
            "decomposition": decomposition,
            "perspectives": perspectives,
            "evidence": evidence,
            "integration": integration,
            "advanced_reasoning": advanced_reasoning_results,
            "solution": solution,
            "verification": verification
        })
        
        # Construct the result
        result = {
            "final_answer": final_answer,
            "cognitive_graph": self.cognitive_graph.to_dict()
        }
        
        # Include reasoning process if requested
        if show_work:
            result["reasoning_process"] = {
                "decomposition": decomposition,
                "perspectives": perspectives,
                "evidence": evidence,
                "integration": integration,
                "solution": solution,
                "verification": verification,
                "reflection": reflection
            }
            
            # Include advanced reasoning results if available
            if advanced_reasoning_results:
                advanced_reasoning_summary = {}
                for module_name, result_data in advanced_reasoning_results["results"].items():
                    if "error" in result_data:
                        advanced_reasoning_summary[module_name] = {"error": result_data["error"]}
                    else:
                        advanced_reasoning_summary[module_name] = {
                            "analysis_summary": result_data["analysis"][:500] + "..." 
                            if len(result_data["analysis"]) > 500 else result_data["analysis"],
                            "node_id": result_data["node_id"]
                        }
                
                result["reasoning_process"]["advanced_reasoning"] = advanced_reasoning_summary
        
        return result
    
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

# ------------------------------------------------------------------------------------------
# ENHANCED GUI COMPONENTS
# ------------------------------------------------------------------------------------------

class InteractiveModeWidget(QWidget):
    """Widget for enabling interactive learning with clarification requests."""
    
    clarification_submitted = pyqtSignal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Interactive mode toggle
        self.interactive_check = QCheckBox("Interactive Mode")
        self.interactive_check.setChecked(True)
        self.interactive_check.setToolTip(
            "When enabled, the system will ask for clarification on ambiguous queries."
        )
        layout.addWidget(self.interactive_check)
        
        # Clarification panel (initially hidden)
        self.clarification_group = QGroupBox("Clarification Needed")
        self.clarification_group.setVisible(False)
        clarification_layout = QVBoxLayout()
        
        self.clarification_text = QTextEdit()
        self.clarification_text.setReadOnly(True)
        clarification_layout.addWidget(self.clarification_text)
        
        self.clarification_input = QTextEdit()
        self.clarification_input.setPlaceholderText("Enter your clarification here...")
        clarification_layout.addWidget(self.clarification_input)
        
        self.submit_btn = QPushButton("Submit Clarification")
        self.submit_btn.clicked.connect(self.submit_clarification)
        clarification_layout.addWidget(self.submit_btn)
        
        self.clarification_group.setLayout(clarification_layout)
        layout.addWidget(self.clarification_group)
        
        self.setLayout(layout)
        
    def display_clarification_request(self, request_data):
        """Display a clarification request to the user."""
        self.original_query = request_data.get("original_query", "")
        
        # Format clarification questions
        questions = request_data.get("clarification_questions", [])
        questions_text = "\n".join([f"- {q}" for q in questions])
        
        # Show the clarification panel
        self.clarification_text.setHtml(f"""
            <h3>I need some clarification:</h3>
            <p>{questions_text}</p>
            <p>Please provide additional details to help me understand your question better.</p>
        """)
        self.clarification_group.setVisible(True)
        
    def submit_clarification(self):
        """Submit the user's clarification."""
        clarification = self.clarification_input.toPlainText().strip()
        if not clarification:
            QMessageBox.warning(self, "Empty Clarification", 
                               "Please provide some clarification text.")
            return
            
        # Hide the clarification panel
        self.clarification_group.setVisible(False)
        self.clarification_input.clear()
        
        # Emit signal with original query and clarification
        self.clarification_submitted.emit(self.original_query, clarification)
    
    def is_interactive_mode(self):
        """Check if interactive mode is enabled."""
        return self.interactive_check.isChecked()

class AdvancedReasoningWidget(QWidget):
    """Widget for controlling advanced reasoning modules."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.module_checkboxes = {}
        
        # Create checkboxes for advanced reasoning modules
        modules = [
            ("temporal", "Temporal Reasoning", True),
            ("counterfactual", "Counterfactual Reasoning", True),
            ("ethical", "Ethical Reasoning", True),
            ("visual_spatial", "Visual-Spatial Reasoning", False),
            ("systems", "Systems Reasoning", False)
        ]
        
        for module_id, module_name, default_state in modules:
            checkbox = QCheckBox(module_name)
            checkbox.setChecked(default_state)
            layout.addWidget(checkbox)
            self.module_checkboxes[module_id] = checkbox
        
        # Add ensemble method controls
        ensemble_group = QGroupBox("Ensemble Methods")
        ensemble_layout = QVBoxLayout()
        
        self.ensemble_check = QCheckBox("Enable Ensemble Generation")
        self.ensemble_check.setChecked(True)
        ensemble_layout.addWidget(self.ensemble_check)
        
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Ensemble Size:"))
        self.ensemble_size = QSpinBox()
        self.ensemble_size.setRange(2, 5)
        self.ensemble_size.setValue(3)
        size_layout.addWidget(self.ensemble_size)
        ensemble_layout.addLayout(size_layout)
        
        ensemble_group.setLayout(ensemble_layout)
        layout.addWidget(ensemble_group)
        
        # Add meta-learning toggle
        self.meta_learning_check = QCheckBox("Enable Meta-Learning")
        self.meta_learning_check.setChecked(True)
        self.meta_learning_check.setToolTip(
            "When enabled, the system learns from past reasoning attempts to improve future performance."
        )
        layout.addWidget(self.meta_learning_check)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def get_active_modules(self):
        """Get list of active advanced reasoning modules."""
        return [module_id for module_id, checkbox in self.module_checkboxes.items() 
                if checkbox.isChecked()]
    
    def is_ensemble_enabled(self):
        """Check if ensemble methods are enabled."""
        return self.ensemble_check.isChecked()
    
    def get_ensemble_size(self):
        """Get the ensemble size."""
        return self.ensemble_size.value()
    
    def is_meta_learning_enabled(self):
        """Check if meta-learning is enabled."""
        return self.meta_learning_check.isChecked()

class EnhancedReasoningThread(QThread):
    """Enhanced thread for running the NeoCortex reasoning process."""
    
    progress_update = pyqtSignal(str, int)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, process_func, description="Processing"):
        super().__init__()
        self.process_func = process_func
        self.description = description
    
    def run(self):
        try:
            self.progress_update.emit(f"Starting {self.description}...", 10)
            
            # Call the process function
            result = self.process_func()
            
            self.progress_update.emit(f"Completed {self.description}", 100)
            self.result_ready.emit(result)
            
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Error: {str(e)}\n{traceback.format_exc()}")

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
        "specialized_reasoning": "#8BE9FD",
        "advanced_reasoning_container": "#BD93F9",
        "temporal_reasoning": "#FF79C6",
        "counterfactual_reasoning": "#F1FA8C",
        "ethical_reasoning": "#FF5555",
        "visual_spatial_reasoning": "#50FA7B",
        "systems_reasoning": "#FFABFF",
        "clarification": "#8BE9FD"
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

class EnhancedReasoningPanel(QWidget):
    """Panel showing all available reasoning types and their status."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create group box for standard reasoning
        standard_group = QGroupBox("Standard Reasoning")
        standard_layout = QVBoxLayout()
        
        self.standard_reasoning_labels = {}
        
        standard_modules = [
            "problem_decomposition", "multiple_perspectives", 
            "evidence_gathering", "integration", "solution_generation",
            "verification", "metacognition"
        ]
        
        for module in standard_modules:
            label = QLabel(f"• {module.replace('_', ' ').title()}: Inactive")
            standard_layout.addWidget(label)
            self.standard_reasoning_labels[module] = label
        
        standard_group.setLayout(standard_layout)
        layout.addWidget(standard_group)
        
        # Create group box for specialized reasoning
        specialized_group = QGroupBox("Specialized Reasoning")
        specialized_layout = QVBoxLayout()
        
        self.specialized_reasoning_labels = {}
        
        specialized_modules = [
            "mathematical", "ethical", "scientific"
        ]
        
        for module in specialized_modules:
            label = QLabel(f"• {module.title()}: Inactive")
            specialized_layout.addWidget(label)
            self.specialized_reasoning_labels[module] = label
        
        specialized_group.setLayout(specialized_layout)
        layout.addWidget(specialized_group)
        
        # Create group box for advanced reasoning
        advanced_group = QGroupBox("Advanced Reasoning")
        advanced_layout = QVBoxLayout()
        
        self.advanced_reasoning_labels = {}
        
        advanced_modules = [
            "temporal", "counterfactual", "ethical", 
            "visual_spatial", "systems"
        ]
        
        for module in advanced_modules:
            label = QLabel(f"• {module.title()}: Inactive")
            advanced_layout.addWidget(label)
            self.advanced_reasoning_labels[module] = label
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # Add status indicator
        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def update_standard_module(self, module_name, active=True, progress=0):
        """Update the status of a standard reasoning module."""
        if module_name in self.standard_reasoning_labels:
            status = "Active" if active else "Inactive"
            progress_text = f" ({progress}%)" if progress > 0 else ""
            self.standard_reasoning_labels[module_name].setText(
                f"• {module_name.replace('_', ' ').title()}: {status}{progress_text}"
            )
    
    def update_specialized_module(self, module_name, active=True):
        """Update the status of a specialized reasoning module."""
        if module_name in self.specialized_reasoning_labels:
            status = "Active" if active else "Inactive"
            self.specialized_reasoning_labels[module_name].setText(
                f"• {module_name.title()}: {status}"
            )
    
    def update_advanced_module(self, module_name, active=True):
        """Update the status of an advanced reasoning module."""
        if module_name in self.advanced_reasoning_labels:
            status = "Active" if active else "Inactive"
            self.advanced_reasoning_labels[module_name].setText(
                f"• {module_name.title()}: {status}"
            )
    
    def set_status(self, status_text):
        """Set the overall status text."""
        self.status_label.setText(f"Status: {status_text}")
    
    def reset_all(self):
        """Reset all modules to inactive."""
        for label_dict in [self.standard_reasoning_labels, 
                          self.specialized_reasoning_labels, 
                          self.advanced_reasoning_labels]:
            for module, label in label_dict.items():
                if isinstance(module, str):
                    module_name = module.replace("_", " ").title()
                    label.setText(f"• {module_name}: Inactive")
        
        self.status_label.setText("Status: Idle")

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
        self.api_key_input = QLineEdit("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")
        self.api_key_input.setReadOnly(True)
        api_layout.addWidget(self.api_key_input)
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
    
    node_selected = pyqtSignal(str)
    
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
        self.canvas.mpl_connect('button_press_event', self.on_node_click)
        layout.addWidget(self.canvas)
        
        # Node information panel
        self.node_info = QTextEdit()
        self.node_info.setReadOnly(True)
        self.node_info.setMaximumHeight(150)
        self.node_info.setPlaceholderText("Select a node to view details")
        layout.addWidget(self.node_info)
        
        self.setLayout(layout)
        
        # Store node positions and labels for click detection
        self.node_positions = {}
        self.node_ids = {}
    
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
        
        # Store positions for click detection
        self.node_positions = pos
        self.node_ids = {node: node_id for node_id, node in enumerate(G.nodes())}
        
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
    
    def on_node_click(self, event):
        """Handle node click event."""
        if not self.graph_data or not hasattr(event, 'xdata') or event.xdata is None:
            return
        
        # Find the closest node to the click
        min_dist = float('inf')
        closest_node = None
        
        for node, (x, y) in self.node_positions.items():
            dist = ((event.xdata - x) ** 2 + (event.ydata - y) ** 2) ** 0.5
            if dist < min_dist and dist < 0.1:  # Threshold for click proximity
                min_dist = dist
                closest_node = node
        
        if closest_node:
            # Get node data
            nodes = self.graph_data.get("nodes", {})
            if closest_node in nodes:
                node_data = nodes[closest_node]
                
                # Display node info
                node_type = node_data.get("node_type", "Unknown")
                content = node_data.get("content", "No content")
                
                self.node_info.setHtml(f"""
                    <h3>Node: {closest_node}</h3>
                    <p><strong>Type:</strong> {node_type}</p>
                    <p><strong>Content:</strong> {content}</p>
                """)
                
                # Emit node selected signal
                self.node_selected.emit(closest_node)

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
        
        # Add specialized reasoning tab
        self.specialized_widget = QTextEdit()
        self.specialized_widget.setReadOnly(True)
        self.reasoning_widget.addTab(self.specialized_widget, "Specialized Reasoning")
        
        # Add advanced reasoning tab
        self.advanced_widget = QTextEdit()
        self.advanced_widget.setReadOnly(True)
        self.reasoning_widget.addTab(self.advanced_widget, "Advanced Reasoning")
        
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
            
            # Display specialized reasoning
            if "specialized_reasoning" in rp:
                spec = rp["specialized_reasoning"]
                if spec:
                    html = "<h2>Specialized Reasoning</h2>"
                    html += f"<h3>Primary Module: {spec.get('primary_module', 'Unknown')}</h3>"
                    html += f"<p><strong>Confidence:</strong> {spec.get('confidence', 0) * 100:.1f}%</p>"
                    html += f"<p><strong>Result:</strong></p><p>{spec.get('result', 'No result')}</p>"
                    self.specialized_widget.setHtml(html)
                else:
                    self.specialized_widget.setHtml("<h2>Specialized Reasoning</h2><p>No specialized reasoning was applied.</p>")
            
            # Display advanced reasoning
            if "advanced_reasoning" in rp:
                adv = rp["advanced_reasoning"]
                if adv:
                    html = "<h2>Advanced Reasoning</h2>"
                    for module_name, module_data in adv.items():
                        html += f"<h3>{module_name.replace('_', ' ').title()} Analysis</h3>"
                        if "error" in module_data:
                            html += f"<p><strong>Error:</strong> {module_data['error']}</p>"
                        else:
                            html += f"<p>{module_data.get('analysis_summary', 'No analysis available')}</p>"
                        html += "<hr>"
                    self.advanced_widget.setHtml(html)
                else:
                    self.advanced_widget.setHtml("<h2>Advanced Reasoning</h2><p>No advanced reasoning was applied.</p>")
            
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
                    html += self.specialized_widget.toHtml()
                    html += self.advanced_widget.toHtml()
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
                    text += "## Specialized Reasoning\n\n" + self.specialized_widget.toPlainText() + "\n\n"
                    text += "## Advanced Reasoning\n\n" + self.advanced_widget.toPlainText() + "\n\n"
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
        
        # Add export button
        self.export_btn = QPushButton("Export History")
        self.export_btn.clicked.connect(self.export_history)
        layout.addWidget(self.export_btn)
        
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

class MetaLearningWidget(QWidget):
    """Widget for displaying meta-learning information."""
    
    def __init__(self, parent=None, neocortex=None):
        super().__init__(parent)
        self.neocortex = neocortex
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Add meta-learning status
        status_group = QGroupBox("Meta-Learning Status")
        status_layout = QVBoxLayout()
        
        # Add learning stats
        self.stats_label = QLabel("No learning data available yet.")
        status_layout.addWidget(self.stats_label)
        
        # Add feedback section
        feedback_layout = QHBoxLayout()
        feedback_layout.addWidget(QLabel("Last Response:"))
        self.feedback_good = QPushButton("👍 Good")
        self.feedback_good.clicked.connect(lambda: self.provide_feedback(True))
        self.feedback_bad = QPushButton("👎 Poor")
        self.feedback_bad.clicked.connect(lambda: self.provide_feedback(False))
        feedback_layout.addWidget(self.feedback_good)
        feedback_layout.addWidget(self.feedback_bad)
        status_layout.addLayout(feedback_layout)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Add performance visualization
        perf_group = QGroupBox("Performance by Query Type")
        perf_layout = QVBoxLayout()
        
        self.figure = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        perf_layout.addWidget(self.canvas)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        self.setLayout(layout)
    
    def provide_feedback(self, is_good):
        """Provide feedback on the last reasoning process."""
        if not self.neocortex or not hasattr(self.neocortex, 'last_query'):
            QMessageBox.information(self, "No Data", "No previous query to provide feedback on.")
            return
        
        # Update meta-learning with feedback
        self.neocortex._update_meta_learning(
            self.neocortex.last_query, 
            self.neocortex.last_result,
            feedback=f"User rated the response as {'good' if is_good else 'poor'}",
            success=is_good
        )
        
        # Update stats display
        self.update_stats()
        
        QMessageBox.information(self, "Feedback Recorded", 
                               f"Your feedback has been recorded. The system will learn from this.")
    
    def update_stats(self):
        """Update the statistics display."""
        if not self.neocortex or not hasattr(self.neocortex, 'reasoning_history'):
            self.stats_label.setText("No learning data available yet.")
            return
        
        # Get performance summary
        summary = self.neocortex.reasoning_history.get_performance_summary()
        
        # Update stats label
        total = summary["overall"]["total"]
        if total > 0:
            success_rate = summary["overall"]["success_rate"] * 100
            stats_text = f"Total queries: {total}\nSuccess rate: {success_rate:.1f}%"
            self.stats_label.setText(stats_text)
        else:
            self.stats_label.setText("No learning data available yet.")
        
        # Update performance chart
        self.update_performance_chart(summary)
    
    def update_performance_chart(self, summary):
        """Update the performance chart with summary data."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Extract data by query type
        query_types = []
        success_rates = []
        
        for query_type, stats in summary.get("by_query_type", {}).items():
            if stats.get("total", 0) > 0:
                query_types.append(query_type)
                success_rates.append(stats.get("success_rate", 0) * 100)
        
        # Create bar chart
        if query_types:
            ax.bar(query_types, success_rates, color=COLOR_SCHEME["accent"])
            ax.set_xlabel("Query Type")
            ax.set_ylabel("Success Rate (%)")
            ax.set_title("Performance by Query Type")
            ax.set_ylim([0, 100])
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        else:
            ax.text(0.5, 0.5, "No query data available", 
                   horizontalalignment='center', verticalalignment='center')
        
        self.canvas.draw()

class EnhancedNeoCortexGUI(QMainWindow):
    """Enhanced GUI application for NeoCortex."""
    
    def __init__(self):
        super().__init__()
        self.neocortex = NeoCortex()
        self.current_result = None
        self.reasoning_thread = None
        self.init_ui()
        self.apply_dark_theme()
    
    def init_ui(self):
        self.setWindowTitle("NeoCortex: Advanced Cognitive Architecture (DeepSeek V3 Edition)")
        self.setGeometry(100, 100, 1400, 900)
        
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
        
        # Advanced Reasoning tab
        self.advanced_reasoning_widget = AdvancedReasoningWidget()
        left_tabs.addTab(self.advanced_reasoning_widget, "Advanced Reasoning")
        
        # Interactive Learning tab
        self.interactive_widget = InteractiveModeWidget()
        self.interactive_widget.clarification_submitted.connect(self.continue_with_clarification)
        left_tabs.addTab(self.interactive_widget, "Interactive")
        
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
        
        # Meta-learning tab
        self.meta_learning_widget = MetaLearningWidget(neocortex=self.neocortex)
        left_tabs.addTab(self.meta_learning_widget, "Meta-Learning")
        
        left_layout.addWidget(left_tabs)
        left_widget.setLayout(left_layout)
        
        # Middle - Results display
        self.result_display_widget = ResultDisplayWidget()
        
        # Right side - Real-time reasoning panel
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        self.reasoning_panel = EnhancedReasoningPanel()
        right_layout.addWidget(self.reasoning_panel)
        right_widget.setLayout(right_layout)
        
        # Add widgets to splitter
        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(self.result_display_widget)
        self.main_splitter.addWidget(right_widget)
        
        # Set initial splitter sizes (30% - 50% - 20%)
        self.main_splitter.setSizes([420, 700, 280])
        
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
        """Start the reasoning process with enhanced capabilities."""
        query = self.query_edit.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Empty Query", "Please enter a problem or question.")
            return
        
        # Check if interactive mode is enabled and if the query needs clarification
        if self.interactive_widget.is_interactive_mode():
            clarification_check = self.neocortex.check_for_ambiguities(query)
            if clarification_check.get("requires_clarification", False):
                # Show clarification request to user and pause reasoning
                clarification_check["original_query"] = query
                self.interactive_widget.display_clarification_request(clarification_check)
                self.status_bar.showMessage("Waiting for clarification...")
                return
        
        # Disable UI elements
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.query_edit.setReadOnly(True)
        
        # Reset progress
        self.progress_bar.setValue(0)
        self.progress_status.setText("Starting reasoning process...")
        self.status_bar.showMessage("Processing query...")
        
        # Reset reasoning panel
        self.reasoning_panel.reset_all()
        self.reasoning_panel.set_status("Processing")
        
        # Configure NeoCortex with current settings
        use_fast_mode = self.fast_mode_toggle.is_fast_mode()
        self.neocortex.max_tokens = 600 if use_fast_mode else 1000
        self.neocortex.reasoning_depth = "minimal" if use_fast_mode else "balanced"
        
        # Set ensemble settings
        self.neocortex.ensemble_enabled = self.advanced_reasoning_widget.is_ensemble_enabled()
        self.neocortex.ensemble_size = self.advanced_reasoning_widget.get_ensemble_size()
        
        # Set active advanced reasoning modules
        self.neocortex.enabled_advanced_modules = self.advanced_reasoning_widget.get_active_modules()
        
        # Set meta-learning status
        self.neocortex.meta_learning_enabled = self.advanced_reasoning_widget.is_meta_learning_enabled()
        
        # Get show work flag
        show_work = self.show_work_check.isChecked()
        
        # Choose which reasoning method to use based on active features
        if self.advanced_reasoning_widget.is_meta_learning_enabled():
            # Use solve_with_meta_learning
            process_func = lambda: self.neocortex.solve_with_meta_learning(query, show_work)
            description = "meta-learning enhanced reasoning"
        elif self.advanced_reasoning_widget.get_active_modules():
            # Use solve_with_advanced_reasoning
            process_func = lambda: self.neocortex.solve_with_advanced_reasoning(query, show_work)
            description = "advanced reasoning"
        else:
            # Use standard solve method
            process_func = lambda: self.neocortex.solve(query, show_work)
            description = "standard reasoning"
        
        # Create and start reasoning thread
        self.reasoning_thread = EnhancedReasoningThread(process_func, description)
        self.reasoning_thread.progress_update.connect(self.update_progress)
        self.reasoning_thread.result_ready.connect(self.handle_result)
        self.reasoning_thread.error_occurred.connect(self.handle_error)
        self.reasoning_thread.start()
        
        # Simulate module activity for UI demonstration
        self.simulate_module_activity()
    
    def continue_with_clarification(self, original_query, clarification):
        """Continue processing after receiving user clarification."""
        # Disable UI elements
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.query_edit.setReadOnly(True)
        
        # Reset progress
        self.progress_bar.setValue(0)
        self.progress_status.setText("Processing with clarification...")
        self.status_bar.showMessage("Continuing with clarification...")
        
        # Reset reasoning panel
        self.reasoning_panel.reset_all()
        self.reasoning_panel.set_status("Processing with clarification")
        
        # Get show work flag
        show_work = self.show_work_check.isChecked()
        
        # Create process function for the thread
        process_func = lambda: self.neocortex.continue_with_clarification(
            original_query, clarification, show_work
        )
        
        # Create and start reasoning thread
        self.reasoning_thread = EnhancedReasoningThread(
            process_func, "interactive clarification"
        )
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
            self.reasoning_panel.set_status("Stopped")
        
        # Re-enable UI elements
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.query_edit.setReadOnly(False)
    
    def update_progress(self, status, progress):
        """Update the progress bar and status."""
        self.progress_bar.setValue(progress)
        self.progress_status.setText(status)
        self.status_bar.showMessage(f"Processing: {status}")
        
        # Update reasoning panel based on status text
        if "problem" in status.lower():
            self.reasoning_panel.update_standard_module("problem_decomposition", True, progress)
        elif "perspective" in status.lower():
            self.reasoning_panel.update_standard_module("multiple_perspectives", True, progress)
        elif "evidence" in status.lower():
            self.reasoning_panel.update_standard_module("evidence_gathering", True, progress)
        elif "integrat" in status.lower():
            self.reasoning_panel.update_standard_module("integration", True, progress)
        elif "solution" in status.lower():
            self.reasoning_panel.update_standard_module("solution_generation", True, progress)
        elif "verif" in status.lower():
            self.reasoning_panel.update_standard_module("verification", True, progress)
        elif "metacognitive" in status.lower() or "reflection" in status.lower():
            self.reasoning_panel.update_standard_module("metacognition", True, progress)
    
    def handle_result(self, result):
        """Handle the reasoning result."""
        self.current_result = result
        
        # Display the result
        self.result_display_widget.display_result(result)
        
        # Add to history
        self.history_widget.add_history_item(self.query_edit.toPlainText(), result)
        
        # Update meta-learning stats if available
        if hasattr(self, 'meta_learning_widget'):
            self.meta_learning_widget.update_stats()
        
        # Re-enable UI elements
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.query_edit.setReadOnly(False)
        
        # Update status
        self.status_bar.showMessage("Processing complete")
        self.reasoning_panel.set_status("Complete")
        
        # Update reasoning panel with modules used
        if "reasoning_process" in result:
            rp = result["reasoning_process"]
            
            # Update specialized modules
            if "specialized_reasoning" in rp and rp["specialized_reasoning"]:
                module = rp["specialized_reasoning"].get("primary_module", "").lower()
                if module:
                    self.reasoning_panel.update_specialized_module(module)
            
            # Update advanced modules
            if "advanced_reasoning" in rp and rp["advanced_reasoning"]:
                for module in rp["advanced_reasoning"].keys():
                    self.reasoning_panel.update_advanced_module(module)
    
    def handle_error(self, error_msg):
        """Handle errors in the reasoning process."""
        QMessageBox.critical(self, "Error", f"An error occurred during reasoning:\n\n{error_msg}")
        
        # Re-enable UI elements
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.query_edit.setReadOnly(False)
        
        self.progress_status.setText("Error encountered")
        self.status_bar.showMessage("Error: Reasoning process failed")
        self.reasoning_panel.set_status("Error")
    
    def update_model_settings(self, settings):
        """Update the NeoCortex model settings."""
        try:
            # Apply settings to NeoCortex instance
            self.neocortex.model_name = settings["model"]
            self.neocortex.temperature = settings["temperature"]
            
            # Apply additional performance settings
            self.neocortex.max_tokens = settings.get("max_tokens", 800)
            
            # Set performance mode
            perf_mode = settings.get("performance_mode", "balanced")
            if perf_mode == "fast":
                # Fast mode - minimal API calls, short responses
                self.neocortex.max_tokens = min(self.neocortex.max_tokens, 600)
                self.neocortex.reasoning_depth = "minimal"
            elif perf_mode == "thorough":
                # Thorough mode - more detailed analysis
                self.neocortex.max_tokens = max(self.neocortex.max_tokens, 1000)
                self.neocortex.reasoning_depth = "thorough"
            else:
                self.neocortex.reasoning_depth = "balanced"
            
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
            self.reasoning_panel.reset_all()
    
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
        <h1>NeoCortex: Enhanced Cognitive Architecture (DeepSeek V3 Edition)</h1>
        
        <h2>def show_help(self):
        """Show the help dialog."""
        help_text = """
        <h1>NeoCortex: Enhanced Cognitive Architecture (DeepSeek V3 Edition)</h1>
        
        <h2>Overview</h2>
        <p>NeoCortex is a revolutionary cognitive architecture that integrates multiple specialized reasoning modules 
        to achieve AGI-like reasoning capabilities. It uses DeepSeek's V3 language models via OpenRouter to implement a neuromorphic 
        approach to problem-solving that mimics the human brain's parallel processing and self-regulatory functions.</p>
        
        <h2>New Enhanced Features</h2>
        <ul>
            <li><strong>Ensemble Methods:</strong> Generates multiple perspectives and aggregates them for more robust reasoning</li>
            <li><strong>Specialized Reasoning Modules:</strong> Domain-specific modules for math, ethics, science, etc.</li>
            <li><strong>Meta-Learning:</strong> System learns from past reasoning to improve future performance</li>
            <li><strong>Interactive Learning:</strong> Can ask for clarification on ambiguous queries</li>
            <li><strong>Advanced Prompt Engineering:</strong> Dynamic templates for optimal responses</li>
            <li><strong>Enhanced Cognitive Architecture:</strong> Support for temporal, counterfactual, ethical reasoning</li>
            <li><strong>Improved Visualization:</strong> Enhanced cognitive graph and real-time module activity</li>
        </ul>
        
        <h2>Using the Interface</h2>
        <ol>
            <li><strong>Problem Input:</strong> Enter your question or problem in the text area at the top.</li>
            <li><strong>Process:</strong> Click "Process with NeoCortex" to start the reasoning process.</li>
            <li><strong>Results:</strong> View the final answer and detailed reasoning in the tabs on the right.</li>
            <li><strong>Cognitive Graph:</strong> Explore the reasoning structure visually in the Cognitive Graph tab.</li>
            <li><strong>Module Controls:</strong> Enable or disable specific reasoning modules in the Modules tab.</li>
            <li><strong>Advanced Reasoning:</strong> Configure ensemble methods and specialized reasoning in the Advanced Reasoning tab.</li>
            <li><strong>Interactive Mode:</strong> Enable interactive clarification requests in the Interactive tab.</li>
            <li><strong>Settings:</strong> Adjust model parameters in the Settings tab.</li>
            <li><strong>History:</strong> Access previous reasoning sessions in the History tab.</li>
            <li><strong>Meta-Learning:</strong> Provide feedback to help the system learn in the Meta-Learning tab.</li>
        </ol>
        
        <h2>Tips for Best Results</h2>
        <ul>
            <li>Formulate clear, specific questions</li>
            <li>For complex problems, disable fast mode and enable detailed reasoning</li>
            <li>Use interactive mode for ambiguous queries that might need clarification</li>
            <li>Enable the specialized modules relevant to your problem domain</li>
            <li>Provide feedback in the Meta-Learning tab to help the system improve</li>
            <li>For mathematical problems, the Mathematical Reasoning module excels</li>
            <li>For ethical questions, enable both the Ethical Reasoning module and advanced ethical reasoning</li>
        </ul>
        """
        
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("NeoCortex Help")
        help_dialog.setMinimumSize(700, 600)
        
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
    app.setApplicationName("NeoCortex Enhanced Edition")
    
    window = EnhancedNeoCortexGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()