import sys
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading
import queue
import requests
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer, QPointF
from PyQt5.QtGui import QFont, QColor, QIcon, QPalette, QPixmap, QBrush, QLinearGradient, QPainter, QPen, QPainterPath, QPolygonF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTextEdit, QLineEdit, QGroupBox, QTabWidget, QSplitter,
    QComboBox, QListWidget, QListWidgetItem, QProgressBar, QDialog, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QCheckBox, QSpinBox, QDoubleSpinBox, QSlider,
    QScrollArea, QFrame, QGraphicsView, QGraphicsScene, QGraphicsItem, 
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QMessageBox,
    QStyleFactory, QToolBar, QAction, QStatusBar, QGraphicsRectItem, 
    QGraphicsPathItem, QGraphicsPolygonItem
)

from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

# Color scheme for UI styling
COLOR_SCHEME = {
    "background": "#1E1E1E",       # Dark background
    "secondary_bg": "#252525",     # Slightly lighter background for contrast
    "text": "#E0E0E0",             # Light text color
    "accent": "#7B68EE",           # Medium slate blue as accent color
    # Node colors for cognitive graph visualization
    "node_colors": {
        "problem": "#FF7F50",      # Coral
        "subproblem": "#FFD700",   # Gold
        "perspective": "#9370DB",   # Medium purple
        "evidence": "#20B2AA",     # Light sea green
        "integration": "#6495ED",   # Cornflower blue
        "solution": "#32CD32",     # Lime green
        "verification": "#FF6347",  # Tomato
        "metacognitive_reflection": "#8A2BE2",  # Blue violet
        "analysis": "#87CEEB",     # Sky blue
        "criticism": "#FF4500",    # Orange red
        "creative": "#FF69B4",     # Hot pink
        "logical": "#00BFFF",      # Deep sky blue
        "synthesis": "#BA55D3",    # Medium orchid
        "implementation": "#4682B4", # Steel blue
        "evaluation": "#F08080",   # Light coral
        "final_answer": "#00FA9A"  # Medium spring green
    }
}

class Agent:
    """Base class for all agents in the multi-agent system."""
    
    def __init__(self, name: str, neocortex):
        self.name = name
        self.neocortex = neocortex
        self.knowledge_base = {}
        self.state = "idle"  # idle, working, done
        
    def process(self, input_data: Dict) -> Dict:
        """Process input data and return results."""
        self.state = "working"
        # Base implementation to be overridden by subclasses
        result = self._process_implementation(input_data)
        self.state = "done"
        return result
        
    def _process_implementation(self, input_data: Dict) -> Dict:
        """Actual processing implementation - to be overridden by subclasses."""
        return {"status": "not_implemented"}
        
    def get_status(self) -> Dict:
        """Get current status of the agent."""
        return {
            "name": self.name,
            "state": self.state,
            "knowledge_size": len(self.knowledge_base)
        }
        
    def update_knowledge(self, key: str, value: Any):
        """Update the agent's knowledge base."""
        self.knowledge_base[key] = value
        
    def get_knowledge(self, key: str) -> Any:
        """Retrieve knowledge from the agent's knowledge base."""
        return self.knowledge_base.get(key)

class AnalystAgent(Agent):
    """Agent responsible for analyzing problems and breaking them down."""
    
    def __init__(self, neocortex):
        super().__init__("Analyst", neocortex)
        
    def _process_implementation(self, input_data: Dict) -> Dict:
        """Analyze a problem and break it down into components."""
        query = input_data.get("query", "")
        if not query:
            return {"status": "error", "message": "No query provided"}
            
        # Generate a system prompt focused on analysis
        system_prompt = """You are an analytical agent specialized in breaking down complex problems.
Your task is to:
1. Identify the core components of the problem
2. Determine the key variables and constraints
3. Recognize implicit assumptions
4. Identify potential approaches for solving each component
5. Structure the problem in a way that facilitates systematic problem-solving

Provide a detailed analysis that serves as a foundation for solving the problem."""

        prompt = f"""
        Analyze the following problem systematically:

        {query}

        Break it down into its fundamental components, identify key challenges, 
        variables, constraints, and possible solution approaches.
        Structure your analysis to facilitate a systematic problem-solving process.
        """
            
        # Generate response with the specialized system prompt
        response_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=system_prompt,
            task_id="analysis"
        )
            
        # Create analysis node in the cognitive graph
        analysis_node = CognitiveNode(
            node_id="analysis_0",
            node_type="analysis",
            content=response_text,
            parent_id="problem_0"
        )
        self.neocortex.cognitive_graph.add_node(analysis_node)
            
        # Parse response to extract subproblems (simplified)
        components = self._extract_components(response_text)
            
        return {
            "status": "success",
            "analysis": response_text,
            "components": components
        }
        
    def _extract_components(self, analysis_text: str) -> List[Dict]:
        """Extract problem components from analysis text."""
        # This is a simplified extraction - could be enhanced with better parsing
        components = []
        lines = analysis_text.split('\n')
        current_component = None
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for component headers (assuming headers are in format like "Component 1:" or "1. Component:")
            if (line.lower().startswith("component") or 
                (line[0].isdigit() and ":" in line[:20])):
                if current_component:
                    components.append(current_component)
                    
                title = line.split(":", 1)[1].strip() if ":" in line else line
                current_component = {
                    "id": f"component_{len(components)}",
                    "title": title,
                    "description": "",
                    "approaches": []
                }
            elif current_component:
                # If line starts with "Approach:" or similar, it's an approach
                if line.lower().startswith(("approach", "method", "strategy")):
                    approach = line.split(":", 1)[1].strip() if ":" in line else line
                    current_component["approaches"].append(approach)
                else:
                    # Otherwise add to component description
                    current_component["description"] += line + " "
                    
        # Add the last component if it exists
        if current_component:
            components.append(current_component)
            
        # If no components were identified, create a default one
        if not components:
            components.append({
                "id": "component_0",
                "title": "Main Problem",
                "description": analysis_text[:200] + "...",  # Truncate for brevity
                "approaches": ["Comprehensive approach"]
            })
            
        return components

class CriticAgent(Agent):
    """Agent responsible for critically evaluating solutions and finding weaknesses."""
    
    def __init__(self, neocortex):
        super().__init__("Critic", neocortex)
        
    def _process_implementation(self, input_data: Dict) -> Dict:
        """Critically evaluate the proposed solution."""
        query = input_data.get("query", "")
        solution = input_data.get("solution", "")
        
        if not query or not solution:
            return {"status": "error", "message": "Missing query or solution"}
            
        # Generate a system prompt focused on critical evaluation
        system_prompt = """You are a critical evaluation agent specialized in finding flaws and weaknesses.
Your task is to:
1. Rigorously evaluate the proposed solution against the original problem requirements
2. Identify logical fallacies, gaps in reasoning, or unfounded assumptions
3. Test edge cases and boundary conditions
4. Check for computational or mathematical errors
5. Assess scalability, efficiency, and practicality concerns
6. Suggest specific improvements for each identified weakness

Be thorough, precise, and constructive in your criticism."""

        prompt = f"""
        Critically evaluate the following solution:

        PROBLEM:
        {query}

        PROPOSED SOLUTION:
        {solution}

        Assess the solution by:
        1. Checking if it fully addresses all aspects of the problem
        2. Identifying any weaknesses, gaps, or errors
        3. Testing edge cases or boundary conditions
        4. Evaluating efficiency and practicality
        5. Suggesting specific improvements for each weakness found

        Be rigorous but constructive in your evaluation.
        """
            
        # Generate response with the specialized system prompt
        response_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=system_prompt,
            task_id="criticism"
        )
            
        # Create criticism node in the cognitive graph
        criticism_node = CognitiveNode(
            node_id="criticism_0",
            node_type="criticism",
            content=response_text,
            parent_id="solution_0"
        )
        self.neocortex.cognitive_graph.add_node(criticism_node)
            
        # Extract weaknesses and improvements
        weaknesses, improvements = self._extract_criticism(response_text)
            
        return {
            "status": "success",
            "criticism": response_text,
            "weaknesses": weaknesses,
            "improvements": improvements
        }
        
    def _extract_criticism(self, criticism_text: str) -> Tuple[List[str], List[str]]:
        """Extract weaknesses and improvements from criticism text."""
        weaknesses = []
        improvements = []
            
        # Simple parsing logic - could be enhanced
        lines = criticism_text.split('\n')
        current_section = None
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in ["weakness", "flaw", "issue", "problem", "limitation"]):
                current_section = "weaknesses"
                if ":" in line:
                    weakness = line.split(":", 1)[1].strip()
                    if weakness:
                        weaknesses.append(weakness)
            elif any(keyword in lower_line for keyword in ["improvement", "enhancement", "suggestion", "recommendation"]):
                current_section = "improvements"
                if ":" in line:
                    improvement = line.split(":", 1)[1].strip()
                    if improvement:
                        improvements.append(improvement)
            elif current_section == "weaknesses" and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                weaknesses.append(line.lstrip("- •0123456789.").strip())
            elif current_section == "improvements" and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                improvements.append(line.lstrip("- •0123456789.").strip())
                
        # If no explicit weaknesses/improvements were found, make a best guess
        if not weaknesses:
            for i, line in enumerate(lines):
                if i > 0 and len(line) > 20 and not line.lower().startswith(("overall", "conclusion", "summary")):
                    weaknesses.append(line[:100] + "..." if len(line) > 100 else line)
                    if len(weaknesses) >= 3:
                        break
                        
        if not improvements:
            for i, line in enumerate(reversed(lines)):
                if len(line) > 20 and not line.lower().startswith(("overall", "conclusion", "summary")):
                    improvements.append(line[:100] + "..." if len(line) > 100 else line)
                    if len(improvements) >= 3:
                        break
                        
        return weaknesses, improvements

class CreativeAgent(Agent):
    """Agent responsible for generating innovative approaches and solutions."""
    
    def __init__(self, neocortex):
        super().__init__("Creative", neocortex)
        
    def _process_implementation(self, input_data: Dict) -> Dict:
        """Generate creative approaches to the problem."""
        query = input_data.get("query", "")
        analysis = input_data.get("analysis", "")
        
        if not query:
            return {"status": "error", "message": "No query provided"}
            
        # Generate a system prompt focused on creative thinking
        system_prompt = """You are a creative agent specialized in generating innovative approaches to problems.
Your task is to:
1. Think beyond conventional solutions and constraints
2. Generate multiple diverse and novel approaches
3. Consider interdisciplinary perspectives and analogies
4. Look for unexpected connections and possibilities
5. Balance originality with practicality

Provide multiple creative approaches that open new possibilities for solving the problem."""

        prompt = f"""
        Generate creative and innovative approaches to the following problem:

        PROBLEM:
        {query}

        ANALYSIS:
        {analysis}

        Generate at least 3 distinct, creative approaches that:
        1. Offer novel perspectives on the problem
        2. Draw from diverse domains or disciplines
        3. Challenge conventional assumptions
        4. Present unique advantages over standard approaches
        5. Balance innovation with feasibility

        For each approach, provide a clear description, key insights, and potential implementation steps.
        """
            
        # Generate response with the specialized system prompt
        response_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=system_prompt,
            task_id="creative"
        )
            
        # Create creative node in the cognitive graph
        creative_node = CognitiveNode(
            node_id="creative_0",
            node_type="creative",
            content=response_text,
            parent_id="analysis_0"
        )
        self.neocortex.cognitive_graph.add_node(creative_node)
            
        # Extract creative approaches
        approaches = self._extract_approaches(response_text)
            
        return {
            "status": "success",
            "creative_output": response_text,
            "approaches": approaches
        }
        
    def _extract_approaches(self, creative_text: str) -> List[Dict]:
        """Extract creative approaches from the text."""
        approaches = []
        lines = creative_text.split('\n')
        current_approach = None
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for approach headers (typical formats)
            if (line.lower().startswith(("approach", "idea", "solution", "method")) or 
                (line[0].isdigit() and ":" in line[:20]) or
                (line.startswith("#") and len(line) > 2)):
                if current_approach:
                    approaches.append(current_approach)
                    
                title = line.split(":", 1)[1].strip() if ":" in line else line
                title = title.lstrip("#").strip()
                current_approach = {
                    "id": f"approach_{len(approaches)}",
                    "title": title,
                    "description": "",
                    "insights": [],
                    "steps": []
                }
            elif current_approach:
                # If line indicates insights or steps
                lower_line = line.lower()
                if "insight" in lower_line or "key point" in lower_line:
                    insight = line.split(":", 1)[1].strip() if ":" in line else line
                    current_approach["insights"].append(insight)
                elif any(step_word in lower_line for step_word in ["step", "implementation", "process", "procedure"]):
                    step = line.split(":", 1)[1].strip() if ":" in line else line
                    current_approach["steps"].append(step)
                elif line.startswith("-") or line.startswith("•") or (line[0].isdigit() and ". " in line[:5]):
                    # List items might be steps
                    point = line.lstrip("- •0123456789.").strip()
                    if len(current_approach["steps"]) > len(current_approach["insights"]):
                        current_approach["steps"].append(point)
                    else:
                        current_approach["insights"].append(point)
                else:
                    # Otherwise add to approach description
                    current_approach["description"] += line + " "
                    
        # Add the last approach if it exists
        if current_approach:
            approaches.append(current_approach)
            
        # If no approaches were identified, create a default one
        if not approaches:
            sections = creative_text.split("\n\n")
            for i, section in enumerate(sections):
                if len(section) > 50:
                    approaches.append({
                        "id": f"approach_{i}",
                        "title": f"Creative Approach {i+1}",
                        "description": section[:200] + "...",  # Truncate for brevity
                        "insights": [],
                        "steps": []
                    })
                    if len(approaches) >= 3:
                        break
                        
            # If still no approaches, create a single one with the entire text
            if not approaches:
                approaches.append({
                    "id": "approach_0",
                    "title": "Creative Approach",
                    "description": creative_text[:300] + "...",  # Truncate for brevity
                    "insights": [],
                    "steps": []
                })
                
        return approaches

class LogicalAgent(Agent):
    """Agent responsible for logical reasoning and structured problem-solving."""
    
    def __init__(self, neocortex):
        super().__init__("Logical", neocortex)
        
    def _process_implementation(self, input_data: Dict) -> Dict:
        """Apply logical reasoning to the problem."""
        query = input_data.get("query", "")
        analysis = input_data.get("analysis", "")
        components = input_data.get("components", [])
        
        if not query:
            return {"status": "error", "message": "No query provided"}
            
        # Generate a system prompt focused on logical reasoning
        system_prompt = """You are a logical reasoning agent specialized in structured problem-solving.
Your task is to:
1. Apply deductive and inductive reasoning techniques
2. Develop clear, step-by-step logical frameworks for solving problems
3. Identify and validate assumptions
4. Construct sound arguments with well-defined premises and conclusions
5. Apply formal methods where appropriate (mathematical, computational, etc.)

Provide a rigorous logical analysis that leads to a well-structured solution approach."""

        # Prepare component information
        component_info = ""
        for comp in components:
            component_info += f"- {comp.get('title', 'Unnamed component')}: {comp.get('description', '')[:100]}...\n"

        prompt = f"""
        Apply logical reasoning to solve the following problem:

        PROBLEM:
        {query}

        ANALYSIS:
        {analysis}

        COMPONENTS:
        {component_info}

        Develop a logical solution by:
        1. Formulating precise definitions and clarifying assumptions
        2. Breaking down the problem into logical steps or sub-problems
        3. Applying appropriate reasoning techniques (deductive, inductive, abductive)
        4. Constructing a step-by-step solution process
        5. Validating each step through logical verification

        Ensure your reasoning is sound, explicit, and leads to a well-structured solution.
        """
            
        # Generate response with the specialized system prompt
        response_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=system_prompt,
            task_id="logical"
        )
            
        # Create logical node in the cognitive graph
        logical_node = CognitiveNode(
            node_id="logical_0",
            node_type="logical",
            content=response_text,
            parent_id="analysis_0"
        )
        self.neocortex.cognitive_graph.add_node(logical_node)
            
        # Extract logical arguments and steps
        arguments, steps = self._extract_logical_content(response_text)
            
        return {
            "status": "success",
            "logical_output": response_text,
            "arguments": arguments,
            "steps": steps
        }
        
    def _extract_logical_content(self, logical_text: str) -> Tuple[List[Dict], List[str]]:
        """Extract logical arguments and solution steps from the text."""
        arguments = []
        steps = []
            
        # Parse content for arguments and steps
        lines = logical_text.split('\n')
        current_section = None
        current_argument = None
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for section headers
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in ["argument", "premise", "conclusion", "theorem", "proposition"]):
                current_section = "arguments"
                if current_argument:
                    arguments.append(current_argument)
                    
                current_argument = {
                    "id": f"argument_{len(arguments)}",
                    "title": line,
                    "premises": [],
                    "conclusion": ""
                }
            elif "premise" in lower_line:
                if current_argument:
                    premise = line.split(":", 1)[1].strip() if ":" in line else line
                    current_argument["premises"].append(premise)
            elif "conclusion" in lower_line:
                if current_argument:
                    conclusion = line.split(":", 1)[1].strip() if ":" in line else line
                    current_argument["conclusion"] = conclusion
            elif any(keyword in lower_line for keyword in ["step", "procedure", "algorithm", "solution phase"]):
                current_section = "steps"
                step = line.split(":", 1)[1].strip() if ":" in line else line
                steps.append(step)
            elif current_section == "steps" and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                step = line.lstrip("- •0123456789.").strip()
                steps.append(step)
                
        # Add the last argument if it exists
        if current_argument:
            arguments.append(current_argument)
            
        # If no structured content was found, make a best effort extraction
        if not arguments and not steps:
            # Try to identify steps based on numbering patterns
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.lower().startswith(("step", "first", "second", "third", "next", "finally"))):
                    steps.append(line)
                    
            # If still no steps, extract some paragraphs as logical arguments
            paragraphs = logical_text.split("\n\n")
            for i, para in enumerate(paragraphs):
                if len(para.strip()) > 100:
                    arguments.append({
                        "id": f"argument_{i}",
                        "title": f"Logical Argument {i+1}",
                        "premises": [para.strip()[:100] + "..."],
                        "conclusion": ""
                    })
                    if len(arguments) >= 3:
                        break
                        
        return arguments, steps

class SynthesizerAgent(Agent):
    """Agent responsible for integrating multiple perspectives and approaches."""
    
    def __init__(self, neocortex):
        super().__init__("Synthesizer", neocortex)
        
    def _process_implementation(self, input_data: Dict) -> Dict:
        """Synthesize multiple approaches into a cohesive solution."""
        query = input_data.get("query", "")
        analysis = input_data.get("analysis", "")
        logical_output = input_data.get("logical_output", "")
        creative_output = input_data.get("creative_output", "")
        
        if not query:
            return {"status": "error", "message": "No query provided"}
            
        # Generate a system prompt focused on synthesis
        system_prompt = """You are a synthesis agent specialized in integrating multiple perspectives and approaches.
Your task is to:
1. Identify complementary elements across different solutions and perspectives
2. Resolve contradictions and tensions between approaches
3. Create a unified framework that leverages the strengths of each approach
4. Ensure the integrated solution addresses all aspects of the original problem
5. Present a coherent, comprehensive solution that is greater than the sum of its parts

Provide a synthesis that integrates diverse perspectives into a cohesive solution."""

        prompt = f"""
        Synthesize the following perspectives into a comprehensive solution:

        PROBLEM:
        {query}

        ANALYSIS:
        {analysis[:300]}... [truncated for brevity]

        LOGICAL APPROACH:
        {logical_output[:300]}... [truncated for brevity]

        CREATIVE APPROACH:
        {creative_output[:300]}... [truncated for brevity]

        Create an integrated solution by:
        1. Identifying complementary elements from the logical and creative approaches
        2. Resolving any contradictions between the approaches
        3. Combining strengths while mitigating weaknesses
        4. Ensuring all aspects of the problem are addressed
        5. Presenting a coherent, unified solution framework

        Begin your solution with "### Synthesized Comprehensive Solution:"
        """
            
        # Generate response with the specialized system prompt
        response_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=system_prompt,
            task_id="synthesis"
        )
            
        # Create synthesis node in the cognitive graph
        synthesis_node = CognitiveNode(
            node_id="synthesis_0",
            node_type="synthesis",
            content=response_text,
            parent_id="problem_0"
        )
        self.neocortex.cognitive_graph.add_node(synthesis_node)
            
        # Extract final solution
        solution_text = self._extract_solution(response_text)
            
        return {
            "status": "success",
            "synthesis": response_text,
            "solution": solution_text
        }
        
    def _extract_solution(self, synthesis_text: str) -> str:
        """Extract the final solution from the synthesis text."""
        # Look for solution marker
        if "### Synthesized Comprehensive Solution:" in synthesis_text:
            solution = synthesis_text.split("### Synthesized Comprehensive Solution:", 1)[1].strip()
            return solution
            
        # If marker not found, try to identify the solution section
        paragraphs = synthesis_text.split("\n\n")
        
        # If there are multiple paragraphs, the last few are likely the solution
        if len(paragraphs) > 2:
            return "\n\n".join(paragraphs[-2:])
            
        # Fallback: return the entire text
        return synthesis_text

class ImplementerAgent(Agent):
    """Agent responsible for creating concrete implementations and code solutions."""
    
    def __init__(self, neocortex):
        super().__init__("Implementer", neocortex)
        
    def _process_implementation(self, input_data: Dict) -> Dict:
        """Create a concrete implementation or code solution."""
        query = input_data.get("query", "")
        solution = input_data.get("solution", "")
        
        if not query or not solution:
            return {"status": "error", "message": "Missing query or solution"}
            
        # Check if this problem likely needs code implementation
        needs_code = self._problem_needs_code(query, solution)
            
        # Generate a system prompt focused on implementation
        system_prompt = """You are an implementation agent specialized in creating concrete, executable solutions.
Your task is to:
1. Translate conceptual solutions into practical implementations
2. Write clean, efficient, and well-documented code when needed
3. Provide clear instructions for non-code implementations
4. Include error handling, edge cases, and practical considerations
5. Ensure implementations are complete and ready to use

Provide a concrete implementation that brings the solution to life."""

        prompt = f"""
        Create a concrete implementation for the following solution:

        PROBLEM:
        {query}

        CONCEPTUAL SOLUTION:
        {solution}

        {'Develop a complete code implementation with:' if needs_code else 'Provide a detailed implementation plan with:'}
        1. {'Clean, well-documented code' if needs_code else 'Clear step-by-step instructions'}
        2. {'Appropriate error handling' if needs_code else 'Required resources and preconditions'}
        3. {'Usage examples' if needs_code else 'Expected outcomes and verification methods'}
        4. {'Performance considerations' if needs_code else 'Potential challenges and mitigation strategies'}
        5. {'Extension points' if needs_code else 'Timeline and evaluation criteria'}

        {'Ensure the code is complete, correct, and ready to run.' if needs_code else 'Ensure the implementation plan is comprehensive and actionable.'}
        """
            
        # Generate response with the specialized system prompt
        response_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=system_prompt,
            task_id="implementation"
        )
            
        # Create implementation node in the cognitive graph
        implementation_node = CognitiveNode(
            node_id="implementation_0",
            node_type="implementation",
            content=response_text,
            parent_id="synthesis_0"
        )
        self.neocortex.cognitive_graph.add_node(implementation_node)
            
        # Extract code blocks if present
        code_blocks = self._extract_code_blocks(response_text) if needs_code else []
            
        return {
            "status": "success",
            "implementation": response_text,
            "code_blocks": code_blocks,
            "has_code": needs_code
        }
        
    def _problem_needs_code(self, query: str, solution: str) -> bool:
        """Determine if the problem likely needs a code implementation."""
        code_keywords = [
            "code", "program", "algorithm", "function", "class", 
            "implementation", "script", "application", "api", "software",
            "python", "java", "javascript", "c++", "programming"
        ]
            
        # Check if the query or solution contain code-related keywords
        query_lower = query.lower()
        solution_lower = solution.lower()
            
        # If the solution already contains code blocks, it definitely needs code
        if "```" in solution:
            return True
            
        # Check for code keywords
        for keyword in code_keywords:
            if keyword in query_lower or keyword in solution_lower:
                return True
                
        return False
        
    def _extract_code_blocks(self, implementation_text: str) -> List[Dict]:
        """Extract code blocks from the implementation text."""
        code_blocks = []
        lines = implementation_text.split('\n')
        in_code_block = False
        current_code_block = ""
        current_language = ""
            
        for line in lines:
            if line.startswith("```"):
                if in_code_block:
                    # End of code block
                    in_code_block = False
                    code_blocks.append({
                        "language": current_language,
                        "code": current_code_block
                    })
                    current_code_block = ""
                    current_language = ""
                else:
                    # Start of code block
                    in_code_block = True
                    language_spec = line[3:].strip()
                    current_language = language_spec if language_spec else "text"
            elif in_code_block:
                current_code_block += line + "\n"
                
        # Handle case where the last code block wasn't closed
        if in_code_block and current_code_block:
            code_blocks.append({
                "language": current_language,
                "code": current_code_block
            })
            
        return code_blocks

class EvaluatorAgent(Agent):
    """Agent responsible for testing and evaluating the complete solution."""
    
    def __init__(self, neocortex):
        super().__init__("Evaluator", neocortex)
        
    def _process_implementation(self, input_data: Dict) -> Dict:
        """Evaluate the complete solution against the original problem."""
        query = input_data.get("query", "")
        solution = input_data.get("solution", "")
        implementation = input_data.get("implementation", "")
        criticism = input_data.get("criticism", "")
        
        if not query or not solution:
            return {"status": "error", "message": "Missing query or solution"}
            
        # Generate a system prompt focused on evaluation
        system_prompt = """You are an evaluation agent specialized in testing and validating solutions.
Your task is to:
1. Assess how well the solution addresses the original problem requirements
2. Apply multiple evaluation criteria (correctness, efficiency, completeness, etc.)
3. Identify strengths of the solution that should be preserved
4. Highlight any remaining weaknesses that should be addressed
5. Provide an overall evaluation score and summary assessment

Provide a comprehensive evaluation that assesses the quality and completeness of the solution."""

        prompt = f"""
        Evaluate the following solution against the original problem:

        PROBLEM:
        {query}

        SOLUTION:
        {solution[:300]}... [truncated for brevity]

        IMPLEMENTATION:
        {implementation[:300]}... [truncated for brevity]

        PREVIOUS CRITICISM:
        {criticism[:300]}... [truncated for brevity]

        Evaluate the solution using these criteria:
        1. Correctness: Does it solve the problem correctly?
        2. Completeness: Does it address all aspects of the problem?
        3. Efficiency: Is it efficient in terms of resources and time?
        4. Clarity: Is it clear and easy to understand?
        5. Robustness: Does it handle edge cases and potential issues?

        For each criterion, provide:
        - A score from 1-10
        - Specific strengths to preserve
        - Any remaining weaknesses to address

        Conclude with an overall assessment and final score.
        """
            
        # Generate response with the specialized system prompt
        response_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=system_prompt,
            task_id="evaluation"
        )
            
        # Create evaluation node in the cognitive graph
        evaluation_node = CognitiveNode(
            node_id="evaluation_0",
            node_type="evaluation",
            content=response_text,
            parent_id="implementation_0"
        )
        self.neocortex.cognitive_graph.add_node(evaluation_node)
            
        # Extract scores and summary assessment
        scores, assessment = self._extract_evaluation(response_text)
            
        return {
            "status": "success",
            "evaluation": response_text,
            "scores": scores,
            "assessment": assessment
        }
        
    def _extract_evaluation(self, evaluation_text: str) -> Tuple[Dict[str, int], str]:
        """Extract scores and assessment from the evaluation text."""
        scores = {
            "correctness": 0,
            "completeness": 0,
            "efficiency": 0,
            "clarity": 0,
            "robustness": 0,
            "overall": 0
        }
            
        # Parse evaluation text for scores
        lines = evaluation_text.split('\n')
        assessment = ""
        in_assessment = False
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for score indicators
            lower_line = line.lower()
            
            # Check for overall assessment section
            if any(keyword in lower_line for keyword in ["overall assessment", "final assessment", "conclusion", "summary"]):
                in_assessment = True
                assessment = line + "\n"
            elif in_assessment:
                assessment += line + "\n"
                
            # Extract scores
            for criterion in scores.keys():
                if criterion.lower() in lower_line and any(char.isdigit() for char in line):
                    # Extract the score (assumed to be a number 1-10)
                    score_match = None
                    if ":" in line:
                        after_colon = line.split(":", 1)[1]
                        for word in after_colon.split():
                            if word.isdigit() and 1 <= int(word) <= 10:
                                score_match = int(word)
                                break
                    if not score_match:
                        # Look for a pattern like "7/10" or "7 out of 10"
                        if "/" in line:
                            for part in line.split():
                                if "/" in part:
                                    num, denom = part.split("/", 1)
                                    if num.isdigit() and denom.isdigit():
                                        score_match = int(num)
                                        break
                                        
                    if score_match:
                        scores[criterion] = score_match
                        
        # If we didn't find an explicit assessment, extract the last few paragraphs
        if not assessment:
            paragraphs = evaluation_text.split("\n\n")
            if paragraphs:
                assessment = paragraphs[-1]
                
        # Calculate overall score if not explicitly found
        if scores["overall"] == 0 and any(scores.values()):
            # Calculate average of other scores
            valid_scores = [s for s in scores.values() if s > 0]
            if valid_scores:
                scores["overall"] = sum(valid_scores) // len(valid_scores)
            else:
                scores["overall"] = 5  # Default middle score
                
        return scores, assessment

class ExecutiveAgent(Agent):
    """Executive agent that coordinates the multi-agent system."""
    
    def __init__(self, neocortex):
        super().__init__("Executive", neocortex)
        
        # Initialize specialized agents
        self.analyst = AnalystAgent(neocortex)
        self.creative = CreativeAgent(neocortex)
        self.logical = LogicalAgent(neocortex)
        self.synthesizer = SynthesizerAgent(neocortex)
        self.critic = CriticAgent(neocortex)
        self.implementer = ImplementerAgent(neocortex)
        self.evaluator = EvaluatorAgent(neocortex)
        
        # Agent workflow configuration
        self.workflow = {
            "analysis": {"agent": self.analyst, "next": ["creative", "logical"]},
            "creative": {"agent": self.creative, "next": ["synthesis"]},
            "logical": {"agent": self.logical, "next": ["synthesis"]},
            "synthesis": {"agent": self.synthesizer, "next": ["critic"]},
            "critic": {"agent": self.critic, "next": ["implementation"]},
            "implementation": {"agent": self.implementer, "next": ["evaluation"]},
            "evaluation": {"agent": self.evaluator, "next": ["final"]}
        }
        
        # Workspace for sharing data between agents
        self.workspace = {}
        
    def process(self, query: str) -> Dict:
        """Process a query through the multi-agent system."""
        self.state = "working"
        
        # Reset workspace
        self.workspace = {"query": query}
        
        # Create problem node in the cognitive graph
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.neocortex.cognitive_graph.add_node(problem_node)
        
        try:
            # Execute the multi-agent workflow
            self._execute_workflow()
            
            # Prepare final answer
            final_answer = self._generate_final_answer()
            
            # Update workspace with final answer
            self.workspace["final_answer"] = final_answer
            
            # Create final answer node in the cognitive graph
            final_answer_node = CognitiveNode(
                node_id="final_answer_0",
                node_type="final_answer",
                content=final_answer,
                parent_id="problem_0"
            )
            self.neocortex.cognitive_graph.add_node(final_answer_node)
            
            self.state = "done"
            return {
                "status": "success",
                "final_answer": final_answer,
                "workspace": self.workspace,
                "cognitive_graph": self.neocortex.cognitive_graph.to_dict()
            }
            
        except Exception as e:
            self.state = "error"
            return {
                "status": "error",
                "message": str(e),
                "workspace": self.workspace
            }
            
    def _execute_workflow(self):
        """Execute the agent workflow based on the configuration."""
        # Start with analysis stage
        current_stages = ["analysis"]
        completed_stages = set()
        
        while current_stages:
            next_stages = []
            
            # Process current stages
            for stage in current_stages:
                if stage in completed_stages:
                    continue
                    
                # Skip if this is the final stage
                if stage == "final":
                    continue
                    
                # Get stage configuration
                stage_config = self.workflow.get(stage)
                if not stage_config:
                    continue
                    
                # Execute agent for this stage
                agent = stage_config["agent"]
                result = agent.process(self.workspace)
                
                # Update workspace with agent result
                self.workspace.update(result)
                
                # Mark stage as completed
                completed_stages.add(stage)
                
                # Add next stages to queue
                next_stages.extend(stage_config["next"])
                
            # Set up next iteration
            current_stages = next_stages
            
    def _generate_final_answer(self) -> str:
        """Generate a final, coherent answer based on the agent outputs."""
        # Collect key components from the workspace
        synthesis = self.workspace.get("synthesis", "")
        solution = self.workspace.get("solution", "")
        implementation = self.workspace.get("implementation", "")
        evaluation = self.workspace.get("evaluation", "")
        assessment = self.workspace.get("assessment", "")
        
        # Generate a final comprehensive answer
        system_prompt = """You are the executive coordinator of a multi-agent cognitive system.
Your task is to:
1. Integrate the outputs from multiple specialized agents
2. Produce a coherent, comprehensive final answer
3. Ensure all key insights and solutions are included
4. Present the information in a clear, structured format
5. Address the original query completely and precisely

Provide a final answer that represents the collective intelligence of the system."""

        prompt = f"""
        Generate a comprehensive final answer based on these agent outputs:

        ORIGINAL QUERY:
        {self.workspace.get('query', '')}

        SYNTHESIS:
        {synthesis[:300]}... [truncated for brevity]

        SOLUTION:
        {solution[:300]}... [truncated for brevity]

        IMPLEMENTATION:
        {implementation[:300]}... [truncated for brevity]

        EVALUATION:
        {assessment}

        Create a final answer that:
        1. Addresses the original query directly and completely
        2. Incorporates the key insights from all agents
        3. Presents the solution and implementation clearly
        4. Acknowledges any limitations or areas for improvement
        5. Is well-structured and easy to understand

        Begin your answer with "### Final Comprehensive Answer:"
        """
            
        # Generate response with the specialized system prompt
        response_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=system_prompt,
            task_id="final_answer"
        )
            
        # Extract the final answer
        if "### Final Comprehensive Answer:" in response_text:
            final_answer = response_text.split("### Final Comprehensive Answer:", 1)[1].strip()
        else:
            final_answer = response_text
            
        return final_answer

class MultiAgentSystem:
    """Manager class for the multi-agent system."""
    
    def __init__(self, neocortex):
        self.neocortex = neocortex
        self.executive = ExecutiveAgent(neocortex)
        
    def process_query(self, query: str) -> Dict:
        """Process a query using the multi-agent system."""
        return self.executive.process(query)
        
    def get_agent_status(self) -> Dict:
        """Get the status of all agents in the system."""
        executive = self.executive
        return {
            "executive": executive.get_status(),
            "analyst": executive.analyst.get_status() if hasattr(executive, 'analyst') else None,
            "creative": executive.creative.get_status() if hasattr(executive, 'creative') else None,
            "logical": executive.logical.get_status() if hasattr(executive, 'logical') else None,
            "synthesizer": executive.synthesizer.get_status() if hasattr(executive, 'synthesizer') else None,
            "critic": executive.critic.get_status() if hasattr(executive, 'critic') else None,
            "implementer": executive.implementer.get_status() if hasattr(executive, 'implementer') else None,
            "evaluator": executive.evaluator.get_status() if hasattr(executive, 'evaluator') else None
        }

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

class PromptLibrary:
    """Class that contains specialized prompt templates for different reasoning tasks."""

    def __init__(self):
        # Base system prompts
        self.base_system_prompt = """You are an advanced reasoning AI assistant with exceptional problem-solving abilities.
Be methodical and clear in your thinking. Avoid redundancy, and strive for precision.
Share your reasoning process explicitly, starting reasoning steps with 'Thinking:' and
providing your final answers clearly.

For solutions, especially complex ones, begin with '### Refined Comprehensive Solution:' and
ensure completeness. Structure your answers with clear section markers and avoid truncating
your responses."""

        self.advanced_system_prompt = """You are an advanced cognitive reasoning system with exceptional problem-solving capabilities.
Follow these explicit reasoning stages when analyzing complex problems:

STAGE 1: PROBLEM DECOMPOSITION
- Break the problem into constituent parts
- Identify hidden assumptions and constraints
- Define clear objectives and success criteria

STAGE 2: MULTI-PERSPECTIVE ANALYSIS
- Examine the problem through different conceptual frameworks
- Consider mathematical, logical, empirical, and creative perspectives
- Identify domain-specific principles that apply

STAGE 3: EVIDENCE GATHERING
- Identify relevant facts and principles
- Perform necessary calculations and logical analyses
- Evaluate the reliability of different evidence sources

STAGE 4: HYPOTHESIS GENERATION
- Formulate multiple potential solutions
- Consider both conventional and unconventional approaches
- Predict outcomes for each potential solution

STAGE 5: CRITICAL EVALUATION
- Scrutinize each hypothesis for weaknesses and errors
- Apply falsification attempts to proposed solutions
- Assess tradeoffs between different approaches

STAGE 6: SYNTHESIS AND INTEGRATION
- Combine insights from multiple perspectives
- Resolve contradictions between different analyses
- Create a coherent framework that addresses all aspects

STAGE 7: SOLUTION REFINEMENT
- Optimize the chosen solution based on evaluation criteria
- Address edge cases and potential failures
- Simplify without losing essential complexity

STAGE 8: METACOGNITIVE REFLECTION
- Evaluate the reasoning process itself
- Identify potential cognitive biases in your analysis
- Consider what could be improved in future analyses

When solving problems, explicitly share your reasoning starting with 'Thinking:' and clearly 
mark your final answers with '### Refined Comprehensive Solution:'. Never truncate your responses.
"""

        # Task-specific prompts
        self.task_specific_prompts = {
            "mathematical": {
                "system_prompt": """You are a mathematical reasoning expert capable of solving complex mathematical problems.
Follow these steps for mathematical problem-solving:

1. PROBLEM FORMULATION
   - Identify the key variables, constraints, and objectives
   - Translate the problem into precise mathematical notation
   - Determine which branch of mathematics is most relevant

2. APPROACH SELECTION
   - Consider multiple solution methods (algebraic, geometric, numerical, etc.)
   - Select the most appropriate mathematical tools for the problem
   - Plan a step-by-step approach to reach the solution

3. EXECUTION
   - Apply mathematical operations with careful attention to detail
   - Maintain precise notation and track units throughout
   - Double-check calculations to avoid arithmetic errors

4. VERIFICATION
   - Test the solution against the original constraints
   - Perform dimensional analysis to ensure consistency
   - Consider edge cases and boundary conditions

5. INTERPRETATION
   - Translate the mathematical result back to the problem context
   - Explain the significance of the solution
   - Address any approximations or simplifications made

Always show your complete reasoning process. For complex problems, use 'Thinking:' 
to share your mathematical reasoning and clearly mark your final answers with 
'### Refined Comprehensive Solution:'.
""",
                "user_prompt_template": """Solve the following mathematical problem:

{query}

Provide a clear, step-by-step solution showing all calculations and reasoning. 
If there are multiple approaches, identify the most elegant or efficient solution.
"""
            },
            
            "creative": {
                "system_prompt": """You are a creative problem-solving expert skilled at generating innovative solutions.
Approach creative challenges using these guidelines:

1. DIVERGENT EXPLORATION
   - Generate a wide range of possible ideas without judgment
   - Consider unusual, metaphorical, and cross-domain approaches
   - Challenge assumptions and conventional boundaries

2. CONCEPTUAL COMBINATION
   - Connect and combine disparate ideas in novel ways
   - Look for patterns and analogies across different domains
   - Develop hybrid concepts that merge multiple approaches

3. PERSPECTIVE SHIFTING
   - View the problem from multiple stakeholder perspectives
   - Consider how the problem might be solved in different contexts
   - Apply principles from unrelated fields to the current domain

4. CONSTRAINT REFRAMING
   - Identify which constraints are rigid and which are flexible
   - Transform apparent limitations into enabling constraints
   - Find advantages in seeming disadvantages

5. ITERATIVE REFINEMENT
   - Develop promising ideas through progressive enhancement
   - Combine elements from multiple concepts into stronger solutions
   - Balance novelty with practicality and feasibility

Share your creative thinking process, starting lines with 'Thinking:' when exploring 
creative possibilities and clearly mark your final solution with '### Refined Comprehensive Solution:'.
""",
                "user_prompt_template": """Address this creative challenge:

{query}

Generate multiple innovative approaches and develop the most promising ones into 
practical, imaginative solutions. Balance originality with feasibility.
"""
            },
            
            "analytical": {
                "system_prompt": """You are an analytical reasoning expert skilled at investigating complex situations.
Follow these analytical reasoning guidelines:

1. PROBLEM DEFINITION
   - Clearly articulate the core analytical question
   - Identify implicit assumptions that might bias analysis
   - Establish clear criteria for evaluating potential conclusions

2. DATA ASSESSMENT
   - Evaluate the relevance, reliability, and sufficiency of available information
   - Identify critical information gaps and their implications
   - Distinguish between facts, inferences, and speculations

3. PATTERN RECOGNITION
   - Identify recurring themes, correlations, and anomalies
   - Recognize structural similarities to previously analyzed situations
   - Detect potential causal relationships versus coincidences

4. LOGICAL INFERENCE
   - Draw warranted conclusions based on available evidence
   - Apply deductive, inductive, and abductive reasoning as appropriate
   - Identify logical fallacies and avoid them in your own reasoning

5. ALTERNATIVE EXPLANATIONS
   - Generate multiple hypotheses that could explain the situation
   - Evaluate competing explanations using objective criteria
   - Determine which explanation has the strongest evidential support

6. CONCLUSION QUALIFICATION
   - Assign appropriate confidence levels to your conclusions
   - Acknowledge limitations and uncertainty in your analysis
   - Identify what additional information could strengthen conclusions

Share your analytical reasoning process, using 'Thinking:' when working through your 
analysis and clearly mark your final conclusions with '### Refined Comprehensive Solution:'.
""",
                "user_prompt_template": """Analyze the following situation:

{query}

Provide a thorough analysis that considers multiple interpretations of the evidence, 
identifies key patterns and relationships, and reaches well-supported conclusions. 
Address potential counterarguments and qualify your conclusions appropriately.
"""
            },
            
            "ethical": {
                "system_prompt": """You are an ethical reasoning expert capable of navigating complex moral dilemmas.
Follow these ethical reasoning guidelines:

1. STAKEHOLDER IDENTIFICATION
   - Identify all parties affected by the situation
   - Consider immediate, secondary, and tertiary impacts
   - Recognize power dynamics and vulnerable populations

2. PRINCIPLE CLARIFICATION
   - Identify relevant ethical principles (autonomy, justice, beneficence, etc.)
   - Consider applicable professional codes and ethical frameworks
   - Recognize when principles come into conflict

3. CONSEQUENCE ANALYSIS
   - Assess short and long-term outcomes of different options
   - Consider both intended and potential unintended consequences
   - Evaluate distributive effects across different stakeholders

4. PRECEDENT EVALUATION
   - Consider how similar situations have been handled
   - Assess the implications of setting new precedents
   - Evaluate consistency with established ethical standards

5. MORAL IMAGINATION
   - Consider the situation from diverse ethical perspectives
   - Develop creative solutions that might resolve apparent dilemmas
   - Imagine how the situation appears to each stakeholder

6. BALANCED JUDGMENT
   - Weigh competing considerations fairly
   - Avoid simplistic solutions to complex ethical questions
   - Acknowledge legitimate disagreement where it exists

Share your ethical reasoning process, using 'Thinking:' when working through complex 
ethical dimensions and clearly mark your final ethical analysis with 
'### Refined Comprehensive Solution:'.
""",
                "user_prompt_template": """Consider the following ethical dilemma:

{query}

Provide a nuanced ethical analysis that considers multiple perspectives, examines 
relevant principles and consequences, and offers a balanced assessment. Acknowledge 
areas of legitimate disagreement and the values underlying different positions.
"""
            },
            
            "scientific": {
                "system_prompt": """You are a scientific reasoning expert skilled at applying the scientific method.
Follow these scientific reasoning guidelines:

1. OBSERVATION & QUESTION FORMULATION
   - Clearly articulate the scientific question or phenomenon
   - Place the question in context of existing scientific knowledge
   - Identify observable, measurable aspects of the phenomenon

2. HYPOTHESIS DEVELOPMENT
   - Formulate testable hypotheses that explain observations
   - Ensure hypotheses make specific, falsifiable predictions
   - Consider multiple alternative hypotheses

3. PREDICTION GENERATION
   - Derive logical consequences that would follow if hypotheses are true
   - Specify conditions under which predictions would hold or fail
   - Identify key variables that would influence outcomes

4. EVIDENCE EVALUATION
   - Assess quality, relevance, and strength of available evidence
   - Distinguish between correlation and causation
   - Consider sample sizes, methodological rigor, and replicability

5. MODEL CONSTRUCTION
   - Develop models that explain relationships between variables
   - Balance simplicity with explanatory power
   - Address limitations and boundary conditions of models

6. INFERENCE & CONCLUSION
   - Draw warranted conclusions based on evidence and reasoning
   - Quantify uncertainty and confidence levels appropriately
   - Identify implications for broader scientific understanding

Share your scientific reasoning process, using 'Thinking:' when working through your
scientific analysis and clearly mark your conclusions with '### Refined Comprehensive Solution:'.
""",
                "user_prompt_template": """Investigate the following scientific question or phenomenon:

{query}

Apply scientific reasoning to evaluate existing evidence, consider alternative explanations, 
and reach well-supported conclusions. Clearly distinguish between established facts, 
reasonable inferences, and speculative possibilities.
"""
            }
        }
        
    def get_system_prompt(self, task_type: str = "general", complexity: str = "standard") -> str:
        """Get the appropriate system prompt based on task type and complexity."""
        if task_type in self.task_specific_prompts:
            return self.task_specific_prompts[task_type]["system_prompt"]
        elif complexity == "advanced":
            return self.advanced_system_prompt
        else:
            return self.base_system_prompt
            
    def get_user_prompt(self, query: str, task_type: str = "general") -> str:
        """Get the appropriate user prompt template and format it with the query."""
        if task_type in self.task_specific_prompts:
            template = self.task_specific_prompts[task_type]["user_prompt_template"]
            return template.format(query=query)
        return query

class ContextWindowManager:
    """Manages token usage and context window for large reasoning tasks."""
    
    def __init__(self, max_tokens_per_call: int = 8000):
        self.max_tokens_per_call = max_tokens_per_call
        self.token_usage = {}
        self.context_memory = {}
        
    def estimate_tokens(self, text: str) -> int:
        """Roughly estimate the number of tokens in a text."""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
        
    def should_split_task(self, task_id: str, input_text: str) -> bool:
        """Determine if a task should be split based on token estimates."""
        estimated_tokens = self.estimate_tokens(input_text)
        return estimated_tokens > self.max_tokens_per_call * 0.8  # 80% threshold
        
    def store_context(self, task_id: str, key: str, content: str):
        """Store context for a specific task."""
        if task_id not in self.context_memory:
            self.context_memory[task_id] = {}
        
        self.context_memory[task_id][key] = content
        
        # Update token usage
        if task_id not in self.token_usage:
            self.token_usage[task_id] = 0
        
        self.token_usage[task_id] += self.estimate_tokens(content)
        
    def get_context(self, task_id: str, key: str) -> str:
        """Retrieve context for a specific task."""
        if task_id not in self.context_memory:
            return ""
        
        return self.context_memory[task_id].get(key, "")
        
    def get_summary_context(self, task_id: str) -> str:
        """Get a summarized context for a task, optimized for token usage."""
        if task_id not in self.context_memory:
            return ""
            
        # If token count is manageable, return all context
        total_tokens = self.token_usage.get(task_id, 0)
        if total_tokens < self.max_tokens_per_call * 0.5:  # If using less than 50% of capacity
            return "\n\n".join(self.context_memory[task_id].values())
            
        # Otherwise, provide a summarized version focusing on key insights
        summary = []
        
        # Always include problem definition if available
        if "problem" in self.context_memory[task_id]:
            summary.append(f"PROBLEM DEFINITION: {self.context_memory[task_id]['problem']}")
            
        # Include latest reasoning steps with priority
        priorities = {
            "final_answer": 1,
            "solution": 2,
            "verification": 3,
            "integration": 4,
            "evidence": 5
        }
        
        # Sort context by priority
        sorted_context = sorted(
            self.context_memory[task_id].items(),
            key=lambda x: priorities.get(x[0], 99)
        )
        
        # Add as many contexts as can fit within token limit
        token_count = self.estimate_tokens("\n\n".join(summary))
        for key, content in sorted_context:
            if key == "problem":  # Already added
                continue
                
            content_tokens = self.estimate_tokens(content)
            if token_count + content_tokens < self.max_tokens_per_call * 0.7:  # Stay under 70% capacity
                summary.append(f"{key.upper()}: {content}")
                token_count += content_tokens
            else:
                # If can't fit full content, add a summary marker
                summary.append(f"{key.upper()}: [Content summarized due to length constraints]")
                
        return "\n\n".join(summary)
        
    def clear_context(self, task_id: str):
        """Clear context for a specific task."""
        if task_id in self.context_memory:
            del self.context_memory[task_id]
        
        if task_id in self.token_usage:
            del self.token_usage[task_id]

class PromptChainer:
    """Handles breaking complex problems into sub-problems and chaining their solutions."""
    
    def __init__(self, neocortex):
        self.neocortex = neocortex
        self.subproblems = {}
        self.dependencies = {}
        self.results = {}
        
    def decompose_problem(self, main_problem: str) -> List[Dict]:
        """Decompose a complex problem into simpler sub-problems."""
        prompt = f"""
        Analyze this complex problem and break it down into 2-5 distinct sub-problems:

        {main_problem}

        For each sub-problem:
        1. Provide a clear title
        2. Write a specific, focused question
        3. Explain why solving this sub-problem helps solve the main problem
        4. List any other sub-problems this depends on (if any)

        Format your response as a numbered list of sub-problems.
        """
        
        response = self.neocortex._generate_response(prompt, emit_thoughts=True)
        
        # Process the response to extract sub-problems
        # This is a simplified parsing for illustration
        lines = response.split('\n')
        subproblems = []
        current_subproblem = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for numbered items like "1.", "2.", etc.
            if line[0].isdigit() and line[1:].startswith('. '):
                if current_subproblem:
                    subproblems.append(current_subproblem)
                
                title = line[3:].strip()
                current_subproblem = {
                    "id": f"sub_{len(subproblems) + 1}",
                    "title": title,
                    "question": "",
                    "rationale": "",
                    "dependencies": []
                }
            elif current_subproblem:
                # Process additional information about the sub-problem
                if "question:" in line.lower():
                    current_subproblem["question"] = line.split(":", 1)[1].strip()
                elif "why:" in line.lower() or "rationale:" in line.lower():
                    current_subproblem["rationale"] = line.split(":", 1)[1].strip()
                elif "depends on:" in line.lower() or "dependencies:" in line.lower():
                    deps = line.split(":", 1)[1].strip()
                    if deps and deps.lower() != "none":
                        current_subproblem["dependencies"] = [d.strip() for d in deps.split(",")]
        
        # Add the last subproblem if it exists
        if current_subproblem:
            subproblems.append(current_subproblem)
            
        # Store the subproblems and their dependencies
        for sp in subproblems:
            self.subproblems[sp["id"]] = sp
            self.dependencies[sp["id"]] = sp["dependencies"]
            
        return subproblems
        
    def solve_subproblem(self, subproblem_id: str) -> str:
        """Solve an individual sub-problem."""
        if subproblem_id not in self.subproblems:
            return "Error: Sub-problem not found"
            
        subproblem = self.subproblems[subproblem_id]
        
        # Check if we've already solved this subproblem
        if subproblem_id in self.results:
            return self.results[subproblem_id]
            
        # Check if all dependencies have been solved
        dependency_results = {}
        for dep_id in self.dependencies.get(subproblem_id, []):
            if dep_id not in self.results:
                # Solve the dependency first (recursive call)
                self.solve_subproblem(dep_id)
            
            # Get the result of the dependency
            dependency_results[dep_id] = self.results.get(dep_id, "Dependency result not available")
            
        # Construct the prompt with dependency information
        prompt = f"""
        Solve this sub-problem:

        {subproblem["question"]}

        Context about this sub-problem:
        - Title: {subproblem["title"]}
        - Rationale: {subproblem["rationale"]}
        """
        
        # Add dependency results if any
        if dependency_results:
            prompt += "\nResults from dependent sub-problems:\n"
            for dep_id, result in dependency_results.items():
                dep_title = self.subproblems.get(dep_id, {}).get("title", dep_id)
                # Truncate very long results to manage token usage
                summary = result if len(result) < 500 else result[:500] + "... [truncated for brevity]"
                prompt += f"- {dep_title}: {summary}\n"
                
        prompt += "\nProvide a clear, comprehensive solution to this specific sub-problem."
        
        # Generate the solution
        solution = self.neocortex._generate_response(prompt, emit_thoughts=True)
        
        # Store the result
        self.results[subproblem_id] = solution
        
        return solution
        
    def integrate_solutions(self, main_problem: str) -> str:
        """Integrate solutions to sub-problems into a comprehensive solution for the main problem."""
        if not self.results:
            return "No sub-problem solutions to integrate"
            
        prompt = f"""
        Integrate these solutions to sub-problems into a comprehensive solution for the main problem:

        MAIN PROBLEM:
        {main_problem}

        SUB-PROBLEM SOLUTIONS:
        """
        
        for subproblem_id, solution in self.results.items():
            subproblem = self.subproblems.get(subproblem_id, {"title": subproblem_id})
            # Summarize very long solutions to manage token usage
            summary = solution if len(solution) < 800 else solution[:800] + "... [full details in sub-problem solution]"
            prompt += f"\n[{subproblem['title']}]\n{summary}\n"
            
        prompt += """
        Provide a cohesive, integrated solution to the main problem that synthesizes all the sub-problem solutions.
        Your integrated solution should:
        1. Connect all the sub-solutions logically
        2. Resolve any inconsistencies between sub-solutions
        3. Present a unified approach to the main problem
        4. Be more valuable than the sum of its parts

        Begin your solution with "### Refined Comprehensive Solution:"
        """
        
        integrated_solution = self.neocortex._generate_response(prompt, emit_thoughts=True)
        
        return integrated_solution
        
    def solve_with_chaining(self, problem: str) -> Dict:
        """Solve a complex problem using prompt chaining."""
        # Step 1: Reset state
        self.subproblems = {}
        self.dependencies = {}
        self.results = {}
        
        # Step 2: Decompose the problem
        subproblems = self.decompose_problem(problem)
        
        # Step 3: Create a cognitive graph node for the main problem
        main_problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=problem
        )
        self.neocortex.cognitive_graph.add_node(main_problem_node)
        
        # Step 4: Add subproblem nodes to the cognitive graph
        for sp in subproblems:
            subproblem_node = CognitiveNode(
                node_id=sp["id"],
                node_type="subproblem",
                content=sp["title"] + ": " + sp["question"],
                parent_id="problem_0"
            )
            self.neocortex.cognitive_graph.add_node(subproblem_node)
        
        # Step 5: Identify and solve subproblems in dependency order
        solved = set()
        while len(solved) < len(subproblems):
            progress_made = False
            
            for sp in subproblems:
                sp_id = sp["id"]
                if sp_id in solved:
                    continue
                
                # Check if all dependencies are solved
                deps_solved = all(dep in solved for dep in self.dependencies.get(sp_id, []))
                
                if deps_solved:
                    # Solve this subproblem
                    solution = self.solve_subproblem(sp_id)
                    
                    # Add solution node to cognitive graph
                    solution_node = CognitiveNode(
                        node_id=f"solution_{sp_id}",
                        node_type="solution",
                        content=solution,
                        parent_id=sp_id
                    )
                    self.neocortex.cognitive_graph.add_node(solution_node)
                    
                    solved.add(sp_id)
                    progress_made = True
            
            # If no progress was made in this iteration, there might be a dependency cycle
            if not progress_made and len(solved) < len(subproblems):
                unsolved = [sp["id"] for sp in subproblems if sp["id"] not in solved]
                print(f"Warning: Possible dependency cycle detected among sub-problems: {unsolved}")
                
                # Force solve one of the unsolved problems to break the cycle
                if unsolved:
                    forced_sp = unsolved[0]
                    print(f"Forcing solution of {forced_sp} to break dependency cycle")
                    solution = self.solve_subproblem(forced_sp)
                    
                    # Add solution node to cognitive graph
                    solution_node = CognitiveNode(
                        node_id=f"solution_{forced_sp}",
                        node_type="solution",
                        content=solution,
                        parent_id=forced_sp
                    )
                    self.neocortex.cognitive_graph.add_node(solution_node)
                    
                    solved.add(forced_sp)
                else:
                    break  # Something went wrong, exit the loop
        
        # Step 6: Integrate all solutions
        integrated_solution = self.integrate_solutions(problem)
        
        # Add integration node to cognitive graph
        integration_node = CognitiveNode(
            node_id="integration_0",
            node_type="integration",
            content="Integrated solution",
            parent_id="problem_0"
        )
        self.neocortex.cognitive_graph.add_node(integration_node)
        
        # Add final answer node to cognitive graph
        final_answer_node = CognitiveNode(
            node_id="final_answer_0",
            node_type="final_answer",
            content=integrated_solution,
            parent_id="integration_0"
        )
        self.neocortex.cognitive_graph.add_node(final_answer_node)
        
        # Return the results
        result = {
            "final_answer": integrated_solution,
            "cognitive_graph": self.neocortex.cognitive_graph.to_dict(),
            "reasoning_process": {
                "decomposition": {
                    "full_decomposition": str(subproblems),
                    "subproblems": subproblems
                },
                "perspectives": [],
                "evidence": {},
                "integration": {"full_integration": integrated_solution},
                "solution": {"full_solution": integrated_solution},
                "verification": {"full_verification": "Verification integrated into sub-problem solutions."},
                "reflection": "Problem solved through decomposition and prompt chaining."
            }
        }
        
        return result

class DynamicPromptConstructor:
    """Constructs customized prompts based on query complexity and type."""
    
    def __init__(self, prompt_library):
        self.prompt_library = prompt_library
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query to determine its complexity and type."""
        analysis = {
            "complexity": "standard",
            "task_type": "general",
            "requires_decomposition": False,
            "estimated_tokens": len(query) // 4  # Rough token estimation
        }
        
        # Determine complexity based on query length and structure
        word_count = len(query.split())
        if word_count > 200:
            analysis["complexity"] = "advanced"
        elif word_count > 50:
            analysis["complexity"] = "standard"
        else:
            analysis["complexity"] = "basic"
            
        # Check if the query might require problem decomposition
        if word_count > 150 or query.count("?") > 3 or "steps" in query.lower():
            analysis["requires_decomposition"] = True
            
        # Determine task type based on keywords and patterns
        keywords = query.lower()
        
        if any(term in keywords for term in ["calculate", "solve", "equation", "math", "computation",
                                            "algebra", "geometry", "calculus", "probability",
                                            "theorem", "formula", "proof"]):
            analysis["task_type"] = "mathematical"
            
        elif any(term in keywords for term in ["create", "design", "novel", "innovative", "idea",
                                             "imagination", "original", "brainstorm", "invent"]):
            analysis["task_type"] = "creative"
            
        elif any(term in keywords for term in ["analyze", "investigate", "examine", "evaluate",
                                             "assess", "critique", "comparison", "pattern"]):
            analysis["task_type"] = "analytical"
            
        elif any(term in keywords for term in ["ethical", "moral", "right", "wrong", "should",
                                             "obligation", "responsibility", "virtue", "justice"]):
            analysis["task_type"] = "ethical"
            
        elif any(term in keywords for term in ["scientific", "experiment", "hypothesis", "theory",
                                             "observation", "empirical", "research", "evidence"]):
            analysis["task_type"] = "scientific"
            
        return analysis
        
    def construct_prompt(self, query: str, task_id: str = None) -> Dict[str, str]:
        """Construct a customized prompt based on query analysis."""
        # Analyze the query
        analysis = self.analyze_query(query)
        
        # Get the appropriate system prompt
        system_prompt = self.prompt_library.get_system_prompt(
            task_type=analysis["task_type"],
            complexity=analysis["complexity"]
        )
        
        # Get the appropriate user prompt
        user_prompt = self.prompt_library.get_user_prompt(
            query=query,
            task_type=analysis["task_type"]
        )
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "analysis": analysis
        }

class NeoCortex:
    """Main class for the advanced cognitive architecture."""

    def __init__(self, model_name: str = "deepseek/deepseek-chat-v3", temperature: float = 0.2):
        """Initialize NeoCortex with specific model settings."""
        self.model_name = model_name
        self.temperature = temperature
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        # Updated API key
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
        self.max_tokens = 8000  # Substantially increased token generation limit
        self.concurrent_requests = True  # Enable parallel processing
        self.reasoning_depth = "balanced"  # Options: "minimal", "balanced", "thorough"

        # Initialize session
        self.session = requests.Session()

        # Thread pool for parallel requests
        self.executor = ThreadPoolExecutor(max_workers=3)

        # This will be assigned by the ReasoningThread
        self.thought_generated = None
        
        # Initialize enhanced components
        self.prompt_library = PromptLibrary()
        self.context_manager = ContextWindowManager(max_tokens_per_call=self.max_tokens)
        self.prompt_constructor = DynamicPromptConstructor(self.prompt_library)
        self.prompt_chainer = PromptChainer(self)
        
        # Initialize multi-agent system
        self.multi_agent_system = MultiAgentSystem(self)

    def _generate_response(self, prompt: str, emit_thoughts=False, system_prompt=None, task_id=None) -> str:
        """Generate a response using the OpenRouter API with retry logic for rate limits."""
        # Check cache first for faster responses
        cache_key = f"{prompt}{self.model_name}{self.temperature}"
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

                # Use provided system prompt or default to the enhanced system prompt
                if system_prompt is None:
                    system_prompt = self.prompt_library.advanced_system_prompt

                # Create a more efficient payload with max_tokens limit
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "extra_params": {
                        "provider_order": ["deepseek"]
                    }
                }

                # Add timeout to prevent hanging - increased for larger responses
                response = self.session.post(self.api_url, headers=headers, json=payload, timeout=120)

                if response.status_code == 200:
                    result = response.json()
                    # Add detailed error handling for API response structure
                    if "choices" not in result:
                        error_msg = f"API response missing 'choices' key. Response: {result}"
                        print(error_msg)
                        return f"Error: {error_msg}"
                    
                    if len(result["choices"]) == 0:
                        error_msg = "API returned empty choices array"
                        print(error_msg)
                        return f"Error: {error_msg}"
                    
                    if "message" not in result["choices"][0]:
                        error_msg = f"API response missing 'message' in choice. Response: {result}"
                        print(error_msg)
                        return f"Error: {error_msg}"
                    
                    if "content" not in result["choices"][0]["message"]:
                        error_msg = f"API response missing 'content' in message. Response: {result}"
                        print(error_msg)
                        return f"Error: {error_msg}"
                        
                    content = result["choices"][0]["message"]["content"]

                    # Check for truncated response
                    if self._is_truncated(content):
                        # Try to generate a completion with instructions to be more concise
                        completion = self._generate_completion(content, prompt)
                        if completion:
                            content = content + "\n\n" + completion

                    # Extract thinking portions for realtime reasoning display
                    if emit_thoughts and hasattr(self, 'thought_generated') and self.thought_generated:
                        # Extract all lines starting with "Thinking:"
                        thoughts = []
                        lines = content.split('\n')
                        for line in lines:
                            if line.strip().startswith("Thinking:"):
                                thought = line.strip()[9:].strip()  # Remove "Thinking:" prefix
                                thoughts.append(thought)
                                # Emit the thought
                                self.thought_generated.emit(thought)

                    # Store context if task_id is provided
                    if task_id:
                        # Extract key insights to store in context
                        key_insights = content
                        if "### Refined Comprehensive Solution:" in content:
                            key_insights = content.split("### Refined Comprehensive Solution:")[1].strip()
                        self.context_manager.store_context(task_id, "latest_insight", key_insights)

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

    def _generate_completion(self, truncated_content: str, original_prompt: str) -> str:
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

        # Use a different cache key for completions
        cache_key = f"completion{original_prompt}{self.model_name}{self.temperature}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Use fewer tokens for the completion
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are an assistant that concisely completes truncated responses."},
                    {"role": "user", "content": completion_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": min(4000, self.max_tokens // 2),  # Use fewer tokens for completion
                "extra_params": {
                    "provider_order": ["deepseek"]
                }
            }

            response = self.session.post(self.api_url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                # Add error handling for API response structure
                if "choices" not in result or len(result["choices"]) == 0 or "message" not in result["choices"][0] or "content" not in result["choices"][0]["message"]:
                    return "\n\n[Note: The response appears to be truncated, but a completion couldn't be generated due to API response format.]"
                
                completion = result["choices"][0]["message"]["content"]

                # Cache the completion
                self.response_cache[cache_key] = completion
                return f"\n\n[Note: The previous response appeared to be truncated. Here's the completion:]\n\n{completion}"

        except Exception as e:
            print(f"Error generating completion: {str(e)}")
            return "\n\n[Note: The response appears to be truncated, but a completion couldn't be generated.]"

        return ""

    def _analyze_problem(self, query: str) -> Dict:
        """Analyze the problem structure."""
        # Use the dynamic prompt constructor to create an appropriate prompt
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="problem_analysis")
        
        prompt = f"""
        Analyze the following problem using a structured approach:

        {query}

        Break it down into its core components and subproblems. Identify key concepts, constraints, and goals.
        Provide a systematic decomposition that will enable thorough problem-solving.
        """

        # Generate response with the constructed system prompt
        response_text = self._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="problem_analysis"
        )

        # Create a problem node in the cognitive graph
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)

        # Store the problem in the context manager
        self.context_manager.store_context("main", "problem", query)
        self.context_manager.store_context("main", "decomposition", response_text)

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
        # Use the dynamic prompt constructor to create an appropriate prompt
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="perspectives")
        
        prompt = f"""
        Consider the following problem from multiple conceptual frameworks and perspectives:

        {query}

        Problem decomposition:
        {decomposition["full_decomposition"]}

        Generate at least 3 distinct perspectives or approaches to this problem, each with:
        1. A distinct conceptual framework or mental model
        2. Key insights this perspective provides
        3. Potential limitations or blind spots of this perspective
        4. How this perspective would approach problem-solving

        Ensure the perspectives are truly distinct and offer complementary views of the problem.
        """

        # Generate response with the specialized system prompt
        response_text = self._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="perspectives"
        )

        # Store perspectives in context manager
        self.context_manager.store_context("main", "perspectives", response_text)

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

            # Get appropriate prompt for this evidence-gathering task
            constructed_prompt = self.prompt_constructor.construct_prompt(
                subproblem_desc, 
                task_id=f"evidence_{subproblem_id}"
            )

            prompt = f"""
            Analyze this aspect of the problem to gather comprehensive evidence:

            {subproblem_desc}

            Original problem: {query}

            Gather evidence using these approaches:
            1. FACTUAL EVIDENCE: Identify relevant facts, principles, and established knowledge
            2. LOGICAL ANALYSIS: Apply deductive and inductive reasoning
            3. COMPUTATIONAL EVIDENCE: Perform any necessary calculations or algorithmic analyses
            4. EMPIRICAL CONSIDERATIONS: Consider what empirical observations would be relevant

            Synthesize all evidence types into a cohesive analysis.
            """

            # Generate response with the specialized system prompt
            response_text = self._generate_response(
                prompt, 
                emit_thoughts=True, 
                system_prompt=constructed_prompt["system_prompt"],
                task_id=f"evidence_{subproblem_id}"
            )

            # Store evidence in context manager
            self.context_manager.store_context("main", f"evidence_{subproblem_id}", response_text)

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
        # Get current context for integration
        context_summary = self.context_manager.get_summary_context("main")
        
        # Use the dynamic prompt constructor for the integration task
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="integration")
        
        prompt = f"""
        Integrate multiple perspectives and evidence into a coherent understanding of this problem:

        PROBLEM: {query}

        CONTEXT FROM PREVIOUS ANALYSIS:
        {context_summary}

        Your task is to synthesize all perspectives and evidence into a unified understanding that:
        1. Resolves contradictions between different perspectives
        2. Weighs evidence appropriately based on relevance and strength
        3. Creates a cohesive framework that addresses all aspects of the problem
        4. Identifies key insights that emerge from the integration process

        Provide a thorough integration that sets the stage for an effective solution.
        """

        # Generate response with the specialized system prompt
        response_text = self._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="integration"
        )

        # Store integration in context manager
        self.context_manager.store_context("main", "integration", response_text)

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
        # Get current context for solution generation
        context_summary = self.context_manager.get_summary_context("main")
        
        # Use the dynamic prompt constructor for the solution task
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="solution")
        
        prompt = f"""
        Generate a comprehensive solution for this problem based on all previous analysis:

        PROBLEM: {query}

        INTEGRATED UNDERSTANDING:
        {integration['full_integration']}

        CONTEXT FROM PREVIOUS ANALYSIS:
        {context_summary}

        Your solution should:
        1. Address all aspects of the problem identified in the decomposition
        2. Incorporate insights from multiple perspectives
        3. Be supported by the evidence gathered
        4. Include specific, actionable steps or implementations
        5. Consider potential challenges and how to overcome them

        Begin your solution with "### Refined Comprehensive Solution:" 
        and ensure all code blocks and sections are complete. Do not truncate or abbreviate your response.
        """

        # Generate response with the specialized system prompt
        response_text = self._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="solution"
        )

        # Store solution in context manager
        self.context_manager.store_context("main", "solution", response_text)

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
        # Use the dynamic prompt constructor for the verification task
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="verification")
        
        prompt = f"""
        Critically evaluate and verify this solution to the original problem:

        PROBLEM: {query}

        PROPOSED SOLUTION:
        {solution['full_solution']}

        Your verification should:
        1. Rigorously test the solution against all requirements and constraints
        2. Identify any weaknesses, errors, or limitations
        3. Assess edge cases and potential failure modes
        4. Evaluate efficiency, elegance, and practicality
        5. Suggest specific improvements to address any identified issues

        Be thorough and constructive in your verification.
        """

        # Generate response with the specialized system prompt
        response_text = self._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="verification"
        )

        # Store verification in context manager
        self.context_manager.store_context("main", "verification", response_text)

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
        # Use the dynamic prompt constructor for the final answer
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="final_answer")
        
        prompt = f"""
        Generate a refined, final solution based on the original solution and verification:

        PROBLEM: {query}

        ORIGINAL SOLUTION:
        {solution['full_solution']}

        VERIFICATION AND IMPROVEMENT SUGGESTIONS:
        {verification['full_verification']}

        Your refined final solution should:
        1. Incorporate all valid improvements identified in verification
        2. Address any weaknesses or limitations that were found
        3. Be complete, precise, and practical
        4. Include any necessary code, algorithms, or specific steps
        5. Represent your best possible answer to the original problem

        Begin your refined solution with "### Refined Comprehensive Solution:" 
        and ensure all code blocks and sections are complete.
        """

        # Generate response with the specialized system prompt
        response_text = self._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="final_answer"
        )

        # Store final answer in context manager
        self.context_manager.store_context("main", "final_answer", response_text)

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
        # Use the dynamic prompt constructor for metacognitive reflection
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="reflection")
        
        # Get a summary of the reasoning process from context manager
        context_summary = self.context_manager.get_summary_context("main")
        
        prompt = f"""
        Perform a metacognitive reflection on the reasoning process used to solve this problem:

        PROBLEM: {query}

        REASONING PROCESS SUMMARY:
        {context_summary}

        Your metacognitive reflection should:
        1. Evaluate the effectiveness of the overall reasoning strategy
        2. Identify strengths and weaknesses in the approach
        3. Detect potential cognitive biases that influenced the analysis
        4. Consider alternative approaches that could have been used
        5. Extract generalizable principles or lessons for future problem-solving

        This reflection should demonstrate critical awareness of the reasoning process itself.
        """

        # Generate response with the specialized system prompt
        response_text = self._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="reflection"
        )

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
        # Use the dynamic prompt constructor to analyze the query
        analysis = self.prompt_constructor.analyze_query(query)
        
        # Consider a query simple if it's short, basic complexity, and doesn't require decomposition
        return (analysis["complexity"] == "basic" and 
                not analysis["requires_decomposition"] and
                len(query.split()) < 30)

    def _fast_response(self, query: str, show_work: bool = True) -> Dict:
        """Generate a fast response for simple queries."""
        # Use the dynamic prompt constructor to create an appropriate prompt
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="fast_response")
        
        prompt = f"""Provide a direct, efficient answer to this question: {query}

If the question is straightforward, just give the answer without elaborate explanation.
If the question requires some reasoning, briefly show your work.
"""

        # Generate response with the appropriate system prompt
        response = self._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="fast_response"
        )

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
        # Use the dynamic prompt constructor to analyze the query
        analysis = self.prompt_constructor.analyze_query(query)
        
        # If this query seems appropriate for problem decomposition, use prompt chaining
        if analysis["requires_decomposition"]:
            return self.prompt_chainer.solve_with_chaining(query)
            
        # Otherwise, use parallel processing of key reasoning components
        
        # Reset cognitive graph for new problem
        self.cognitive_graph = CognitiveGraph()
        
        # Reset context manager for new problem
        self.context_manager.clear_context("main")
        
        # Create problem node
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        # Store the problem
        self.context_manager.store_context("main", "problem", query)
        
        # Submit tasks to thread pool for parallel processing
        
        # Task 1: Problem decomposition and perspective generation (combined)
        decomp_prompt = f"""
        Analyze this problem from multiple angles:

        {query}

        First, decompose the problem into core components.
        Then, provide 2-3 distinct perspectives on approaching this problem.
        """
        
        decomp_future = self.executor.submit(
            self._generate_response,
            decomp_prompt,
            True,  # emit_thoughts
            self.prompt_constructor.construct_prompt(query, task_id="decomp_perspectives")["system_prompt"],
            "decomp_perspectives"
        )
        
        # Task 2: Direct solution generation
        solution_prompt = f"""
        Solve this problem directly and efficiently:

        {query}

        Provide a comprehensive solution that addresses all aspects of the problem.
        Begin your solution with "### Refined Comprehensive Solution:"
        """
        
        solution_future = self.executor.submit(
            self._generate_response,
            solution_prompt,
            True,  # emit_thoughts
            self.prompt_constructor.construct_prompt(query, task_id="direct_solution")["system_prompt"],
            "direct_solution"
        )
        
        # Collect results as they complete
        decomp_result = decomp_future.result()
        solution_result = solution_future.result()
        
        # Store results in context manager
        self.context_manager.store_context("main", "decomposition", decomp_result)
        self.context_manager.store_context("main", "solution", solution_result)
        
        # Task 3: Verification and refinement (sequential after getting solution)
        verification_prompt = f"""
        Verify and improve this solution:

        PROBLEM: {query}

        PROPOSED SOLUTION:
        {solution_result}

        Critically evaluate the solution, identify any issues, and suggest improvements.
        """
        
        verification_result = self._generate_response(
            verification_prompt,
            True,  # emit_thoughts
            self.prompt_constructor.construct_prompt(query, task_id="verification")["system_prompt"],
            "verification"
        )
        
        self.context_manager.store_context("main", "verification", verification_result)
        
        # Task 4: Generate final answer based on verification
        final_prompt = f"""
        Create a refined final solution incorporating verification feedback:

        PROBLEM: {query}

        ORIGINAL SOLUTION:
        {solution_result}

        VERIFICATION AND IMPROVEMENTS:
        {verification_result}

        Provide a comprehensive final solution that incorporates the improvements.
        Begin with "### Refined Comprehensive Solution:"
        """
        
        final_answer = self._generate_response(
            final_prompt,
            True,  # emit_thoughts
            self.prompt_constructor.construct_prompt(query, task_id="final_answer")["system_prompt"],
            "final_answer"
        )
        
        self.context_manager.store_context("main", "final_answer", final_answer)
        
        # Build cognitive graph nodes for UI
        
        # Add decomposition node
        decomp_node = CognitiveNode(
            node_id="decomposition_0",
            node_type="subproblem",
            content="Problem decomposition",
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(decomp_node)
        
        # Add perspective node
        perspective_node = CognitiveNode(
            node_id="perspective_0",
            node_type="perspective",
            content="Multiple perspectives",
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(perspective_node)
        
        # Add solution node
        solution_node = CognitiveNode(
            node_id="solution_0",
            node_type="solution",
            content=solution_result,
            parent_id="problem_0"
        )
        self.cognitive_graph.add_node(solution_node)
        
        # Add verification node
        verification_node = CognitiveNode(
            node_id="verification_0",
            node_type="verification",
            content=verification_result,
            parent_id="solution_0"
        )
        self.cognitive_graph.add_node(verification_node)
        
        # Add final answer node
        final_answer_node = CognitiveNode(
            node_id="final_answer_0",
            node_type="final_answer",
            content=final_answer,
            parent_id="verification_0"
        )
        self.cognitive_graph.add_node(final_answer_node)
        
        # Construct the result
        result = {
            "final_answer": final_answer,
            "cognitive_graph": self.cognitive_graph.to_dict()
        }
        
        # Include detailed reasoning process if requested
        if show_work:
            result["reasoning_process"] = {
                "decomposition": {"full_decomposition": decomp_result},
                "perspectives": [{"name": "Parallel Processing", "description": "Perspectives processed in parallel with solution"}],
                "evidence": {"main": {"synthesis": "Evidence gathered during parallel processing"}},
                "integration": {"full_integration": "Integration performed during solution verification"},
                "solution": {"full_solution": solution_result},
                "verification": {"full_verification": verification_result},
                "reflection": "Problem solved using parallel processing for efficiency."
            }
            
        return result

    def _minimal_reasoning(self, query: str, show_work: bool = True) -> Dict:
        """Generate a minimalistic reasoning approach with fewer API calls."""
        # Use the dynamic prompt constructor to create an appropriate prompt
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="minimal_reasoning")
        
        # Single API call for both analysis and solution
        prompt = f"""
        Analyze and solve the following problem with efficient reasoning:

        {query}

        Follow these steps in your response:
        1. First, briefly decompose the problem into its core components
        2. Then, outline your approach to solving it
        3. Finally, provide a comprehensive solution

        Be efficient but thorough, focusing on the most important aspects of the problem.
        Begin your final solution with "### Refined Comprehensive Solution:"
        """

        # Generate response with the specialized system prompt
        response = self._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="minimal_reasoning"
        )

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
    
    def solve_with_agents(self, query: str, show_work: bool = True) -> Dict:
        """Solve a problem using the multi-agent system."""
        # Reset cognitive graph for new problem
        self.cognitive_graph = CognitiveGraph()
        
        # Process with multi-agent system
        result = self.multi_agent_system.process_query(query)
        
        if result["status"] != "success":
            # If multi-agent processing failed, fall back to regular solver
            print(f"Multi-agent system failed: {result.get('message', 'Unknown error')}. Falling back to standard solver.")
            return self.solve(query, show_work)
            
        # Return the result from the multi-agent system
        return {
            "final_answer": result["final_answer"],
            "cognitive_graph": result["cognitive_graph"],
            "reasoning_process": {
                "decomposition": {"full_decomposition": "Problem processed by multi-agent system."},
                "perspectives": [],
                "evidence": {},
                "integration": {"full_integration": "Integration performed by Synthesizer agent."},
                "solution": {"full_solution": result["final_answer"]},
                "verification": {"full_verification": "Verification performed by Critic and Evaluator agents."},
                "reflection": "Problem solved using coordinated multi-agent system."
            },
            "multi_agent_workspace": result["workspace"]
        }

    def solve(self, query: str, show_work: bool = True, use_agents: bool = True) -> Dict:
        """Solve a problem using the cognitive architecture - optimized for speed."""
        # If agent-based processing is enabled, use the multi-agent system
        if use_agents and not self._is_simple_query(query):
            return self.solve_with_agents(query, show_work)
            
        # Use the dynamic prompt constructor to analyze the query
        analysis = self.prompt_constructor.analyze_query(query)
        
        # Check if this is a simple query that doesn't need the full reasoning pipeline
        if self._is_simple_query(query):
            return self._fast_response(query, show_work)

        # Reset cognitive graph for new problem
        self.cognitive_graph = CognitiveGraph()
        
        # Reset context manager for new problem
        self.context_manager.clear_context("main")

        if self.reasoning_depth == "minimal" or len(query.split()) < 15:
            # Super fast mode - single API call for simple to moderate queries
            return self._minimal_reasoning(query, show_work)

        # Use parallel API calls for faster processing if enabled
        if self.concurrent_requests:
            return self._solve_parallel(query, show_work)

        # If query seems to require decomposition into subproblems, use prompt chaining
        if analysis["requires_decomposition"] and len(query.split()) > 100:
            return self.prompt_chainer.solve_with_chaining(query)

        # Standard optimized approach with sequential API calls
        # Step 1: Analyze problem and generate perspectives (combined)
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="analysis")
        
        combined_prompt = f"""
        Analyze the following problem in a concise manner:

        {query}

        First, break down the problem into core components.
        Then, provide 2-3 different perspectives on this problem.
        Keep your analysis focused and brief.
        """

        combined_response = self._generate_response(
            combined_prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="analysis"
        )

        # Create problem node in the cognitive graph
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=query
        )
        self.cognitive_graph.add_node(problem_node)
        
        # Store problem in context manager
        self.context_manager.store_context("main", "problem", query)
        self.context_manager.store_context("main", "analysis", combined_response)

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
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="solution")
        
        solution_prompt = f"""
        Solve the following problem concisely:

        {query}

        Initial analysis: {combined_response}

        Provide a direct, efficient solution. Focus on practicality and clarity.
        Begin your solution with "### Refined Comprehensive Solution:"
        and ensure all sections and code blocks are complete.
        """

        solution_response = self._generate_response(
            solution_prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="solution"
        )
        
        # Store solution in context manager
        self.context_manager.store_context("main", "solution", solution_response)

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
        constructed_prompt = self.prompt_constructor.construct_prompt(query, task_id="verification")
        
        verification_prompt = f"""
        Quickly verify and improve this solution:

        Problem: {query}

        Solution: {solution_response}

        Identify any key weaknesses and provide a refined final answer.
        Begin your refined solution with "### Refined Comprehensive Solution:"
        """

        verification_response = self._generate_response(
            verification_prompt, 
            emit_thoughts=True, 
            system_prompt=constructed_prompt["system_prompt"],
            task_id="verification"
        )
        
        # Store verification in context manager
        self.context_manager.store_context("main", "verification", verification_response)

        # Extract final answer (second half of the verification response)
        response_parts = verification_response.split("### Refined Comprehensive Solution:", 1)
        if len(response_parts) > 1:
            final_answer = "### Refined Comprehensive Solution:" + response_parts[1].strip()
        else:
            response_parts = verification_response.split("Refined final answer:", 1)
            if len(response_parts) > 1:
                final_answer = response_parts[1].strip()
            else:
                final_answer = verification_response
                
        # Store final answer in context manager
        self.context_manager.store_context("main", "final_answer", final_answer)

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
# GUI Classes for Multi-Agent System
# ============================================================================

# Agent Status Widget
class AgentStatusWidget(QWidget):
    """Widget for displaying the status of agents in the multi-agent system."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Multi-Agent System Status")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Status grid
        self.status_grid = QTreeWidget()
        self.status_grid.setHeaderLabels(["Agent", "Status", "Knowledge"])
        self.status_grid.setRootIsDecorated(False)
        self.status_grid.setAlternatingRowColors(True)
        layout.addWidget(self.status_grid)
        
        # Add default agents
        self.add_agent_item("Executive", "idle", 0)
        self.add_agent_item("Analyst", "idle", 0)
        self.add_agent_item("Creative", "idle", 0)
        self.add_agent_item("Logical", "idle", 0)
        self.add_agent_item("Synthesizer", "idle", 0)
        self.add_agent_item("Critic", "idle", 0)
        self.add_agent_item("Implementer", "idle", 0)
        self.add_agent_item("Evaluator", "idle", 0)
        
        # Resize columns to content
        for i in range(3):
            self.status_grid.resizeColumnToContents(i)
            
        self.setLayout(layout)
        
    def add_agent_item(self, name, status, knowledge_size):
        item = QTreeWidgetItem(self.status_grid)
        item.setText(0, name)
        item.setText(1, status)
        item.setText(2, str(knowledge_size))
        
        # Set status-dependent styling
        if status == "working":
            item.setBackground(1, QColor("#FFD700"))  # Gold for working
        elif status == "done":
            item.setBackground(1, QColor("#90EE90"))  # Light green for done
        elif status == "error":
            item.setBackground(1, QColor("#FFA07A"))  # Light salmon for error
            
    def update_status(self, agent_status):
        """Update the status display with current agent statuses."""
        self.status_grid.clear()
        
        for agent_name, status in agent_status.items():
            if status:
                self.add_agent_item(
                    status.get("name", agent_name),
                    status.get("state", "unknown"),
                    status.get("knowledge_size", 0)
                )
                
        # Resize columns to content
        for i in range(3):
            self.status_grid.resizeColumnToContents(i)

# Agent Interaction Visualization
class AgentInteractionViz(QWidget):
    """Widget for visualizing interactions between agents."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.interactions = []
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Agent Interactions")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Canvas for network visualization
        self.figure = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def add_interaction(self, from_agent, to_agent, data_type, timestamp=None):
        """Add an interaction between agents."""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.interactions.append({
            "from": from_agent,
            "to": to_agent,
            "type": data_type,
            "timestamp": timestamp
        })
        
        self.redraw()
        
    def redraw(self):
        """Redraw the interaction network."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all agent nodes
        agents = set()
        for interaction in self.interactions:
            agents.add(interaction["from"])
            agents.add(interaction["to"])
            
        for agent in agents:
            G.add_node(agent)
            
        # Add edges for interactions
        edge_labels = {}
        for interaction in self.interactions:
            from_agent = interaction["from"]
            to_agent = interaction["to"]
            data_type = interaction["type"]
            
            # Add or increment edge weight
            if G.has_edge(from_agent, to_agent):
                G[from_agent][to_agent]["weight"] += 1
            else:
                G.add_edge(from_agent, to_agent, weight=1)
                
            # Update edge label
            edge_key = (from_agent, to_agent)
            if edge_key in edge_labels:
                if data_type not in edge_labels[edge_key]:
                    edge_labels[edge_key] += f", {data_type}"
            else:
                edge_labels[edge_key] = data_type
                
        # Get positions for nodes
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color="#7B68EE",  # Medium slate blue
            node_size=700,
            alpha=0.8
        )
        
        # Draw edges with varying thickness based on weight
        edge_widths = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.5,
            edge_color="#555555",
            arrows=True,
            arrowsize=15
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_family="sans-serif",
            font_color="white"
        )
        
        # Draw edge labels (simplified to avoid cluttering)
        simplified_labels = {}
        for edge, label in edge_labels.items():
            if len(label) > 10:
                simplified_labels[edge] = label[:10] + "..."
            else:
                simplified_labels[edge] = label
                
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=simplified_labels,
            font_size=8
        )
        
        ax.set_title("Agent Interaction Network")
        ax.axis("off")
        
        self.canvas.draw()
        
    def clear(self):
        """Clear all interactions."""
        self.interactions = []
        self.redraw()

# Agent Workflow Visualization
class AgentWorkflowWidget(QWidget):
    """Widget for visualizing the multi-agent workflow."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Multi-Agent Workflow")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Canvas for workflow visualization
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        layout.addWidget(self.view)
        
        # Create default workflow
        self.create_default_workflow()
        
        self.setLayout(layout)
        
    def create_default_workflow(self):
        """Create default multi-agent workflow visualization."""
        self.scene.clear()
        
        # Define agent nodes and their positions
        agents = [
            {"name": "Executive", "x": 0, "y": 0, "color": "#FFD700"},  # Gold
            {"name": "Analyst", "x": -150, "y": 100, "color": "#87CEFA"},  # Light sky blue
            {"name": "Creative", "x": -75, "y": 100, "color": "#FFA07A"},  # Light salmon
            {"name": "Logical", "x": 75, "y": 100, "color": "#90EE90"},  # Light green
            {"name": "Synthesizer", "x": 0, "y": 200, "color": "#D8BFD8"},  # Thistle
            {"name": "Critic", "x": 0, "y": 300, "color": "#F08080"},  # Light coral
            {"name": "Implementer", "x": 0, "y": 400, "color": "#ADD8E6"},  # Light blue
            {"name": "Evaluator", "x": 0, "y": 500, "color": "#FAFAD2"}   # Light goldenrod
        ]
        
        # Define connections between agents
        connections = [
            {"from": "Executive", "to": "Analyst"},
            {"from": "Analyst", "to": "Creative"},
            {"from": "Analyst", "to": "Logical"},
            {"from": "Creative", "to": "Synthesizer"},
            {"from": "Logical", "to": "Synthesizer"},
            {"from": "Synthesizer", "to": "Critic"},
            {"from": "Critic", "to": "Implementer"},
            {"from": "Implementer", "to": "Evaluator"},
            {"from": "Evaluator", "to": "Executive"}
        ]
        
        # Add agent nodes
        agent_nodes = {}
        for agent in agents:
            # Create ellipse
            ellipse = QGraphicsEllipseItem(-40, -25, 80, 50)
            ellipse.setBrush(QColor(agent["color"]))
            ellipse.setPen(QPen(Qt.black, 1))
            ellipse.setPos(agent["x"], agent["y"])
            
            # Add text
            text = QGraphicsTextItem(agent["name"])
            text.setDefaultTextColor(Qt.black)
            text_rect = text.boundingRect()
            text.setPos(
                agent["x"] - text_rect.width() / 2,
                agent["y"] - text_rect.height() / 2
            )
            
            # Add to scene
            self.scene.addItem(ellipse)
            self.scene.addItem(text)
            
            agent_nodes[agent["name"]] = (agent["x"], agent["y"])
            
        # Add connections
        for conn in connections:
            from_pos = agent_nodes[conn["from"]]
            to_pos = agent_nodes[conn["to"]]
            
            # Create arrow line
            line = QGraphicsLineItem(
                from_pos[0], from_pos[1],
                to_pos[0], to_pos[1]
            )
            line.setPen(QPen(Qt.black, 2))
            
            # Add arrowhead
            angle = math.atan2(to_pos[1] - from_pos[1], to_pos[0] - from_pos[0])
            arrowhead_size = 10
            arrowhead = QPolygonF([
                QPointF(to_pos[0], to_pos[1]),
                QPointF(
                    to_pos[0] - arrowhead_size * math.cos(angle - math.pi/6),
                    to_pos[1] - arrowhead_size * math.sin(angle - math.pi/6)
                ),
                QPointF(
                    to_pos[0] - arrowhead_size * math.cos(angle + math.pi/6),
                    to_pos[1] - arrowhead_size * math.sin(angle + math.pi/6)
                ),
                QPointF(to_pos[0], to_pos[1])
            ])
            arrowhead_item = QGraphicsPolygonItem(arrowhead)
            arrowhead_item.setBrush(Qt.black)
            
            # Add to scene
            self.scene.addItem(line)
            self.scene.addItem(arrowhead_item)
            
        # Set scene rect to ensure all elements are visible
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        
    def update_workflow_status(self, agent_status):
        """Update the workflow visualization with current agent statuses."""
        # This would highlight the currently active agents
        pass

# Multi-Agent System Tab
class MultiAgentSystemTab(QWidget):
    """Tab for the Multi-Agent System in the GUI."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Multi-agent controls
        controls_layout = QHBoxLayout()
        
        self.enable_agents_check = QCheckBox("Enable Multi-Agent System")
        self.enable_agents_check.setChecked(True)
        self.enable_agents_check.setToolTip(
            "Enable the multi-agent architecture for more thorough and diverse reasoning"
        )
        controls_layout.addWidget(self.enable_agents_check)
        
        self.debug_mode_check = QCheckBox("Debug Mode")
        self.debug_mode_check.setToolTip(
            "Show detailed agent interactions and intermediate outputs"
        )
        controls_layout.addWidget(self.debug_mode_check)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Add splitter for agent status and visualizations
        splitter = QSplitter(Qt.Vertical)
        
        # Agent status widget
        self.agent_status_widget = AgentStatusWidget()
        splitter.addWidget(self.agent_status_widget)
        
        # Tab widget for visualizations
        viz_tabs = QTabWidget()
        
        # Agent interaction network
        self.agent_interaction_viz = AgentInteractionViz()
        viz_tabs.addTab(self.agent_interaction_viz, "Interaction Network")
        
        # Agent workflow visualization
        self.agent_workflow_widget = AgentWorkflowWidget()
        viz_tabs.addTab(self.agent_workflow_widget, "Workflow Diagram")
        
        splitter.addWidget(viz_tabs)
        
        # Set initial splitter sizes
        splitter.setSizes([200, 400])
        
        layout.addWidget(splitter)
        
        self.setLayout(layout)
        
    def is_enabled(self):
        """Check if the multi-agent system is enabled."""
        return self.enable_agents_check.isChecked()
        
    def is_debug_mode(self):
        """Check if debug mode is enabled."""
        return self.debug_mode_check.isChecked()
        
    def update_agent_status(self, agent_status):
        """Update the agent status widget with current status."""
        self.agent_status_widget.update_status(agent_status)
        self.agent_workflow_widget.update_workflow_status(agent_status)
        
    def add_interaction(self, from_agent, to_agent, data_type):
        """Add an interaction to the visualization."""
        self.agent_interaction_viz.add_interaction(from_agent, to_agent, data_type)

# Add Multi-Agent System tab to the GUI
def add_multi_agent_tab(self):
    """Add the Multi-Agent System tab to the left panel."""
    self.multi_agent_tab = MultiAgentSystemTab()
    
    # Find the left tabs widget
    left_tabs = None
    for widget in self.findChildren(QTabWidget):
        if hasattr(widget, 'count') and widget.count() > 0:
            if widget.tabText(0) == "Modules":
                left_tabs = widget
                break
                
    if left_tabs:
        left_tabs.addTab(self.multi_agent_tab, "Multi-Agent")
        
        # Connect multi-agent system status updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_agent_status)
        self.timer.start(1000)  # Update every second

# Add update_agent_status method to NeoCortexGUI
def update_agent_status(self):
    """Update the multi-agent system status display."""
    if hasattr(self, 'multi_agent_tab') and hasattr(self, 'neocortex'):
        if hasattr(self.neocortex, 'multi_agent_system'):
            agent_status = self.neocortex.multi_agent_system.get_agent_status()
            self.multi_agent_tab.update_agent_status(agent_status)


# ============================================================================
# Other GUI Components
# ============================================================================

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
            self.thought_generated.emit("Beginning analysis of the problem...")

            # Step 1: Analyze the problem structure
            self.thought_generated.emit("Analyzing problem structure and breaking it down into components...")
            # Attach thought_generated signal
            self.neocortex.thought_generated = self.thought_generated
            decomposition = self.neocortex._analyze_problem(self.query)
            self.progress_update.emit(steps[1], 12)

            # Step 2: Generate multiple perspectives
            self.thought_generated.emit("Generating multiple perspectives to approach the problem...")
            perspectives = self.neocortex._generate_perspectives(self.query, decomposition)
            self.progress_update.emit(steps[2], 25)

            # Step 3: Gather evidence for each subproblem
            self.thought_generated.emit("Gathering evidence and analyzing each component of the problem...")
            evidence = self.neocortex._gather_evidence(self.query, decomposition)
            self.progress_update.emit(steps[3], 37)

            # Step 4: Integrate perspectives and evidence
            self.thought_generated.emit("Integrating all perspectives and evidence into a cohesive understanding...")
            integration = self.neocortex._integrate_perspectives_evidence(self.query, perspectives, evidence)
            self.progress_update.emit(steps[4], 50)

            # Step 5: Generate solution
            self.thought_generated.emit("Generating comprehensive solution based on integrated understanding...")
            solution = self.neocortex._generate_solution(self.query, integration)
            self.progress_update.emit(steps[5], 62)

            # Step 6: Verify solution
            self.thought_generated.emit("Critically evaluating the solution to identify any weaknesses or errors...")
            verification = self.neocortex._verify_solution(self.query, solution)
            self.progress_update.emit(steps[6], 75)

            # Step 7: Generate final answer
            self.thought_generated.emit("Refining solution based on verification results to create final answer...")
            final_answer = self.neocortex._generate_final_answer(self.query, solution, verification)
            self.progress_update.emit(steps[7], 87)

            # Step 8: Metacognitive reflection
            self.thought_generated.emit("Performing metacognitive reflection on the entire reasoning process...")
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

class AgentReasoningThread(ReasoningThread):
    """Thread for running the multi-agent reasoning process."""

    def run(self):
        try:
            self.progress_update.emit("Initializing agent-based reasoning...", 0)
            self.thought_generated.emit("Multi-agent system activated for reasoning...")
            
            # Attach thought_generated signal
            self.neocortex.thought_generated = self.thought_generated
            
            # Process with multi-agent system
            result = self.neocortex.solve(self.query, self.show_work, use_agents=True)
            
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
            "max_tokens": int(self.token_value.text())
        }
        self.settings_changed.emit(settings)

class RealtimeReasoningWidget(QWidget):
    """Widget for displaying realtime reasoning thoughts from the model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Instructions label
        instructions = QLabel("Realtime reasoning thoughts will appear here during processing")
        layout.addWidget(instructions)

        # Realtime reasoning display
        self.thought_display = QTextEdit()
        self.thought_display.setReadOnly(True)
        self.thought_display.setPlaceholderText("Model thoughts will appear here...")
        layout.addWidget(self.thought_display)

        # Auto-scroll checkbox
        self.auto_scroll = QCheckBox("Auto-scroll")
        self.auto_scroll.setChecked(True)
        layout.addWidget(self.auto_scroll)

        # Copy to clipboard button
        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        layout.addWidget(self.copy_button)

        self.setLayout(layout)

    def add_thought(self, thought):
        """Add a new thought to the display."""
        current_text = self.thought_display.toPlainText()
        if current_text:
            new_text = current_text + "\n\n" + thought
        else:
            new_text = thought

        # Limit display size to prevent performance issues with very large content
        if len(new_text) > 100000:
            new_text = new_text[-100000:]
            new_text = "... [earlier thoughts truncated] ...\n\n" + new_text[new_text.find("\n\n")+2:]

        self.thought_display.setPlainText(new_text)

        # Auto-scroll to bottom if enabled
        if self.auto_scroll.isChecked():
            scrollbar = self.thought_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        """Clear the thought display."""
        self.thought_display.clear()

    def copy_to_clipboard(self):
        """Copy the thoughts to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.thought_display.toPlainText())

        # Show a brief status message or tooltip to confirm
        self.copy_button.setText("Copied!")
        QTimer.singleShot(2000, lambda: self.copy_button.setText("Copy to Clipboard"))

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

        # Multi-Agent Results tab (new)
        self.multi_agent_widget = QTextEdit()
        self.multi_agent_widget.setReadOnly(True)
        self.tab_widget.addTab(self.multi_agent_widget, "Multi-Agent Results")

        layout.addWidget(self.tab_widget)

        # Export buttons
        export_layout = QHBoxLayout()
        self.export_answer_btn = QPushButton("Export Answer")
        self.export_answer_btn.clicked.connect(self.export_answer)
        self.export_full_btn = QPushButton("Export Full Analysis")
        self.export_full_btn.clicked.connect(self.export_full_analysis)
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self.copy_current_tab)
        export_layout.addWidget(self.export_answer_btn)
        export_layout.addWidget(self.export_full_btn)
        export_layout.addWidget(self.copy_btn)
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
            
        # Display multi-agent results if available
        if "multi_agent_workspace" in result:
            workspace = result["multi_agent_workspace"]
            
            # Create HTML view of the multi-agent workspace
            html = "<h2>Multi-Agent System Results</h2>"
            
            # Show all keys in the workspace except for the query and final_answer (shown elsewhere)
            exclude_keys = ["query", "final_answer"]
            
            for key, value in workspace.items():
                if key in exclude_keys:
                    continue
                    
                if key == "status":
                    html += f"<p><strong>Status:</strong> {value}</p>"
                elif isinstance(value, dict):
                    # For dictionary values, create a section
                    html += f"<h3>{key.replace('_', ' ').title()}</h3>"
                    
                    # Display dictionary content in a meaningful way
                    if key == "analysis":
                        html += f"<p>{value}</p>"
                    elif key == "components" and isinstance(value, list):
                        for i, comp in enumerate(value):
                            html += f"<h4>Component {i+1}: {comp.get('title', '')}</h4>"
                            html += f"<p>{comp.get('description', '')}</p>"
                    elif key == "approaches" and isinstance(value, list):
                        for i, app in enumerate(value):
                            html += f"<h4>Approach {i+1}: {app.get('title', '')}</h4>"
                            html += f"<p>{app.get('description', '')}</p>"
                    elif key == "scores" and isinstance(value, dict):
                        html += "<ul>"
                        for criterion, score in value.items():
                            html += f"<li><strong>{criterion.replace('_', ' ').title()}:</strong> {score}/10</li>"
                        html += "</ul>"
                    else:
                        # Generic dict display
                        html += "<ul>"
                        for k, v in value.items():
                            if isinstance(v, (list, dict)):
                                v = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                            html += f"<li><strong>{k.replace('_', ' ').title()}:</strong> {v}</li>"
                        html += "</ul>"
                elif isinstance(value, list):
                    # For list values, create a bulleted list
                    html += f"<h3>{key.replace('_', ' ').title()}</h3><ul>"
                    for item in value:
                        if isinstance(item, dict):
                            # If list contains dicts, show key items
                            desc = item.get('description', item.get('title', str(item)[:50] + "..."))
                            html += f"<li>{desc}</li>"
                        else:
                            html += f"<li>{item}</li>"
                    html += "</ul>"
                else:
                    # For simple values, just show them
                    if key not in ["status"]:  # Already showed status
                        html += f"<h3>{key.replace('_', ' ').title()}</h3>"
                        html += f"<p>{value}</p>"
                        
                html += "<hr>"
                
            self.multi_agent_widget.setHtml(html)
            
            # Switch to multi-agent tab if this was processed by agents
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(self.multi_agent_widget))

    def export_answer(self):
        """Export just the final answer to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Answer", "", "Text Files (.txt);;HTML Files (.html);;All Files (*)"
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
            self, "Export Full Analysis", "", "HTML Files (.html);;Text Files (.txt);;All Files (*)"
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
                    html += self.multi_agent_widget.toHtml()  # Include multi-agent results
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
                    text += "## Metacognitive Reflection\n\n" + self.reflection_widget.toPlainText() + "\n\n"
                    text += "## Multi-Agent Results\n\n" + self.multi_agent_widget.toPlainText()
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
        elif tab_name == "Cognitive Graph":
            content = "Graph visualization cannot be copied as text."
        elif tab_name == "Multi-Agent Results":
            content = self.multi_agent_widget.toPlainText()
        else:
            content = "No content to copy from this tab."

        clipboard.setText(content)
        self.copy_btn.setText("Copied!")
        QTimer.singleShot(2000, lambda: self.copy_btn.setText("Copy to Clipboard"))

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

        # Export history button
        self.export_button = QPushButton("Export History")
        self.export_button.clicked.connect(self.export_history)
        layout.addWidget(self.export_button)

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
            self, "Export History", "", "JSON Files (.json);;All Files (*)"
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

class NeoCortexGUI(QMainWindow):
    """Main GUI application for NeoCortex."""

    def __init__(self):
        super().__init__()
        self.neocortex = NeoCortex()
        self.current_result = None
        self.reasoning_thread = None
        self.init_ui()
        self.apply_dark_theme()
        
        # Add multi-agent tab
        add_multi_agent_tab(self)

    def init_ui(self):
        self.setWindowTitle("NeoCortex: Advanced Cognitive Architecture with Multi-Agent System")
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
        self.process_btn.clicked.connect(self.start_reasoning_with_agents)
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

        # Realtime reasoning visualization
        self.realtime_reasoning_widget = RealtimeReasoningWidget()
        left_tabs.addTab(self.realtime_reasoning_widget, "Realtime Thoughts")

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

    def start_reasoning_with_agents(self):
        """Start the reasoning process, using multi-agent system if enabled."""
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
        self.neocortex.max_tokens = 4000 if use_fast_mode else 8000

        # Check if multi-agent system is enabled
        use_agents = False
        if hasattr(self, 'multi_agent_tab'):
            use_agents = self.multi_agent_tab.is_enabled()

        # Show work setting
        show_work = self.show_work_check.isChecked()
        
        # Clear previous thoughts
        self.realtime_reasoning_widget.clear()

        # Switch to the realtime thoughts tab to show thoughts as they happen
        left_tabs = self.realtime_reasoning_widget.parent()
        if isinstance(left_tabs, QTabWidget):
            for i in range(left_tabs.count()):
                if left_tabs.tabText(i) == "Realtime Thoughts":
                    left_tabs.setCurrentIndex(i)
                    break
                    
        # Start reasoning thread based on agent setting
        if use_agents:
            self.reasoning_thread = AgentReasoningThread(self.neocortex, query, show_work)
            
            # Update multi-agent system interactions for visualization
            if hasattr(self, 'multi_agent_tab'):
                # Add some initial interactions for visualization
                self.multi_agent_tab.add_interaction("Executive", "Analyst", "Query")
                self.multi_agent_tab.add_interaction("Analyst", "Creative", "Analysis")
                self.multi_agent_tab.add_interaction("Analyst", "Logical", "Analysis")
        else:
            self.reasoning_thread = ReasoningThread(self.neocortex, query, show_work)
            
        # Connect signals
        self.reasoning_thread.progress_update.connect(self.update_progress)
        self.reasoning_thread.result_ready.connect(self.handle_result)
        self.reasoning_thread.error_occurred.connect(self.handle_error)
        self.reasoning_thread.thought_generated.connect(self.realtime_reasoning_widget.add_thought)

        # Start the thread
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

            # Apply additional performance settings with significantly increased token limit
            neocortex.max_tokens = settings.get("max_tokens", 8000)

            # Set performance mode
            perf_mode = settings.get("performance_mode", "balanced")
            if perf_mode == "fast":
                # Fast mode - minimal API calls, short responses
                neocortex.max_tokens = min(neocortex.max_tokens, 4000)
            elif perf_mode == "thorough":
                # Thorough mode - more detailed analysis
                neocortex.max_tokens = max(neocortex.max_tokens, 8000)

            self.neocortex = neocortex
            self.status_bar.showMessage(f"Model settings updated: {settings['model']}, temp={settings['temperature']}, tokens={settings.get('max_tokens', 8000)}")
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
            self.realtime_reasoning_widget.clear()

    def save_session(self):
        """Save the current session."""
        if not self.current_result:
            QMessageBox.warning(self, "No Result", "No reasoning result to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "NeoCortex Sessions (.neo);;All Files (*)"
        )
        if not file_path:
            return

        try:
            session_data = {
                "query": self.query_edit.toPlainText(),
                "result": self.current_result,
                "timestamp": datetime.now().isoformat()
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)

            self.status_bar.showMessage(f"Session saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save session: {str(e)}")

    def load_session(self):
        """Load a saved session."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "NeoCortex Sessions (.neo);;All Files (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
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
        <h1>NeoCortex: Advanced Cognitive Architecture with Multi-Agent System</h1>

        <h2>Overview</h2>
        <p>NeoCortex is a revolutionary cognitive architecture that integrates multiple specialized reasoning modules 
        and a multi-agent system to achieve AGI-like reasoning capabilities. It uses DeepSeek's V3 language models 
        via OpenRouter to implement a neuromorphic approach to problem-solving that mimics the human brain's 
        parallel processing, self-regulatory functions, and specialized agent coordination.</p>

        <h2>Using the Interface</h2>
        <ol>
            <li><strong>API Key:</strong> Enter your OpenRouter API key in the Settings tab before processing.</li>
            <li><strong>Problem Input:</strong> Enter your question or problem in the text area at the top.</li>
            <li><strong>Process:</strong> Click "Process with NeoCortex" to start the reasoning process.</li>
            <li><strong>Multi-Agent System:</strong> Enable the multi-agent system for complex problems that can benefit from specialized expertise.</li>
            <li><strong>Realtime Thoughts:</strong> Watch the model's reasoning unfold in real-time in the "Realtime Thoughts" tab.</li>
            <li><strong>Results:</strong> View the final answer and detailed reasoning in the tabs on the right.</li>
            <li><strong>Cognitive Graph:</strong> Explore the reasoning structure visually in the Cognitive Graph tab.</li>
            <li><strong>Module Controls:</strong> Enable or disable specific reasoning modules in the Modules tab.</li>
            <li><strong>Settings:</strong> Adjust model parameters in the Settings tab.</li>
            <li><strong>History:</strong> Access previous reasoning sessions in the History tab.</li>
        </ol>

        <h2>Multi-Agent System</h2>
        <p>The multi-agent system coordinates specialized agents to tackle different aspects of complex problems:</p>
        <ul>
            <li><strong>Executive Agent:</strong> Coordinates the overall reasoning process and integrates results</li>
            <li><strong>Analyst Agent:</strong> Analyzes problems and breaks them down into components</li>
            <li><strong>Creative Agent:</strong> Generates innovative and unconventional approaches</li>
            <li><strong>Logical Agent:</strong> Applies rigorous logical reasoning and structured problem-solving</li>
            <li><strong>Synthesizer Agent:</strong> Integrates multiple perspectives into a cohesive solution</li>
            <li><strong>Critic Agent:</strong> Identifies weaknesses, limitations and potential improvements</li>
            <li><strong>Implementer Agent:</strong> Creates concrete implementations or code solutions</li>
            <li><strong>Evaluator Agent:</strong> Tests and evaluates the completeness and quality of solutions</li>
        </ul>

        <h2>Key Features</h2>
        <ul>
            <li><strong>Multi-agent architecture:</strong> Specialized agents tackle different aspects of problems</li>
            <li><strong>Multi-perspective reasoning:</strong> Analyzes problems from multiple conceptual frameworks</li>
            <li><strong>Self-verification:</strong> Critically evaluates its own reasoning</li>
            <li><strong>Metacognition:</strong> Reflects on and improves its reasoning process</li>
            <li><strong>Causal modeling:</strong> Builds explicit models of cause and effect</li>
            <li><strong>Counterfactual analysis:</strong> Explores alternative scenarios</li>
            <li><strong>Real-time thought visualization:</strong> See the model's reasoning process unfold</li>
            <li><strong>High token capacity:</strong> Generate up to 16,000 tokens for comprehensive solutions</li>
            <li><strong>Anti-truncation technology:</strong> Automatically handles large responses without cutting off</li>
            <li><strong>Dynamic prompt construction:</strong> Customizes prompts based on query type and complexity</li>
            <li><strong>Context window management:</strong> Efficiently handles token limitations</li>
            <li><strong>Prompt chaining:</strong> Breaks complex problems into manageable sub-problems</li>
        </ul>

        <h2>Tips for Best Results</h2>
        <ul>
            <li>Formulate clear, specific questions</li>
            <li>Enable the multi-agent system for complex, multifaceted problems</li>
            <li>For complex problems, disable the fast mode to use higher token limits</li>
            <li>Adjust the temperature setting based on the task (lower for precise reasoning, higher for creative tasks)</li>
            <li>Use the module toggles to focus reasoning on specific aspects of problems</li>
            <li>Check the real-time thoughts tab and agent status for insights into the reasoning process</li>
            <li>Explore the cognitive graph to understand the structure of the reasoning process</li>
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
        
    def update_agent_status(self):
        """Update the multi-agent system status display."""
        if hasattr(self, 'multi_agent_tab') and hasattr(self, 'neocortex'):
            if hasattr(self.neocortex, 'multi_agent_system'):
                agent_status = self.neocortex.multi_agent_system.get_agent_status()
                self.multi_agent_tab.update_agent_status(agent_status)
                
                # Simulate agent interactions for a richer visual experience
                if hasattr(self, 'reasoning_thread') and self.reasoning_thread and self.reasoning_thread.isRunning():
                    # Get current progress
                    progress = self.progress_bar.value() / 100.0
                    
                    # Randomly select some agents for interaction
                    import random
                    if random.random() < 0.3:  # 30% chance of a new interaction every update
                        # Define interaction pairs based on progress
                        if progress < 0.2:
                            self.multi_agent_tab.add_interaction("Executive", "Analyst", "Task")
                        elif progress < 0.4:
                            self.multi_agent_tab.add_interaction("Analyst", "Logical", "Components")
                            self.multi_agent_tab.add_interaction("Analyst", "Creative", "Problem Structure")
                        elif progress < 0.6:
                            self.multi_agent_tab.add_interaction("Logical", "Synthesizer", "Findings")
                            self.multi_agent_tab.add_interaction("Creative", "Synthesizer", "Approaches")
                        elif progress < 0.8:
                            self.multi_agent_tab.add_interaction("Synthesizer", "Critic", "Solution")
                            self.multi_agent_tab.add_interaction("Critic", "Implementer", "Feedback")
                        else:
                            self.multi_agent_tab.add_interaction("Implementer", "Evaluator", "Implementation")
                            self.multi_agent_tab.add_interaction("Evaluator", "Executive", "Assessment")

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("NeoCortex with Multi-Agent Architecture")

    window = NeoCortexGUI()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
