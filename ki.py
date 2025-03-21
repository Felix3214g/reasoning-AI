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
import uuid
import re
import random
import psutil

import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer, QPointF, QDateTime
from PyQt5.QtGui import QFont, QColor, QIcon, QPalette, QPixmap, QBrush, QLinearGradient, QPainter, QPen, QPainterPath, QPolygonF, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTextEdit, QLineEdit, QGroupBox, QTabWidget, QSplitter,
    QComboBox, QListWidget, QListWidgetItem, QProgressBar, QDialog, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QCheckBox, QSpinBox, QDoubleSpinBox, QSlider,
    QScrollArea, QFrame, QGraphicsView, QGraphicsScene, QGraphicsItem, 
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QMessageBox,
    QStyleFactory, QToolBar, QAction, QStatusBar, QGraphicsRectItem, 
    QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsItemGroup, QTableWidget, QTableWidgetItem, QHeaderView,
    QGridLayout
)

from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from flask_cors import CORS

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
    """Base class for specialized cognitive agents."""
    
    def __init__(self, name: str, neocortex):
        self.name = name
        self.neocortex = neocortex
        self.state = "idle"  # idle, working, done, error
        self.knowledge = {}
        self.message_queue = []  # Queue for messages from other agents
        
    def process(self, input_data: Dict) -> Dict:
        """Process input data and return results."""
        self.state = "working"
        
        try:
            # Process incoming messages if any
            if self.message_queue:
                self._process_messages(input_data)
                
            # Run the agent's core processing logic
            result = self._process_implementation(input_data)
            
            self.state = "done"
            return result
        except Exception as e:
            self.state = "error"
            return {
                "status": "error",
                "message": str(e)
            }
            
    def _process_implementation(self, input_data: Dict) -> Dict:
        """Implementation of the agent's processing logic. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def get_status(self) -> Dict:
        """Get the current status of the agent."""
        return {
            "name": self.name,
            "state": self.state,
            "knowledge_size": len(self.knowledge),
            "message_count": len(self.message_queue)
        }
        
    def update_knowledge(self, key: str, value: Any):
        """Update the agent's knowledge base."""
        self.knowledge[key] = value
        
    def get_knowledge(self, key: str) -> Any:
        """Retrieve information from the agent's knowledge base."""
        return self.knowledge.get(key)
        
    def send_message(self, recipient_agent: 'Agent', message_type: str, content: Any):
        """Send a message to another agent."""
        message = {
            "sender": self.name,
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        recipient_agent.receive_message(message)
        
        # Log the interaction in the cognitive graph
        interaction_node = CognitiveNode(
            node_id=f"interaction_{uuid.uuid4().hex[:8]}",
            node_type="agent_interaction",
            content=f"{self.name} -> {recipient_agent.name}: {message_type}"
        )
        self.neocortex.cognitive_graph.add_node(interaction_node)
        
        return message
        
    def receive_message(self, message: Dict):
        """Receive a message from another agent."""
        self.message_queue.append(message)
        
    def _process_messages(self, input_data: Dict):
        """Process messages in the queue and incorporate relevant information."""
        processed_messages = []
        
        for message in self.message_queue:
            # Process the message based on its type
            if message["type"] == "feedback":
                # Store feedback in knowledge base
                self.update_knowledge(f"feedback_from_{message['sender']}", message["content"])
                
            elif message["type"] == "request":
                # Handle information request
                if "query" in message["content"]:
                    # Generate a response to the query
                    response = self._generate_response_to_query(message["content"]["query"])
                    # Send response back to the requester
                    sender_agent = self._find_agent_by_name(message["sender"])
                    if sender_agent:
                        self.send_message(sender_agent, "response", response)
                        
            elif message["type"] == "update":
                # Update input data with the new information
                if isinstance(message["content"], dict):
                    key = message["content"].get("key", "update_from_" + message["sender"])
                    input_data[key] = message["content"].get("value", message["content"])
                else:
                    input_data[f"update_from_{message['sender']}"] = message["content"]
                    
            # Mark message as processed
            processed_messages.append(message)
            
        # Remove processed messages from the queue
        for message in processed_messages:
            self.message_queue.remove(message)
            
        return input_data
        
    def _generate_response_to_query(self, query: str) -> Dict:
        """Generate a response to a query from another agent."""
        # Default implementation - can be overridden by specialized agents
        context = {k: v for k, v in self.knowledge.items() if isinstance(v, str)}
        context_str = "\n".join([f"{k}: {v[:100]}..." if len(v) > 100 else f"{k}: {v}" 
                                for k, v in context.items()])
        
        prompt = f"""
        Using your specialized knowledge as {self.name} agent, respond to this query:
        
        QUERY: {query}
        
        RELEVANT CONTEXT:
        {context_str}
        
        Provide a concise, helpful response that leverages your specific expertise.
        """
        
        response_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=False,
            task_id=f"{self.name.lower()}_query_response"
        )
        
        return {
            "text": response_text,
            "agent": self.name,
            "timestamp": datetime.now().isoformat()
        }
        
    def _find_agent_by_name(self, agent_name: str) -> Optional['Agent']:
        """Find an agent by name in the executive agent's workflow."""
        # This requires access to the executive agent or multi-agent system
        if hasattr(self.neocortex, "multi_agent_system"):
            executive = self.neocortex.multi_agent_system.executive
            if hasattr(executive, "workflow"):
                for stage_config in executive.workflow.values():
                    agent = stage_config.get("agent")
                    if agent and agent.name == agent_name:
                        return agent
        return None

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
        
        # Get approaches from creative agent
        creative_approaches = input_data.get("creative_approaches", [])
        
        # Get reasoning from logical agent
        logical_reasoning = input_data.get("logical_reasoning", [])
        
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
    Synthesize the following approaches and perspectives into a cohesive solution:

    PROBLEM:
    {query}

    ANALYSIS:
    {analysis}

    CREATIVE APPROACHES:
    {creative_output}

    LOGICAL REASONING:
    {logical_output}

    Your synthesis should:
    1. Identify the most valuable elements from each approach
    2. Resolve any contradictions or tensions between different perspectives
    3. Create an integrated solution framework that combines strengths from all approaches
    4. Address all aspects of the original problem comprehensively
    5. Present a clear, unified solution that builds upon all previous analyses

    Be thorough yet clear in your synthesis, creating a solution that is greater than the sum of its parts.
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
        
        # Extract key components from the synthesis
        solution = self._extract_solution(response_text)
        
        return {
            "status": "success",
            "synthesis": response_text,
            "solution": solution
        }
    
    def _extract_solution(self, synthesis_text: str) -> str:
        """Extract the core solution from synthesis text."""
        # Look for solution section or return the whole text if no clear section exists
        solution_markers = ["solution:", "integrated solution:", "proposed solution:", "synthesis:"]
        
        lower_text = synthesis_text.lower()
        for marker in solution_markers:
            if marker in lower_text:
                # Find the position of the marker
                pos = lower_text.find(marker)
                # Extract text after the marker
                after_marker = synthesis_text[pos + len(marker):].strip()
                
                # Find the end of this section (next section marker or end of text)
                next_section = float('inf')
                for next_marker in solution_markers:
                    next_pos = lower_text.find(next_marker, pos + len(marker))
                    if next_pos > -1 and next_pos < next_section:
                        next_section = next_pos
                
                if next_section < float('inf'):
                    return after_marker[:next_section - pos - len(marker)].strip()
                else:
                    return after_marker
        
        # If no explicit solution section, return the whole text or a reasonable portion
        if len(synthesis_text) > 1000:
            # If text is very long, return a reasonable portion from the middle
            # This assumes the core solution is likely in the middle after initial analysis
            middle_start = len(synthesis_text) // 3
            middle_end = (len(synthesis_text) * 2) // 3
            return synthesis_text[middle_start:middle_end].strip()
        else:
            return synthesis_text

class ImplementerAgent(Agent):
    """Agent specialized in implementing solutions as executable code."""
    
    def __init__(self, neocortex):
        super().__init__("Implementer", neocortex)
        
    def _process_implementation(self, input_data: Dict) -> Dict:
        """Convert solution concepts into executable implementation."""
        query = input_data.get("query", "")
        solution = input_data.get("solution", "")
        criticism = input_data.get("criticism", "")
        
        if not query or not solution:
            return {"status": "error", "message": "Missing required input data"}
            
        # Generate a system prompt focused on implementation
        system_prompt = """You are an implementation agent specialized in converting conceptual solutions into executable code.
Your task is to:
1. Translate the provided solution into practical, runnable code
2. Consider the criticisms provided and address them in your implementation
3. Create implementation that adheres to best practices in the relevant programming language
4. Include comments to explain important aspects of the implementation
5. Ensure the code is efficient, readable, and maintainable

Provide implementation as code blocks with appropriate language specifications."""
        
        # Generate the implementation prompt
        prompt = f"""
ORIGINAL QUERY: {query}

SOLUTION TO IMPLEMENT:
{solution}

CRITICISM TO ADDRESS:
{criticism}

Create a practical implementation of the solution. Focus on producing:
1. Clear, executable code in the most appropriate language
2. Necessary helper functions or utilities
3. Implementation that addresses the criticisms
4. Code that would actually work in a real-world environment

Format your response with clear code blocks, and explain any important design decisions briefly.
"""
        
        # Generate the implementation
        implementation_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=False,
            system_prompt=system_prompt,
            task_id="implementation_generation"
        )
        
        # Extract code blocks and explanations from the implementation
        code_blocks = self._extract_code(implementation_text)
        
        return {
            "status": "success",
            "implementation": implementation_text,
            "code_blocks": code_blocks
        }
        
    def _extract_code(self, implementation_text: str) -> List[Dict]:
        """Extract code blocks from the implementation text."""
        code_blocks = []
        
        # Match markdown code blocks (```language ... ```)
        code_pattern = r"```(\w*)\n(.*?)```"
        matches = re.finditer(code_pattern, implementation_text, re.DOTALL)
        
        for match in matches:
            language = match.group(1).strip() or "text"
            code = match.group(2).strip()
            
            code_blocks.append({
                "language": language,
                "code": code
            })
            
        # If no code blocks found but there is content, add it as plaintext
        if not code_blocks and implementation_text.strip():
            code_blocks.append({
                "language": "text",
                "code": implementation_text.strip()
            })
            
        return code_blocks

class EvaluatorAgent(Agent):
    """Agent specialized in evaluating implementations and solutions."""
    
    def __init__(self, neocortex):
        super().__init__("Evaluator", neocortex)
        
    def _process_implementation(self, input_data: Dict) -> Dict:
        """Evaluate the implementation against the original query and solution."""
        query = input_data.get("query", "")
        solution = input_data.get("solution", "")
        implementation = input_data.get("implementation", "")
        code_blocks = input_data.get("code_blocks", [])
        
        if not query or not implementation:
            return {"status": "error", "message": "Missing required input data"}
            
        # Generate a system prompt focused on evaluation
        system_prompt = """You are an evaluation agent specialized in assessing implementations against requirements.
Your task is to:
1. Evaluate how well the implementation addresses the original query
2. Assess the implementation's alignment with the proposed solution
3. Identify strengths and weaknesses of the implementation
4. Suggest improvements or refinements
5. Provide an overall assessment of effectiveness

Be thorough, fair, and constructive in your evaluation."""
        
        # Generate the evaluation prompt
        prompt = f"""
ORIGINAL QUERY: {query}

PROPOSED SOLUTION:
{solution}

IMPLEMENTATION:
{implementation}

Evaluate the implementation by addressing these aspects:
1. Effectiveness: How well does the implementation address the original query?
2. Alignment: Is the implementation consistent with the proposed solution?
3. Quality: Assess the code quality, readability, and maintainability
4. Completeness: Does the implementation cover all necessary aspects?
5. Potential Issues: What problems might arise with this implementation?

Provide a balanced assessment with specific examples from the implementation.
"""
        
        # Generate the evaluation
        evaluation_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=False,
            system_prompt=system_prompt,
            task_id="implementation_evaluation"
        )
        
        # Extract evaluation components
        evaluation_components = self._extract_evaluation(evaluation_text)
        
        return {
            "status": "success",
            "assessment": evaluation_text,
            "evaluation_components": evaluation_components
        }
        
    def _extract_evaluation(self, evaluation_text: str) -> Dict:
        """Extract structured evaluation components from the text."""
        components = {
            "strengths": [],
            "weaknesses": [],
            "improvements": [],
            "overall_rating": None
        }
        
        # Extract strengths
        strengths_match = re.search(r"(?:Strengths|STRENGTHS|Pros|PROS):(.*?)(?:\n\n|\n[A-Z])", evaluation_text, re.DOTALL)
        if strengths_match:
            strengths_text = strengths_match.group(1).strip()
            components["strengths"] = [s.strip() for s in re.split(r"\n[-*•]|\n\d+\.", strengths_text) if s.strip()]
            
        # Extract weaknesses
        weaknesses_match = re.search(r"(?:Weaknesses|WEAKNESSES|Cons|CONS|Issues|ISSUES):(.*?)(?:\n\n|\n[A-Z])", evaluation_text, re.DOTALL)
        if weaknesses_match:
            weaknesses_text = weaknesses_match.group(1).strip()
            components["weaknesses"] = [w.strip() for w in re.split(r"\n[-*•]|\n\d+\.", weaknesses_text) if w.strip()]
            
        # Extract improvements
        improvements_match = re.search(r"(?:Improvements|IMPROVEMENTS|Suggestions|SUGGESTIONS):(.*?)(?:\n\n|\n[A-Z])", evaluation_text, re.DOTALL)
        if improvements_match:
            improvements_text = improvements_match.group(1).strip()
            components["improvements"] = [i.strip() for i in re.split(r"\n[-*•]|\n\d+\.", improvements_text) if i.strip()]
            
        # Extract overall rating if present
        rating_match = re.search(r"(?:Rating|RATING|Overall|OVERALL|Score|SCORE):\s*(\d+(?:\.\d+)?\/\d+|\d+%)", evaluation_text)
        if rating_match:
            components["overall_rating"] = rating_match.group(1).strip()
            
        # If structured extraction failed, use simple paragraph splitting
        if not any(components.values()):
            paragraphs = [p.strip() for p in evaluation_text.split("\n\n") if p.strip()]
            
            if len(paragraphs) >= 3:
                # Assume first few paragraphs are about strengths, weaknesses, and improvements
                components["strengths"] = [paragraphs[0]]
                components["weaknesses"] = [paragraphs[1]]
                components["improvements"] = [paragraphs[2]]
                
        return components

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
        
        # Check if the query is too large and needs preprocessing
        context_window_manager = self.neocortex.context_window_manager
        query_tokens = context_window_manager.estimate_tokens(query)
        
        # Reset workspace
        self.workspace = {}
        
        # If query is too large, summarize it first
        if query_tokens > 2000:  # Threshold for large queries
            self.update_knowledge("preprocessing", "Large query detected, summarizing...")
            
            # Summarize the query while preserving key details
            summarize_prompt = f"""
            This query is very long. Create a concise version that:
            1. Captures all key requirements and details
            2. Maintains specific constraints or parameters
            3. Preserves any specific instructions or formats
            4. Omits unnecessary explanations or repetitions
            
            Original query:
            {query[:4000]}  # Use first 4000 chars to avoid token issues in summarization
            
            Respond with only the summarized query.
            """
            
            summarized_query = self.neocortex._generate_response(summarize_prompt, emit_thoughts=False)
            
            # Store both versions in workspace
            self.workspace = {
                "query": summarized_query,
                "original_query": query,
                "preprocessing_note": "Query was summarized due to length while preserving key details."
            }
        else:
            # Store query directly if within token limits
            self.workspace = {"query": query}
        
        # Create problem node in the cognitive graph
        problem_node = CognitiveNode(
            node_id="problem_0",
            node_type="problem",
            content=self.workspace["query"]
        )
        self.neocortex.cognitive_graph.add_node(problem_node)
        
        # Initialize token tracking
        self.update_knowledge("token_usage", {})
        
        try:
            # Execute the multi-agent workflow
            self._execute_workflow()
            
            # Check if we need to handle any token warnings before final answer
            token_warning = self.get_knowledge("token_warning")
            if token_warning:
                self.update_knowledge("workflow_note", 
                                     "Token limits approached during processing. Some intermediate results were compressed.")
            
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
            
            # If we're using the original query, make sure it's referenced in the final output
            if "original_query" in self.workspace:
                self.workspace["query"] = self.workspace["original_query"]
            
            self.state = "done"
            return {
                "status": "success",
                "final_answer": final_answer,
                "workspace": self.workspace,
                "cognitive_graph": self.neocortex.cognitive_graph.to_dict(),
                "token_usage": self.get_knowledge("token_usage")
            }
            
        except Exception as e:
            self.state = "error"
            return {
                "status": "error",
                "message": str(e),
                "workspace": self.workspace,
                "token_usage": self.get_knowledge("token_usage")
            }
            
    def _execute_workflow(self):
        """Execute the agent workflow based on the configuration."""
        # Start with analysis stage
        current_stages = ["analysis"]
        completed_stages = set()
        failed_stages = set()
        retry_attempts = {}
        max_retries = 2
        
        # Track token usage across all agents
        total_token_usage = 0
        max_tokens_per_workflow = self.neocortex.context_window_manager.max_tokens_per_call * 3  # Allow for multiple calls
        
        # Create a log of token usage for monitoring
        token_usage_log = {}
        
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
                
                # Check if we're approaching token limits
                if total_token_usage > max_tokens_per_workflow * 0.8:  # 80% threshold
                    self.update_knowledge("token_warning", 
                                         f"Warning: Approaching token limit ({total_token_usage}/{max_tokens_per_workflow})")
                    
                    # Compact previous results to save tokens
                    self._compact_workspace()
                
                # Execute agent for this stage
                agent = stage_config["agent"]
                
                # Prepare input data with token limits in mind
                prepared_input = self._prepare_agent_input(stage, self.workspace)
                
                # Pass messages between agents before processing
                self._coordinate_agent_communication(stage, agent)
                
                try:
                    # Process through the agent
                    result = agent.process(prepared_input)
                    
                    # Check if the result indicates an error
                    if result.get("status") == "error":
                        raise Exception(result.get("message", f"Error in {stage} stage"))
                    
                    # Estimate token usage for this stage
                    result_tokens = self._estimate_stage_tokens(result)
                    total_token_usage += result_tokens
                    token_usage_log[stage] = result_tokens
                    
                    # Update workspace with agent result
                    self.workspace.update(result)
                    
                    # Store token usage information
                    self.update_knowledge("token_usage", token_usage_log)
                    
                    # Mark stage as completed
                    completed_stages.add(stage)
                    
                    # Add next stages to queue
                    next_stages.extend(stage_config["next"])
                    
                except Exception as e:
                    # Log the error
                    error_message = str(e)
                    self.update_knowledge(f"{stage}_error", error_message)
                    
                    # Track retry attempts
                    if stage not in retry_attempts:
                        retry_attempts[stage] = 0
                        
                    # Decide whether to retry or mark as failed
                    if retry_attempts[stage] < max_retries:
                        # Implement recovery strategy
                        recovery_success = self._attempt_recovery(stage, agent, error_message)
                        retry_attempts[stage] += 1
                        
                        if recovery_success:
                            # Try this stage again
                            next_stages.append(stage)
                        else:
                            # If recovery failed, mark as failed and try to continue
                            failed_stages.add(stage)
                            self._handle_stage_failure(stage, error_message)
                            
                            # Add next stages to continue workflow if possible
                            next_stages.extend(stage_config["next"])
                    else:
                        # Max retries reached, mark as failed
                        failed_stages.add(stage)
                        self._handle_stage_failure(stage, error_message)
                        
                        # Add next stages to continue workflow
                        next_stages.extend(stage_config["next"])
                
            # Set up next iteration
            current_stages = next_stages
            
        # Record any failed stages in the workspace
        if failed_stages:
            self.workspace["failed_stages"] = list(failed_stages)
    
    def _coordinate_agent_communication(self, current_stage: str, current_agent: Agent):
        """Coordinate communication between agents before processing."""
        # Define which agents should communicate with the current agent
        communication_map = {
            "analysis": [],  # Analysis doesn't need input from other agents
            "creative": ["analyst"],  # Creative gets input from analyst
            "logical": ["analyst"],  # Logical gets input from analyst
            "synthesis": ["creative", "logical"],  # Synthesis gets input from creative and logical
            "critic": ["synthesizer"],  # Critic gets input from synthesizer
            "implementation": ["synthesizer", "critic"],  # Implementation gets input from synthesizer and critic
            "evaluation": ["implementer"]  # Evaluation gets input from implementer
        }
        
        # Get the agents that should provide input to the current agent
        input_agents = []
        for agent_name in communication_map.get(current_stage, []):
            for stage, config in self.workflow.items():
                if config["agent"].name.lower() == agent_name.lower():
                    input_agents.append(config["agent"])
                    break
        
        # Facilitate communication between agents
        for input_agent in input_agents:
            # Skip if the input agent hasn't completed its work
            if input_agent.state != "done":
                continue
                
            # Determine what type of message to send based on agent roles
            if input_agent.name == "Analyst" and current_agent.name in ["Creative", "Logical"]:
                # Send analysis components to creative/logical agents
                if "components" in self.workspace:
                    input_agent.send_message(
                        current_agent, 
                        "update", 
                        {"key": "analysis_components", "value": self.workspace["components"]}
                    )
                    
            elif input_agent.name in ["Creative", "Logical"] and current_agent.name == "Synthesizer":
                # Send approaches/reasoning to synthesizer
                if input_agent.name == "Creative" and "approaches" in self.workspace:
                    input_agent.send_message(
                        current_agent,
                        "update",
                        {"key": "creative_approaches", "value": self.workspace["approaches"]}
                    )
                if input_agent.name == "Logical" and "reasoning" in self.workspace:
                    input_agent.send_message(
                        current_agent,
                        "update",
                        {"key": "logical_reasoning", "value": self.workspace["reasoning"]}
                    )
                    
            elif input_agent.name == "Synthesizer" and current_agent.name == "Critic":
                # Send solution to critic
                if "solution" in self.workspace:
                    input_agent.send_message(
                        current_agent,
                        "update",
                        {"key": "proposed_solution", "value": self.workspace["solution"]}
                    )
                    
            elif input_agent.name == "Critic" and current_agent.name == "Implementer":
                # Send feedback to implementer
                if "strengths" in self.workspace and "weaknesses" in self.workspace:
                    input_agent.send_message(
                        current_agent,
                        "feedback",
                        {
                            "strengths": self.workspace["strengths"],
                            "weaknesses": self.workspace["weaknesses"],
                            "improvements": self.workspace.get("improvements", [])
                        }
                    )
                    
            elif input_agent.name == "Implementer" and current_agent.name == "Evaluator":
                # Send implementation to evaluator
                if "code_blocks" in self.workspace:
                    input_agent.send_message(
                        current_agent,
                        "update",
                        {"key": "implementation", "value": self.workspace["code_blocks"]}
                    )
    
    def _attempt_recovery(self, failed_stage: str, agent: Agent, error_message: str) -> bool:
        """Attempt to recover from an error in a stage."""
        # Log recovery attempt
        self.update_knowledge(f"{failed_stage}_recovery_attempt", f"Attempting recovery: {error_message}")
        
        # Different recovery strategies based on the stage
        if failed_stage == "analysis":
            # For analysis failures, simplify the query and retry
            original_query = self.workspace.get("query", "")
            simplify_prompt = f"""
            The analysis of this query has failed with error: {error_message}
            Please simplify this query while preserving its core intent:
            
            {original_query}
            
            Provide a simplified version that removes complexity but keeps the essential request.
            """
            
            simplified_query = self.neocortex._generate_response(simplify_prompt, emit_thoughts=False)
            self.workspace["original_query"] = original_query
            self.workspace["query"] = simplified_query
            self.update_knowledge("recovery_action", "Simplified query for analysis")
            return True
            
        elif failed_stage in ["creative", "logical"]:
            # For creative/logical failures, provide more focused instructions
            self.update_knowledge(f"{failed_stage}_guidance", "Focus on core aspects only, limit scope")
            return True
            
        elif failed_stage == "synthesis":
            # For synthesis failures, try with just the strongest approaches
            if "approaches" in self.workspace and isinstance(self.workspace["approaches"], list):
                # Use only the top approaches
                self.workspace["approaches"] = self.workspace["approaches"][:2]
                self.update_knowledge("recovery_action", "Reduced number of approaches for synthesis")
                return True
                
        elif failed_stage == "critic":
            # For critic failures, skip detailed criticism and just provide basic feedback
            self.workspace["strengths"] = ["The solution addresses the core problem"]
            self.workspace["weaknesses"] = ["Limited scope due to previous processing issues"]
            self.update_knowledge("recovery_action", "Created minimal criticism to continue workflow")
            return True
            
        elif failed_stage == "implementation":
            # For implementation failures, generate pseudo-code instead of full implementation
            if "solution" in self.workspace:
                pseudo_code_prompt = f"""
                Generate simple pseudocode for this solution. Avoid complex syntax:
                
                {self.workspace.get('solution', '')}
                
                Provide basic pseudocode steps only.
                """
                
                pseudo_code = self.neocortex._generate_response(pseudo_code_prompt, emit_thoughts=False)
                self.workspace["code_blocks"] = [{
                    "language": "pseudocode",
                    "code": pseudo_code,
                    "purpose": "Simplified implementation due to error recovery"
                }]
                self.update_knowledge("recovery_action", "Generated pseudocode instead of full implementation")
                return True
                
        elif failed_stage == "evaluation":
            # For evaluation failures, provide a basic evaluation
            self.workspace["score"] = {"overall": 5, "correctness": 5, "efficiency": 5, "robustness": 5}
            self.workspace["assessment"] = "Basic evaluation due to error recovery process"
            self.update_knowledge("recovery_action", "Created minimal evaluation to complete workflow")
            return True
            
        # Default case - recovery not possible
        return False
    
    def _handle_stage_failure(self, failed_stage: str, error_message: str):
        """Handle a failed stage when recovery attempts have been exhausted."""
        # Log the permanent failure
        self.update_knowledge(f"{failed_stage}_permanent_failure", 
                             f"Stage failed after recovery attempts: {error_message}")
        
        # Create a failure node in the cognitive graph
        failure_node = CognitiveNode(
            node_id=f"failure_{failed_stage}_{uuid.uuid4().hex[:8]}",
            node_type="failure",
            content=f"Stage {failed_stage} failed: {error_message}",
            parent_id="problem_0"
        )
        self.neocortex.cognitive_graph.add_node(failure_node)
        
        # Add minimal data to workspace to allow continuation
        # These are fallback values that will let the workflow continue
        if failed_stage == "analysis":
            self.workspace["components"] = [{"title": "Core Problem", "description": self.workspace.get("query", "")}]
            
        elif failed_stage == "creative":
            self.workspace["approaches"] = [{
                "title": "Basic Approach",
                "description": "Direct approach based on problem statement",
                "pros": ["Simple", "Direct"],
                "cons": ["Limited", "May not address all aspects"]
            }]
            
        elif failed_stage == "logical":
            self.workspace["reasoning"] = [{
                "step": "Direct Solution",
                "description": "Proceed with available information"
            }]
            self.workspace["constraints"] = ["Limited by available information"]
            
        elif failed_stage == "synthesis":
            self.workspace["solution"] = "Basic solution using available approaches. " + \
                                       "Note: This is a fallback solution due to processing limitations."
            
        elif failed_stage == "critic":
            self.workspace["strengths"] = ["Addresses core problem"]
            self.workspace["weaknesses"] = ["Limited by previous processing issues"]
            self.workspace["improvements"] = ["Consider alternative approaches"]
            
        elif failed_stage == "implementation":
            self.workspace["code_blocks"] = [{
                "language": "text",
                "code": "Implementation unavailable due to processing limitations",
                "purpose": "Placeholder due to failure"
            }]
            
        elif failed_stage == "evaluation":
            self.workspace["score"] = {"overall": 3, "correctness": 3, "efficiency": 3, "robustness": 3}
            self.workspace["assessment"] = "Limited assessment due to processing issues"
            
    def _prepare_agent_input(self, stage: str, workspace: Dict) -> Dict:
        """Prepare input data for an agent, managing token usage."""
        # Create a copy of the workspace to modify
        prepared_input = {"query": workspace.get("query", "")}
        
        # Get token manager
        context_window_manager = self.neocortex.context_window_manager
        
        # Define which previous stages are relevant for each stage
        relevant_stages = {
            "analysis": [],  # Analysis only needs the query
            "creative": ["analysis"],
            "logical": ["analysis"],
            "synthesis": ["analysis", "creative", "logical"],
            "critic": ["synthesis"],
            "implementation": ["synthesis", "critic"],
            "evaluation": ["implementation"]
        }
        
        # Define max tokens for each input field
        max_tokens_per_field = 1000
        
        # Include only relevant data from previous stages
        for prev_stage in relevant_stages.get(stage, []):
            if prev_stage in workspace:
                data = workspace[prev_stage]
                
                # Check if data needs truncation
                if context_window_manager.estimate_tokens(str(data)) > max_tokens_per_field:
                    if isinstance(data, str):
                        # Truncate string data
                        char_limit = max_tokens_per_field * 4  # Approximation: 4 chars ≈ 1 token
                        truncated = data[:char_limit]
                        
                        # Find a good cutting point (end of sentence)
                        last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
                        if last_period > 0:
                            truncated = truncated[:last_period+1]
                            
                        prepared_input[prev_stage] = truncated + "... [truncated for token efficiency]"
                    elif isinstance(data, dict):
                        # For dictionaries, include only essential keys
                        essential_keys = self._get_essential_keys(prev_stage)
                        prepared_input[prev_stage] = {k: data[k] for k in data if k in essential_keys}
                    elif isinstance(data, list):
                        # For lists, truncate to the first few items
                        prepared_input[prev_stage] = data[:5]  # Limit to 5 items
                    else:
                        # For other types, use as is
                        prepared_input[prev_stage] = data
                else:
                    # Use data as is if within token limits
                    prepared_input[prev_stage] = data
                    
        # Add any other required fields
        for key in workspace:
            if key not in prepared_input and key not in ["analysis", "creative", "logical", 
                                                         "synthesis", "critic", "implementation",
                                                         "evaluation"]:
                prepared_input[key] = workspace[key]
                
        return prepared_input
    
    def _get_essential_keys(self, stage: str) -> List[str]:
        """Get the essential keys for a given stage."""
        essential_keys = {
            "analysis": ["components", "complexity", "key_aspects"],
            "creative": ["approaches", "insights"],
            "logical": ["reasoning", "constraints", "steps"],
            "synthesis": ["solution", "explanation"],
            "critic": ["strengths", "weaknesses", "improvements"],
            "implementation": ["code_blocks", "explanation"],
            "evaluation": ["score", "feedback"]
        }
        return essential_keys.get(stage, [])
    
    def _estimate_stage_tokens(self, result: Dict) -> int:
        """Estimate the token usage for a stage result."""
        # Serialize the result to string for token estimation
        result_str = str(result)
        return self.neocortex.context_window_manager.estimate_tokens(result_str)
    
    def _compact_workspace(self):
        """Compact the workspace to save tokens."""
        # Identify stages that can be compacted
        stages_to_compact = ["analysis", "creative", "logical"]
        
        for stage in stages_to_compact:
            if stage in self.workspace:
                # Extract only the essential information
                if isinstance(self.workspace[stage], dict):
                    essential_keys = self._get_essential_keys(stage)
                    self.workspace[stage] = {k: self.workspace[stage][k] 
                                          for k in self.workspace[stage] 
                                          if k in essential_keys}
                elif isinstance(self.workspace[stage], str) and len(self.workspace[stage]) > 500:
                    # Summarize long text
                    summary_prompt = f"Summarize this content in 3-5 key points: {self.workspace[stage][:1000]}"
                    summary = self.neocortex._generate_response(summary_prompt, emit_thoughts=False)
                    self.workspace[stage] = f"[Compacted for token efficiency] {summary}"
        
    def _generate_final_answer(self) -> str:
        """Generate a final, coherent answer based on the agent outputs."""
        # Collect key components from the workspace
        synthesis = self.workspace.get("synthesis", "")
        solution = self.workspace.get("solution", "")
        implementation = self.workspace.get("implementation", "")
        evaluation = self.workspace.get("evaluation", "")
        assessment = self.workspace.get("assessment", "")
        
        # Check for failed stages
        failed_stages = self.workspace.get("failed_stages", [])
        error_summary = ""
        if failed_stages:
            error_details = []
            for stage in failed_stages:
                error_msg = self.get_knowledge(f"{stage}_permanent_failure")
                recovery_action = self.get_knowledge(f"recovery_action")
                if error_msg:
                    error_details.append(f"{stage.capitalize()}: {error_msg}")
                    if recovery_action:
                        error_details.append(f"Recovery action: {recovery_action}")
            
            error_summary = f"""
            Note: The system encountered issues in the following stages: {', '.join(failed_stages)}.
            Fallback mechanisms were employed to continue processing.
            Some aspects of the solution may be simplified as a result.
            """
        
        # Get token estimates using ContextWindowManager
        context_window_manager = self.neocortex.context_window_manager
        query_tokens = context_window_manager.estimate_tokens(self.workspace.get('query', ''))
        
        # Calculate available tokens for each section
        total_available_tokens = context_window_manager.max_tokens_per_call * 0.7  # Use 70% of available tokens
        base_tokens = 500  # Base tokens for prompt structure
        remaining_tokens = total_available_tokens - base_tokens - query_tokens
        
        # Distribute tokens proportionally to each section
        section_weights = {
            "synthesis": 0.25,
            "solution": 0.3,
            "implementation": 0.3,
            "assessment": 0.15
        }
        
        # Calculate max tokens for each section
        section_max_tokens = {
            section: int(remaining_tokens * weight)
            for section, weight in section_weights.items()
        }
        
        # Function to truncate text based on token limit
        def truncate_to_tokens(text, max_tokens):
            if not text:
                return ""
                
            # If text is within token limit, return as is
            if context_window_manager.estimate_tokens(text) <= max_tokens:
                return text
                
            # Truncate to character estimate (4 chars ≈ 1 token)
            char_limit = max_tokens * 4
            truncated = text[:char_limit]
            
            # Ensure we don't cut off mid-sentence or code block
            last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
            last_code_block = truncated.rfind('```')
            
            # Find a good truncation point
            if last_code_block > 0 and '```' in text[last_code_block+3:last_code_block+100]:
                # We're in the middle of a code block, truncate at the previous block end
                prev_block_end = truncated.rfind('```', 0, last_code_block-1)
                if prev_block_end > 0:
                    truncated = truncated[:prev_block_end+3]
            elif last_period > 0:
                # Truncate at sentence end
                truncated = truncated[:last_period+1]
            
            return truncated + "... [truncated due to length]"
        
        # Truncate sections to fit token limits
        truncated_synthesis = truncate_to_tokens(synthesis, section_max_tokens["synthesis"])
        truncated_solution = truncate_to_tokens(solution, section_max_tokens["solution"])
        truncated_implementation = truncate_to_tokens(implementation, section_max_tokens["implementation"])
        truncated_assessment = truncate_to_tokens(assessment, section_max_tokens["assessment"])
        
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
        {truncated_synthesis}

        SOLUTION:
        {truncated_solution}

        IMPLEMENTATION:
        {truncated_implementation}

        EVALUATION:
        {truncated_assessment}

        {error_summary if error_summary else ""}

        Create a final answer that:
        1. Addresses the original query directly and completely
        2. Incorporates the key insights from all agents
        3. Presents the solution and implementation clearly
        4. Acknowledges any limitations or areas for improvement
        5. Is well-structured and easy to understand
        {f"6. Acknowledges the processing limitations that occurred" if failed_stages else ""}

        Begin your answer with "### Final Comprehensive Answer:"
        """
            
        # Generate response with the specialized system prompt
        response_text = self.neocortex._generate_response(
            prompt, 
            emit_thoughts=True, 
            system_prompt=system_prompt,
            task_id="final_answer"
        )
        
        # Check if response is truncated and handle it
        if self.neocortex._is_truncated(response_text):
            # Try to generate a completion
            response_text = self.neocortex._generate_completion(response_text, 
                                                               f"Final answer for query: {self.workspace.get('query', '')}")
            
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
        self.agent_communication_log = []
        self.error_log = []
        
    def process_query(self, query: str) -> Dict:
        """Process a query through the multi-agent system."""
        # Clear previous logs
        self.agent_communication_log = []
        self.error_log = []
        
        # Register message interceptor to log agent communications
        self._register_communication_interceptor()
        
        # Process query through executive agent
        result = self.executive.process(query)
        
        # Add communication log to result
        result["agent_communications"] = self.agent_communication_log
        
        # Add error log if there were errors
        if "failed_stages" in result:
            self._collect_error_logs(result["failed_stages"])
            result["error_log"] = self.error_log
            
        return result
        
    def get_agent_status(self) -> Dict:
        """Get current status of all agents in the system."""
        agent_status = {
            "executive": self.executive.get_status()
        }
        
        # Get status of all specialized agents
        if hasattr(self.executive, "workflow"):
            for stage, config in self.executive.workflow.items():
                agent = config.get("agent")
                if agent:
                    agent_status[stage] = agent.get_status()
                    
        return agent_status
        
    def _register_communication_interceptor(self):
        """Register a communication interceptor to log agent messages."""
        # Patch the send_message method on Agent class to log communications
        original_send_message = Agent.send_message
        
        def intercepted_send_message(self, recipient_agent, message_type, content):
            # Call the original method
            message = original_send_message(self, recipient_agent, message_type, content)
            
            # Log the communication
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "sender": self.name,
                "recipient": recipient_agent.name,
                "type": message_type,
                "content_summary": str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
            }
            
            # Add to the communication log
            if hasattr(self.neocortex, "multi_agent_system"):
                self.neocortex.multi_agent_system.agent_communication_log.append(log_entry)
                
            return message
            
        # Replace the method temporarily
        Agent.send_message = intercepted_send_message
        
    def _collect_error_logs(self, failed_stages):
        """Collect error logs from failed stages."""
        for stage in failed_stages:
            # Get error messages from executive agent's knowledge
            error_msg = self.executive.get_knowledge(f"{stage}_permanent_failure")
            recovery_attempts = self.executive.get_knowledge(f"{stage}_recovery_attempt")
            recovery_action = self.executive.get_knowledge("recovery_action")
            
            if error_msg:
                error_entry = {
                    "stage": stage,
                    "error": error_msg,
                    "recovery_attempts": recovery_attempts,
                    "recovery_action": recovery_action
                }
                self.error_log.append(error_entry)

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
        # Add compatibility alias for context_window_manager
        self.context_window_manager = self.context_manager
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
    """Widget for visualizing agent interactions."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.interactions = []
        self.message_types = {}  # Track message types for coloring
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        self.layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Agent Interactions")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(title)
        
        # Scene and view for interaction visualization
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setMinimumHeight(200)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.view)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        controls_layout.addWidget(self.clear_btn)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_visualization)
        controls_layout.addWidget(self.export_btn)
        
        self.layout.addLayout(controls_layout)
        
        # Legend
        self.legend_group = QGroupBox("Message Types")
        self.legend_layout = QVBoxLayout()
        self.legend_group.setLayout(self.legend_layout)
        self.layout.addWidget(self.legend_group)
        
        # Apply styling
        self.setStyleSheet("""
        QGroupBox {
            border: 1px solid #444;
            border-radius: 5px;
            margin-top: 1ex;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
        }
        """)
        
    def add_interaction(self, from_agent, to_agent, data_type, timestamp=None, message_content=None):
        """Add an interaction between agents."""
        # Create a new interaction record
        interaction = {
            "from": from_agent,
            "to": to_agent,
            "type": data_type,
            "timestamp": timestamp or datetime.now().isoformat(),
            "content": message_content
        }
        
        # Add to interactions list
        self.interactions.append(interaction)
        
        # Update message type tracking
        if data_type not in self.message_types:
            # Assign a color to this message type
            hue = (len(self.message_types) * 60) % 360
            color = QColor.fromHsv(hue, 200, 240)
            self.message_types[data_type] = {
                "color": color,
                "count": 1
            }
        else:
            self.message_types[data_type]["count"] += 1
            
        # Update legend
        self._update_legend()
        
        # Redraw the visualization
        self.redraw()
        
    def redraw(self):
        """Redraw the interaction visualization."""
        # Clear the scene
        self.scene.clear()
        
        # Skip if no interactions
        if not self.interactions:
            return
            
        # Track node positions
        agents = set()
        for interaction in self.interactions:
            agents.add(interaction["from"])
            agents.add(interaction["to"])
            
        agents = sorted(list(agents))
        node_positions = {}
        
        # Calculate node positions in a circular layout
        radius = 150
        center_x = 200
        center_y = 150
        
        for i, agent in enumerate(agents):
            angle = 2 * math.pi * i / len(agents)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Create agent node
            node = self._create_agent_node(agent, x, y)
            self.scene.addItem(node)
            
            # Store position
            node_positions[agent] = (x, y)
            
        # Draw connections
        for interaction in self.interactions:
            from_pos = node_positions[interaction["from"]]
            to_pos = node_positions[interaction["to"]]
            
            # Get color for this message type
            color = self.message_types[interaction["type"]]["color"]
            
            # Create edge
            edge = self._create_edge(
                from_pos[0], from_pos[1], 
                to_pos[0], to_pos[1],
                color, interaction["type"],
                interaction.get("content")
            )
            self.scene.addItem(edge)
            
        # Fit scene in view
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        
    def _create_agent_node(self, agent_name, x, y):
        """Create a node representing an agent."""
        # Create ellipse for the agent
        radius = 40
        ellipse = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
        ellipse.setBrush(QColor(60, 60, 60))
        ellipse.setPen(QPen(QColor(200, 200, 200), 2))
        
        # Create text for agent name
        text = QGraphicsTextItem(agent_name)
        text.setDefaultTextColor(QColor(255, 255, 255))
        
        # Center text
        text_rect = text.boundingRect()
        text.setPos(x - text_rect.width() / 2, y - text_rect.height() / 2)
        
        # Group node
        group = QGraphicsItemGroup()
        group.addToGroup(ellipse)
        group.addToGroup(text)
        
        return group
        
    def _create_edge(self, x1, y1, x2, y2, color, message_type, message_content=None):
        """Create an edge between agents."""
        # Calculate arrow
        angle = math.atan2(y2 - y1, x2 - x1)
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Adjust start and end points to be on the edge of the agent circles
        radius = 40
        dx = math.cos(angle) * radius
        dy = math.sin(angle) * radius
        
        start_x = x1 + dx
        start_y = y1 + dy
        end_x = x2 - dx
        end_y = y2 - dy
        
        # Create the arrow line
        line = QGraphicsLineItem(start_x, start_y, end_x, end_y)
        pen = QPen(color, 2)
        pen.setStyle(QtCore.Qt.DashLine if message_type.lower() == "feedback" else QtCore.Qt.SolidLine)
        line.setPen(pen)
        
        # Create arrowhead
        arrowhead = QPolygonF()
        arrow_size = 10
        arrowhead.append(QPointF(end_x, end_y))
        arrowhead.append(QPointF(
            end_x - arrow_size * math.cos(angle - math.pi / 6),
            end_y - arrow_size * math.sin(angle - math.pi / 6)
        ))
        arrowhead.append(QPointF(
            end_x - arrow_size * math.cos(angle + math.pi / 6),
            end_y - arrow_size * math.sin(angle + math.pi / 6)
        ))
        
        arrow_head_item = QGraphicsPolygonItem(arrowhead)
        arrow_head_item.setBrush(color)
        arrow_head_item.setPen(QPen(color, 1))
        
        # Create message label
        message_label = QGraphicsTextItem(message_type)
        message_label.setDefaultTextColor(color)
        
        # Position label
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        label_rect = message_label.boundingRect()
        message_label.setPos(mid_x - label_rect.width() / 2, mid_y - label_rect.height() - 5)
        
        # Create tooltip rectangle for message content
        if message_content:
            tooltip_text = self._format_message_content(message_content)
            tooltip = QGraphicsTextItem(tooltip_text)
            tooltip.setDefaultTextColor(QColor(220, 220, 220))
            tooltip.setTextWidth(200)
            
            # Background for tooltip
            tooltip_rect = tooltip.boundingRect()
            background = QGraphicsRectItem(tooltip_rect)
            background.setBrush(QColor(40, 40, 40, 220))
            background.setPen(QPen(color, 1))
            
            # Position tooltip (hidden by default)
            background.setPos(mid_x, mid_y)
            tooltip.setPos(mid_x, mid_y)
            
            # Both are invisible by default
            background.setVisible(False)
            tooltip.setVisible(False)
            
            # Interaction for hovering over message label
            class HoverableTextItem(QGraphicsTextItem):
                def __init__(self, text, tooltip_items, parent=None):
                    super().__init__(text, parent)
                    self.tooltip_items = tooltip_items
                    self.setAcceptHoverEvents(True)
                    
                def hoverEnterEvent(self, event):
                    for item in self.tooltip_items:
                        item.setVisible(True)
                    super().hoverEnterEvent(event)
                    
                def hoverLeaveEvent(self, event):
                    for item in self.tooltip_items:
                        item.setVisible(False)
                    super().hoverLeaveEvent(event)
            
            # Replace message label with hoverable version
            message_label = HoverableTextItem(message_type, [background, tooltip])
            message_label.setDefaultTextColor(color)
            message_label.setPos(mid_x - label_rect.width() / 2, mid_y - label_rect.height() - 5)
        
        # Group all components
        group = QGraphicsItemGroup()
        group.addToGroup(line)
        group.addToGroup(arrow_head_item)
        group.addToGroup(message_label)
        
        if message_content:
            group.addToGroup(background)
            group.addToGroup(tooltip)
            
        return group
    
    def _format_message_content(self, content):
        """Format message content for tooltip display."""
        if isinstance(content, dict):
            return "\n".join([f"{k}: {str(v)[:30]}..." if len(str(v)) > 30 else f"{k}: {v}" 
                             for k, v in content.items()])
        elif isinstance(content, str):
            return content[:100] + "..." if len(content) > 100 else content
        else:
            return str(content)
            
    def _update_legend(self):
        """Update the legend with current message types."""
        # Clear existing items
        for i in reversed(range(self.legend_layout.count())):
            self.legend_layout.itemAt(i).widget().deleteLater()
            
        # Add legend items for each message type
        for message_type, data in self.message_types.items():
            color = data["color"]
            count = data["count"]
            
            # Create legend item
            legend_item = QWidget()
            item_layout = QHBoxLayout(legend_item)
            item_layout.setContentsMargins(5, 2, 5, 2)
            
            # Color indicator
            color_indicator = QLabel()
            color_indicator.setFixedSize(16, 16)
            color_indicator.setStyleSheet(f"background-color: {color.name()}; border-radius: 8px;")
            item_layout.addWidget(color_indicator)
            
            # Type label
            type_label = QLabel(f"{message_type} ({count})")
            item_layout.addWidget(type_label)
            
            self.legend_layout.addWidget(legend_item)
            
    def clear(self):
        """Clear all interactions."""
        self.interactions = []
        self.message_types = {}
        self._update_legend()
        self.scene.clear()
        
    def export_visualization(self):
        """Export the visualization as an image."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Agent Interaction Visualization", "", "PNG Files (*.png);;All Files (*)"
        )
        
        if filename:
            # Create an image to render to
            image = QImage(self.scene.sceneRect().size().toSize(), QImage.Format_ARGB32)
            image.fill(QtCore.Qt.transparent)
            
            # Create a painter to render with
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Render the scene
            self.scene.render(painter)
            painter.end()
            
            # Save the image
            image.save(filename)
            QMessageBox.information(self, "Export Complete", f"Visualization exported to {filename}")

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
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.history_list = QListWidget()
        layout.addWidget(self.history_list)
        
        self.history_list.itemClicked.connect(self.on_item_selected)
        
        export_btn = QPushButton("Export History")
        export_btn.clicked.connect(self.export_history)
        layout.addWidget(export_btn)
        
    def add_history_item(self, query, result):
        """Add a new item to the history list."""
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        item = QListWidgetItem(f"{timestamp} - {query[:50]}{'...' if len(query) > 50 else ''}")
        item.setData(Qt.UserRole, {"query": query, "result": result})
        self.history_list.insertItem(0, item)  # Add at the top
        
    def on_item_selected(self, item):
        """Handle history item selection."""
        data = item.data(Qt.UserRole)
        self.item_selected.emit(data)
        
    def export_history(self):
        """Export history to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export History", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            history_data = []
            for i in range(self.history_list.count()):
                item = self.history_list.item(i)
                history_data.append(item.data(Qt.UserRole))
                
            with open(file_path, 'w') as f:
                json.dump(history_data, f, indent=2)
                
            QMessageBox.information(self, "Export Successful", 
                                   f"History exported successfully to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export: {str(e)}")

class DashboardWidget(QWidget):
    """Dashboard widget for system overview."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # System status card
        status_card = QGroupBox("System Status")
        status_layout = QGridLayout(status_card)
        
        # Model info
        self.model_label = QLabel("Model: Unknown")
        status_layout.addWidget(self.model_label, 0, 0)
        
        # Temperature
        self.temp_label = QLabel("Temperature: 0.0")
        status_layout.addWidget(self.temp_label, 0, 1)
        
        # System resource meters
        resource_group = QGroupBox("Resources")
        resource_layout = QVBoxLayout(resource_group)
        
        # CPU usage
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU:"))
        self.cpu_meter = QProgressBar()
        self.cpu_meter.setRange(0, 100)
        self.cpu_meter.setValue(0)
        cpu_layout.addWidget(self.cpu_meter)
        resource_layout.addLayout(cpu_layout)
        
        # Memory usage
        mem_layout = QHBoxLayout()
        mem_layout.addWidget(QLabel("Memory:"))
        self.memory_meter = QProgressBar()
        self.memory_meter.setRange(0, 100)
        self.memory_meter.setValue(0)
        mem_layout.addWidget(self.memory_meter)
        resource_layout.addLayout(mem_layout)
        
        status_layout.addWidget(resource_group, 1, 0, 1, 2)
        
        main_layout.addWidget(status_card)
        
        # Agent status dashboard
        agents_card = QGroupBox("Agent Status")
        agents_layout = QVBoxLayout(agents_card)
        
        self.agent_table = QTableWidget(0, 3)
        self.agent_table.setHorizontalHeaderLabels(["Agent", "Status", "Task"])
        self.agent_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        agents_layout.addWidget(self.agent_table)
        
        main_layout.addWidget(agents_card)
        
        # Workflow visualization
        workflow_card = QGroupBox("Cognitive Workflow")
        workflow_layout = QVBoxLayout(workflow_card)
        
        self.workflow_view = QGraphicsView()
        self.workflow_scene = QGraphicsScene()
        self.workflow_view.setScene(self.workflow_scene)
        self.workflow_view.setMinimumHeight(120)
        workflow_layout.addWidget(self.workflow_view)
        
        main_layout.addWidget(workflow_card)
        
        # Statistics card
        stats_card = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_card)
        
        self.queries_label = QLabel("Queries Processed: 0")
        stats_layout.addWidget(self.queries_label, 0, 0)
        
        self.avg_time_label = QLabel("Avg. Response Time: 0 ms")
        stats_layout.addWidget(self.avg_time_label, 0, 1)
        
        self.token_label = QLabel("Total Tokens: 0")
        stats_layout.addWidget(self.token_label, 1, 0)
        
        self.success_label = QLabel("Success Rate: 0%")
        stats_layout.addWidget(self.success_label, 1, 1)
        
        main_layout.addWidget(stats_card)
        
        # Activity log
        activity_card = QGroupBox("Recent Activity")
        activity_layout = QVBoxLayout(activity_card)
        
        self.activity_log = QListWidget()
        self.activity_log.setMaximumHeight(150)
        activity_layout.addWidget(self.activity_log)
        
        main_layout.addWidget(activity_card)
        
    def update_model_info(self, model_name, temperature):
        """Update model information display."""
        self.model_label.setText(f"Model: {model_name}")
        self.temp_label.setText(f"Temperature: {temperature:.2f}")
        
    def update_resources(self, cpu_usage, memory_usage):
        """Update resource meters."""
        self.cpu_meter.setValue(int(cpu_usage))
        self.memory_meter.setValue(int(memory_usage))
        
    def update_agent_status(self, agent_status):
        """Update agent status table."""
        self.agent_table.setRowCount(0)
        row = 0
        
        for agent_name, status in agent_status.items():
            if not status:
                continue
                
            self.agent_table.insertRow(row)
            
            # Agent name
            name_item = QTableWidgetItem(agent_name)
            self.agent_table.setItem(row, 0, name_item)
            
            # Status with color coding
            state = status.get("state", "unknown")
            status_item = QTableWidgetItem(state)
            
            if state == "done":
                status_item.setBackground(QColor(100, 200, 100))
            elif state == "processing":
                status_item.setBackground(QColor(255, 200, 0))
            elif state == "error":
                status_item.setBackground(QColor(255, 100, 100))
            
            self.agent_table.setItem(row, 1, status_item)
            
            # Current task
            task_item = QTableWidgetItem(status.get("current_task", ""))
            self.agent_table.setItem(row, 2, task_item)
            
            row += 1
            
    def update_workflow(self, stages, current_stage):
        """Update workflow visualization."""
        self.workflow_scene.clear()
        
        if not stages:
            return
            
        # Draw the workflow stages
        stage_width = 100
        stage_height = 60
        stage_spacing = 30
        y_pos = 10
        
        total_width = stage_width * len(stages) + stage_spacing * (len(stages) - 1)
        x_start = 10
        
        for i, stage in enumerate(stages):
            x_pos = x_start + i * (stage_width + stage_spacing)
            
            # Create the stage box
            rect = QGraphicsRectItem(x_pos, y_pos, stage_width, stage_height)
            
            # Color based on status
            if current_stage == -2:  # Error state
                if i == 0:  # Highlight first stage in red for error
                    rect.setBrush(QColor(255, 100, 100))  # Error red
                else:
                    rect.setBrush(QColor(200, 200, 200))  # Pending
            elif current_stage == -1:  # Initial state
                rect.setBrush(QColor(200, 200, 200))  # Pending
            elif i < current_stage:
                rect.setBrush(QColor(100, 200, 100))  # Completed
            elif i == current_stage:
                rect.setBrush(QColor(255, 200, 0))    # In progress
            else:
                rect.setBrush(QColor(200, 200, 200))  # Pending
                
            self.workflow_scene.addItem(rect)
            
            # Stage name
            text = QGraphicsTextItem(stage)
            text.setPos(x_pos + (stage_width - text.boundingRect().width()) / 2, 
                       y_pos + (stage_height - text.boundingRect().height()) / 2)
            self.workflow_scene.addItem(text)
            
            # Draw connecting arrow
            if i > 0:
                prev_x = x_pos - stage_spacing
                line = QGraphicsLineItem(prev_x, y_pos + stage_height/2, 
                                        x_pos, y_pos + stage_height/2)
                self.workflow_scene.addItem(line)
                
                # Arrow head
                arrow = QPolygonF()
                arrow.append(QPointF(x_pos, y_pos + stage_height/2))
                arrow.append(QPointF(x_pos - 5, y_pos + stage_height/2 - 5))
                arrow.append(QPointF(x_pos - 5, y_pos + stage_height/2 + 5))
                arrow_item = QGraphicsPolygonItem(arrow)
                arrow_item.setBrush(QColor(0, 0, 0))
                self.workflow_scene.addItem(arrow_item)
        
        # Add error indicator if in error state
        if current_stage == -2:
            error_text = QGraphicsTextItem("Error occurred")
            error_text.setDefaultTextColor(QColor(255, 0, 0))
            error_text.setPos(x_start, y_pos + stage_height + 10)
            self.workflow_scene.addItem(error_text)
        
        # Fit view to content
        self.workflow_view.fitInView(self.workflow_scene.sceneRect(), Qt.KeepAspectRatio)
        
    def update_statistics(self, stats):
        """Update statistics display."""
        if "queries_processed" in stats:
            self.queries_label.setText(f"Queries Processed: {stats['queries_processed']}")
            
        if "avg_time" in stats:
            self.avg_time_label.setText(f"Avg. Response Time: {stats['avg_time']:.2f} ms")
            
        if "total_tokens" in stats:
            self.token_label.setText(f"Total Tokens: {stats['total_tokens']}")
            
        if "success_rate" in stats:
            self.success_label.setText(f"Success Rate: {stats['success_rate']:.1f}%")
            
    def add_activity(self, activity_text):
        """Add an entry to the activity log."""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.activity_log.insertItem(0, f"{timestamp} - {activity_text}")
        
        # Limit the number of entries
        while self.activity_log.count() > 20:
            self.activity_log.takeItem(self.activity_log.count() - 1)

class NeoCortexGUI(QMainWindow):
    """Main GUI application for NeoCortex."""

    def __init__(self):
        super().__init__()
        self.neocortex = NeoCortex()
        self.current_result = None
        self.reasoning_thread = None
        self.init_ui()
        self.apply_dark_theme()
        
        # Create multi-agent tab as a proper MultiAgentTabWidget
        self.multi_agent_tab = MultiAgentTabWidget()
        
        # Add the multi-agent tab to the result display widget's tab widget
        if hasattr(self.result_display_widget, 'tab_widget'):
            self.result_display_widget.tab_widget.addTab(self.multi_agent_tab, "Multi-Agent System")
        
        # Setup timer for agent status updates
        self.agent_status_timer = QTimer()
        self.agent_status_timer.timeout.connect(self.update_agent_status)
        self.agent_status_timer.start(1000)  # Update every second

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

        # Create the main splitter for the workspace with three panels
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Left side - Module controls and settings
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # Create a tab widget for left panel
        left_tabs = QTabWidget()

        # Modules tab
        self.module_toggle_widget = ModuleToggleWidget(self.neocortex)
        left_tabs.addTab(self.module_toggle_widget, "Modules")

        # Settings tab
        self.model_settings_widget = ModelSettingsWidget(self.neocortex)
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

        # Middle - Dashboard and overview
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        
        # Create Dashboard
        self.dashboard_widget = DashboardWidget()
        middle_layout.addWidget(self.dashboard_widget)
        
        # Set initial model info
        self.dashboard_widget.update_model_info(
            self.neocortex.model_name, 
            self.neocortex.temperature
        )
        
        # Initialize the workflow view
        workflow_stages = ["Analysis", "Criticism", "Creative", "Logical", "Synthesis", "Implementation", "Evaluation"]
        self.dashboard_widget.update_workflow(workflow_stages, -1)  # -1 means no current stage

        # Right - Results display
        self.result_display_widget = ResultDisplayWidget()

        # Add widgets to splitter
        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(middle_widget)
        self.main_splitter.addWidget(self.result_display_widget)

        # Set initial splitter sizes (30% | 40% | 30%)
        total_width = 1200  # Approximate initial width
        self.main_splitter.setSizes([int(total_width * 0.25), int(total_width * 0.4), int(total_width * 0.35)])

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
            
        # Disable query input and start button during processing
        self.query_edit.setReadOnly(True)
        self.reason_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Clear previous result displays
        self.realtime_reasoning.clear()
        if hasattr(self, 'multi_agent_tab'):
            self.multi_agent_tab.agent_interaction_viz.clear()
            self.multi_agent_tab.error_recovery_widget.clear()
            
        # Set up progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_status.setText("Initializing...")
        self.progress_status.setVisible(True)
        
        # Use multi-agent system if enabled in the tab
        use_agents = True
        if hasattr(self, 'multi_agent_tab'):
            use_agents = self.multi_agent_tab.is_enabled()
            
        # Get fast mode setting
        show_work = True
        if hasattr(self, 'fast_mode_toggle'):
            show_work = not self.fast_mode_toggle.is_fast_mode()
            
        # Create the reasoning thread
        if use_agents:
            self.reasoning_thread = AgentReasoningThread(self.neocortex, query, show_work)
            
            # Connect agent-specific signals if multi-agent tab exists
            if hasattr(self, 'multi_agent_tab'):
                self.reasoning_thread.agent_communication.connect(self._handle_agent_communication)
                self.reasoning_thread.error_recovery.connect(self._handle_error_recovery)
        else:
            self.reasoning_thread = ReasoningThread(self.neocortex, query, show_work)
            
        # Connect signals for progress updates and results
        self.reasoning_thread.progress_update.connect(self.update_progress)
        self.reasoning_thread.result_ready.connect(self.handle_result)
        self.reasoning_thread.error_occurred.connect(self.handle_error)
        self.reasoning_thread.thought_generated.connect(self.realtime_reasoning.add_thought)
        
        # Start the thread
        self.reasoning_thread.start()
        
        # Update status 
        if use_agents:
            status_text = "Processing with multi-agent system..."
        else:
            status_text = "Processing with neural reasoning..."
        self.statusBar().showMessage(status_text)
        
    def _handle_agent_communication(self, from_agent, to_agent, message_type, content):
        """Handle agent communication signals."""
        if hasattr(self, 'multi_agent_tab'):
            self.multi_agent_tab.add_interaction(from_agent, to_agent, message_type, content)
            
    def _handle_error_recovery(self, stage, error_message, recovery_action, success):
        """Handle error recovery signals."""
        if hasattr(self, 'multi_agent_tab'):
            self.multi_agent_tab.add_error_recovery(stage, error_message, recovery_action, success)

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
        
        # Update the dashboard workflow visualization
        if hasattr(self, 'dashboard_widget'):
            # Define the workflow stages
            workflow_stages = ["Analysis", "Criticism", "Creative", "Logical", "Synthesis", "Implementation", "Evaluation"]
            
            # Calculate current stage based on progress
            if progress > 0:
                progress_per_stage = 100 / len(workflow_stages)
                current_stage = min(int(progress / progress_per_stage), len(workflow_stages))
                
                # Update the workflow visualization
                self.dashboard_widget.update_workflow(workflow_stages, current_stage)
                
                # Add activity based on current stage
                if progress > 0 and progress % 20 == 0:  # Add activity at 20%, 40%, 60%, 80%, 100%
                    stage_index = min(int(progress / 20), len(workflow_stages) - 1)
                    self.dashboard_widget.add_activity(f"Starting {workflow_stages[stage_index]} phase...")

    def handle_result(self, result):
        """Handle the reasoning result."""
        self.current_result = result

        # Display the result
        self.result_display_widget.display_result(result)

        # Add to history
        self.history_widget.add_history_item(self.query_edit.toPlainText(), result)

        # Update dashboard
        if hasattr(self, 'dashboard_widget'):
            # Add completion activity
            self.dashboard_widget.add_activity(f"Query completed: {self.query_edit.toPlainText()[:30]}...")
            
            # Update statistics
            if not hasattr(self.neocortex, 'queries_processed'):
                self.neocortex.queries_processed = 0
                self.neocortex.avg_response_time = 0
                self.neocortex.total_tokens_used = 0
                
            # Update query count
            self.neocortex.queries_processed += 1
            
            # Update response time tracking
            processing_time = result.get("processing_time", 0)
            self.neocortex.avg_response_time = (
                (self.neocortex.avg_response_time * (self.neocortex.queries_processed - 1) + processing_time) 
                / self.neocortex.queries_processed
            )
            
            # Update token usage
            token_count = result.get("token_count", 0)
            if token_count:
                if not hasattr(self.neocortex, 'total_tokens_used'):
                    self.neocortex.total_tokens_used = 0
                self.neocortex.total_tokens_used += token_count
            
            # Update statistics on dashboard
            stats = {
                "queries_processed": self.neocortex.queries_processed,
                "avg_time": self.neocortex.avg_response_time,
                "total_tokens": self.neocortex.total_tokens_used,
                "success_rate": getattr(self.neocortex, 'success_rate', 98.0)
            }
            self.dashboard_widget.update_statistics(stats)
            
            # Update the workflow to show completion
            workflow_stages = ["Analysis", "Criticism", "Creative", "Logical", "Synthesis", "Implementation", "Evaluation"]
            self.dashboard_widget.update_workflow(workflow_stages, len(workflow_stages))

        # Re-enable UI elements
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.query_edit.setReadOnly(False)

        self.status_bar.showMessage("Processing complete")

    def handle_error(self, error_message):
        """Handle errors that occur during reasoning."""
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        
        # Update UI elements
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.query_edit.setReadOnly(False)
        self.progress_bar.setValue(0)
        self.progress_status.setText("Error")
        self.status_bar.showMessage(f"Error: {error_message[:50]}{'...' if len(error_message) > 50 else ''}")
        
        # Update dashboard
        if hasattr(self, 'dashboard_widget'):
            self.dashboard_widget.add_activity(f"Error occurred: {error_message[:50]}{'...' if len(error_message) > 50 else ''}")
            
            # Log the error in statistics
            if not hasattr(self.neocortex, 'error_count'):
                self.neocortex.error_count = 0
            self.neocortex.error_count += 1
            
            # Update success rate
            if hasattr(self.neocortex, 'queries_processed'):
                total_queries = self.neocortex.queries_processed + 1  # Include this failed query
                success_rate = ((total_queries - self.neocortex.error_count) / total_queries) * 100
                self.neocortex.success_rate = success_rate
                
                # Update statistics
                stats = {
                    "queries_processed": total_queries,
                    "avg_time": getattr(self.neocortex, 'avg_response_time', 0),
                    "total_tokens": getattr(self.neocortex, 'total_tokens_used', 0),
                    "success_rate": success_rate
                }
                self.dashboard_widget.update_statistics(stats)
            
            # Reset workflow visualization to indicate error
            workflow_stages = ["Analysis", "Criticism", "Creative", "Logical", "Synthesis", "Implementation", "Evaluation"]
            # Set current stage to -2 to indicate error state
            self.dashboard_widget.update_workflow(workflow_stages, -2)

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
            
            # Update dashboard with new model info if available
            if hasattr(self, 'dashboard_widget'):
                self.dashboard_widget.update_model_info(
                    self.neocortex.model_name,
                    self.neocortex.temperature
                )
                
            self.status_bar.showMessage(f"Model settings updated: {settings['model']}, temp={settings['temperature']}, tokens={settings.get('max_tokens', 8000)}")
        except Exception as e:
            QMessageBox.critical(self, "Settings Error", f"Failed to update settings: {str(e)}")

    def load_history_item(self, item):
        """Load a history item into the input field."""
        self.query_edit.setPlainText(item["query"])
        self.result_display_widget.display_result(item["result"])

    def new_session(self):
        """Start a new session."""
        self.query_edit.clear()
        self.result_display_widget.clear()
        self.history_widget.clear()
        self.status_bar.showMessage("New session started")

    def save_session(self):
        """Save the current session."""
        # Implement session saving functionality
        pass

    def load_session(self):
        """Load a previous session."""
        # Implement session loading functionality
        pass

    def show_settings(self):
        """Show the settings dialog."""
        # Implement settings dialog functionality
        pass

    def show_help(self):
        """Show help documentation."""
        # Implement help documentation functionality
        pass

    def update_agent_status(self):
        """Update the status of all agents."""
        agent_status = self.neocortex.multi_agent_system.get_agent_status()
        self.dashboard_widget.update_agent_status(agent_status)
        self.multi_agent_tab.update_agent_status(agent_status)

class ReasoningThread(QThread):
    """Thread for running the reasoning process."""
    progress_update = pyqtSignal(str, int)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    thought_generated = pyqtSignal(str)
    
    def __init__(self, neocortex, query, show_work=True):
        super().__init__()
        self.neocortex = neocortex
        self.query = query
        self.show_work = show_work
        
    def run(self):
        try:
            self.progress_update.emit("Starting reasoning process...", 0)
            result = self.neocortex.solve(self.query, self.show_work)
            self.progress_update.emit("Reasoning complete!", 100)
            self.result_ready.emit(result)
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Error: {str(e)}\n{traceback.format_exc()}")

class AgentReasoningThread(ReasoningThread):
    """Thread for running the multi-agent reasoning process."""
    agent_communication = pyqtSignal(str, str, str, object)  # from_agent, to_agent, message_type, content
    error_recovery = pyqtSignal(str, str, str, bool)  # stage, error_message, recovery_action, success

    def run(self):
        try:
            self.progress_update.emit("Initializing agent-based reasoning...", 0)
            self.thought_generated.emit("Multi-agent system activated for reasoning...")
            
            # Attach thought_generated signal
            self.neocortex.thought_generated = self.thought_generated
            
            # Set up hooks for agent communications and error recovery
            if hasattr(self.neocortex, "multi_agent_system"):
                original_process_query = self.neocortex.multi_agent_system.process_query
                
                def intercepted_process_query(query):
                    """Intercept the process_query method to monitor agent activities."""
                    result = original_process_query(query)
                    
                    # Emit agent communications signals
                    if "agent_communications" in result:
                        for comm in result["agent_communications"]:
                            self.agent_communication.emit(
                                comm["sender"],
                                comm["recipient"],
                                comm["type"],
                                comm.get("content_summary", "")
                            )
                    
                    # Emit error recovery signals
                    if "error_log" in result:
                        for error in result["error_log"]:
                            self.error_recovery.emit(
                                error["stage"],
                                error.get("error", "Unknown error"),
                                error.get("recovery_action", ""),
                                "recovery_action" in error
                            )
                    
                    return result
                
                # Apply the interceptor
                self.neocortex.multi_agent_system.process_query = intercepted_process_query
            
            # Process with multi-agent system
            result = self.neocortex.solve(self.query, self.show_work, use_agents=True)
            
            # Update progress for each agent
            if hasattr(self.neocortex, "multi_agent_system"):
                agent_status = self.neocortex.multi_agent_system.get_agent_status()
                completed = sum(1 for status in agent_status.values() 
                               if status and status.get("state") == "done")
                total = len(agent_status)
                if total > 0:
                    progress = int((completed / total) * 100)
                    self.progress_update.emit(f"Agents completed: {completed}/{total}", progress)
            
            self.progress_update.emit("Processing complete!", 100)
            self.result_ready.emit(result)
            
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Error: {str(e)}\n{traceback.format_exc()}")

    error_recovery = pyqtSignal(str, str, str, bool)  # stage, error_message, recovery_action, success

    def run(self):
        try:
            self.progress_update.emit("Initializing agent-based reasoning...", 0)
            self.thought_generated.emit("Multi-agent system activated for reasoning...")
            
            # Attach thought_generated signal
            self.neocortex.thought_generated = self.thought_generated
            
            # Set up hooks for agent communications and error recovery
            if hasattr(self.neocortex, "multi_agent_system"):
                original_process_query = self.neocortex.multi_agent_system.process_query
                
                def intercepted_process_query(query):
                    """Intercept the process_query method to monitor agent activities."""
                    result = original_process_query(query)
                    
                    # Emit agent communications signals
                    if "agent_communications" in result:
                        for comm in result["agent_communications"]:
                            self.agent_communication.emit(
                                comm["sender"],
                                comm["recipient"],
                                comm["type"],
                                comm.get("content_summary", "")
                            )
                    
                    # Emit error recovery signals
                    if "error_log" in result:
                        for error in result["error_log"]:
                            self.error_recovery.emit(
                                error["stage"],
                                error.get("error", "Unknown error"),
                                error.get("recovery_action", ""),
                                "recovery_action" in error
                            )
                    
                    return result
                
                # Apply the interceptor
                self.neocortex.multi_agent_system.process_query = intercepted_process_query
            
            # Process with multi-agent system
            result = self.neocortex.solve(self.query, self.show_work, use_agents=True)
            
            # Update progress for each agent
            if hasattr(self.neocortex, "multi_agent_system"):
                agent_status = self.neocortex.multi_agent_system.get_agent_status()
                completed = sum(1 for status in agent_status.values() 
                               if status and status.get("state") == "done")
                total = len(agent_status)
                if total > 0:
                    progress = int((completed / total) * 100)
                    self.progress_update.emit(f"Agents completed: {completed}/{total}", progress)
            
            self.progress_update.emit("Processing complete!", 100)
            self.result_ready.emit(result)
            
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Error: {str(e)}\n{traceback.format_exc()}")

class CognitiveGraphVisualizer(QWidget):
    """Widget for visualizing the cognitive graph."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph_data = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create matplotlib figure
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Empty placeholder
        self.placeholder_label = QLabel("No cognitive graph data available")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.placeholder_label)
        
        # Set initial visibility
        self.canvas.setVisible(False)
        self.placeholder_label.setVisible(True)
        
    def update_graph(self, graph_data):
        """Update the visualization with new graph data."""
        self.graph_data = graph_data
        
        if not graph_data or "nodes" not in graph_data or not graph_data["nodes"]:
            # No valid data, show placeholder
            self.canvas.setVisible(False)
            self.placeholder_label.setVisible(True)
            return
            
        # Hide placeholder, show canvas
        self.canvas.setVisible(True)
        self.placeholder_label.setVisible(False)
        
        # Clear previous figure
        self.figure.clear()
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in graph_data["nodes"].items():
            G.add_node(node_id, 
                      node_type=node.get("node_type", "unknown"),
                      content=node.get("content", "")[:50])  # Truncate for display
        
        # Add edges
        for node_id, node in graph_data["nodes"].items():
            parent_id = node.get("parent_id")
            if parent_id:
                G.add_edge(parent_id, node_id)
            
            # Add edges to children if specified
            for child_id in node.get("children", []):
                G.add_edge(node_id, child_id)
        
        # Create plot
        ax = self.figure.add_subplot(111)
        
        # Define position layout
        pos = nx.spring_layout(G, seed=42)
        
        # Define node colors based on node_type
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node]["node_type"]
            color = COLOR_SCHEME["node_colors"].get(node_type, "#CCCCCC")
            node_colors.append(color)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=15, alpha=0.7)
        
        # Add labels
        labels = {}
        for node in G.nodes():
            node_type = G.nodes[node]["node_type"]
            labels[node] = f"{node_type}"
        
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        # Remove axis
        ax.set_axis_off()
        
        # Update canvas
        self.canvas.draw()

class RealtimeReasoningWidget(QWidget):
    """Widget for displaying real-time reasoning updates."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Realtime Reasoning")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Thought display area
        self.thought_display = QTextEdit()
        self.thought_display.setReadOnly(True)
        layout.addWidget(self.thought_display)
        
        # Apply styling
        self.setStyleSheet("""
        QTextEdit {
            background-color: #252525;
            color: #E0E0E0;
            border: 1px solid #444;
            border-radius: 4px;
        }
        """)
        
    def add_thought(self, thought_text):
        """Add a thought to the display."""
        # Format the thought with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_thought = f"[{timestamp}] {thought_text}\n"
        
        # Add to display
        self.thought_display.append(formatted_thought)
        
        # Scroll to bottom
        scrollbar = self.thought_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def clear(self):
        """Clear all thoughts."""
        self.thought_display.clear()

class FastModeToggle(QWidget):
    """Toggle switch for fast mode vs detailed reasoning."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_fast = False
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        self.label = QLabel("Reasoning Mode:")
        layout.addWidget(self.label)
        
        # Toggle button
        self.toggle_btn = QPushButton("Detailed")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(False)
        self.toggle_btn.clicked.connect(self.toggle_mode)
        layout.addWidget(self.toggle_btn)
        
        # Apply styling
        self.toggle_btn.setStyleSheet("""
        QPushButton {
            background-color: #5D8AA8;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 5px 10px;
        }
        QPushButton:checked {
            background-color: #FF7F50;
        }
        """)
        
    def toggle_mode(self):
        """Toggle between fast and detailed mode."""
        self.is_fast = self.toggle_btn.isChecked()
        self.toggle_btn.setText("Fast" if self.is_fast else "Detailed")
        
    def is_fast_mode(self):
        """Return True if in fast mode, False if in detailed mode."""
        return self.is_fast

class ModuleToggleWidget(QWidget):
    """Widget for toggling cognitive modules on/off."""
    
    def __init__(self, neocortex, parent=None):
        super().__init__(parent)
        self.neocortex = neocortex
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Cognitive Modules")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Create toggle switches for each module
        self.module_toggles = {}
        
        for module, state in self.neocortex.module_states.items():
            # Create module row
            module_row = QWidget()
            row_layout = QHBoxLayout(module_row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            # Module label (convert snake_case to Title Case)
            module_label = " ".join(word.capitalize() for word in module.split("_"))
            label = QLabel(module_label)
            row_layout.addWidget(label)
            
            # Spacer
            row_layout.addStretch()
            
            # Toggle checkbox
            toggle = QCheckBox()
            toggle.setChecked(state)
            toggle.stateChanged.connect(lambda checked, m=module: self.toggle_module(m, checked))
            row_layout.addWidget(toggle)
            
            # Add to layout
            layout.addWidget(module_row)
            
            # Store reference
            self.module_toggles[module] = toggle
            
        # Apply styling
        self.setStyleSheet("""
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
        QCheckBox::indicator:unchecked {
            background-color: #444;
            border: 2px solid #666;
            border-radius: 4px;
        }
        QCheckBox::indicator:checked {
            background-color: #7B68EE;
            border: 2px solid #8A2BE2;
            border-radius: 4px;
        }
        """)
        
    def toggle_module(self, module, state):
        """Toggle a cognitive module on/off."""
        self.neocortex.module_states[module] = bool(state)

class ModelSettingsWidget(QWidget):
    """Widget for adjusting model settings."""
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, neocortex, parent=None):
        super().__init__(parent)
        self.neocortex = neocortex
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Model Settings")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "deepseek/deepseek-chat-v3",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "meta-llama/Llama-3-70b-chat",
            "gpt-4-turbo"
        ])
        self.model_combo.setCurrentText(self.neocortex.model_name)
        self.model_combo.currentTextChanged.connect(self.update_model)
        model_layout.addWidget(self.model_combo)
        
        layout.addLayout(model_layout)
        
        # Temperature setting
        temp_layout = QHBoxLayout()
        temp_label = QLabel("Temperature:")
        temp_layout.addWidget(temp_label)
        
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(0)
        self.temp_slider.setMaximum(100)
        self.temp_slider.setValue(int(self.neocortex.temperature * 100))
        self.temp_slider.valueChanged.connect(self.update_temperature)
        temp_layout.addWidget(self.temp_slider)
        
        self.temp_value = QLabel(f"{self.neocortex.temperature:.2f}")
        temp_layout.addWidget(self.temp_value)
        
        layout.addLayout(temp_layout)
        
        # Max tokens setting
        tokens_layout = QHBoxLayout()
        tokens_label = QLabel("Max Tokens:")
        tokens_layout.addWidget(tokens_label)
        
        self.tokens_spinner = QSpinBox()
        self.tokens_spinner.setMinimum(1000)
        self.tokens_spinner.setMaximum(16000)
        self.tokens_spinner.setSingleStep(1000)
        self.tokens_spinner.setValue(self.neocortex.max_tokens)
        self.tokens_spinner.valueChanged.connect(self.update_max_tokens)
        tokens_layout.addWidget(self.tokens_spinner)
        
        layout.addLayout(tokens_layout)
        
        # Reasoning depth
        depth_layout = QHBoxLayout()
        depth_label = QLabel("Reasoning Depth:")
        depth_layout.addWidget(depth_label)
        
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["minimal", "balanced", "thorough"])
        self.depth_combo.setCurrentText(self.neocortex.reasoning_depth)
        self.depth_combo.currentTextChanged.connect(self.update_reasoning_depth)
        depth_layout.addWidget(self.depth_combo)
        
        layout.addLayout(depth_layout)
        
        # Apply styling
        self.setStyleSheet("""
        QComboBox, QSpinBox {
            background-color: #333;
            color: #E0E0E0;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 4px;
        }
        QSlider::groove:horizontal {
            height: 8px;
            background: #333;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #7B68EE;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        """)
        
    def update_model(self, model_name):
        """Update the model name."""
        self.neocortex.model_name = model_name
        # Emit settings changed signal
        settings = {
            "model": model_name,
            "temperature": self.neocortex.temperature,
            "max_tokens": self.neocortex.max_tokens,
            "reasoning_depth": self.neocortex.reasoning_depth
        }
        self.settings_changed.emit(settings)
        
    def update_temperature(self, value):
        """Update the temperature setting."""
        temperature = value / 100.0
        self.neocortex.temperature = temperature
        self.temp_value.setText(f"{temperature:.2f}")
        # Emit settings changed signal
        settings = {
            "model": self.neocortex.model_name,
            "temperature": temperature,
            "max_tokens": self.neocortex.max_tokens,
            "reasoning_depth": self.neocortex.reasoning_depth
        }
        self.settings_changed.emit(settings)
        
    def update_max_tokens(self, value):
        """Update the max tokens setting."""
        self.neocortex.max_tokens = value
        self.neocortex.context_manager.max_tokens_per_call = value
        # Emit settings changed signal
        settings = {
            "model": self.neocortex.model_name,
            "temperature": self.neocortex.temperature,
            "max_tokens": value,
            "reasoning_depth": self.neocortex.reasoning_depth
        }
        self.settings_changed.emit(settings)
        
    def update_reasoning_depth(self, depth):
        """Update the reasoning depth setting."""
        self.neocortex.reasoning_depth = depth
        # Emit settings changed signal
        settings = {
            "model": self.neocortex.model_name,
            "temperature": self.neocortex.temperature,
            "max_tokens": self.neocortex.max_tokens,
            "reasoning_depth": depth
        }
        self.settings_changed.emit(settings)

class ModuleActivityWidget(QWidget):
    """Widget for displaying cognitive module activity."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.activity_data = {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Module Activity")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Create module activity bars
        self.module_bars = {}
        
        # Default modules
        default_modules = [
            "Concept Network", "Self-Regulation", "Spatial Reasoning",
            "Causal Reasoning", "Counterfactual", "Metacognition"
        ]
        
        for module in default_modules:
            # Create module row
            module_row = QWidget()
            row_layout = QHBoxLayout(module_row)
            row_layout.setContentsMargins(5, 2, 5, 2)
            
            # Module label
            label = QLabel(module)
            label.setMinimumWidth(120)
            row_layout.addWidget(label)
            
            # Activity bar
            activity_bar = QProgressBar()
            activity_bar.setMinimum(0)
            activity_bar.setMaximum(100)
            activity_bar.setValue(0)
            activity_bar.setTextVisible(False)
            activity_bar.setFixedHeight(12)
            row_layout.addWidget(activity_bar)
            
            # Add to layout
            layout.addWidget(module_row)
            
            # Store reference
            self.module_bars[module] = activity_bar
            
        # Apply styling
        self.setStyleSheet("""
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #333;
        }
        QProgressBar::chunk {
            background-color: #7B68EE;
            border-radius: 2px;
        }
        """)
        
    def update_activity(self, activity_data):
        """Update the activity display with new data."""
        self.activity_data = activity_data
        
        for module, value in activity_data.items():
            if module in self.module_bars:
                # Update the progress bar
                self.module_bars[module].setValue(int(value * 100))
                
                # Color based on activity level
                if value < 0.3:
                    color = "#5D8AA8"  # Low activity (blue)
                elif value < 0.7:
                    color = "#DDA0DD"  # Medium activity (purple)
                else:
                    color = "#FF7F50"  # High activity (orange)
                    
                self.module_bars[module].setStyleSheet(f"""
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 2px;
                }}
                """)

class MultiAgentTabWidget(QWidget):
    """Widget for displaying the multi-agent system status and interactions."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        
        # Agent status section
        status_group = QGroupBox("Agent Status")
        status_layout = QVBoxLayout()
        
        # Create grid for agent information
        self.status_grid = QTableWidget()
        self.status_grid.setColumnCount(3)
        self.status_grid.setHorizontalHeaderLabels(["Agent", "Status", "Knowledge Size"])
        self.status_grid.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.status_grid.setSelectionBehavior(QTableWidget.SelectRows)
        self.status_grid.verticalHeader().setVisible(False)
        
        status_layout.addWidget(self.status_grid)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Agent interaction visualization
        interaction_group = QGroupBox("Agent Interactions")
        interaction_layout = QVBoxLayout()
        self.agent_interaction_viz = AgentInteractionViz()
        interaction_layout.addWidget(self.agent_interaction_viz)
        interaction_group.setLayout(interaction_layout)
        layout.addWidget(interaction_group)
        
        # Error recovery section
        error_group = QGroupBox("Error Recovery")
        error_layout = QVBoxLayout()
        self.error_recovery_widget = QTextEdit()
        self.error_recovery_widget.setReadOnly(True)
        error_layout.addWidget(self.error_recovery_widget)
        error_group.setLayout(error_layout)
        layout.addWidget(error_group)
        
        self.setLayout(layout)
        
    def update_agent_status(self, agent_status):
        """Update the agent status display."""
        # Clear existing rows
        self.status_grid.setRowCount(0)
        
        # Add new rows for each agent
        for agent_name, status in agent_status.items():
            if status:
                row = self.status_grid.rowCount()
                self.status_grid.insertRow(row)
                
                # Set agent name
                name_item = QTableWidgetItem(status.get("name", agent_name))
                self.status_grid.setItem(row, 0, name_item)
                
                # Set status with color coding
                state = status.get("state", "unknown")
                status_item = QTableWidgetItem(state)
                if state == "working":
                    status_item.setBackground(QColor("#FFD700"))  # Gold
                elif state == "done":
                    status_item.setBackground(QColor("#90EE90"))  # Light green
                elif state == "error":
                    status_item.setBackground(QColor("#FFA07A"))  # Light salmon
                self.status_grid.setItem(row, 1, status_item)
                
                # Set knowledge size
                knowledge_size = status.get("knowledge_size", 0)
                knowledge_item = QTableWidgetItem(str(knowledge_size))
                self.status_grid.setItem(row, 2, knowledge_item)
                
    def add_interaction(self, from_agent, to_agent, message_type, content=None):
        """Add an agent interaction to the visualization."""
        self.agent_interaction_viz.add_interaction(from_agent, to_agent, message_type, content=content)
        
    def add_error_recovery(self, stage, error_message, recovery_action, success):
        """Add an error recovery event to the display."""
        # Format the recovery information
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = "#90EE90" if success else "#FFA07A"  # Green if successful, red if failed
        
        html = f"""
        <div style="margin-bottom: 10px; border-left: 4px solid {color}; padding-left: 10px;">
            <p><b>[{timestamp}] Error in {stage.capitalize()} stage:</b></p>
            <p>{error_message}</p>
            <p><b>Recovery Action:</b> {recovery_action or "None"}</p>
            <p><b>Status:</b> <span style="color: {color};">{
                "Recovery Successful" if success else "Recovery Failed"
            }</span></p>
        </div>
        """
        
        self.error_recovery_widget.append(html)
        
    def clear(self):
        """Clear all visualizations and data."""
        self.status_grid.setRowCount(0)
        self.agent_interaction_viz.clear()
        self.error_recovery_widget.clear()
        
    def is_enabled(self):
        """Check if multi-agent system should be used."""
        return True  # Could add a checkbox to toggle this in the future

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("NeoCortex with Multi-Agent Architecture")

    window = NeoCortexGUI()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()