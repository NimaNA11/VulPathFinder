import json
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from gensim.models import Word2Vec
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import os
import html
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('test_gnn.log'), logging.StreamHandler()])

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Node type mapping
type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpr': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpr': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryOp': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostfixExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'ClassStaticIdentifier': 58, 'ForRangeInit': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69, 'ThrowExpression': 70, 'IncDecOp': 71, 'NewExpression': 72, 'DeleteExpression': 73,
    'BoolExpression': 74, 'CharExpression': 75, 'DoubleExpression': 76, 'IntegerExpression': 77,
    'PointerExpression': 78, 'StringExpression': 79, 'ExpressionHolderStatement': 80
}

# Parser class for data configuration
class DataParser:
    def __init__(self):
        self.num_workers = 4
        self.test_batch_size = 32
        self.device = device
        self.num_classes = 2  # sink, none

# GNN model definition
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layer_num=6):
        super(GNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(layer_num - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.layers.append(GCNConv(hidden_channels, out_channels))
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, (layer, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            x_new = layer(x, edge_index)
            x_new = norm(x_new)
            x_new = torch.relu(x_new)
            x = x_new + x if i > 0 else x_new
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x

@dataclass
class VisualizationConfig:
    """Configuration class for visualization settings."""
    base_dir: str = "/home/nimana11/Thesis/codes/VulExplainerExp-84ED_2/datasets/buffer overflow/function/testcases"
    output_dir: str = "visualizations"
    output_filename: str = "samples_visualization.html"
    max_node_content_length: int = 30
    network_height: int = 700
    
    # Color schemes
    role_colors: Dict[str, str] = None
    edge_colors: Dict[str, str] = None
    node_colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.role_colors is None:
            self.role_colors = {
                "sink": "border-left: 4px solid #1e88e5; background-color: #e3f2fd;",
                "none": "background-color: #ffffff;"
            }
        
        if self.edge_colors is None:
            self.edge_colors = {
                "cfgEdges": "#1976d2",
                "ddgEdges": "#d32f2f", 
                "cdgEdges": "#388e3c"
            }
            
        if self.node_colors is None:
            self.node_colors = {
                "sink": "#bbdefb",
                "none": "#e0e0e0"
            }

class CodeFileLoader:
    """Handles loading and caching of source code files."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self._file_cache = {}
    
    def load_code_lines(self, filename: str) -> List[str]:
        """Load source code lines with caching and error handling."""
        if filename in self._file_cache:
            return self._file_cache[filename]
        
        if not filename:
            error_content = ["/* Error: No filename specified */"]
            self._file_cache[filename] = error_content
            return error_content
        
        try:
            cwe_category = filename.split('__')[0]
            cwe_dir = self.base_dir / cwe_category
            
            if not cwe_dir.exists():
                raise FileNotFoundError(f"CWE directory {cwe_dir} not found")
            
            for subfolder in cwe_dir.iterdir():
                if subfolder.is_dir() and subfolder.name.startswith('s') and subfolder.name[1:].isdigit():
                    file_path = subfolder / filename
                    if file_path.exists():
                        with file_path.open('r', encoding='utf-8') as f:
                            code_lines = f.readlines()
                        self._file_cache[filename] = code_lines
                        return code_lines
            
            raise FileNotFoundError(f"File {filename} not found in any subfolder of {cwe_dir}")
            
        except Exception as e:
            logging.error(f"Failed to load file {filename}: {e}")
            error_content = [f"/* Error: Could not load file {filename}: {str(e)} */"]
            self._file_cache[filename] = error_content
            return error_content

class NodeProcessor:
    """Processes node data and creates mappings."""
    
    @staticmethod
    def create_line_mappings(nodes: List[str], predictions: List[int], labels: List[int]) -> Tuple[Dict[int, Tuple[str, str]], set]:
        """Create line number to role mappings and node lines set."""
        idx_to_role = {0: "none", 1: "sink"}
        line_to_role = {}
        node_lines = set()
        
        for node_idx, node in enumerate(nodes):
            try:
                node_dict = json.loads(node)
                line_num = node_dict.get("line")
                if line_num is None:
                    logging.warning(f"No line number found for node {node_idx}")
                    continue
                    
                node_lines.add(line_num)
                true_role = node_dict.get("role", "none")
                pred_role = idx_to_role.get(predictions[node_idx], "none") if node_idx < len(predictions) else "none"
                line_to_role[line_num] = (pred_role, true_role)
                
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Error processing node {node_idx}: {e}")
                continue
        
        return line_to_role, node_lines

class HTMLGenerator:
    """Generates HTML components for visualization."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def generate_html_header(self) -> str:
        """Generate HTML header with styles and scripts."""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Vulnerability Analysis Visualization</title>
            <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Fira+Code&display=swap" rel="stylesheet">
            <style>
                body {{ font-family: 'Inter', sans-serif; margin: 20px; background-color: #f7fafc; }}
                .sample {{ display: none; margin-bottom: 40px; }}
                .sample.active {{ display: block; }}
                
                /* Tab Styles */
                .tabs {{
                    display: flex;
                    border-bottom: 2px solid #e2e8f0;
                    margin-bottom: 20px;
                    background: white;
                    border-radius: 8px 8px 0 0;
                    overflow: hidden;
                }}
                
                .tab {{
                    flex: 1;
                    padding: 16px 24px;
                    background: #f7fafc;
                    border: none;
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: 600;
                    color: #4a5568;
                    transition: all 0.2s ease;
                    border-right: 1px solid #e2e8f0;
                }}
                
                .tab:last-child {{
                    border-right: none;
                }}
                
                .tab.active {{
                    background: white;
                    color: #2d3748;
                    border-bottom: 2px solid #3182ce;
                }}
                
                .tab:hover:not(.active) {{
                    background: #edf2f7;
                }}
                
                .tab-content {{
                    display: none;
                    background: white;
                    border-radius: 0 0 8px 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 24px;
                    min-height: 600px;
                }}
                
                .tab-content.active {{
                    display: block;
                }}
                
                .network {{ 
                    width: 100%; 
                    height: {self.config.network_height}px; 
                    border: 1px solid #e2e8f0; 
                    border-radius: 8px; 
                    background: white;
                }}
                
                .legend, .metrics {{ 
                    background: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                    margin-top: 20px; 
                }}
                
                .correct {{ color: #48bb78; font-weight: bold; }}
                .incorrect {{ color: #f56565; font-weight: bold; }}
                
                select {{ 
                    padding: 8px; 
                    font-size: 16px; 
                    border-radius: 4px; 
                    border: 1px solid #e2e8f0; 
                    width: 100%; 
                    max-width: 500px; 
                    margin-bottom: 20px;
                }}
                
                select:focus {{ outline: none; border-color: #3182ce; }}
                
                pre {{ 
                    font-family: 'Fira Code', monospace; 
                    font-size: 14px; 
                    line-height: 1.6; 
                    margin: 0; 
                    white-space: pre-wrap; 
                    max-height: 600px;
                    overflow-y: auto;
                    background: #f8f9fa;
                    padding: 16px;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                }}
                
                .code-line {{ 
                    padding: 2px 0; 
                    position: relative; 
                }}
                
                .line-number {{ 
                    display: inline-block; 
                    width: 50px; 
                    text-align: right; 
                    padding-right: 15px; 
                    color: #718096; 
                    user-select: none;
                }}
                
                .code-content {{ 
                    display: inline-block; 
                    white-space: pre; 
                }}
                
                .indicator {{ 
                    display: inline-block; 
                    width: 20px; 
                    text-align: center; 
                }}
                
                .ground-truth {{ 
                    border-bottom: 2px dashed #f56565; 
                }}
                
                .vuln-indicator {{ 
                    color: #f56565; 
                    font-weight: bold; 
                    margin-left: 5px; 
                }}
                
                .code-line[data-tooltip]:not([data-tooltip=""]):hover::after {{
                    content: attr(data-tooltip); 
                    position: absolute; 
                    top: -28px; 
                    left: 70px;
                    background: #2d3748; 
                    color: white; 
                    padding: 5px 10px; 
                    border-radius: 4px;
                    font-size: 12px; 
                    z-index: 1000; 
                    white-space: nowrap;
                }}
                
                .vis-tooltip {{ 
                    background: #2d3748; 
                    color: white; 
                    padding: 8px; 
                    border-radius: 4px; 
                    font-size: 12px; 
                }}
                
                .sample-header {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                
                @media (max-width: 768px) {{
                    .tabs {{
                        flex-direction: column;
                    }}
                    .tab {{
                        border-right: none;
                        border-bottom: 1px solid #e2e8f0;
                    }}
                    .tab:last-child {{
                        border-bottom: none;
                    }}
                }}
            </style>
        </head>
        <body>
        """
    
    def generate_select_options(self, sample_indices: List[int], samples: List[Dict]) -> str:
        """Generate dropdown options for sample selection."""
        options = []
        for sample_idx, sample in zip(sample_indices, samples):
            filename = html.escape(sample.get('fileName', 'unknown'))
            options.append(f'<option value="{sample_idx}">{sample_idx}: {filename}</option>')
        return '\n'.join(options)
    
    def generate_code_html(self, code_lines: List[str], line_to_role: Dict[int, Tuple[str, str]], node_lines: set) -> str:
        """Generate HTML for code display with highlighting."""
        code_html = "<pre>"
        for line_num, line in enumerate(code_lines, 1):
            line = line.rstrip('\n')
            line_class = "code-line"
            style = ""
            tooltip = ""
            indicator = ""
            
            if line_num in node_lines:
                pred_role, true_role = line_to_role.get(line_num, ("none", "none"))
                style = self.config.role_colors.get(pred_role, "")
                tooltip = f"{pred_role} ({'Correct' if pred_role == true_role else 'Incorrect'})"
                indicator = '<span class="correct">✓</span>' if pred_role == true_role else '<span class="incorrect">✗</span>'
            
            ground_truth = line_num in node_lines and line_to_role.get(line_num, ("none", "none"))[1] == "sink"
            ground_class = " ground-truth" if ground_truth else ""
            vuln_indicator = '<span class="vuln-indicator">❗VULN</span>' if ground_truth else ''
            
            code_html += f'<div class="{line_class}{ground_class}" style="{style}" data-tooltip="{tooltip}">' + \
                        f'{indicator}<span class="line-number">{line_num:>3}</span>' + \
                        f'<span class="code-content">{html.escape(line)}{vuln_indicator}</span></div>'
        code_html += "</pre>"
        return code_html
    
    def generate_vis_data(self, nodes: List[str], predictions: List[int], labels: List[int], sample: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Generate Vis.js nodes and edges data."""
        idx_to_role = {0: "none", 1: "sink"}
        vis_nodes = []
        vis_edges = []
        
        for i, node in enumerate(nodes):
            try:
                node_dict = json.loads(node)
                node_type = node_dict.get("contents", [["unknown", ""]])[0][0]
                node_content = node_dict.get("contents", [["", ""]])[0][1][:self.config.max_node_content_length]
                
                pred_role = idx_to_role.get(predictions[i], "none") if i < len(predictions) else "none"
                true_role = idx_to_role.get(labels[i], "none") if i < len(labels) else "none"
                
                color = self.config.node_colors.get(pred_role, "#e0e0e0")
                
                vis_nodes.append({
                    "id": i,
                    "label": f"{node_type}\\n{node_content}{'...' if len(node_dict.get('contents', [['', '']])[0][1]) > self.config.max_node_content_length else ''}",
                    "color": {"background": color, "border": "#2d3748"},
                    "title": f"{pred_role} ({'Correct' if pred_role == true_role else 'Incorrect'})",
                    "font": {"size": 12}
                })
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logging.error(f"Error processing node {i}: {e}")
                continue
        
        for edge_type in ["cfgEdges", "ddgEdges", "cdgEdges"]:
            if edge_type not in sample:
                continue
                
            color = self.config.edge_colors.get(edge_type, "#666666")
            edge_label = edge_type[:3].upper()
            
            for edge in sample[edge_type]:
                try:
                    edge_data = json.loads(edge)
                    if len(edge_data) >= 2:
                        vis_edges.append({
                            "from": edge_data[0],
                            "to": edge_data[1],
                            "color": {"color": color},
                            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                            "label": edge_label,
                            "font": {"size": 10}
                        })
                except (json.JSONDecodeError, IndexError) as e:
                    logging.error(f"Error processing edge in {edge_type}: {e}")
                    continue
        
        return vis_nodes, vis_edges



def generate_html_visualization(samples: List[Dict], sample_indices: List[int], 
                              sample_predictions: List[Tuple[List[int], List[int]]], 
                              dataset: Any, test_indices: List[int], 
                              metrics: Dict[str, float],
                              config: Optional[VisualizationConfig] = None) -> str:
    """
    Generate a publication-ready HTML visualization with tabbed interface.
    """
    if config is None:
        config = VisualizationConfig()
    
    file_loader = CodeFileLoader(config.base_dir)
    html_gen = HTMLGenerator(config)
    
    html_content = html_gen.generate_html_header()
    
    html_content += f"""
    <div class="max-w-7xl mx-auto">
        <h1 class="text-3xl font-bold mb-6 text-gray-800">Vulnerability Analysis Results</h1>
        
        <div class="mb-6">
            <label for="sampleSelect" class="block text-sm font-medium text-gray-700 mb-2">Select Sample:</label>
            <select id="sampleSelect" onchange="showSample(this.value)">
                {html_gen.generate_select_options(sample_indices, samples)}
            </select>
        </div>
    """
    
    for idx, (sample, sample_idx, (predictions, labels)) in enumerate(zip(samples, sample_indices, sample_predictions)):
        nodes = sample.get("nodes", [])
        filename = sample.get('fileName', 'unknown')
        
        code_lines = file_loader.load_code_lines(filename)
        line_to_role, node_lines = NodeProcessor.create_line_mappings(nodes, predictions, labels)
        code_html = html_gen.generate_code_html(code_lines, line_to_role, node_lines)
        vis_nodes, vis_edges = html_gen.generate_vis_data(nodes, predictions, labels, sample)
        
        # Create numbered nodes version for the third tab
        numbered_nodes = []
        for i, node in enumerate(vis_nodes):
            numbered_node = node.copy()
            numbered_node['label'] = str(i + 1)  # Just show node number
            numbered_node['title'] = f"Node {i + 1}\n" + node.get('title', '')  # Keep full info in tooltip
            numbered_nodes.append(numbered_node)
        
        html_content += f"""
        <div class="sample" id="sample_{sample_idx}">
            <div class="sample-header">
                <h2 class="text-xl font-semibold text-gray-800">Sample {sample_idx}: {html.escape(filename)}</h2>
                <p class="text-sm text-gray-600 mt-2">
                    Nodes: {len(nodes)} | 
                    Sink Nodes: {sum(1 for pred in predictions if pred == 1)} predicted, {sum(1 for label in labels if label == 1)} actual
                </p>
            </div>
            
            <div class="tabs">
                <button class="tab active" onclick="switchTab('{sample_idx}', 'code')">
                    📄 Source Code
                </button>
                <button class="tab" onclick="switchTab('{sample_idx}', 'graph')">
                    🔗 Detailed Graph
                </button>
                <button class="tab" onclick="switchTab('{sample_idx}', 'numbered-graph')">
                    🔢 Numbered Graph
                </button>
            </div>
            
            <div id="code-tab-{sample_idx}" class="tab-content active">
                <div class="mb-4">
                    <h3 class="text-lg font-medium text-gray-700 mb-2">Source Code Analysis</h3>
                    <p class="text-sm text-gray-600">
                        Lines are highlighted based on model predictions. Hover over highlighted lines for details.
                    </p>
                </div>
                {code_html}
            </div>
            
            <div id="graph-tab-{sample_idx}" class="tab-content">
                <div class="mb-4">
                    <h3 class="text-lg font-medium text-gray-700 mb-2">Detailed Control Flow Graph</h3>
                    <p class="text-sm text-gray-600">
                        Interactive graph showing the relationships between code elements. Hover over nodes for prediction details.
                    </p>
                </div>
                <div id="network_{sample_idx}" class="network"></div>
            </div>
            
            <div id="numbered-graph-tab-{sample_idx}" class="tab-content">
                <div class="mb-4">
                    <h3 class="text-lg font-medium text-gray-700 mb-2">Numbered Control Flow Graph</h3>
                    <p class="text-sm text-gray-600">
                        Simplified view showing just node numbers. Hover for details.
                    </p>
                </div>
                <div id="numbered-network_{sample_idx}" class="network"></div>
            </div>
            
            <script>
                (function() {{
                    // Detailed graph
                    var nodes_{sample_idx} = new vis.DataSet({json.dumps(vis_nodes)});
                    var edges_{sample_idx} = new vis.DataSet({json.dumps(vis_edges)});
                    var container_{sample_idx} = document.getElementById('network_{sample_idx}');
                    var data_{sample_idx} = {{ nodes: nodes_{sample_idx}, edges: edges_{sample_idx} }};
                    
                    // Numbered graph
                    var numbered_nodes_{sample_idx} = new vis.DataSet({json.dumps(numbered_nodes)});
                    var numbered_edges_{sample_idx} = new vis.DataSet({json.dumps(vis_edges)});
                    var numbered_container_{sample_idx} = document.getElementById('numbered-network_{sample_idx}');
                    var numbered_data_{sample_idx} = {{ nodes: numbered_nodes_{sample_idx}, edges: numbered_edges_{sample_idx} }};
                    
                    var graphOptions = {{
                        nodes: {{ 
                            shape: 'box',
                            margin: 10,
                            font: {{ 
                                size: 12, 
                                face: 'monospace',
                                multi: 'html',
                                align: 'left'
                            }},
                            borderWidth: 2,
                            shadow: {{ enabled: true, color: 'rgba(0,0,0,0.1)', size: 5 }},
                            shapeProperties: {{ borderRadius: 6 }}
                        }},
                        edges: {{ 
                            font: {{ size: 10, color: '#666' }},
                            smooth: {{ type: 'cubicBezier', forceDirection: 'vertical', roundness: 0.4 }},
                            arrows: {{ to: {{ scaleFactor: 0.8 }} }},
                            width: 2
                        }},
                        physics: {{ 
                            enabled: true,
                            hierarchicalRepulsion: {{ 
                                nodeDistance: 150,
                                centralGravity: 0.1,
                                springLength: 100,
                                springConstant: 0.01,
                                damping: 0.09
                            }},
                            stabilization: {{ iterations: 200 }}
                        }},
                        layout: {{ 
                            hierarchical: {{ 
                                direction: 'UD', 
                                sortMethod: 'directed',
                                levelSeparation: 120,
                                nodeSpacing: 180,
                                treeSpacing: 200
                            }} 
                        }},
                        interaction: {{ 
                            hover: true, 
                            selectConnectedEdges: false,
                            tooltipDelay: 200,
                            zoomView: true
                        }}
                    }};
                    
                    // Create both networks
                    var network_{sample_idx} = new vis.Network(container_{sample_idx}, data_{sample_idx}, graphOptions);
                    var numbered_network_{sample_idx} = new vis.Network(numbered_container_{sample_idx}, numbered_data_{sample_idx}, graphOptions);
                    
                    // Store network references for lazy loading
                    window.networks = window.networks || {{}};
                    window.networks['{sample_idx}'] = {{
                        detailed: network_{sample_idx},
                        numbered: numbered_network_{sample_idx},
                        initialized: false
                    }};
                }})();
            </script>
        </div>
        """
    
    # Rest of the HTML content (legend, metrics, etc.) remains the same...
    html_content += f"""
        <div class="legend">
            <h3 class="text-lg font-semibold mb-4 text-gray-700">Legend</h3>
            <div class="grid grid-cols-2 gap-6">
                <div>
                    <h4 class="font-medium mb-2">Node Types</h4>
                    <div class="space-y-2">
                        <div class="flex items-center">
                            <span class="w-4 h-4 mr-3 rounded" style="background-color: {config.node_colors['sink']}; border: 1px solid #2d3748;"></span>
                            <span>Sink Node (Vulnerability)</span>
                        </div>
                        <div class="flex items-center">
                            <span class="w-4 h-4 mr-3 rounded" style="background-color: {config.node_colors['none']}; border: 1px solid #2d3748;"></span>
                            <span>Non-Sink Node</span>
                        </div>
                    </div>
                </div>
                <div>
                    <h4 class="font-medium mb-2">Edge Types</h4>
                    <div class="space-y-2">
                        <div class="flex items-center">
                            <span class="w-4 h-1 mr-3 rounded" style="background-color: {config.edge_colors['cfgEdges']};"></span>
                            <span>CFG Edge (Control Flow)</span>
                        </div>
                        <div class="flex items-center">
                            <span class="w-4 h-1 mr-3 rounded" style="background-color: {config.edge_colors['ddgEdges']};"></span>
                            <span>DDG Edge (Data Dependency)</span>
                        </div>
                        <div class="flex items-center">
                            <span class="w-4 h-1 mr-3 rounded" style="background-color: {config.edge_colors['cdgEdges']};"></span>
                            <span>CDG Edge (Control Dependency)</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="metrics">
            <h3 class="text-lg font-semibold mb-4 text-gray-700">Performance Metrics</h3>
            <div class="grid grid-cols-4 gap-4">
                <div class="p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm text-gray-500">Accuracy</p>
                    <p class="text-2xl font-bold">{metrics.get('accuracy', 0):.4f}</p>
                </div>
                <div class="p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm text-gray-500">F1-Macro</p>
                    <p class="text-2xl font-bold">{metrics.get('f1_macro', 0):.4f}</p>
                </div>
                <div class="p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm text-gray-500">Precision</p>
                    <p class="text-2xl font-bold">{metrics.get('precision', 0):.4f}</p>
                </div>
                <div class="p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm text-gray-500">Recall</p>
                    <p class="text-2xl font-bold">{metrics.get('recall', 0):.4f}</p>
                </div>
            </div>
        </div>
        
        <script>
            function showSample(sampleId) {{
                document.querySelectorAll('.sample').forEach(sample => {{
                    sample.classList.remove('active');
                }});
                var selectedSample = document.getElementById('sample_' + sampleId);
                if (selectedSample) {{
                    selectedSample.classList.add('active');
                    
                    // Initialize the networks if they haven't been initialized yet
                    if (window.networks && window.networks[sampleId] && !window.networks[sampleId].initialized) {{
                        window.networks[sampleId].detailed.redraw();
                        window.networks[sampleId].detailed.fit();
                        window.networks[sampleId].numbered.redraw();
                        window.networks[sampleId].numbered.fit();
                        window.networks[sampleId].initialized = true;
                    }}
                }}
            }}
            
            function switchTab(sampleId, tabName) {{
                // Hide all tabs for this sample
                document.querySelectorAll('#sample_' + sampleId + ' .tab-content').forEach(tab => {{
                    tab.classList.remove('active');
                }});
                
                // Show selected tab
                document.getElementById(tabName + '-tab-' + sampleId).classList.add('active');
                
                // Update tab buttons
                document.querySelectorAll('#sample_' + sampleId + ' .tab').forEach(tab => {{
                    tab.classList.remove('active');
                }});
                event.currentTarget.classList.add('active');
                
                // Initialize networks if needed
                if (window.networks && window.networks[sampleId] && !window.networks[sampleId].initialized) {{
                    if (tabName === 'graph') {{
                        window.networks[sampleId].detailed.redraw();
                        window.networks[sampleId].detailed.fit();
                    }} else if (tabName === 'numbered-graph') {{
                        window.networks[sampleId].numbered.redraw();
                        window.networks[sampleId].numbered.fit();
                    }}
                    window.networks[sampleId].initialized = true;
                }}
            }}
            
            document.addEventListener('DOMContentLoaded', function() {{
                // Show first sample by default
                showSample('{sample_indices[0] if sample_indices else ''}');
            }});
        </script>
    </div>
    </body>
    </html>
    """
    
    # Create output directory if it doesn't exist
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write HTML file
    filepath = output_path / config.output_filename
    with filepath.open("w", encoding="utf-8") as f:
        f.write(html_content)
    
    logging.info(f"Generated visualization: {filepath}")
    return str(filepath)



def generate_html_visualization_legacy(*args, **kwargs):
    """Legacy wrapper for backward compatibility."""
    return generate_html_visualization(*args, **kwargs)

def load_dataset(dataset_path, word2vec_model):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    try:
        with open('splits.json', 'r') as f:
            splits = json.load(f)
        test_indices = splits['test_indices']
    except Exception as e:
        logging.error(f"Failed to load splits.json: {e}")
        return None, None, None
    
    test_dataset = [dataset[i] for i in test_indices]
    
    embedding_dim = word2vec_model.vector_size
    num_features = embedding_dim
    default_embedding = np.zeros(embedding_dim)
    
    def process_dataset(data_subset, desc):
        data_list = []
        for sample in tqdm(data_subset, desc=desc):
            nodes = sample["nodes"]
            x = []
            for node in nodes:
                node_dict = json.loads(node)
                node_type = node_dict["contents"][0][0]
                node_content = node_dict["contents"][0][1].replace(' ', '_')[:50]
                token = f"{node_type}_{node_content}"
                embedding = word2vec_model.wv[token] if token in word2vec_model.wv else default_embedding
                x.append(embedding)
            x = torch.tensor(np.array(x), dtype=torch.float).to(device)
            
            node_roles = {i: json.loads(node).get("role", "none") for i, node in enumerate(nodes)}
            role_to_idx = {"none": 0, "sink": 1}
            y = torch.tensor([role_to_idx[role] for role in node_roles.values()], dtype=torch.long).to(device)
            
            edge_index = []
            for edge_type in ["ddgEdges", "cdgEdges", "cfgEdges"]:
                for edge in sample[edge_type]:
                    edge_data = json.loads(edge)
                    src, dst = edge_data[0], edge_data[1]
                    edge_index.append([src, dst])
                    edge_index.append([dst, src])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        return data_list
    
    test_data_list = process_dataset(test_dataset, "Processing test samples")
    return test_data_list, dataset, test_indices

def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_test.png')
    plt.close()

def evaluate_gnn(dataset_path, word2vec_model_path, gnn_model_path):
    try:
        word2vec_model = Word2Vec.load(word2vec_model_path)
        logging.info("Word2Vec model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load Word2Vec model: {e}")
        return
    
    try:
        test_data_list, dataset, test_indices = load_dataset(dataset_path, word2vec_model)
        if test_data_list is None:
            return
        logging.info(f"Loaded {len(test_data_list)} test samples")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return
    
    data_args = DataParser()
    
    test_loader = DataLoader(
        test_data_list,
        batch_size=data_args.test_batch_size,
        shuffle=False,
        num_workers=data_args.num_workers
    )
    
    model = GNN(
        in_channels=word2vec_model.vector_size,
        hidden_channels=256,
        out_channels=data_args.num_classes,
        layer_num=6
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(gnn_model_path))
        logging.info("GNN model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load GNN model: {e}")
        return
    
    model.eval()
    all_preds, all_labels = [], []
    classes = ["none", "sink"]
    sample_predictions = []
    
    with torch.no_grad():
        for i, data in enumerate(test_data_list):
            out = model(data)
            preds = out.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            sample_predictions.append((preds, labels))
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=range(len(classes)))
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    
    logging.info(f"\nTest Accuracy: {accuracy:.4f}")
    logging.info(f"Test F1-Macro: {f1_macro:.4f}")
    logging.info(f"Test Precision: {precision:.4f}")
    logging.info(f"Test Recall: {recall:.4f}")
    logging.info(f"Test Per-class F1: {dict(zip(classes, f1_per_class.round(4)))}")
    logging.info(f"Test Confusion Matrix:\n{cm}")
    
    plot_confusion_matrix(all_labels, all_preds, classes)
    
    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall
    }
    samples_to_visualize = [dataset[idx] for idx in test_indices[:300]]
    generate_html_visualization(samples_to_visualize, test_indices[:300], sample_predictions[:300], dataset, test_indices, metrics)
    
    logging.info("\nSample Predictions:")
    idx_to_role = {0: "none", 1: "sink"}
    for i, data in enumerate(test_data_list[:300]):
        sample_idx = test_indices[i]
        sample = dataset[sample_idx]
        preds, labels = sample_predictions[i]
        logging.info(f"\nSample {sample_idx} ({sample.get('fileName', 'unknown')}):")
        for j in range(min(500, len(preds))):
            node_dict = json.loads(sample["nodes"][j])
            node_content = node_dict["contents"][0][1][:50]
            logging.info(f"  Node {j}: Predicted={idx_to_role[preds[j]]}, True={idx_to_role[labels[j]]}, Content={node_content}")

if __name__ == "__main__":
    dataset_path = "/home/nimana11/Thesis/codes/VulExplainerExp-84ED_2/runs/run_full_bufferoverflow_sink_none/labeled_dataset.json"
    word2vec_model_path = "word2vec_model.model"
    gnn_model_path = "gnn_word2vec_model.pth"
    evaluate_gnn(dataset_path, word2vec_model_path, gnn_model_path)