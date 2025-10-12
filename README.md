VulPathFinder




Learning to Locate: GNN-Powered Vulnerability Path Discovery in Open Source Code

VulPathFinder is an explainable framework for discovering vulnerability paths in source code using Graph Neural Networks (GNNs).
It improves upon traditional tools like SliceLocator by employing a GNN model to detect sink statements based on semantic and syntactic dependencies rather than fixed rules.

Potential sink points (PSPs) are identified, vulnerable paths are extracted via program slicing, and paths are ranked using a graph-based detector to provide interpretable explanations of vulnerabilities.

This repository contains the implementation for the paper:
📄 Learning to Locate: GNN-Powered Vulnerability Path Discovery in Open Source Code

by Nima Atashin (2025).

🔍 Key Features

🧠 GNN-based sink detection for better generalization

🧩 Program slicing for extracting vulnerability paths

🔎 Path ranking for explainability and visualization

🧪 Evaluated on buffer overflow CWEs from the SARD dataset

📈 Outperforms SliceLocator and GNNExplainer in path discovery

Note: The SliceLocator implementation is adapted from this public repository
.

⚙️ Installation
Requirements

Python 3.8+

Key libraries:

torch (for GNNs)

networkx (for graph handling)

pycparser (or similar parser for C/C++ code)

Setup
# Clone the repository
git clone https://github.com/NimaNA11/VulPathFinder.git
cd VulPathFinder

# Install dependencies
pip install -r requirements.txt

🚀 Usage
Quick Start

Prepare your dataset (C/C++ source files with known vulnerabilities)

Detect Potential Sink Points (PSPs):

python src/detect_psps.py --input path/to/code.c --model models/gnn_model.pth


Extract and rank paths:

python src/extract_and_rank.py --psp-file psps.json --detector graph_detector


Visualize top-ranked paths:

python src/visualize_path.py --path top_path.json

Reproduce Paper Results

Download SARD dataset buffer overflow CWEs

Run evaluation:

python eval/evaluate.py --dataset sard_buffer_overflow --metrics precision recall

💡 Example

Input: Vulnerable C file containing a buffer overflow
Output: Ranked paths highlighting the root cause

Top Path: Source (line 10) → ... → Sink (line 42)
Explanation: Untrusted data flow leading to overflow.


See the /examples/
 directory for sample vulnerable code and outputs.

📂 Repository Structure
/src/      # Core scripts for detection, slicing, and ranking
/models/   # Pre-trained GNN models
/data/     # Sample datasets or preprocessing scripts
/eval/     # Evaluation scripts and metrics
/examples/ # Example vulnerable code and outputs

🧪 Datasets

Primary Dataset: Buffer overflow CWEs from the SARD dataset

Preprocessed graph data can be found (or generated) in /data/graphs/

🤝 Contributing

Contributions are welcome!

Fork the repository

Create a new branch:

git checkout -b feature/your-feature-name


Commit and push your changes:

git commit -am "Add your feature"
git push origin feature/your-feature-name


Open a Pull Request

For issues, please use the GitHub Issues section.

📚 Citation

If you use VulPathFinder in your research, please cite:

@misc{atashin2025learning,
  title={Learning to Locate: GNN-Powered Vulnerability Path Discovery in Open Source Code},
  author={Nima Atashin},
  year={2025},
  eprint={2507.17888},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  doi={10.48550/arXiv.2507.17888}
}

📜 License

This project is licensed under the MIT License.
See the LICENSE
 file for more details.

🙏 Acknowledgments

Adapted in part from SliceLocator (VulExplainerExp)

Thanks to the SARD dataset team for providing benchmark vulnerability data

💬 For questions or collaborations, contact:
📧 NimaNA11
