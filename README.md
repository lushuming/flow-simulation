# Fractured Flow Simulation

This is my personal project for simulating flow in fractured porous media. It's mainly Python + Jupyter Notebook experiments, with both fitted mesh and unfitted mesh methods.

---

## Project Structure

```
fractured-flow-simulation/
├── README.md
├── notebooks/         # Experiment notebooks
│   ├── fitted/        # Fitted mesh examples
│   └── unfitted/      # Unfitted mesh examples
├── src/               # Python source code (including convergence order checks)
│   ├── fitted/
│   └── unfitted/
├── notes/             # Personal research notes and derivations
└── requirements.txt   # Python dependencies
```

---

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/lushuming/flow-simulation.git
cd flow-simulation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open notebooks in the `notebooks/` folder and run them to see the results.

---

## About the Folders

* **notebooks/fitted/** : experiments using fitted mesh
* **notebooks/unfitted/** : experiments using unfitted mesh
* **src/** : Python implementations corresponding to notebooks
* **notes/** : personal notes on papers about computational mathematics, etc.

---

## Notes

* This repo is mainly for my own research and experiments; updates are ongoing.
