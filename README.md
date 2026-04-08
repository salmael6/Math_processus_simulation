# 📊 Stochastic Processes Simulator

This project is an interactive web application built with **Streamlit** that allows users to explore and simulate different stochastic processes:

* 📌 Poisson Process (Homogeneous)
* 📌 Poisson Process (Non-Homogeneous)
* 📌 Discrete-Time Markov Chains

---

## 🚀 Features

### 🔹 1. Homogeneous Poisson Process

* Compute probability
* Visualize:

  * Probability Mass Function (PMF)
  * Cumulative Distribution Function (CDF)
* Interactive parameter inputs:

  * λ (rate)
  * t (time interval)
  * k (number of events)

---

### 🔹 2. Non-Homogeneous Poisson Process

* Define a custom rate function λ(t)
* Compute probability 
* Plot λ(t) over time
* Flexible time interval input

---

### 🔹 3. Discrete-Time Markov Process

* Simulate Markov chains
* Define transition matrix:

  * Manually
  * Randomly (Uniform, Dirichlet, Normal distributions)
* Validate transition matrix
* Visualize state transitions using graphs (NetworkX)

---

## 🛠️ Technologies Used

* **Python**
* **Streamlit** – Web interface
* **NumPy** – Numerical computations
* **SciPy** – Probability distributions & integration
* **Matplotlib** – Data visualization
* **NetworkX** – Graph visualization

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/salmael6/Math_processus_simulation.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```


## ▶️ Running the App

Run the Streamlit app with:

```bash
streamlit run ine.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 📖 How to Use

1. Select a process from the sidebar:

   * Poisson Homogeneous
   * Poisson Non-Homogeneous
   * Markov Process

2. Choose a sub-section:

   * Simulation
   * Graphs
   * Resources

3. Enter parameters and click the corresponding button to:

   * Compute probabilities
   * Generate plots
   * Run simulations

---

## ⚠️ Notes

* Ensure transition matrices are valid:

  * Square matrix
  * Rows sum to 1
  * No negative probabilities
* For non-homogeneous Poisson:

  * λ(t) must be a valid Python expression (e.g., `2 + 0.1*t`)
* The app uses `eval()` for λ(t), so be cautious with inputs.

---

## 📚 Educational Purpose

This project is designed for:

* Students studying probability & stochastic processes
* Understanding Poisson processes and Markov chains
* Interactive learning through visualization

---

## 🔮 Possible Improvements

* Save simulation results
* Add continuous-time Markov chains
* Improve graph labeling (real transition probabilities)
* Add Monte Carlo simulations
* Enhance UI/UX design

---



**GitHub-ready project structure** 👍
