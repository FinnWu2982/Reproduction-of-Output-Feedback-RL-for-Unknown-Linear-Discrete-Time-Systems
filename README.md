# ECE411 Final Project: Output-Feedback RL Tracking

This repository contains the MATLAB source code for my ECE411 final project. It reproduces the core output-feedback reinforcement learning pipeline from the following paper on a double-integrator benchmark:

> C. Chen et al., "Robust Output Regulation and Reinforcement Learning-Based Output Tracking Design for Unknown Linear Discrete-Time Systems," IEEE Transactions on Automatic Control, 2023.

## Files Included

* `main_OutputFeedbackRL_Tracking.m`: The main script. It executes the pre-collection phase, collects input-output data, verifies the rank condition, runs the Bellman-equation-based gain update iteration, and evaluates the learned controller on a fresh test.
* `helper_Generate_Initial_Gains.m`: A helper script used to compute and verify the stabilizing initial behavior and output-feedback gains required to initialize the learning phase. 

## How to Run

1. Open MATLAB.
2. Run `main_OutputFeedbackRL_Tracking.m`. 
3. The script is entirely self-contained. It will print the iterative gain update norms to the console and automatically generate the evaluation figures (gain convergence, output tracking, tracking error, and control input) discussed in the final report.

**Author:** Feiyang Wu
**Course:** ECE411 (Winter 2026), University of Toronto
