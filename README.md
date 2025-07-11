# Parallel Graph-Based Approach for Policy Learning in Railway Networks

This repository contains the source code and experiments related to the thesis:

**Parallel Graph-Based Approach for Policy Learning in Railway Networks**  
by Alessia Grandetti  
Supervised by Prof. Barbara Re, Professor Manuel Renold and PhD student Julia Usher
University of Camerino, A.A. 2024/2025

---

## 📌 Project Overview

This project implements a parallel training framework for Deep Reinforcement Learning (DRL) agents in complex railway environments simulated with Flatland.  
The key contribution is a tree-structured evolutionary approach, where the best-performing policy at each generation is propagated to the next one, significantly improving convergence speed and scalability.

---

## 🛠️ Project Structure

- `main_train.py`: orchestrates the training process and evolutionary tree  
- `train_dddqn.py`: defines the Double Dueling Deep Q-Learning training loop  
- `DDDQNPolicy.py`: policy class implementing the agent  
- `DuelingQNetwork.py`: dueling architecture for the Q-network  
- `ReplayBuffer.py`: experience replay memory  
- `environment.py`: Flatland environment builder  
- `obs_utils.py`: observation pre-processing

---

