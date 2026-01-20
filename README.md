# 🤖 Agentic Data Scientist

### An Autonomous AI Agent that Reasons, Codes, and Trains ML Models.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-magenta)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Supabase](https://img.shields.io/badge/Database-Supabase-green)

---

## 📖 Overview

The **Agentic Data Scientist** is an end-to-end autonomous system designed to act as a virtual teammate for data analysis tasks. Unlike standard "Chat with PDF" tools, this agent does not just retrieve information—it **executes actions**.

It writes its own Python code to perform Exploratory Data Analysis (EDA), cleans dirty datasets, trains Machine Learning models, and generates visualizations. It features a **Human-in-the-Loop (HITL)** architecture to ensure safety and control over high-stakes operations.

## ✨ Key Features

* **🔄 Cyclic Reasoning (LangGraph):** Uses a graph-based Finite State Machine (FSM) architecture, allowing the agent to write code, encounter an error, and *loop back* to self-correct automatically.
* **🛡️ Human-in-the-Loop Safety:** Implements **interrupts** for sensitive actions. The agent *must* pause and ask for user approval before running computation-heavy training (`.fit()`) or model serialization (`joblib.dump`).
* **🧠 Long-Term Memory:** Integrates **Supabase** to persist conversation history, allowing users to reload sessions without losing context (state rehydration).
* **📊 Autonomous Visualization:** Can generate Matplotlib charts and render them directly in the Streamlit UI.
* **⚡ Code Sandboxing (Local):** Executes generated Python code in a controlled local environment with state management for variables (DataFrames, Models).

---

## 🏗️ Architecture

The system follows the **ReAct (Reason + Act)** pattern implemented via **LangGraph**:

```mermaid
graph LR
    A[Start] --> B(Agent Node);
    B -- Generates Code --> C{Router};
    C -- "Has Code" --> D(Tool Node);
    C -- "Text Only" --> E((End));
    D -- "Success/Error" --> B;
    
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
