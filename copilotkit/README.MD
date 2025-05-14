<div align="center">
  <h1>CopilotKit</h1>

  <br/>

  <h3>
    Build deeply-integrated AI assistants & agents<br/>
    that work <em>alongside</em> your users inside your applications.
  </h3>
</div>

---

## Overview

CopilotKit is an open-source framework for integrating AI copilots directly into your applications. It enables users to collaborate with Large Language Models (LLMs) to accomplish tasks within your app, moving beyond passive content generation to active, context-aware assistance.

With CopilotKit, you can:

- Embed AI copilots that understand and interact with your app's state.
- Implement human-in-the-loop workflows for greater control and reliability.
- Utilize agent frameworks like LangGraph to build complex, multi-step agents.

---

## Key Features

### 🧠 Agentic Agents:

CoAgents is a framework within CopilotKit designed for building AI agents that collaborate with users in real-time. It provides the infrastructure to create agents that are transparent, controllable, and integrated into your application's UI.

**Core capabilities include:**

- Agentic Chat UI: Fully customizable components with headless UI options
- Human-in-the-Loop Flows: Add human approval, plan editing, and more with both in-chat and out-of-chat support
- Agentic Generative UI: Render your agent's state, progress, and outputs with custom UI components in real time
- Tool-Based Generative UI: Create dynamic UI components triggered by your agent's tool calls
- Shared State between Agent and UI: Give your agents awareness of what users see in your application
- Predictive State Updates: Improve responsiveness by rendering predicted agent states before completion

Learn more in the [CoAgents Documentation](https://docs.copilotkit.ai/coagents).

### ⚙️ Seamless Integration with LangGraph

CopilotKit integrates with LangGraph, a framework for defining and executing LLM agents using graphs. This integration allows you to:

- Build agents tailored to specific workflows.
- Manage complex agent behaviors with structured graphs.
- Combine LangGraph's capabilities with CopilotKit's UI components for a cohesive user experience.

Quickstart guide: [LangGraph Integration](https://docs.copilotkit.ai/coagents/quickstart/langgraph)

### 🧩 React Components for AI Interaction

CopilotKit offers a suite of React components to facilitate AI interactions within your app:

- `<CopilotChat />`: A customizable chat interface for user-agent conversations.
- `<CopilotSidebar />`: A sidebar component to house your AI copilot.
- `useCopilotAction()`: A hook to define actions that the copilot can perform within your app.
- `useCopilotReadable()`: A hook to provide the copilot with read access to your app's state.

These components are designed to be flexible and easy to integrate, allowing you to tailor the AI experience to your application's needs.

---

## Getting Started

To get started with CopilotKit:

1. **Explore the Documentation:** [docs.copilotkit.ai](https://docs.copilotkit.ai/)
2. **Clone the Starter Repo:** [CoAgents Starter](https://github.com/CopilotKit/CopilotKit/tree/main/examples/coagents-starter)

## Resources

- **GitHub Repository:** [CopilotKit on GitHub](https://github.com/CopilotKit/CopilotKit)
- **Blog:** [Latest Updates and Tutorials](https://www.copilotkit.ai/blog)

