#!/usr/bin/env python3
"""
Agents Package

Provides agent implementations for the packet distribution protocol.
Each agent defines a strategy for requesting and distributing packets
during satellite software updates.

Class Hierarchy
---------------
BaseAgent (base class)
    The abstract base class that all agents should subclass.
    Provides the 4-phase protocol interface and default implementations.
    The default make_requests() returns {} (no requests), useful as control.

MinAgent(BaseAgent)
    Example subclass that overrides make_requests().
    Orders neighbors by completion (lowest first), requests lowest packets.

Creating Custom Agents
----------------------
To create a custom agent, subclass BaseAgent and override make_requests():

    from agents import BaseAgent, register_agent

    class MyAgent(BaseAgent):
        name = "my_agent"
        description = "My custom distribution strategy"

        def make_requests(self, neighbor_broadcasts):
            requests = {}
            missing = self.get_missing_packets()
            for neighbor_id, broadcast in neighbor_broadcasts.items():
                available = missing & broadcast.get("packets", set())
                if available:
                    requests[neighbor_id] = min(available)
                    missing.discard(requests[neighbor_id])
            return requests

    # Register so it can be used via --agent-controller my_agent
    register_agent("my_agent", MyAgent)

Usage
-----
    from agents import get_agent_class, list_agents, BaseAgent

    # Get agent class by name
    AgentClass = get_agent_class("min")

    # Create agent instance
    agent = AgentClass(
        agent_id=1,
        num_packets=100,
        num_satellites=12,
        is_base_station=False
    )

    # List available agents
    for name, desc in list_agents().items():
        print(f"{name}: {desc}")
"""

from typing import Dict, Type

from .base_agent import BaseAgent
from .min_agent import MinAgent


# Registry mapping agent names to classes
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "base": BaseAgent,
    "min": MinAgent,
}

# Default agent to use if none specified
DEFAULT_AGENT = "min"


def get_agent_class(name: str) -> Type[BaseAgent]:
    """
    Get an agent class by name.

    Parameters
    ----------
    name : str
        Agent name (e.g., "base", "min").

    Returns
    -------
    Type[BaseAgent]
        The agent class (a subclass of BaseAgent).

    Raises
    ------
    ValueError
        If agent name is not found in registry.
    """
    if name not in AGENT_REGISTRY:
        available = ", ".join(AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown agent type: '{name}'. Available agents: {available}"
        )
    return AGENT_REGISTRY[name]


def list_agents() -> Dict[str, str]:
    """
    List all available agents with their descriptions.

    Returns
    -------
    dict
        Mapping of agent names to descriptions.
    """
    return {
        name: cls.description
        for name, cls in AGENT_REGISTRY.items()
    }


def register_agent(name: str, agent_class: Type[BaseAgent]) -> None:
    """
    Register a new agent type.

    The agent class should be a subclass of BaseAgent.

    Parameters
    ----------
    name : str
        Name for the agent (used with --agent-controller).
    agent_class : Type[BaseAgent]
        The agent class to register.

    Raises
    ------
    TypeError
        If agent_class is not a subclass of BaseAgent.
    """
    if not issubclass(agent_class, BaseAgent):
        raise TypeError(
            f"Agent class must be a subclass of BaseAgent, got {agent_class}"
        )
    AGENT_REGISTRY[name] = agent_class


# For backwards compatibility, also export Agent as alias to default
Agent = AGENT_REGISTRY[DEFAULT_AGENT]


__all__ = [
    # Base class for subclassing
    "BaseAgent",
    # Example subclass
    "MinAgent",
    # Alias to default agent
    "Agent",
    # Registry
    "AGENT_REGISTRY",
    "DEFAULT_AGENT",
    # Functions
    "get_agent_class",
    "list_agents",
    "register_agent",
]