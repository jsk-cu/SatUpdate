#!/usr/bin/env python3
"""
Base Agent - Abstract Base Class for Packet Distribution Agents

Provides the base class that all agent implementations should subclass.
The BaseAgent defines:
1. The 4-phase protocol interface (broadcast, request, respond, receive)
2. Common state management (packets, completion tracking)
3. Default implementations that can be overridden

Subclasses typically only need to override `make_requests()` to implement
their distribution strategy. The default implementation makes no requests,
which can serve as a control case for testing.
"""

from typing import Dict, Set, Optional, Any


class BaseAgent:
    """
    Base class for packet distribution agents.

    This class provides the complete protocol interface and default
    implementations. Subclasses should override `make_requests()` to
    implement custom distribution strategies.

    The default `make_requests()` returns an empty dictionary (no requests),
    making this usable as a "null" agent for control experiments.

    Parameters
    ----------
    agent_id : int
        Unique identifier. 0 = base station, 1+ = satellites.
    num_packets : int
        Total packets in the software update.
    num_satellites : int
        Total satellites in the constellation.
    is_base_station : bool
        True if this is the base station agent.

    Attributes
    ----------
    agent_id : int
        Agent identifier.
    num_packets : int
        Total packets in update.
    packets : Set[int]
        Packet indices this agent has received.
    is_base_station : bool
        Whether this is the base station.

    Examples
    --------
    Subclassing to create a custom agent:

    >>> class MyAgent(BaseAgent):
    ...     name = "my_agent"
    ...     description = "My custom strategy"
    ...
    ...     def make_requests(self, neighbor_broadcasts):
    ...         # Custom logic here
    ...         requests = {}
    ...         missing = self.get_missing_packets()
    ...         for neighbor_id, broadcast in neighbor_broadcasts.items():
    ...             available = missing & broadcast.get("packets", set())
    ...             if available:
    ...                 requests[neighbor_id] = min(available)
    ...         return requests
    """

    # Agent type name for registry - subclasses should override
    name = "base"
    description = "Base agent: makes no requests (no distribution occurs)"

    def __init__(
        self,
        agent_id: int,
        num_packets: int,
        num_satellites: int,
        is_base_station: bool = False,
    ):
        self.agent_id = agent_id
        self.num_packets = num_packets
        self.num_satellites = num_satellites
        self.is_base_station = is_base_station

        # Base station starts with all packets; satellites start empty
        if is_base_station:
            self.packets: Set[int] = set(range(num_packets))
        else:
            self.packets: Set[int] = set()

    def broadcast_state(self) -> Dict[str, Any]:
        """
        Phase 1: Broadcast current state to neighbors.

        Called at the start of each protocol round. Returns information
        about this agent's current state that neighbors can use to
        decide what to request.

        Returns
        -------
        dict
            State information with keys:
            - agent_id: This agent's ID
            - packets: Set of packet indices held
            - num_packets: Total packets in update
            - is_base_station: Whether this is the base station
            - completion: Completion percentage (0-100)
        """
        return {
            "agent_id": self.agent_id,
            "packets": self.packets.copy(),
            "num_packets": self.num_packets,
            "is_base_station": self.is_base_station,
            "completion": self.get_completion_percentage(),
        }

    def make_requests(
        self, neighbor_broadcasts: Dict[int, Dict[str, Any]]
    ) -> Dict[int, int]:
        """
        Phase 2: Decide which packets to request from neighbors.

        Override this method in subclasses to implement custom
        distribution strategies.

        The default implementation makes no requests, so no packets
        are ever transferred. This is useful as a control case.

        Parameters
        ----------
        neighbor_broadcasts : dict
            Mapping of neighbor agent IDs to their broadcast state.
            Each broadcast contains:
            - packets: Set of packet indices the neighbor has
            - completion: Neighbor's completion percentage
            - is_base_station: Whether neighbor is the base station

        Returns
        -------
        dict
            Mapping of neighbor IDs to requested packet indices.
            Request at most one packet per neighbor.
            Return empty dict to make no requests.
        """
        # Default: no requests (subclasses override this)
        return {}

    def receive_requests_and_update(
        self, requests: Dict[int, int]
    ) -> Dict[int, Optional[int]]:
        """
        Phase 3: Process incoming requests and decide responses.

        Default implementation grants all requests for packets we have.
        Subclasses can override to implement bandwidth limits, priority
        schemes, etc.

        Parameters
        ----------
        requests : dict
            Mapping of requester agent IDs to requested packet indices.

        Returns
        -------
        dict
            Mapping of requester IDs to packet indices being sent,
            or None if request is denied.
        """
        responses = {}
        for requester_id, packet_idx in requests.items():
            if packet_idx in self.packets:
                responses[requester_id] = packet_idx
            else:
                responses[requester_id] = None
        return responses

    def receive_packets_and_update(
        self, received: Dict[int, Optional[int]]
    ) -> None:
        """
        Phase 4: Receive packets and update local state.

        Parameters
        ----------
        received : dict
            Mapping of sender agent IDs to received packet indices
            (None if request was denied).
        """
        for sender_id, packet_idx in received.items():
            if packet_idx is not None:
                self.packets.add(packet_idx)

    # -------------------------------------------------------------------------
    # Utility methods (generally don't need to be overridden)
    # -------------------------------------------------------------------------

    def get_packet_count(self) -> int:
        """Return number of packets this agent has."""
        return len(self.packets)

    def get_completion_percentage(self) -> float:
        """Return percentage of packets received (0-100)."""
        if self.num_packets == 0:
            return 100.0
        return 100.0 * len(self.packets) / self.num_packets

    def has_all_packets(self) -> bool:
        """Check if agent has all packets."""
        return len(self.packets) >= self.num_packets

    def get_missing_packets(self) -> Set[int]:
        """Return set of missing packet indices."""
        return set(range(self.num_packets)) - self.packets

    def reset(self) -> None:
        """Reset agent to initial state."""
        if self.is_base_station:
            self.packets = set(range(self.num_packets))
        else:
            self.packets = set()

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        agent_type = "BaseStation" if self.is_base_station else "Satellite"
        return (
            f"{class_name}({agent_type}-{self.agent_id}, "
            f"packets={len(self.packets)}/{self.num_packets}, "
            f"{self.get_completion_percentage():.1f}%)"
        )