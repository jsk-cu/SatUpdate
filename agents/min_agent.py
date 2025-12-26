#!/usr/bin/env python3
"""
Minimum-First Agent Implementation

A packet distribution strategy that prioritizes requesting from
neighbors with the lowest completion percentage, then selects
the lowest-indexed missing packets from each.

This agent subclasses BaseAgent and only overrides make_requests()
to implement its custom distribution strategy.
"""

from typing import Dict, List, Tuple, Any

from .base_agent import BaseAgent


class MinAgent(BaseAgent):
    """
    Agent that prioritizes neighbors with lowest completion percentage.

    Strategy:
    - Sort neighbors by completion percentage (lowest first)
    - For each neighbor (in order), request the lowest-indexed
      packet that we're missing and they have
    - Continue until we've made a request from each neighbor or
      we have no more missing packets

    This is a subclass of BaseAgent that only overrides make_requests().
    All other protocol methods (broadcast, respond, receive) use the
    default implementations from BaseAgent.

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
    """

    # Agent type name for registry
    name = "min"
    description = "Orders neighbors by completion (lowest first), requests lowest packets"

    def make_requests(
        self, neighbor_broadcasts: Dict[int, Dict[str, Any]]
    ) -> Dict[int, int]:
        """
        Phase 2: Request packets from neighbors, ordered by completion.

        Strategy:
        1. Sort neighbors by completion percentage (lowest first)
        2. For each neighbor, find packets they have that we're missing
        3. Request the lowest-indexed such packet
        4. Continue until all neighbors processed or no missing packets

        Parameters
        ----------
        neighbor_broadcasts : dict
            Neighbor agent IDs mapped to their broadcast state.

        Returns
        -------
        dict
            Mapping of neighbor IDs to requested packet indices.
        """
        if self.has_all_packets():
            return {}

        if not neighbor_broadcasts:
            return {}

        # Sort neighbors by completion percentage (lowest first)
        # Use agent_id as tiebreaker for determinism
        sorted_neighbors: List[Tuple[int, Dict[str, Any]]] = sorted(
            neighbor_broadcasts.items(),
            key=lambda x: (x[1].get("completion", 0.0), x[0])
        )

        requests = {}
        missing_packets = self.get_missing_packets()

        for neighbor_id, broadcast in sorted_neighbors:
            if not missing_packets:
                break

            neighbor_packets = broadcast.get("packets", set())
            available = missing_packets & neighbor_packets

            if available:
                # Request lowest-indexed available packet
                packet_to_request = min(available)
                requests[neighbor_id] = packet_to_request
                # Remove from missing so we don't request same packet twice
                missing_packets.discard(packet_to_request)

        return requests