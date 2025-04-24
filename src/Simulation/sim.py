# Simulation/sim.py

import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .hospital_env import MultiAgentFreeSpaceEnv
from .handler import compute_box_and_boundaries
import json
from typing import List
from fastapi import WebSocket
import logging

from .connections import (active_connections)  # Updated import

logger = logging.getLogger(__name__)

class Simulator:
    def __init__(self, poi, graph, nurses, doctors, patients, steps):
        graph_data, node_positions, box_size, bounding_box = compute_box_and_boundaries(graph)
        self.no_doctors = doctors
        self.no_nurses = nurses
        self.no_patients = patients
        self.env = MultiAgentFreeSpaceEnv(
            poi=poi,
            graph_data=graph_data,
            node_positions=node_positions,
            box_size=box_size,
            bounding_box=bounding_box,
            num_doctors=self.no_doctors,
            num_nurses=self.no_nurses,
            num_patients=self.no_patients
        )
        self.no_steps = steps
        self.stop_simulation = False  # Initialize stop flag
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Initially not paused
        self.stop_event = asyncio.Event()
        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on your needs

    async def run_simulation(self):
        self.env.reset()
        batch = []

        for i in range(self.no_steps):
            if self.stop_simulation:
                logger.info("Simulation stopped by user.")
                break  # Terminate simulation early

            # Wait if paused
            await self.pause_event.wait()

            # Offload CPU-bound simulation steps to a separate thread
            observations = await asyncio.get_event_loop().run_in_executor(self.executor, self.get_observations)

            batch.append({
                "step": i,
                "observations": {k: v.tolist() for k, v in observations[0].items()},
                "collisions": observations[1]['collisions']
            })

            if len(batch) == 5:
                # Send the batch to all active WebSocket clients
                await self.broadcast_batch(batch.copy())
                batch.clear()


        if batch:
            await self.broadcast_batch(batch.copy())

        self.stop_event.set()
        await self.broadcast_message(json.dumps({"message": "END"}))

    def get_observations(self):
        actions = {}
        for agent_type in self.env.agents:
            for agent in self.env.agents[agent_type]:
                actions[agent['name']] = agent['agent'].get_action(np.random.random())
        return self.env.step(actions)

    async def broadcast_batch(self, batch):
        message = json.dumps({"batch": batch})
        await self.broadcast_message(message)

    async def broadcast_message(self, message):
        disconnected_clients: List[WebSocket] = []
        for connection in active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                disconnected_clients.append(connection)
        # Remove disconnected clients
        #for connection in disconnected_clients:
         #  active_connections.remove(connection)
