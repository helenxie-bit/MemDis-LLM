import numpy as np
import random

def generate_workload(lambda_rate, simulation_duration, new_conv_prob):
    """
    Generate a synthetic workload for a system simulating user requests and conversations.
    Args:
        lambda_rate (float): Average number of requests per second.
        simulation_duration (int): Total duration of the simulation in seconds.
        new_conv_prob (float): Probability of starting a new conversation.
    Returns:
        request_id (int): The ID of the last request processed.
    """

    inter_arrival_times = np.random.exponential(scale=1/lambda_rate, size=int(lambda_rate * simulation_duration)) # Generate inter-arrival times
    arrival_times = np.cumsum(inter_arrival_times) # Cumulative sum to get arrival times
    followup_delay_mean = 1.0 # Seconds

    requests = []
    active_conversations = {}
    current_request_id = 0

    for t in arrival_times:
        if len(active_conversations) == 0 or random.random() < new_conv_prob:
            request_id = current_request_id
            current_request_id += 1
            turn_id = 0
        else:
            request_id = random.choice(list(active_conversations.keys()))
            last_time, last_turn = active_conversations[request_id]
            t = max(t, last_time + np.random.exponential(followup_delay_mean)) # Ensure the request is after the last turn
            turn_id = last_turn + 1
        
        active_conversations[request_id] = (t, turn_id)

        requests.append({
            "arrival_time": t,
            "request_id": request_id,
            "turn_id": turn_id
        })

    return requests
