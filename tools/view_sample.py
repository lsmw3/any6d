import numpy as np
import random

# Function to compute camera position in world coordinates
def compute_camera_position(R, t):
    return -np.dot(R.T, t)

# Function to calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to perform Farthest Point Sampling (FPS) on camera positions
def views_FPS(views_data, num_samples):
    # Extract views from the JSON file and store positions
    view_positions = {}
    for key, view_list in views_data.items():
        view = view_list[0]
        R = np.array(view['cam_R_m2c']).reshape(3, 3)
        t = np.array(view['cam_t_m2c'])

        # Compute camera position
        position = compute_camera_position(R, t)
        view_positions[key] = position

    # Get the number of camera positions
    keys = list(view_positions.keys())
    positions = np.array([view_positions[key] for key in keys])
    num_views = len(positions)

    # Randomly pick the first point
    initial_idx = random.choice(range(num_views))

    # Initialize the list of selected points
    selected_indices = [initial_idx]  # Start with the random initial point
    remaining_indices = list(range(num_views))
    remaining_indices.remove(initial_idx)  # Remove the initial point from the remaining list

    # Compute distances between the initial selected point and all others
    distances = np.array([euclidean_distance(positions[initial_idx], positions[i]) for i in remaining_indices])

    while len(selected_indices) < num_samples:
        # Find the point with the maximum minimum distance to the selected points
        farthest_idx_in_remaining = np.argmax(distances)
        farthest_idx = remaining_indices[farthest_idx_in_remaining]
        selected_indices.append(farthest_idx)

        # Remove the selected point's index from remaining_indices
        remaining_indices.pop(farthest_idx_in_remaining)

        if not remaining_indices:
            break

        # Update distances by calculating the new distances from the newly selected point
        new_distances = np.array([euclidean_distance(positions[farthest_idx], positions[i]) for i in remaining_indices])

        # Remove the farthest distance from the distances array
        distances = np.delete(distances, farthest_idx_in_remaining)

        # Update distances: keep the minimum distance to the nearest selected point
        distances = np.minimum(distances, new_distances)

    # Return the keys of the selected camera positions
    selected_keys = [keys[i] for i in selected_indices]
    return selected_keys
