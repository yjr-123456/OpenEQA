import json

def create_reverse_lookup_dict(data):
    """
    Constructs a reverse lookup dictionary from agent names to their type_id.

    Args:
        data (dict): The input dictionary loaded from agent_name.json.

    Returns:
        dict: A dictionary mapping agent names to their "type_id" string.
              Example: {"the_man_in_light_pants_and_sunglasses": "player_1"}
    """
    reverse_lookup = {}
    for agent_type, agents in data.items():
        if isinstance(agents, dict):
            for agent_id, agent_name in agents.items():
                if agent_name in reverse_lookup:
                    # Handle potential duplicate names if necessary,
                    # though in this specific JSON, names seem unique.
                    # For now, we'll just overwrite or log a warning.
                    print(f"Warning: Duplicate agent name '{agent_name}' found. "
                          f"Previous: '{reverse_lookup[agent_name]}', New: '{agent_type}_{agent_id}'. "
                          f"Overwriting.")
                reverse_lookup[agent_name] = f"{agent_type}_{agent_id}"
    return reverse_lookup

# Load the JSON data from the file
# filepath: /home/yjr/UnrealZoo/gym_unrealzoo-E034/example/random/agent_configs_sampler/agent_caption/agent_name.json
file_path = '/home/yjr/UnrealZoo/gym_unrealzoo-E034/example/random/agent_configs_sampler/agent_caption/agent_name.json'
try:
    with open(file_path, 'r') as f:
        agent_data = json.load(f)

    # Create the reverse lookup dictionary
    name_to_type_id_map = create_reverse_lookup_dict(agent_data)

    # Print some examples
    print(f"Reverse lookup for 'the_man_in_light_pants_and_sunglasses': {name_to_type_id_map.get('the_man_in_light_pants_and_sunglasses')}")
    print(f"Reverse lookup for 'Beagle_Dog': {name_to_type_id_map.get('Beagle_Dog')}")
    print(f"Reverse lookup for 'Blue_Car': {name_to_type_id_map.get('Blue_Car')}")
    print(f"Reverse lookup for 'Non_Existent_Agent': {name_to_type_id_map.get('Non_Existent_Agent')}")

    # You can print the whole map if it's not too large
    # import pprint
    # pprint.pprint(name_to_type_id_map)
    with open('/home/yjr/UnrealZoo/gym_unrealzoo-E034/example/random/agent_configs_sampler/agent_caption/reverse_agent_name.json', 'w') as f:
        json.dump(name_to_type_id_map, f, indent=4)
    

except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {file_path}.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
