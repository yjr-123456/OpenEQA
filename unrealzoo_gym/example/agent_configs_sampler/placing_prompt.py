sys_prompt_camera = """
    You are a smart advisor in placing cameras in a 3D environment.
    Your task is to decide the locations of the cameras among candidate locations in the virtual environment.
"""

usr_prompt_camera = """
    We will provide you with a top-down view image of the environment with discrete candidate locations marked in red with their ids and orientations.
    We will place cameras at the locations you selected to capture images of objects shown in the top-down view.
    Your task is to select the most two or three suitable locations for placing the cameras to photograph the objects.

    note:
    1. make sure that the cameras is not blocked by other objects in the environment such as bushes, trees, barriers etc.
    2. try to cover as many objects as possible.
    3. make sure the distribution of the cameras are evenly spread out.
    4. **Do not select ids that are not marked in red in the top-down view image.**

    attention:
    1. the camera id is just a number, do not output it's perfix like "C"

    output format:
    Use xml format, and use a list to represent multiple selected node ids:
    <a>[node_id1,node_id2,node_id3]</a>
"""

sys_prompt_point_sample = """
    You are a smart placing agent in a 3D virtual environment.
    Your task is to iteratively place objects in this virtual environment using a top - down view image
"""

usr_prompt_point_sample = """
    input:
    1. the top-down view image of the environment with discrete placeable locations marked in green with their ids, and there may be objects you placed before that are marked with blue boxes.
       On the right side of image, there is the object you need to place in the environment.We will also provide you with it's size and rotation information.
    2. all the coordinates of placeable locations in the environment corresponding to the ids marked in the top-down image.

    requirements:
    1. Make sure that the object you place does not collide with any other objects(object you place before and other objects in the environment).
    2. Make sure that all the objects you placed are connected with each other.Do not allow any environmental objects (barriers bushes etc) to obstruct the space between the two objects you placed.
    3. Make sure that no object exceeds the boundary of the top-down image.
    4. if you are placing a big object(size > 300*150), please keep a safe distance(> 200cm) from any other objects you placed before (and environmental objects).
       (distance between big and small object > 150cm when placing a small object)
    5. **safe distance explaination**: 
       the safe distance is the minimum distance between the bounding boxes' edges of two objects in the top-down view.
       Thus, the estimation of distance should consider the size and rotation of objects.Because the location you select is the center of the object.

    think step by step:
    1. Perceive the top-down view image to find out the environmental objects.And evaluate their size and rotation.
    2. Perceive the size and rotation of the object you need to place.
    3. Evaluate the free space and distribution of objects in the environment(both placed and environmental).
       Pay attention to the environmental objects (barriers bushes etc) that may obstruct the placement of the object or may cause collisions with your object to be placed.
    4. Pick one free area that is suitable for placing the object.Make sure meet all the requirements in the "requirements" section.
    5. Correlate the coordinates of all points with their respective IDs.And select the most suitable point ID for placing the object based on its size and rotation.

    note:
    1. do not output your reasoning process, only output the final answer in form of output format.
    
    output format:
    Use xml format, and use a single integer to represent the selected node id:
    <a>node_id</a>
    <b>node position in a python list format</b>
"""

sys_prompt_event_plot = """
    You are a cogitative event planner.
    Your task is to design reasonable action sequences for several agents in a 3D virtual environment.
"""

usr_prompt_event_plot = """
    input:
    1. a top-down view of the environment.
    2. a group of agents with their full-body shots.
    3. excutable actions **only for human figures**.

    goal:
    Your task is to organize a coherent event by organizing a series of executable actions for each human figures.

    requirements:
    1. Ensure that the entire event is coherent and logical, and achievable with the provided actions and agents.
    2. Each human figure should have at least one action to perform during the event.


    output format:
    Use xml format:
    <a>event description in natural language</a>
    <b>action list of agents in json format</b>

    Just an example!:
    <a> A person enters the house</a>
    <b>{
        "agent_1": ["action 1", "action 2", "action 3"],
        ....
        "agent_2": ["action 1", "action 2"]
    }
    </b>
"""



PROMPT_TEMPLATES = {
    "sample_point_prompt_cot": {
        "system": sys_prompt_point_sample,
        "user": usr_prompt_point_sample
    },
    "sample_camera_point_cot": {
        "system": sys_prompt_camera,
        "user": usr_prompt_camera
    },
    "event_plot_prompt":{
        "system": sys_prompt_event_plot,
        "user": usr_prompt_event_plot
    }
}


def load_prompt(template_name):
    return PROMPT_TEMPLATES[template_name]["system"], PROMPT_TEMPLATES[template_name]["user"]
