def search_prompt_begin(question):

    p = f"""
        You are a question answering robot, you need analyze target information based on the question:{question}.
        
        Please complete the following tasks:
        1. Analyze and extract the target object and secondary objects from the question;


        Use XML tags to output results in the following format:
        <a>Analysis of target information in the question</a>
        <b>the target object list</b>

        Example:\n
        [input question]:\n
        "which is closer to the man in red shirt, the red car or the white car?"\n
        [output]:\n
        <a>This is a relative distance question about the man in red shirt.Thus the target object are the man in red shirt, red car and blue car.</a>\n
        <b>["the_man_in_red_shirt","red_car", "white_car"]</b>\n
    """
    return p

def search_begin(question):

    p = f"""
        You are a question answering robot, you need analyze target information based on the question:{question}.
        
        Please complete the following tasks:
        1. Analyze and extract the main target object and secondary objects from the question;

        note:the main target object is only one and most important object in the question.However the secondary trarget objects is at least 0.

        Use XML tags to output results in the following format:
        <a>Analysis of target information in the question</a>
        <b>the main target object</b>
        <c>the secondary target objects list.(return none list([]) if there is no secondary object)</c>

        Example:\n
        [input question]:\n
        "which is closer to the man in red shirt, the red car or the white car?"\n
        [output]:\n
        <a>This is a relative distance question about the man in red shirt.Thus the main object is the man in red shirt, the secondary objects are red car and blue car.</a>\n
        <b>the_man_in_red_shirt</b>\n
        <c>["red_car", "white_car"]</c>\n
    """
    return p

def relavance_prompt(question):
    p = f"""
        You are an intelligent analysis robot.
        Your task is to analyze the question and your current observation, and evaluate the relevance between the question and current observation.

        [input]:
        1. question:"{question}"
        2. current observation, which will be provided in a format of base64 encoded image.

        [thinking steps]:
        1. Firstly, analyze whether **the objects in question** appear in the current observation;
        2. Give your relavance score based on your analysis above, the score should be a float between 0 and 1.
        
        Use **XML tags** to output results in the following format:
        <a>Analysis of relevance bewteen question and current observation according to the thinking steps</a>\n
        <b>relevance score</b>\n

        **JUST AN FORMAT EXAMPLE NOT A REFERENCE**:\n
        [input]:\n
        1. question:"which is closer to the man in red shirt, the red car or the white car?"\n
        2. current observation: Provided in a format of base64 encoded image.\n
        [output]:\n
        <a>I can see two cars but cannot identify the man in red shirt mentioned in the question.The red man in shirt may be occluded by the car in front of me.Based on my analysis above,the relevance score is 0.4</a>\n
        <b>0.4</b>\n
    """
    return p


def ask_for_depth(question, target_object):
    target_object = ', '.join(target_object) if isinstance(target_object, list) else target_object
    p = f"""
        You are an intelligent vision analysis robot.
        Your task is to analyze the current observation and determine if depth image is needed for collecting clues for the question:{question}.

        [input]:
        1. question:"{question}"
        2. target object:{target_object}
        2. current observation, which will be provided in a format of base64 encoded image.
        [note]:
        1. clues: visual information that may help answer the question
        2. depth img: which will be provided in a format of base64 encoded image(jpeg).
        [output format]:
        `Use **XML tags** to output results in the following format:
        <a>yes or no</a>\n
    """
    return p

def key_clue_collection(question,target_object):
    target_object = ', '.join(target_object) if isinstance(target_object, list) else target_object
    p = f"""
        You are an intelligent vision analysis robot.
        Your task is to analyze the current observation and find key clues for the question:{question}.

        [input]:
        1. question:"{question}"
        2. target object:{target_object}
        3. current observation, which will be provided in a format of base64 encoded image.We will provide you with a RGB image.By the way,a depth image will probably be provided to you if needed.

        [note]:
        1. Key clues are the decisive evidence that help you answer the question.**If there are no obvious clues,return "none" in <b>key clues</b>**
        2. Please analyze based on the current observation! Do not guess!If you cannot find any clues, return "none" in <b>key clues</b> as case 1 does.
        3. Pay attention to the **relative location** of objects in the question.**For example,"Where is A relative to B", always describe A's position with respect to B as the reference point**. Do not describe B's position relative to A, and do not use the observer's perspective unless explicitly asked.
           That means, firstly, you should determine the orientation of object B in your view.
           Then, estimate the direction of object A relative to object B's orientation.
        4. Explanation of depth image:
            -1. You can qualitatively determine the distance between objects and you by using the depth map. 
            -2. In the picture, the darker areas are farther from you, while the lighter areas are closer to you.
            -3. You can also combine the depth information with the RGB observation to get a more comprehensive understanding of the scene such as object grounding, spatial relationships, and occlusions.

        [thinking steps]:
        1. Analyze the question find out the type of objects relationship in the question, such as "which is closer", "how many", etc
        2. According to the target objects can you find whrere the targets object are in the current observation?
        3. If you can find the target object, can you figure out the key clues of the question such as where the targets are or relationships of the target objects?
           if you can not find the target object, can you find other clues that help you answer the question?
        3. Based on your analysis above, do you think you find out key clues that help you answer the question? **If yes, please give your key clues in <b>key clues</b>, otherwise return "none" in <b>key clues</b>.

        [output format]:
        `Use **XML tags** to output results in the following format:
        <a>Analysis of key clues in the current observation based on the question</a>\n
        <b>key clues</b>\n 
        
        [**JUST AN FORMAT EXAMPLE NOT A REFERENCE**]:
        case 1:\n
        <a>The question asks relative distance of red car and white car from the man.But I can't figure out the target object.So there are no clues in current views.</a>\n
        <b>none</b>\n 
        case 2:\n
        <a>The question asks relative location of red car from the man in red shirt.I can see that the red car is in the right side of the man in red shirt, which is a key clue to answer the question.</a>\n
        <b>the red car is in the right side of the man in red shirt</b>\n
    """
    return p

def direction_planner(question, target_object):
    target_object = ', '.join(target_object) if isinstance(target_object, list) else target_object
    p = f"""
        You are an intelligent exploration robot.
        Your task is to determine direction worth exploring.

        [input]:
        1. question:"{question}"
        2. target object:{target_object}
        3. previous exploration history: which will be provided with the last observations.
        4. last observations, which will be provided in a format of base64 encoded image.

        [thinking steps]:
        1. Firstly, analyze whether **the objects in question** appear in the current observation;
        2. Do you think they all appear in your current view?\n
           If they partly appear in your view, choose one area that you can further explore the objects that are partly appear or unseen in your view\n
           If they there are no objects in your view, which direction might the target object appear?Please choose one area to explore\n
        3. To reach the area you choose, what action should you take? Please choose from the following actions: ["move_forward", "move_backward","turn_left", "turn_right"];
        4. Based on your analysis above, to better explore the area interset,give your next continuous action predictions based on the following actions(no more than three actions).
        
        
        [note]:
            1. actions explaining:
                - "move_forward": move forward in the current direction by 100cm;
                - "move_backward": move backward in the current direction by 100cm;  
                - "turn_left": turn left 30 degrees;
                - "turn_right": turn right 30 degrees;
                **You can only choose the four actions above.So please establish the measurement of distance and angle**
            2. exploration memory contains the last steps of action reasoning, which respectively resulted in the last several observations.\n
               They are all in chronological order.Please note the causality of actions and observations, then combine past actions with observations for analysis.
        [output format]:            
        Use XML tags to output results in the following format:
        <a>Analysis of area of interest choosing</a>\n
        <b>your action reasoning</b>\n
        <c>continuous action list</c>\n
 
        **JUST AN FORMAT EXAMPLE NOT A REFERENCE**:
        <a>I can see the red man in shirt in my view,and half of a white car in my right side.Maybe my right side area contains target object</a>\n
        <b>I should turn around to explore more of the environment and search for the white car.Thus,I should take action1 first and take action2 to figure out whether there are target objects</b>\n
        <c>["action1", "action2", "action3"]</c>\n
        """
    return p

def search_prompt(question, target_object,exploration_memory=None):
    target_object = ', '.join(target_object) if isinstance(target_object, list) else target_object
    p = f"""
        You are exploring the environment and planning actions to answer the **question:{question}**.
        To better explore the environment and make action,you need to make use of your previous exploration history:{exploration_memory} and reason next action in such steps:
        1. Analyze the current observation with **the question and previous exploration history**.Especially, pay attention to the target object:{target_object} in the observation; 
        2. Based on your analysis, reason the next action to take in order to explore environmental clues to answer the question;
        3. Give your next action prediction based on **your current observation analysis and action reasoning** above, please choose from the following actions: ["move_forward", "move_backward","turn_left", "turn_right"],
        4. Give your confidence of answering the question, the score should be a float between 0 and 1, where 1 means very confident and 0 means not confident at all.

        note:
            1. actions explaining:
                - "move_forward": move forward in the current direction by 100cm;
                - "move_backward": move backward in the current direction by 100cm;  
                - "turn_left": turn left 30 degrees;
                - "turn_right": turn right 30 degrees;
                **if it's hard to choose an action, please randomly choose one to explore the environment**
            2. please analyze and explore the environment **in first person view** as if you were in the environment.
        Use XML tags to output results in the following format:
        <a>Analysis of current observation</a>\n
        <b>next action reasoning</b>\n
        <c>next action</c>\n
        <d>confidence</d>\n

        **JUST AN FORMAT EXAMPLE NOT A REFERENCE**:
        <a>I can see a residential street with houses. No target objects are clearly visible yet.</a>\n
        <b>I should move forward to explore more of the environment and search for the target objects.</b>\n
        <c>move_forward</c>\n
        <d>0.2</d>\n
    """
    return p

def question_answer_prompt(question, target_object):
    target_object = ', '.join(target_object) if isinstance(target_object, list) else target_object
    p = f"""
        You are an intelligent information synthesizer question answering robot.
        
        Your task is to sysnthesize the information you have collected from the environment answer the question:{question} and give your confidence of your answer.

        [input]:
        1. question:"{question}"
        2. key observations, which will be provided in a format of base64 encoded image.Both RGB and depth images will be provided.
        3. exploration clues, which will be provided together with the key observations.

        [thinking steps]:
        1. Analyze the question find out the type of objects relationship in the question, such as "which is closer", "how many", etc
        2. Analyze and sysnthesize the information you have collected in your exploration history pay attention to the key clues that help you answer the question;Pay attention to the observations that are related to the target object:{target_object} and their relationship in the question;
        3. Based on your analysis above, give your answer to the question;
        4. Based on your procedure of answering questions, give your confidence of your answer being correct, the score should be a float between 0 and 1.**Please be cautious and not that confident.**
        
        [note]: 
        1. If there are options in the question, you should choose the best option based on your observations and reasoning.Otherwise, you should give your answer based on your observations and reasoning.
        2. **The clues provided to you may not be absolutely correct**, so you should analyze the clues and make your own judgment based on all your previous observations and analysis.
        3. Please don't be overconfident about your answer especially when you have few clues.
        4. Pay attention to the **relative location** question. 
           When answering a question like "Where is A relative to B", the question always describes A's position with respect to B as the reference point.
           Do not describe B's position relative to A, and do not use the observer's perspective unless explicitly asked.
           That means, firstly, you should determine the orientation of object B in your view.
           Then, estimate the direction of object A relative to object B's orientation.

        Use XML tags to output results in the following format:
        <a>Your reasons to answer the question</a>\n
        <b>Your answer</b>\n
        <c>confidence score</c>\n

        **ANSWER GUIDELINES**:
            - If multiple choice: Choose the best option, **give the hole option directly**
            - If counting: Provide exact number in Arabic numerals.
            - Base answer ONLY on what you observed

        **JUST AN FORMAT EXAMPLE NOT A REFERENCE**:
        <a>Based on my exploration, I observed two red cars and one white car. The man in red shirt was standing *****(answer clue) to the white car than to either red car.</a>
        <b>B. The man in red shirt</b>
        <c>0.9</c>
    """
    return p
