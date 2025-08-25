import os.path
import cv2
import gym


from gym import Wrapper

class ShowQuestionWrapper(Wrapper):
    def __init__(self, env, question_stem, question_options=None, question_answer=None): # Modified to take stem and options separately
        super().__init__(env)
        self.question_stem = question_stem
        self.question_options = question_options if question_options is not None else []
        self.question_answer = question_answer  # This can be used later if needed, e.g., for evaluation
    def step(self, action):
        obs, reward, termination,truncation, info = self.env.step(action)
        img_show = self.add_question_on_image(obs[0][...,:3].squeeze())
        info["img_show"] = img_show
        return obs, reward, termination,truncation, info

    def reset(self, **kwargs):
        states,info = self.env.reset(**kwargs)
        img_show = self.add_question_on_image(states[0][...,:3].squeeze())
        info["img_show"] = img_show
        return states,info
    
    def add_question_on_image(self, goal_show):
        import numpy as np
        import cv2
        import textwrap
        goal_show = np.ascontiguousarray(goal_show)
        if goal_show.dtype != np.uint8:
            goal_show = (goal_show * 255).clip(0, 255).astype(np.uint8)

        all_display_lines = [] # This will store all lines to be rendered

        # Wrap the question stem
        if self.question_stem: # Check if stem is not None or empty
            stem_lines = textwrap.wrap(self.question_stem, width=70)
            all_display_lines.extend(stem_lines)

        # Wrap each option (options already have prefixes like "A. ", "B. " from JSON)
        if self.question_options: # Check if options list is not None or empty
            for option_text_with_prefix in self.question_options:
                if option_text_with_prefix: # Check if the option string itself is not None or empty
                    wrapped_option_lines = textwrap.wrap(option_text_with_prefix, width=70)
                    all_display_lines.extend(wrapped_option_lines)
        
        if self.question_answer: # Check if answer is not None or empty
            wrapped_answer_lines = textwrap.wrap(f"Answer: {self.question_answer}", width=70)
            all_display_lines.extend(wrapped_answer_lines)
        
        line_y_start = 30
        line_height = 20

        for i, line_content in enumerate(all_display_lines):
            y = line_y_start + i * line_height
            
            overlay = goal_show.copy()
            # Use cv2.getTextSize to get the exact width of the text for the rectangle
            text_size, _ = cv2.getTextSize(line_content, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            rect_width = text_size[0] + 10 # Add some padding

            # Draw the semi-transparent background rectangle
            cv2.rectangle(overlay, (5, y - 15), (5 + rect_width, y + 5), (128, 128, 128), -1)
            alpha = 0.6 # Transparency level
            blended = cv2.addWeighted(overlay, alpha, goal_show, 1 - alpha, 0)
            goal_show = blended # Update goal_show with the blended image
            
            # Put the text on the image (on top of the blended background)
            cv2.putText(goal_show, line_content, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA) # Black text

        return goal_show
