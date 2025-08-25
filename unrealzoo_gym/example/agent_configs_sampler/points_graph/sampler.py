import unittest
from unittest.mock import patch, mock_open
import json
import random
import string
from collections import Counter
from .agent_sampler import AgentSampler, AGENT_OPTIONS, DEFAULT_DISTRIBUTION

# Relative import for the module to test
# Assumes test_agent_sampler.py is in the same directory as agent_sampler.py
# and this directory is part of a package.

# Mock feature caption data for consistent testing
MOCK_FEATURE_CAPTIONS = {
    "player": {
        "1": "Mock Player 1 Description",
        "2": "Mock Player 2 Description",
        "7": "Mock Bald Man Description"
    },
    "car": {
        "1": "Mock Red Car Description",
        "2": "Mock White Car Description"
    },
    "animal": {
        "0": "Mock Beagle Description"
    },
    "drone": {
        "0": "Mock Drone Description"
    }
}

class TestAgentSampler(unittest.TestCase):

    def setUp(self):
        # Patch load_json_file to return mock data and avoid file dependency
        self.patcher = patch('example.random.agent_configs_sampler.agent_sampler.load_json_file', return_value=MOCK_FEATURE_CAPTIONS)
        self.mock_load_json = self.patcher.start()
        self.sampler = AgentSampler(feature_path="dummy_path.json") # Path is now for mock

    def tearDown(self):
        self.patcher.stop()

    def test_initialization_default(self):
        self.assertEqual(self.sampler.agent_options, AGENT_OPTIONS)
        self.assertEqual(self.sampler.default_distribution, DEFAULT_DISTRIBUTION)
        self.assertEqual(self.sampler.feature_caption, MOCK_FEATURE_CAPTIONS)
        self.mock_load_json.assert_called_once_with("dummy_path.json")

    def test_initialization_custom(self):
        custom_options = {"custom_type": {"app_id": [100], "animation": ["fly"]}}
        custom_dist = {"custom_type": 1.0}
        # Custom feature captions can be tested by re-patching load_json_file or passing feature_caption directly if __init__ supported it
        
        sampler_custom = AgentSampler(custom_options=custom_options, custom_distribution=custom_dist, feature_path="another_dummy.json")
        self.assertEqual(sampler_custom.agent_options, custom_options)
        self.assertEqual(sampler_custom.default_distribution, custom_dist)
        self.mock_load_json.assert_called_with("another_dummy.json")


    def test_get_agent_caption(self):
        self.assertEqual(self.sampler.get_agent_caption("player", 1), "Mock Player 1 Description")
        self.assertEqual(self.sampler.get_agent_caption("player", "1"), "Mock Player 1 Description")
        self.assertEqual(self.sampler.get_agent_caption("car", 2), "Mock White Car Description")
        self.assertEqual(self.sampler.get_agent_caption("player", 99), "unknown_player_99")
        self.assertEqual(self.sampler.get_agent_caption("unknown_type", 1), "unknown_unknown_type_1")

    def test_get_agent_name(self):
        self.sampler.batch_id = "TESTBATCH"
        self.assertEqual(self.sampler.get_agent_name("player", 1), "player_1_TESTBATCH")
        self.assertEqual(self.sampler.get_agent_name("car", "app7"), "car_app7_TESTBATCH")

    @patch.object(AgentSampler, 'sample_with_specific_counts_no_repeat')
    def test_sample_agents_distribution_logic(self, mock_sample_specific_counts):
        # Assuming sample_agents calls sample_with_specific_counts_no_repeat
        # due to sample_with_specific_counts not being defined.
        # If this assumption is wrong, this mock target needs to change.
        mock_sample_specific_counts.return_value = {"mock_agent_type": {"app_id": [1]}} 
        
        total_agents_to_sample = 10
        self.sampler.sample_agents(total_agents=total_agents_to_sample)
        
        args, _ = mock_sample_specific_counts.call_args
        agent_counts = args[0]
        
        self.assertIsInstance(agent_counts, dict)
        self.assertEqual(sum(agent_counts.values()), total_agents_to_sample)

        # Check if counts are somewhat as expected by default distribution
        # Example: player default is 0.3, for 10 agents, count should be 3
        # The exact distribution depends on int casting and remainder handling.
        # For player (0.3 * 10 = 3), animal (0.2 * 10 = 2), car (0.25 * 10 = 2), drone (0.15 * 10 = 1)
        # Remaining for motorbike: 10 - 3 - 2 - 2 - 1 = 2
        # Expected: {'player': 3, 'animal': 2, 'car': 2, 'drone': 1, 'motorbike': 2}
        
        # Re-calculate expected counts based on the implementation's logic
        expected_counts = {}
        remaining_calc = total_agents_to_sample
        dist_items = sorted(self.sampler.default_distribution.items(), key=lambda item: item[0]) # Consistent order
        
        temp_valid_dist = {k:v for k,v in self.sampler.default_distribution.items() if k in self.sampler.agent_options}
        total_norm = sum(temp_valid_dist.values())
        normalized_dist = {k: v/total_norm for k,v in temp_valid_dist.items()}

        agent_types_sorted = sorted(list(normalized_dist.keys()))

        for agent_type in agent_types_sorted[:-1]:
            count = int(total_agents_to_sample * normalized_dist[agent_type])
            expected_counts[agent_type] = count
            remaining_calc -= count
        if agent_types_sorted:
            expected_counts[agent_types_sorted[-1]] = max(0, remaining_calc)
        
        self.assertEqual(agent_counts, expected_counts)


    def test_sample_with_specific_counts_no_repeat_basic(self):
        agent_counts = {"player": 2, "car": 1}
        result = self.sampler.sample_with_specific_counts_no_repeat(agent_counts)

        self.assertIn("player", result)
        self.assertEqual(len(result["player"]["app_id"]), 2)
        self.assertEqual(len(result["player"]["animation"]), 2)
        self.assertIsNotNone(self.sampler.batch_id) # batch_id should be set
        self.assertEqual(len(result["player"]["name"][0]), len(f"player_{result['player']['app_id'][0]}_{self.sampler.batch_id}"))
        self.assertIn(result["player"]["feature_caption"][0], MOCK_FEATURE_CAPTIONS["player"].values())


        self.assertIn("car", result)
        self.assertEqual(len(result["car"]["app_id"]), 1)

    def test_sample_with_specific_counts_no_repeat_no_repetition(self):
        # player has many app_ids, so 2 should be no repeat
        agent_counts = {"player": 2}
        result = self.sampler.sample_with_specific_counts_no_repeat(agent_counts)
        sampled_app_ids = result["player"]["app_id"]
        self.assertEqual(len(sampled_app_ids), 2)
        self.assertEqual(len(set(sampled_app_ids)), 2) # Check for uniqueness

    def test_sample_multiple_identical_objects_for_counting(self):
        # User's specific request: sample multiple identical objects for counting problems
        agent_type_to_test = "car" # Cars have 2 unique app_ids: [1, 2]
        num_to_sample = 5          # Request 5 cars, forcing repetition

        agent_counts = {agent_type_to_test: num_to_sample}
        result = self.sampler.sample_with_specific_counts_no_repeat(agent_counts)

        self.assertIn(agent_type_to_test, result)
        sampled_type_data = result[agent_type_to_test]

        self.assertEqual(len(sampled_type_data["app_id"]), num_to_sample)
        self.assertEqual(len(sampled_type_data["animation"]), num_to_sample)
        self.assertEqual(len(sampled_type_data["name"]), num_to_sample)
        self.assertEqual(len(sampled_type_data["feature_caption"]), num_to_sample)

        allowed_app_ids = AGENT_OPTIONS[agent_type_to_test]["app_id"]
        for app_id in sampled_type_data["app_id"]:
            self.assertIn(app_id, allowed_app_ids)
        
        # Repetition should occur for app_id as num_to_sample > len(allowed_app_ids)
        self.assertLessEqual(len(set(sampled_type_data["app_id"])), len(allowed_app_ids))
        if len(allowed_app_ids) > 0 : # only if there are options to pick from
             self.assertTrue(len(set(sampled_type_data["app_id"])) <= num_to_sample)


        allowed_animations = AGENT_OPTIONS[agent_type_to_test]["animation"] # For car, this is ["None"]
        for anim in sampled_type_data["animation"]:
            self.assertIn(anim, allowed_animations)
        # For car, animation is ["None"], so all 5 animations will be "None"
        self.assertEqual(len(set(sampled_type_data["animation"])), 1 if allowed_animations else 0)


    def test_add_names_to_output(self):
        self.sampler.batch_id = "NAMEBATCH" # Manually set for this test
        sampled_agents_input = {
            "player": {"app_id": [1, 7], "animation": ["stand", "crouch"], "other_data": "test_value"},
            "car": {"app_id": [1], "animation": ["None"]}
        }
        result = self.sampler.add_names_to_output(sampled_agents_input)

        self.assertEqual(result["player"]["name"], ["player_1_NAMEBATCH", "player_7_NAMEBATCH"])
        self.assertEqual(result["player"]["feature_caption"], 
                         ["Mock Player 1 Description", "Mock Bald Man Description"])
        self.assertEqual(result["player"]["other_data"], "test_value") # Check other keys preserved

        self.assertEqual(result["car"]["name"], ["car_1_NAMEBATCH"])
        self.assertEqual(result["car"]["feature_caption"], ["Mock Red Car Description"])

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_to_json_include_names(self, mock_json_dump, mock_file_open):
        self.sampler.batch_id = "JSONTEST1" # Set batch_id as add_names_to_output will be called
        sampled_data = {"player": {"app_id": [1], "animation": ["stand"]}}
        output_file = "dummy_output1.json"

        self.sampler.save_to_json(sampled_data, output_file, include_names=True)

        mock_file_open.assert_called_once_with(output_file, 'w')
        
        args, kwargs = mock_json_dump.call_args
        dumped_container = args[0]
        self.assertIn("target_configs", dumped_container)
        dumped_data = dumped_container["target_configs"]

        self.assertIn("player", dumped_data)
        self.assertIn("name", dumped_data["player"])
        self.assertEqual(dumped_data["player"]["name"][0], "player_1_JSONTEST1")
        self.assertIn("feature_caption", dumped_data["player"])
        self.assertEqual(dumped_data["player"]["feature_caption"][0], MOCK_FEATURE_CAPTIONS["player"]["1"])
        self.assertEqual(kwargs.get("indent"), 4)

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch.object(AgentSampler, 'add_names_to_output')
    def test_save_to_json_no_names(self, mock_add_names, mock_json_dump, mock_file_open):
        input_sampled_data = {"player": {"app_id": [1], "animation": ["stand"], "original_key": "value"}}
        output_file = "dummy_output2.json"

        self.sampler.save_to_json(input_sampled_data, output_file, include_names=False)

        mock_add_names.assert_not_called()
        mock_file_open.assert_called_once_with(output_file, 'w')
        
        args, kwargs = mock_json_dump.call_args
        dumped_container = args[0]
        self.assertIn("target_configs", dumped_container)
        self.assertEqual(dumped_container["target_configs"], input_sampled_data) # Should be original data
        self.assertEqual(kwargs.get("indent"), 4)

    @patch("builtins.print")
    def test_print_stats(self, mock_print):
        self.sampler.batch_id = "STATSID" # get_agent_name uses this
        sampled_data = {
            "player": {"app_id": [1, 1], "animation": ["stand", "crouch"], "name": ["p1", "p2"], "feature_caption": ["c1", "c2"]}, # print_stats uses len(config['app_id'])
            "car": {"app_id": [2], "animation": ["None"], "name": ["c1"], "feature_caption": ["cc1"]}
        }
        # Note: print_stats's name printing logic: self.get_agent_name(agent_type, i+1)
        # This might not reflect the actual sampled names if app_ids are not 1-based sequential.
        # The test will verify its current behavior.
        
        self.sampler.print_stats(sampled_data)

        mock_print.assert_any_call("采样完成:")
        mock_print.assert_any_call("  - player: 2 个")
        mock_print.assert_any_call(f"    app_id分布: {dict(Counter([1,1]))}") # Counter result
        
        # Check the sequence of calls for names for player
        # It will call print("    对应名称: ", end="") then print("player_1_STATSID, ", end="") then print("player_2_STATSID, ", end="") then print()
        # This is hard to assert precisely with assert_any_call for the full line due to multiple prints.
        # We can check parts or use call_args_list.
        
        printed_text = "".join(call_args[0][0] for call_args in mock_print.call_args_list if "对应名称" in call_args[0][0] or "player_" in call_args[0][0])
        self.assertIn("    对应名称: player_1_STATSID, player_2_STATSID, ", printed_text)
        
        mock_print.assert_any_call(f"    动作分布: {dict(Counter(['stand', 'crouch']))}")
        mock_print.assert_any_call("  - car: 1 个")
        mock_print.assert_any_call(f"    app_id分布: {dict(Counter([2]))}")
        mock_print.assert_any_call("总计: 3 个agents")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)