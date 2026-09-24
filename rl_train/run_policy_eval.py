
# Parse command line arguments for log_dir
import sys
if len(sys.argv) > 1:
    log_dir = sys.argv[1]
else:
    log_dir = ""

if log_dir == "":
    log_dir = input("Enter the log directory: ")
    # log_dir = "docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs" # partial obs
    # log_dir = "docs/assets/tutorial_rl_models/train_session_20250729-005528_tutorial_full_obs" # Full obs
show_plot = False
import os

import numpy as np
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils.data_types import DictionableDataclass

import os
from rl_train.utils.train_log_handler import TrainLogHandler
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
import json
from rl_train.utils.data_types import DictionableDataclass
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.analyzer.reward_term_plotter import generate_selected_reward_plots
from rl_train.analyzer.standard_analysis import run_standard_gait_analysis
with open(os.path.join(log_dir, "session_config.json"), 'r') as f:
    config_dict = json.load(f)
config_class = EnvironmentHandler.get_config_type_from_session_id(config_dict["env_params"]["env_id"])
config = DictionableDataclass.create(config_class, config_dict)

for (idx, evaluate_param) in enumerate(config.evaluate_param_list):
    analyze_result_dir = os.path.join(log_dir,f"analyze_results_{idx:02d}")
    if not os.path.exists(analyze_result_dir):
        os.makedirs(analyze_result_dir)

    log_handler = TrainLogHandler(log_dir)
    log_handler.load_log_data(ImitationTrainCheckpointData)

    from rl_train.utils.data_types import DictionableDataclass
    DictionableDataclass.to_dict(log_handler.log_datas[-1])

    import sys
    sys.modules.pop('package.train_log_analyzer', None)
    from rl_train.analyzer.train_log_analyzer import TrainLogAnalyzer
    train_log_analyzer = TrainLogAnalyzer(log_handler)
    train_log_analyzer.plot_reward(result_dir=analyze_result_dir, show_plot=show_plot)


    


    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

        

    import sys
    from rl_train.analyzer.gait_analyze import GaitAnalyzer
    from rl_train.analyzer.gait_evaluate import GaitData


    gait_data_name = f"gait_evaluated_data.json"
    if os.path.exists(os.path.join(analyze_result_dir, gait_data_name)):
        user_input = input(f"Regenerate evaluate data? ({gait_data_name}) (y/n(anything))")
    else:
        user_input = "y"
    is_regen_evaluating_data = True if user_input == "y" else False

    from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
    gait_evaluator: ImitationGaitEvaluator = ImitationGaitEvaluator(log_handler, config)
    try:
        gait_evaluator.load_reference_data()
        gait_evaluator.initialize_env()
        env_id = config.env_params.env_id
        if is_regen_evaluating_data:
            plot_reward_keys = evaluate_param.get("plot_reward_keys", [])
            if env_id in ("myoAssistLegAcceleration-v0", "myoAssistLegAccelerationExo-v0"):
                expected_duration = evaluate_param["num_timesteps"] / config.env_params.control_framerate
                if abs(expected_duration - evaluate_param["episode_duration"]) > 1e-9:
                    print(
                        "Warning: Stage3 num_timesteps/control_framerate duration "
                        f"({expected_duration:.6g}s) differs from episode_duration "
                        f"({evaluate_param['episode_duration']:.6g}s)."
                    )
                gait_data_path = gait_evaluator.evaluate_stage3(
                    result_dir=analyze_result_dir,
                    file_name=gait_data_name,
                    max_timestep=evaluate_param["num_timesteps"],
                    initial_velocity=evaluate_param["initial_velocity"],
                    goal_velocity=evaluate_param["goal_velocity"],
                    target_acceleration=evaluate_param["target_acceleration"],
                    ramp_start_time=evaluate_param["ramp_start_time"],
                    episode_duration=evaluate_param["episode_duration"],
                    reference_index=evaluate_param.get("reference_index", 0),
                    seed=evaluate_param.get("seed", config.env_params.seed),
                    terminate_when_done=False,
                    plot_reward_keys=plot_reward_keys,
                )
                if gait_data_path is None:
                    gait_data_path = os.path.join(analyze_result_dir, gait_data_name)
            else:
                gait_data_path = gait_evaluator.evaluate(result_dir=analyze_result_dir,
                                                        file_name=gait_data_name,
                                                        velocity_mode=MyoAssistLegBase.VelocityMode[evaluate_param["velocity_mode"]],
                                                        target_velocity_period=evaluate_param["target_velocity_period"],
                                                        max_timestep=evaluate_param["num_timesteps"],
                                                        min_target_velocity=evaluate_param["min_target_velocity"],
                                                        max_target_velocity=evaluate_param["max_target_velocity"],
                                                        terminate_when_done=False,
                                                        plot_reward_keys=plot_reward_keys,
                                                        )
        else:
            gait_data_path = os.path.join(analyze_result_dir, gait_data_name)
        gait_data_path = os.path.abspath(gait_data_path)
        if not os.path.isfile(gait_data_path):
            raise FileNotFoundError(f"Evaluated gait data JSON not found: {gait_data_path}")
        if not gait_data_path.endswith(".json"):
            raise ValueError(f"Evaluated gait data path is not a JSON file: {gait_data_path}")

        gait_data = GaitData()
        gait_data.read_json_data(gait_data_path)
        segmented_ref_data = np.load("rl_train/reference_data/segmented.npz", allow_pickle=True)
        segmented_ref_data = {key: segmented_ref_data[key] for key in segmented_ref_data.files}

        if evaluate_param.get("plot_reward_keys", []):
            # Reward term plots are per-step eval traces, separate from existing return plots.
            generate_selected_reward_plots(
                gait_data,
                evaluate_param.get("plot_reward_keys", []),
                os.path.join(analyze_result_dir, "reward_term_plots"),
            )

        if not evaluate_param.get("skip_replay", False):
            replay_video_path = os.path.abspath(os.path.join(analyze_result_dir, "replay.mp4"))
            gait_evaluator.replay(gait_data_path, replay_video_path,
                                                        cam_distance=evaluate_param.get("cam_distance", 2.5),
                                                        # max_time_step=evaluate_param["num_timesteps"],
                                                        use_activation_visualization=evaluate_param.get("visualize_activation", False),
                                                        cam_type=evaluate_param.get("cam_type", "average_speed"),
                                                        realtime_plotting_info=evaluate_param.get("realtime_plotting_info", []),
                                                        video_library="imageio",
                                                        video_fps=config.env_params.control_framerate
                                                        )
        else:
            print("Skipping replay generation because evaluate_param.skip_replay is true.")

        analysis_exceptions = run_standard_gait_analysis(
            gait_data=gait_data,
            segmented_ref_data=segmented_ref_data,
            analyze_result_dir=analyze_result_dir,
            show_plot=show_plot,
            include_reference_plots=True,
        )
        if analysis_exceptions:
            print(f"Analysis completed with {len(analysis_exceptions)} exception(s):")
            for exception in analysis_exceptions:
                print(f"- {exception['function_name']}: {exception['exception']}")

    except Exception as exc:
        print(f"Evaluation failed for evaluate_param index {idx}: {exc}")
        raise
    finally:
        gait_evaluator.close()
