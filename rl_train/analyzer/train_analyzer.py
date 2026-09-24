from rl_train.utils.train_log_handler import TrainLogHandler
import os
import json
from rl_train.utils.data_types import DictionableDataclass
from rl_train.analyzer.train_log_analyzer import TrainLogAnalyzer
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from rl_train.envs.environment_handler import EnvironmentHandler



from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
from rl_train.analyzer.reward_term_plotter import generate_selected_reward_plots


from rl_train.analyzer.gait_analyze import GaitAnalyzer
from rl_train.analyzer.gait_evaluate import GaitData
from rl_train.analyzer.standard_analysis import run_standard_gait_analysis
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from enum import Enum
import numpy as np
import traceback

class TrainAnalyzer:
    class SequenceElement(Enum):
        REWARD_PLOT = 0
        EVALUATE = 1
        ANALYZE = 2
        # Should have detail analysis

    def analyze_in_sequence(self, log_dir:str, show_plot:bool):
        with open(os.path.join(log_dir, "session_config.json"), 'r') as f:
            config_dict = json.load(f)
            config_type = EnvironmentHandler.get_config_type_from_session_id(config_dict["env_params"]["env_id"])
            config = DictionableDataclass.create(config_type, config_dict)

        log_handler = TrainLogHandler(log_dir)
        log_handler.load_log_data(ImitationTrainCheckpointData)
        train_log_analyzer = TrainLogAnalyzer(log_handler)
        from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

        def write_report(report, analyze_result_dir):
            train_analyzer_report_path = os.path.join(analyze_result_dir, "train_analyzer_report.json")
            with open(train_analyzer_report_path, 'w') as f:
                json.dump(report, f)

        gait_evaluator = ImitationGaitEvaluator(log_handler, config)
        try:
            gait_evaluator.load_reference_data()
            gait_evaluator.initialize_env()

            for (eval_idx, evaluate_param) in enumerate(config.evaluate_param_list):
                train_analyzer_report = {
                    "num_timesteps":log_handler.log_datas[-1].num_timesteps,
                    "exceptions":[],
                }

                analyze_result_dir = os.path.join(log_dir,f"analyze_results_{log_handler.log_datas[-1].num_timesteps}_{eval_idx:02d}")
                if not os.path.exists(analyze_result_dir):
                    os.makedirs(analyze_result_dir)

                try:
                    train_log_analyzer.plot_reward(result_dir=analyze_result_dir, show_plot=show_plot)
                    train_log_analyzer.plot_reward_dict(result_dir=analyze_result_dir, show_plot=show_plot, mult_weights=False)
                    train_log_analyzer.plot_reward_dict(result_dir=analyze_result_dir, show_plot=show_plot, mult_weights=True)
                    # train_log_analyzer.plot_reward_weights(result_dir=analyze_result_dir)
                except Exception as exc:
                    train_analyzer_report["exceptions"].append({
                        "function_name": "plot_train_logs",
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                    })

                if "initial_velocity" not in evaluate_param:
                    config.env_params.min_target_velocity = evaluate_param["min_target_velocity"]
                    config.env_params.max_target_velocity = evaluate_param["max_target_velocity"]

                gait_data_name = f"gait_evaluated_data_{eval_idx:02d}.json"
                gait_data_path = os.path.join(analyze_result_dir, gait_data_name)
                is_regen_evaluating_data = not os.path.exists(gait_data_path)

                try:
                    plot_reward_keys = evaluate_param.get("plot_reward_keys", [])
                    if is_regen_evaluating_data:
                        if "initial_velocity" in evaluate_param:
                            gait_data_path = gait_evaluator.evaluate_stage3(
                                result_dir=analyze_result_dir,
                                file_name=gait_data_name,
                                initial_velocity=evaluate_param["initial_velocity"],
                                goal_velocity=evaluate_param["goal_velocity"],
                                target_acceleration=evaluate_param["target_acceleration"],
                                ramp_start_time=evaluate_param["ramp_start_time"],
                                episode_duration=evaluate_param["episode_duration"],
                                reference_index=evaluate_param.get("reference_index"),
                                seed=evaluate_param.get("seed"),
                                max_timestep=evaluate_param.get("num_timesteps"),
                                terminate_when_done=False,
                                plot_reward_keys=plot_reward_keys,
                            )
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

                    gait_data = GaitData()
                    gait_data.read_json_data(gait_data_path)
                except Exception as exc:
                    train_analyzer_report["exceptions"].append({
                        "function_name": "evaluate",
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    write_report(train_analyzer_report, analyze_result_dir)
                    continue

                try:
                    try:
                        segmented_ref_data = np.load("rl_train/reference_data/segmented.npz", allow_pickle=True)
                        segmented_ref_data = {key: segmented_ref_data[key] for key in segmented_ref_data.files}
                    except FileNotFoundError:
                        print("Warning: Reference data file not found. Running analysis without reference plots.")
                        segmented_ref_data = None
                    exception_report_list = self.analyze(gait_data, segmented_ref_data, analyze_result_dir, show_plot)
                except Exception as exc:
                    exception_report_list = [{
                        "function_name": "analyze",
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                    }]

                train_analyzer_report["exceptions"].extend(exception_report_list)

                if evaluate_param.get("plot_reward_keys", []):
                    try:
                        # Reward term plots are per-step eval traces, separate from existing return plots.
                        generate_selected_reward_plots(
                            gait_data,
                            evaluate_param.get("plot_reward_keys", []),
                            os.path.join(analyze_result_dir, "reward_term_plots"),
                        )
                    except Exception as exc:
                        train_analyzer_report["exceptions"].append({
                            "function_name": "plot_reward_terms",
                            "exception": str(exc),
                            "traceback": traceback.format_exc(),
                        })

                if not evaluate_param.get("skip_replay", False):
                    try:
                        file_name = f'replay_{eval_idx:02d}.mp4'
                        gait_evaluator.replay(gait_data_path, os.path.join(analyze_result_dir, file_name),
                                                    cam_distance=evaluate_param["cam_distance"],
                                                    # max_time_step=evaluate_param["num_timesteps"],
                                                    use_activation_visualization=evaluate_param["visualize_activation"],
                                                    cam_type=evaluate_param["cam_type"],
                                                    realtime_plotting_info=evaluate_param.get("realtime_plotting_info", []),
                                                    video_fps=config.env_params.control_framerate
                                                    )
                    except Exception as exc:
                        train_analyzer_report["exceptions"].append({
                            "function_name": "replay",
                            "exception": str(exc),
                            "traceback": traceback.format_exc(),
                        })

                write_report(train_analyzer_report, analyze_result_dir)
        finally:
            gait_evaluator.close()
        
    def analyze(self, gait_data, segmented_ref_data, analyze_result_dir, show_plot:bool):
        return run_standard_gait_analysis(
            gait_data=gait_data,
            segmented_ref_data=segmented_ref_data,
            analyze_result_dir=analyze_result_dir,
            show_plot=show_plot,
            include_reference_plots=segmented_ref_data is not None,
        )
