from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from datetime import datetime
from rl_train.utils import train_log_handler
from rl_train.utils.train_checkpoint_data import TrainCheckpointData
from rl_train.utils.evaluation_runner import EvaluationRunResult, run_serial_evaluation

def _analyze_process(log_dir, *, timeout_seconds: float | None = None) -> EvaluationRunResult:
    """
    Run analyzer/evaluation synchronously in the training process.
    """
    return run_serial_evaluation(log_dir, timeout_seconds=timeout_seconds)

def _print_analyzer_warning(result):
    print("Warning: analyzer/evaluation failed; PPO training will continue.")
    if isinstance(result, EvaluationRunResult):
        print(f"Analyzer error type: {result.error_type or 'unknown'}")
        print(f"Analyzer error message: {result.error_message or ''}")
        if result.timed_out:
            print(
                "SIGALRM can interrupt many Python-level stalls, but it is not "
                "a hard timeout for all native C, CUDA, MuJoCo, or EGL calls."
            )
        if result.traceback_text:
            print("Analyzer traceback:")
            print(result.traceback_text, end="")
    else:
        print(f"Analyzer returned non-dict result: {result!r}")

class BaseCustomLearningCallback(BaseCallback):
    def __init__(self, *,
                 log_rollout_freq: int,
                 evaluate_freq: int,
                 log_handler:train_log_handler.TrainLogHandler,
                 evaluate_timeout_seconds: float | None = None,
                 verbose=1):
        super().__init__(verbose)
        self.log_rollout_freq = log_rollout_freq
        self.evaluate_freq = evaluate_freq
        self.evaluate_timeout_seconds = evaluate_timeout_seconds
        self.train_log_handler:train_log_handler.TrainLogHandler = log_handler
        self.log_count = 0
        
        # Move the analyze_process function to class level
        # self.analyze_process = functools.partial(_analyze_process)
        self.pool = None

    def _init_callback(self):
        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.current_episode_rewards = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)
        self.current_episode_length_counts = np.zeros(self.training_env.num_envs)

        self.current_reward_dict_sum = [{} for _ in range(self.training_env.num_envs)]
        self.episode_reward_dict_sum = [{} for _ in range(self.training_env.num_envs)]

        self.prev_logging_timestep = 0
        self._reset_teacher_task_counters()
        self._reset_mirror_reference_counters()

    def _reset_teacher_task_counters(self):
        self.teacher_task_counters = {
            "episode_count_by_abs_acceleration": {},
            "ramp_steps_by_abs_acceleration": {},
            "task_tuple_count": {},
            "phase_steps": {
                "INITIAL_HOLD": 0,
                "RAMP": 0,
                "FINAL_HOLD": 0,
            },
            "pre_hold_duration_sum": 0.0,
            "ramp_duration_sum": 0.0,
            "post_hold_duration_sum": 0.0,
            "episode_duration_sum": 0.0,
            "duration_sample_count": 0,
            "A_norm": None,
            "exo_torque_limit_nm": None,
        }

    def _reset_mirror_reference_counters(self):
        self.mirror_reference_counters = {
            "mirror_episode_count_original": 0,
            "mirror_episode_count_mirrored": 0,
        }

    @staticmethod
    def _counter_key(value):
        try:
            return f"{float(value):.6g}"
        except Exception:
            return str(value)

    def _accumulate_teacher_task_info(self, info: dict, done: bool) -> None:
        teacher_task = info.get("teacher_task") if isinstance(info, dict) else None
        if not isinstance(teacher_task, dict):
            return

        phase = teacher_task.get("phase")
        if phase in self.teacher_task_counters["phase_steps"]:
            self.teacher_task_counters["phase_steps"][phase] += 1

        abs_acceleration = teacher_task.get("abs_acceleration")
        abs_acceleration_key = self._counter_key(abs_acceleration)
        if phase == "RAMP":
            ramp_steps = self.teacher_task_counters["ramp_steps_by_abs_acceleration"]
            ramp_steps[abs_acceleration_key] = ramp_steps.get(abs_acceleration_key, 0) + 1

        if teacher_task.get("A_norm") is not None:
            self.teacher_task_counters["A_norm"] = float(teacher_task["A_norm"])
        if teacher_task.get("exo_torque_limit_nm") is not None:
            self.teacher_task_counters["exo_torque_limit_nm"] = float(teacher_task["exo_torque_limit_nm"])

        if done:
            episode_counts = self.teacher_task_counters["episode_count_by_abs_acceleration"]
            episode_counts[abs_acceleration_key] = episode_counts.get(abs_acceleration_key, 0) + 1
            tuple_key = teacher_task.get("task_tuple_key")
            if tuple_key is None:
                delta_velocity = teacher_task.get("delta_velocity")
                tuple_key = f"{abs_acceleration_key}:{self._counter_key(delta_velocity)}"
            tuple_counts = self.teacher_task_counters["task_tuple_count"]
            tuple_counts[str(tuple_key)] = tuple_counts.get(str(tuple_key), 0) + 1

            for source_key, sum_key in (
                ("pre_hold_duration", "pre_hold_duration_sum"),
                ("ramp_duration", "ramp_duration_sum"),
                ("post_hold_duration", "post_hold_duration_sum"),
                ("episode_duration", "episode_duration_sum"),
            ):
                value = teacher_task.get(source_key)
                if value is not None:
                    self.teacher_task_counters[sum_key] += float(value)
            self.teacher_task_counters["duration_sample_count"] += 1

    def _accumulate_mirror_reference_info(self, info: dict, done: bool) -> None:
        if not done or not isinstance(info, dict) or "mirror_reference" not in info:
            return
        if bool(info["mirror_reference"]):
            self.mirror_reference_counters["mirror_episode_count_mirrored"] += 1
        else:
            self.mirror_reference_counters["mirror_episode_count_original"] += 1

        
    #called after all envs step done
    def _on_step(self) -> bool:
        self.current_episode_rewards += self.locals["rewards"]
        for idx, done in enumerate(self.locals["dones"]):
            self.current_episode_length_counts[idx] += 1

            # ------------------------------------------------------------------
            # Safeguard: 'info' may be None or may not contain 'rwd_dict'.
            # In such cases we skip per-key reward accumulation instead of
            # triggering a `TypeError: 'NoneType' object is not subscriptable`.
            # ------------------------------------------------------------------
            info_dict = None
            try:
                info_dict = self.locals["infos"][idx]
            except (IndexError, KeyError):
                # Defensive: infos array shape mismatch → just ignore for now
                info_dict = None

            if info_dict and isinstance(info_dict, dict) and "rwd_dict" in info_dict:
                for key, val in info_dict["rwd_dict"].items():
                    if key not in self.current_reward_dict_sum[idx]:
                        self.current_reward_dict_sum[idx][key] = 0
                    self.current_reward_dict_sum[idx][key] += val
            if info_dict and isinstance(info_dict, dict):
                self._accumulate_teacher_task_info(info_dict, bool(done))
                self._accumulate_mirror_reference_info(info_dict, bool(done))
            if done:
                self.rewards_sum[idx] += self.current_episode_rewards[idx]
                self.episode_counts[idx] += 1
                self.current_episode_rewards[idx] = 0.0
                self.episode_length_counts[idx] += self.current_episode_length_counts[idx]
                self.current_episode_length_counts[idx] = 0

                # Aggregate episode-level reward dictionary safely
                for key, val in self.current_reward_dict_sum[idx].items():
                    if key not in self.episode_reward_dict_sum[idx]:
                        self.episode_reward_dict_sum[idx][key] = 0
                    self.episode_reward_dict_sum[idx][key] += val

             
        return True
    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()

    def _on_rollout_end(self, write_log:bool=True) -> TrainCheckpointData|None:
        super()._on_rollout_end()

        self.prev_logging_timestep = self.num_timesteps
        
        log_data = None
        if self.log_count % self.log_rollout_freq == 0:

            model_path = self.train_log_handler.get_path2save_model(self.model.num_timesteps)
            self.model.save(model_path)
            
            def get_logger_value(key, default=-1):
                value = self.logger.name_to_value.get(key, default)
                return value.item() if hasattr(value, 'item') else value
            
            # average_reward_dict_per_episode=[{key:self.episode_reward_dict_sum[idx][key] / self.episode_counts[idx] for key in self.episode_reward_dict_sum[idx].keys()} for idx in range(self.training_env.num_envs)]
            # Calculate the average reward for each key across all environments
            # average_reward_dict_per_episode={key:sum(average_reward_dict_per_episode[idx][key] for idx in range(self.training_env.num_envs)) / self.training_env.num_envs for key in average_reward_dict_per_episode[0].keys()}
            # If there are no keys in self.episode_reward_dict_sum[idx], set all values to 0
            for idx in range(self.training_env.num_envs):
                if not self.episode_reward_dict_sum[idx]:
                    self.episode_reward_dict_sum[idx] = {key: 0 for key in self.episode_reward_dict_sum[0].keys()}

            average_reward_dict_per_episode = {
                key: (
                    sum(
                        self.episode_reward_dict_sum[idx][key]
                        for idx in range(self.training_env.num_envs)
                    ) / sum(self.episode_counts)
                )
                for key in self.episode_reward_dict_sum[0].keys()
            }



            log_data = TrainCheckpointData(
                approx_kl=get_logger_value('train/approx_kl'),
                clip_fraction=get_logger_value('train/clip_fraction'),
                clip_range=get_logger_value('train/clip_range'),
                clip_range_vf=get_logger_value('train/clip_range_vf'),
                entropy_loss=get_logger_value('train/entropy_loss'),
                explained_variance=get_logger_value('train/explained_variance'),
                learning_rate=get_logger_value('train/learning_rate'),
                loss=get_logger_value('train/loss'),
                n_updates=get_logger_value('train/n_updates'),
                policy_gradient_loss=get_logger_value('train/policy_gradient_loss'),
                std=get_logger_value('train/std'),
                value_loss=get_logger_value('train/value_loss'),
                num_timesteps=self.model.num_timesteps,
                average_num_timestep=(np.sum(self.episode_length_counts) / np.sum(self.episode_counts) if np.sum(self.episode_counts) != 0 else np.array(0)).item(),
                average_reward_per_episode=(np.sum(self.rewards_sum) / np.sum(self.episode_counts) if np.sum(self.episode_counts) != 0 else np.array(0)).item(),
                average_reward_dict_per_episode=average_reward_dict_per_episode,
                teacher_task_counters=self._summarize_teacher_task_counters(),
                mirror_reference_counters=self.mirror_reference_counters.copy(),
                time=f"{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}",
            )
            if write_log:
                self.train_log_handler.add_log_data(log_data)
                self.train_log_handler.write_json_file()
            
            self.rewards_sum = np.zeros(self.training_env.num_envs)
            self.episode_counts = np.zeros(self.training_env.num_envs)
            self.episode_length_counts = np.zeros(self.training_env.num_envs)
            self.episode_reward_dict_sum = [{} for _ in range(self.training_env.num_envs)]
            self.current_reward_dict_sum = [{} for _ in range(self.training_env.num_envs)]
            self._reset_teacher_task_counters()
            self._reset_mirror_reference_counters()


        if self.log_count % self.evaluate_freq == 0 and self.log_count != 0:
            result = _analyze_process(
                self.train_log_handler.log_dir,
                timeout_seconds=self.evaluate_timeout_seconds,
            )
            if not result.ok:
                _print_analyzer_warning(result)
        self.log_count += 1
        
        return log_data

    def _summarize_teacher_task_counters(self):
        counters = {
            key: value.copy() if isinstance(value, dict) else value
            for key, value in self.teacher_task_counters.items()
        }
        count = counters.get("duration_sample_count", 0)
        if count:
            counters["mean_pre_hold_duration"] = counters["pre_hold_duration_sum"] / count
            counters["mean_ramp_duration"] = counters["ramp_duration_sum"] / count
            counters["mean_post_hold_duration"] = counters["post_hold_duration_sum"] / count
            counters["mean_episode_duration"] = counters["episode_duration_sum"] / count
        else:
            counters["mean_pre_hold_duration"] = 0.0
            counters["mean_ramp_duration"] = 0.0
            counters["mean_post_hold_duration"] = 0.0
            counters["mean_episode_duration"] = 0.0
        return counters
