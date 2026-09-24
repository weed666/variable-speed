import json
import dm_control.mujoco
import dm_control.mujoco.wrapper
import dm_control.mujoco.wrapper.core
import mujoco
import numpy as np
from rl_train.utils import numpy_utils
class GaitData:
    def __init__(self, 
                #  *,
                #  mj_model,
                #  mj_data,
                 ):
        # self.data = {
        #     "series_data":{}
        # }
        self.series_data = {
            "joint_data":{}, # joint_name: {"qpos":[], "qvel":[]}
            "actuator_data":{}, # actuator_name: {"force":[], "velocity":[], "ctrl":[]}
            "sensor_data":{}, # sensor_name: {"data":[]}
            "physics_data":{
                "contacts":{
                    "data":[] # [{"pos":[], "force":[], "torque":[], "geom1":[], "geom2":[]}]
                }
            },
            "target_data":{
                "sim_time": [],
                "target_velocity": [],
                "phase": [],
                "target_acceleration": [],
                "estimated_acceleration": [],
                "step_average_acceleration": [],
                "activation_cost_raw": [],
                "episode_activation_integral": [],
            },
            "joint_torque_data":{},
            "exo_policy_output_data":{},
        }
        self.metadata = {
            "data_length": 0,
            # "sample_rate": 100,
        }
    def add_data(self,*,
                 mj_model:dm_control.mujoco.wrapper.core.MjModel,
                 mj_data:dm_control.mujoco.wrapper.core.MjData,
                 target_velocity:float,
                 phase:str|None=None,
                 target_acceleration:float|None=None,
                 estimated_acceleration:float|None=None,
                 step_average_acceleration:float|None=None,
                 activation_cost_raw:float|None=None,
                 episode_activation_integral:float|None=None,
                 printing=False
                 ):
        # TODO: there is no lumbar extension ctrl!!
        muscle_act_ind = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.metadata["data_length"] += 1
        self._add_hip_actuator_torque_data(
            mj_model=mj_model,
            mj_data=mj_data,
            muscle_act_ind=muscle_act_ind,
        )

        # for idx in range(mj_model.na):
        for idx in range(mj_model.nu):
            actuator_name = mj_model.actuator(idx).name
            actuator_model = mj_model.actuator(actuator_name)
            actuator_data = mj_data.actuator(actuator_name)
            
            actuator_dict = self.series_data["actuator_data"].setdefault(
                f"{actuator_name}",
                {"force": [], "velocity": [], "ctrl": []}
            )

            actuator_dict["force"].append(numpy_utils.numpy2array(actuator_data.force.copy()))
            actuator_dict["velocity"].append(numpy_utils.numpy2array(actuator_data.velocity.copy()))
            actuator_dict["ctrl"].append(numpy_utils.numpy2array(actuator_data.ctrl.copy()))
        for idx in range(mj_model.njnt):
            joint_name = mj_model.joint(idx).name
            joint_model = mj_model.joint(joint_name)
            joint_data = mj_data.joint(joint_name)

            joint_dict = self.series_data["joint_data"].setdefault(
                f"{joint_name}",
                {"qpos": [], "qvel": []}
            )

            joint_dict["qpos"].append(numpy_utils.numpy2array(joint_data.qpos.copy()))
            joint_dict["qvel"].append(numpy_utils.numpy2array(joint_data.qvel.copy()))
        for idx in range(mj_model.nsensor):
            sensor_name = mj_model.sensor(idx).name
            sensor_model = mj_model.sensor(sensor_name)
            sensor_data = mj_data.sensor(sensor_name)
            sensor_dict = self.series_data["sensor_data"].setdefault(
                f"{sensor_name}",
                {"data": []}
            )
            sensor_dict["data"].append(numpy_utils.numpy2array(sensor_data.data.copy()))
        contacts = []
        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(mj_model.ptr, mj_data.ptr, i, force)
            contacts.append({
                'pos': contact.pos.copy().tolist(),
                'force': force[:3].tolist(),
                'torque': force[3:].tolist(),
                'geom1': mj_model.id2name(contact.geom1, 'geom'),
                'geom2': mj_model.id2name(contact.geom2, 'geom')
            })
        
        # contact_dict = self.series_data["physics_data"].setdefault(
        #     "contacts",
        #     {"data": []}
        # )
        contact_dict = self.series_data["physics_data"]["contacts"]
        contact_dict["data"].append(contacts)

        target_data = self.series_data.setdefault("target_data", {})
        target_data.setdefault("sim_time", []).append([float(mj_data.time)])
        self.series_data["target_data"]["target_velocity"].append([target_velocity])
        if phase is not None:
            target_data.setdefault("phase", []).append([phase])
        if target_acceleration is not None:
            target_data.setdefault("target_acceleration", []).append([target_acceleration])
        if estimated_acceleration is not None:
            target_data.setdefault("estimated_acceleration", []).append([estimated_acceleration])
        if step_average_acceleration is not None:
            target_data.setdefault("step_average_acceleration", []).append([step_average_acceleration])
        if activation_cost_raw is not None:
            target_data.setdefault("activation_cost_raw", []).append([activation_cost_raw])
        if episode_activation_integral is not None:
            target_data.setdefault("episode_activation_integral", []).append([episode_activation_integral])

    def add_exo_policy_output_data(self, exo_teacher_info: dict | None):
        """Record eval-time raw and executed exo policy outputs without changing control."""
        if not isinstance(exo_teacher_info, dict):
            return
        raw_action = exo_teacher_info.get("latest_raw_normalized_exo_action")
        filtered_action = exo_teacher_info.get("latest_normalized_exo_action")
        torque_limit = exo_teacher_info.get("exo_torque_limit_nm")
        if raw_action is None or filtered_action is None or torque_limit is None:
            return

        raw_action = np.asarray(raw_action, dtype=float).reshape(-1)
        filtered_action = np.asarray(filtered_action, dtype=float).reshape(-1)
        if raw_action.size < 2 or filtered_action.size < 2:
            return

        torque_limit = float(torque_limit)
        if not np.isfinite(torque_limit):
            return

        lpf_enabled = bool(exo_teacher_info.get("exo_output_lpf_enabled", False))
        exo_output_data = self.series_data.setdefault("exo_policy_output_data", {})
        exo_output_data.setdefault(
            "metadata",
            {
                "source": "info.exo_teacher",
                "raw_key": "latest_raw_normalized_exo_action",
                "filtered_key": "latest_normalized_exo_action",
                "force_scale_key": "exo_torque_limit_nm",
                "exo_output_lpf_enabled": lpf_enabled,
            },
        )
        exo_output_data["metadata"]["exo_output_lpf_enabled"] = lpf_enabled

        for side_name, idx in (("Exo_R", 0), ("Exo_L", 1)):
            side_data = exo_output_data.setdefault(
                side_name,
                {"raw_force": [], "filtered_force": [], "selected_force": []},
            )
            raw_force = float(raw_action[idx] * torque_limit)
            filtered_force = float(filtered_action[idx] * torque_limit)
            side_data.setdefault("raw_force", []).append([raw_force])
            side_data.setdefault("filtered_force", []).append([filtered_force])
            side_data.setdefault("selected_force", []).append(
                [filtered_force if lpf_enabled else raw_force]
            )

    def _add_hip_actuator_torque_data(self, *, mj_model, mj_data, muscle_act_ind):
        torque_data = self.series_data.setdefault("joint_torque_data", {})
        try:
            actuator_moment_source = mj_data.actuator_moment
            actuator_force_source = mj_data.actuator_force
            actuator_moment = np.asarray(
                actuator_moment_source.copy()
                if hasattr(actuator_moment_source, "copy")
                else actuator_moment_source
            )
            actuator_force = np.asarray(
                actuator_force_source.copy()
                if hasattr(actuator_force_source, "copy")
                else actuator_force_source
            ).reshape(-1)
        except Exception as exc:
            self.metadata.setdefault("joint_torque_data", {})["error"] = str(exc)
            return

        if (
            actuator_moment.ndim == 1
            and actuator_moment.shape[0] != mj_model.nu * mj_model.nv
        ):
            try:
                rowadr = np.asarray(mj_data.moment_rowadr).reshape(-1)
                rownnz = np.asarray(mj_data.moment_rownnz).reshape(-1)
                colind = np.asarray(mj_data.moment_colind).reshape(-1)
                dense_moment = np.zeros((mj_model.nu, mj_model.nv), dtype=float)
                for actuator_id in range(mj_model.nu):
                    start = int(rowadr[actuator_id])
                    stop = start + int(rownnz[actuator_id])
                    dense_moment[actuator_id, colind[start:stop].astype(int)] = actuator_moment[start:stop]
                actuator_moment = dense_moment
            except Exception as exc:
                self.metadata.setdefault("joint_torque_data", {})["error"] = (
                    "unexpected actuator_moment shape "
                    f"{actuator_moment.shape}; expected dense ({mj_model.nu}, {mj_model.nv}) "
                    f"or MuJoCo sparse moment arrays; sparse decode failed: {exc}"
                )
                return
        elif actuator_moment.ndim == 1:
            actuator_moment = actuator_moment.reshape(mj_model.nu, mj_model.nv)

        if actuator_moment.shape != (mj_model.nu, mj_model.nv):
            self.metadata.setdefault("joint_torque_data", {})["error"] = (
                "unexpected actuator_moment shape after decode "
                f"{actuator_moment.shape}; expected ({mj_model.nu}, {mj_model.nv})"
            )
            return
        if actuator_force.shape[0] != mj_model.nu:
            self.metadata.setdefault("joint_torque_data", {})["error"] = (
                "unexpected actuator_force shape "
                f"{actuator_force.shape}; expected first dimension {mj_model.nu}"
            )
            return

        torque_metadata = self.metadata.setdefault("joint_torque_data", {})
        torque_metadata.setdefault("actuator_moment_shape", list(actuator_moment.shape))
        torque_metadata.setdefault("actuator_force_shape", list(actuator_force.shape))
        torque_metadata.setdefault("nu", int(mj_model.nu))
        torque_metadata.setdefault("nv", int(mj_model.nv))

        muscle_actuator_ids = np.where(np.asarray(muscle_act_ind, dtype=bool))[0]
        exo_by_joint = {
            "hip_flexion_r": "Exo_R",
            "hip_flexion_l": "Exo_L",
        }

        for joint_name, exo_actuator_name in exo_by_joint.items():
            try:
                joint_id = int(mj_model.joint(joint_name).id)
                dof_idx = int(mj_model.jnt_dofadr[joint_id])
            except Exception:
                continue

            human_tau = float(
                np.sum(actuator_moment[muscle_actuator_ids, dof_idx] * actuator_force[muscle_actuator_ids])
            )
            exo_tau = 0.0
            exo_actuator_id = None
            try:
                exo_actuator_id = int(mj_model.actuator(exo_actuator_name).id)
                exo_tau = float(actuator_moment[exo_actuator_id, dof_idx] * actuator_force[exo_actuator_id])
            except Exception:
                pass
            total_tau = human_tau + exo_tau

            joint_dict = torque_data.setdefault(
                joint_name,
                {"human": [], "exo": [], "total": [], "qfrc_actuator": []},
            )
            joint_dict.setdefault("human", []).append([human_tau])
            joint_dict.setdefault("exo", []).append([exo_tau])
            joint_dict.setdefault("total", []).append([total_tau])
            try:
                qfrc_source = mj_data.qfrc_actuator
                qfrc_values = np.asarray(
                    qfrc_source.copy() if hasattr(qfrc_source, "copy") else qfrc_source
                ).reshape(-1)
                qfrc_actuator = float(qfrc_values[dof_idx])
                joint_dict.setdefault("qfrc_actuator", []).append([qfrc_actuator])
            except Exception:
                pass

            if joint_name not in torque_metadata:
                active_ids = np.where(np.abs(actuator_moment[:, dof_idx]) > 1e-12)[0]
                torque_metadata[joint_name] = {
                    "dof_idx": dof_idx,
                    "muscle_actuator_count": int(len(muscle_actuator_ids)),
                    "exo_actuator_name": exo_actuator_name if exo_actuator_id is not None else None,
                    "exo_actuator_id": exo_actuator_id,
                    "actuators_with_nonzero_moment": [
                        mj_model.actuator(int(actuator_id)).name for actuator_id in active_ids
                    ],
                }

    def add_reward_data(self, reward_dict: dict, reward_keys: list[str]):
        """Record selected eval-time reward terms without recalculating or modifying rewards."""
        if not reward_keys:
            return
        reward_data = self.series_data.setdefault("reward_data", {})
        reward_metadata = self.metadata.setdefault("reward_data", {"missing_keys": {}})
        missing_keys = reward_metadata.setdefault("missing_keys", {})
        step_index = self.metadata["data_length"] - 1

        for reward_key in reward_keys:
            if reward_key not in reward_dict:
                missing_keys.setdefault(reward_key, []).append(step_index)
                continue
            reward_data.setdefault(reward_key, []).append([float(reward_dict[reward_key])])
    def apply_to_env(self,*,
                     time_index:int,
                     mj_model:dm_control.mujoco.wrapper.core.MjModel,
                     mj_data:dm_control.mujoco.wrapper.core.MjData,):
        for idx in range(mj_model.nu):
            actuator_name = mj_model.actuator(idx).name
            actuator_model = mj_model.actuator(actuator_name)
            actuator_data = mj_data.actuator(actuator_name)
            
            actuator_dict = self.series_data["actuator_data"].setdefault(
                f"{actuator_name}",
                {"force": [], "velocity": [], "ctrl": []}
            )
            actuator_data.force = self.series_data["actuator_data"][actuator_name]["force"][time_index]
            actuator_data.velocity = self.series_data["actuator_data"][actuator_name]["velocity"][time_index]
            actuator_data.ctrl = self.series_data["actuator_data"][actuator_name]["ctrl"][time_index]
        for idx in range(mj_model.njnt):
            joint_name = mj_model.joint(idx).name
            joint_model = mj_model.joint(joint_name)
            joint_data = mj_data.joint(joint_name)

            joint_dict = self.series_data["joint_data"].setdefault(
                f"{joint_name}",
                {"qpos": [], "qvel": []}
            )
            joint_data.qpos = self.series_data["joint_data"][joint_name]["qpos"][time_index]
            joint_data.qvel = self.series_data["joint_data"][joint_name]["qvel"][time_index]
    def save_json_data(self, path):
        # for key in self.series_data.keys():
        #     self.series_data[key] = np.array(self.series_data[key])
        data = {
                "series_data":self.series_data,
                "metadata":self.metadata
            }
        # np.savez(path, **data)
        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    def read_json_data(self, path):
        with open(path,"r") as json_file:
            data_loaded = json.load(json_file)
        self.series_data = data_loaded["series_data"]
        self.metadata = data_loaded["metadata"]
        # data_npz = np.load(path, allow_pickle=True)


        # data_dict = {key: data_npz[key].item() if key == "series_data" else data_npz[key] for key in data_npz.files}

        # # data_dict = dict(data_npz)
        # # print(f"{data_dict=}")
        # # print(f"{data_dict['series_data'].files}")
        # self.series_data = dict(data_dict["series_data"])
        # # self.hip_flexion_r = ref_data_dict["hip_flexion_r"]
    def get_contact_data(self, geom_name1:str, geom_name2:str):
        data = []
        for contact_data_list in self.series_data["physics_data"]["contacts"]["data"]:
            for contact_data in contact_data_list:
                if contact_data["geom1"] == geom_name1 and contact_data["geom2"] == geom_name2:
                    data.append(contact_data["force"])
                    break
                elif contact_data["geom2"] == geom_name1 and contact_data["geom1"] == geom_name2:
                    data.append([-f for f in contact_data["force"]])
                    break
            else:
                data.append([0,0,0])
        return data
    def print_brief_data(self):
        # print("=====================Start of GaitData==================")
        # print(f"{len(self.hip_flexion_r)=}, {self.hip_flexion_r[0]=}, {self.hip_flexion_r[-1]=}")
        # print("=====================Start of Series Data==================")
        for j_key in self.series_data["joint_data"].keys():
            for property_key in self.series_data["joint_data"][j_key]:
                current_data = self.series_data["joint_data"][j_key][property_key]
                # print(f"{j_key=},{property_key=},{len(current_data)=},{np.min(current_data)=},{np.max(current_data)=}")
        for a_key in self.series_data["actuator_data"].keys():
            current_data = self.series_data["actuator_data"][a_key]
            for property_key in self.series_data["actuator_data"][a_key]:
                current_data = self.series_data["actuator_data"][a_key][property_key]
                # print(f"{a_key=},{property_key=},{len(current_data)=},{np.min(current_data)=},{np.max(current_data)=}")
        # print("=====================End of GaitData==================")
    def print_data_structure(self):
        def print_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    print('    ' * indent + str(key) + ": ")
                    print_dict(value, indent+1)
                elif isinstance(value, list):
                    print('    ' * indent + str(key) + ": [length=" + str(len(value)) + "]")
                else:
                    print('    ' * indent + str(key) + ": " + str(value))

        print_dict(self.series_data)
