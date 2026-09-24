import traceback

import numpy as np

from rl_train.analyzer.gait_analyze import GaitAnalyzer


def run_standard_gait_analysis(
    *,
    gait_data,
    segmented_ref_data,
    analyze_result_dir: str,
    show_plot: bool,
    include_reference_plots: bool = True,
):
    exception_report_list = []
    gait_analyzer = GaitAnalyzer(gait_data, segmented_ref_data, show_plot)

    def run_plot(function_name, callback):
        try:
            callback()
        except Exception as exc:
            exception_report_list.append({
                "function_name": function_name,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            })

    right_gait_segments = []
    try:
        right_gait_segments = gait_analyzer.get_gait_segment_index(is_right_foot_based=True)
    except Exception as exc:
        exception_report_list.append({
            "function_name": "get_gait_segment_index_right",
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        })

    if len(right_gait_segments) < 1:
        print("=" * 10 + "Warning" + "=" * 10)
        print("Warning! Not enough gait data to plot gait-cycle analysis. Skipping segmented plots.")
        print("=" * 10 + "Warning" + "=" * 10)
    else:
        run_plot(
            "plot_entire_result_right",
            lambda: gait_analyzer.plot_entire_result(
                result_dir=analyze_result_dir,
                is_right_foot_based=True,
            ),
        )
        run_plot(
            "plot_entire_result_left",
            lambda: gait_analyzer.plot_entire_result(
                result_dir=analyze_result_dir,
                is_right_foot_based=False,
            ),
        )

        exo_policy_output_data = gait_data.series_data.get("exo_policy_output_data", {})
        if "Exo_L" in exo_policy_output_data and "Exo_R" in exo_policy_output_data:
            run_plot(
                "plot_exo_segmented_data",
                lambda: gait_analyzer.plot_exo_segmented_data(result_dir=analyze_result_dir),
            )
        else:
            print("Skipping exoskeleton plots: Exo_L/Exo_R policy output data not present.")

        run_plot(
            "plot_segmented_kinematics_result",
            lambda: gait_analyzer.plot_segmented_kinematics_result(result_dir=analyze_result_dir),
        )
        run_plot(
            "plot_left_right_comparison",
            lambda: gait_analyzer.plot_left_right_comparison(result_dir=analyze_result_dir),
        )
        if include_reference_plots and segmented_ref_data is not None:
            run_plot(
                "plot_right_ref_comparison",
                lambda: gait_analyzer.plot_right_ref_comparison(result_dir=analyze_result_dir),
            )
        run_plot(
            "plot_contact_data",
            lambda: gait_analyzer.plot_contact_data(result_dir=analyze_result_dir),
        )
        run_plot(
            "plot_segmented_muscle_data_right",
            lambda: gait_analyzer.plot_segmented_muscle_data(
                result_dir=analyze_result_dir,
                is_plot_right=True,
            ),
        )
        run_plot(
            "plot_segmented_muscle_data_left",
            lambda: gait_analyzer.plot_segmented_muscle_data(
                result_dir=analyze_result_dir,
                is_plot_right=False,
            ),
        )
        if include_reference_plots and segmented_ref_data is not None:
            run_plot(
                "joint_angle_by_velocity",
                lambda: gait_analyzer.joint_angle_by_velocity(result_dir=analyze_result_dir),
            )

    torque_data = gait_data.series_data.get("joint_torque_data", {})
    if torque_data:
        run_plot(
            "plot_hip_human_exo_torque",
            lambda: gait_analyzer.plot_hip_human_exo_torque(result_dir=analyze_result_dir),
        )
    else:
        print("Skipping hip torque plots: joint_torque_data not present.")

    run_plot(
        "plot_exo_mechanical_power",
        lambda: gait_analyzer.plot_exo_mechanical_power(result_dir=analyze_result_dir),
    )

    return exception_report_list
