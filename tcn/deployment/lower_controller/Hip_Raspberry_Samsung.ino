/*
 * Bilateral hip-exoskeleton Teensy torque execution layer
 * Simplified version: IMU is handled directly by PC / Raspberry Pi.
 *
 * Responsibilities kept on Teensy:
 *   1) Receive left/right desired torque from Raspberry Pi over Serial8.
 *   2) Execute torque/current command at 1 kHz.
 *   3) Request motor realtime feedback at 250 Hz.
 *   4) Request motor status at 20 Hz.
 *   5) Return left/right actual torque + motor position to Raspberry Pi at 100 Hz.
 *   6) Apply command timeout, torque/current saturation, slew-rate limit,
 *      velocity safety, motor-feedback safety and drive-fault safety.
 *
 * ============================================================================
 * UART protocol
 * ============================================================================
 *
 * Raspberry Pi -> Teensy torque command, 15 bytes:
 *
 *   A5 5A | 54 |
 *   seq:uint16 |
 *   left_desired_tau:float32 |
 *   right_desired_tau:float32 |
 *   enable:uint8 |
 *   crc8
 *
 * Total:
 *   header 2
 *   cmd    1
 *   payload 11
 *   crc    1
 *   = 15 bytes
 *
 *
 * Raspberry Pi -> Teensy STOP, 4 bytes:
 *
 *   A5 5A 50 crc8
 *
 *
 * Raspberry Pi -> Teensy CLEAR_FAULT, 4 bytes:
 *
 *   A5 5A 43 crc8
 *
 *
 * Teensy -> Raspberry Pi state feedback, 22 bytes:
 *
 *   A5 5A | 44 |
 *   seq:uint16 |
 *   left_actual_tau:float32 |
 *   right_actual_tau:float32 |
 *   left_position_rad:float32 |
 *   right_position_rad:float32 |
 *   crc8
 *
 * Payload:
 *   2 + 4 + 4 + 4 + 4 = 18 bytes
 *
 * Full frame:
 *   2 + 1 + 18 + 1 = 22 bytes
 *
 * Multi-byte values: little-endian.
 *
 * NOTE:
 * position_rad is the direct Motor_Control_Tmotor::pos value.
 * No gear-ratio correction or left/right sign correction is applied.
 *
 * CRC-8:
 * polynomial 0x07
 * init       0x00
 * calculated over CMD + payload
 */

#include <Arduino.h>
#include <FlexCAN_T4.h>
#include <math.h>
#include <string.h>

#include "Motor_Control_Tmotor.h"

#define LINK_SERIAL Serial8


// ============================================================================
// 1. Configuration
// ============================================================================

namespace Config
{

constexpr uint8_t LEFT_MOTOR_ID  = 0x01;
constexpr uint8_t RIGHT_MOTOR_ID = 0x02;

constexpr uint32_t LINK_BAUD = 115200;
constexpr uint32_t CAN_BAUD  = 1000000;


// ---------------------------------------------------------------------------
// Rates
// ---------------------------------------------------------------------------

constexpr uint32_t CONTROL_PERIOD_US =
    1000;                       // 1 kHz

constexpr uint32_t FEEDBACK_REQUEST_PERIOD_US =
    4000;                       // 250 Hz

constexpr uint32_t STATUS_REQUEST_PERIOD_US =
    50000;                      // 20 Hz

constexpr uint32_t TELEMETRY_PERIOD_US =
    10000;                      // 100 Hz


// ---------------------------------------------------------------------------
// Timeouts
// ---------------------------------------------------------------------------

constexpr uint32_t COMMAND_TIMEOUT_US =
    200000;                     // 200 ms

constexpr uint32_t FEEDBACK_TIMEOUT_US =
    100000;                     // 100 ms


// ---------------------------------------------------------------------------
// Torque/current safety
// ---------------------------------------------------------------------------

// Final joint torque execution limit.
constexpr float MAX_TORQUE_NM =
    13.0f;

// Final q-axis current limit.
constexpr float MAX_Q_CURRENT_A =
    30.0f;

// Incoming Pi command limit.
// Finite commands outside this range are clamped,
// NOT rejected.
constexpr float MAX_INPUT_TORQUE_NM =
    13.0f;

// Torque slew rate.
constexpr float MAX_TORQUE_SLEW_NM_S =
    100.0f;

// Small torque cleanup.
constexpr float ZERO_EPS_NM =
    0.001f;


// ---------------------------------------------------------------------------
// Motor mounting directions
// ---------------------------------------------------------------------------

constexpr float LEFT_TORQUE_SIGN =
     1.0f;

constexpr float RIGHT_TORQUE_SIGN =
    -1.0f;


// Motor velocity uses same mechanical sign convention.
constexpr float LEFT_VELOCITY_SIGN =
     1.0f;

constexpr float RIGHT_VELOCITY_SIGN =
    -1.0f;


// ---------------------------------------------------------------------------
// Velocity safety
// ---------------------------------------------------------------------------

constexpr bool ENABLE_VELOCITY_LIMIT =
    true;

constexpr float MAX_ABS_VELOCITY_RAD_S =
    20.0f;

}


// ============================================================================
// 2. UART protocol
// ============================================================================

namespace Protocol
{

constexpr uint8_t HEADER1 =
    0xA5;

constexpr uint8_t HEADER2 =
    0x5A;

constexpr uint8_t CMD_TORQUE =
    0x54;

constexpr uint8_t CMD_STOP =
    0x50;

constexpr uint8_t CMD_CLEAR_FAULT =
    0x43;

constexpr uint8_t CMD_STATE =
    0x44;


// ---------------------------------------------------------------------------
// Pi -> Teensy CMD_TORQUE
//
// payload:
//   seq:uint16
//   left_tau:float32
//   right_tau:float32
//   enable:uint8
//
// 2 + 4 + 4 + 1 = 11 bytes
// ---------------------------------------------------------------------------

constexpr size_t TORQUE_PAYLOAD_SIZE =
    11;


// ---------------------------------------------------------------------------
// Teensy -> Pi STATE
//
// payload:
//   seq:uint16
//   left_actual_tau:float32
//   right_actual_tau:float32
//   left_position_rad:float32
//   right_position_rad:float32
//
// 2 + 4 + 4 + 4 + 4 = 18 bytes
//
// full frame:
//   header 2
//   cmd    1
//   payload 18
//   crc    1
//
// = 22 bytes
// ---------------------------------------------------------------------------

constexpr size_t STATE_PAYLOAD_SIZE =
    18;

constexpr size_t STATE_FRAME_SIZE =
    22;

static_assert(
    STATE_FRAME_SIZE == 22,
    "State frame must be 22 bytes.");

}


// ============================================================================
// 3. Internal safety / diagnostic flags
// ============================================================================
//
// These remain inside Teensy.
//
// IMPORTANT:
// They are NOT transmitted in the 22-byte STATE frame anymore.
//
// They can still be useful internally for debugging / future extension.
// ============================================================================

namespace ZeroReason
{

constexpr uint16_t NONE =
    0x0000;

constexpr uint16_t COMMAND_DISABLED =
    0x0001;

constexpr uint16_t COMMAND_TIMEOUT =
    0x0002;

constexpr uint16_t FEEDBACK_TIMEOUT =
    0x0004;

constexpr uint16_t MOTOR_FAULT =
    0x0008;

constexpr uint16_t TORQUE_CONSTANT_INVALID =
    0x0010;

constexpr uint16_t VELOCITY_LIMIT =
    0x0020;

constexpr uint16_t MOTOR_COMMAND_FAILED =
    0x0040;

constexpr uint16_t ZERO_EPS =
    0x0080;

constexpr uint16_t STOP_COMMAND =
    0x0100;

constexpr uint16_t CLEAR_FAULT =
    0x0200;

}


namespace RxError
{

constexpr uint16_t NONE =
    0x0000;

constexpr uint16_t MALFORMED_TORQUE =
    0x0001;

constexpr uint16_t CRC_ERROR =
    0x0002;

constexpr uint16_t UNKNOWN_COMMAND =
    0x0004;

constexpr uint16_t PAYLOAD_OVERFLOW =
    0x0008;

}


// ============================================================================
// 4. Hardware
// ============================================================================

Motor_Control_Tmotor left_motor(
    Config::LEFT_MOTOR_ID);

Motor_Control_Tmotor right_motor(
    Config::RIGHT_MOTOR_ID);

CAN_message_t can_message;


// ============================================================================
// 5. Runtime state
// ============================================================================

struct MotorFeedback
{
    float position =
        0.0f;

    float velocity =
        0.0f;

    float actual_torque =
        0.0f;

    uint32_t last_feedback_us =
        0;
};


struct TorqueChannel
{
    float requested =
        0.0f;

    float ramped =
        0.0f;

    float applied =
        0.0f;
};


struct Runtime
{
    MotorFeedback left_feedback;
    MotorFeedback right_feedback;

    TorqueChannel left_torque;
    TorqueChannel right_torque;

    bool command_enabled =
        false;

    uint16_t disable_reason =
        ZeroReason::COMMAND_DISABLED;

    uint16_t last_command_sequence =
        0;

    uint16_t telemetry_sequence =
        0;


    // Internal diagnostics.
    uint16_t left_zero_reason_latched =
        ZeroReason::NONE;

    uint16_t right_zero_reason_latched =
        ZeroReason::NONE;

    uint16_t rx_error_flags_latched =
        RxError::NONE;


    uint32_t last_valid_command_us =
        0;

    uint32_t previous_control_us =
        0;

    uint32_t previous_feedback_request_us =
        0;

    uint32_t previous_status_request_us =
        0;

    uint32_t previous_telemetry_us =
        0;
};


Runtime rt;


// ============================================================================
// 6. Utility functions
// ============================================================================

static float clampf(
    float value,
    float min_value,
    float max_value)
{
    return fminf(
        fmaxf(
            value,
            min_value),
        max_value);
}


static float slew_limit(
    float target,
    float previous,
    float max_rate,
    float dt_s)
{
    const float max_delta =
        fmaxf(
            max_rate,
            0.0f) *
        fmaxf(
            dt_s,
            0.0f);

    return clampf(
        target,
        previous - max_delta,
        previous + max_delta);
}


// ---------------------------------------------------------------------------
// CRC8
// ---------------------------------------------------------------------------

static uint8_t crc8_update(
    uint8_t crc,
    uint8_t data)
{
    crc ^= data;

    for (uint8_t i = 0; i < 8; ++i)
    {
        if (crc & 0x80)
        {
            crc =
                static_cast<uint8_t>(
                    (crc << 1) ^ 0x07);
        }
        else
        {
            crc <<= 1;
        }
    }

    return crc;
}


static uint8_t crc8_compute(
    uint8_t command,
    const uint8_t* payload,
    size_t length)
{
    uint8_t crc =
        crc8_update(
            0x00,
            command);

    for (size_t i = 0; i < length; ++i)
    {
        crc =
            crc8_update(
                crc,
                payload[i]);
    }

    return crc;
}


// ---------------------------------------------------------------------------
// Append arbitrary bytes
// ---------------------------------------------------------------------------

static void append_bytes(
    uint8_t* buffer,
    size_t& index,
    const void* value,
    size_t size)
{
    memcpy(
        buffer + index,
        value,
        size);

    index += size;
}


// ============================================================================
// 7. Motor safety
// ============================================================================

static bool feedback_is_fresh(
    uint32_t last_feedback_us,
    uint32_t now_us)
{
    return
        last_feedback_us != 0 &&
        (uint32_t)(
            now_us -
            last_feedback_us)
        <= Config::FEEDBACK_TIMEOUT_US;
}


static void latch_zero_reason_both(
    uint16_t reason)
{
    rt.left_zero_reason_latched |=
        reason;

    rt.right_zero_reason_latched |=
        reason;
}


static uint16_t motor_system_zero_reason(
    uint32_t now_us)
{
    uint16_t reason =
        ZeroReason::NONE;


    // Torque constant invalid
    if (
        !left_motor.torque_constant_is_valid() ||
        !right_motor.torque_constant_is_valid())
    {
        reason |=
            ZeroReason::TORQUE_CONSTANT_INVALID;
    }


    // Feedback timeout
    if (
        !feedback_is_fresh(
            rt.left_feedback.last_feedback_us,
            now_us) ||
        !feedback_is_fresh(
            rt.right_feedback.last_feedback_us,
            now_us))
    {
        reason |=
            ZeroReason::FEEDBACK_TIMEOUT;
    }


    // Motor fault
    if (
        left_motor.fault_code != 0 ||
        right_motor.fault_code != 0)
    {
        reason |=
            ZeroReason::MOTOR_FAULT;
    }


    return reason;
}


static bool motors_ready(
    uint32_t now_us)
{
    return
        motor_system_zero_reason(
            now_us)
        == ZeroReason::NONE;
}


static void command_zero_current()
{
    left_motor.command_q_current_A(
        0.0f);

    right_motor.command_q_current_A(
        0.0f);


    rt.left_torque.ramped =
        0.0f;

    rt.right_torque.ramped =
        0.0f;


    rt.left_torque.applied =
        0.0f;

    rt.right_torque.applied =
        0.0f;
}


static void stop_immediately(
    uint16_t reason)
{
    rt.command_enabled =
        false;

    rt.disable_reason =
        reason;


    rt.left_torque.requested =
        0.0f;

    rt.right_torque.requested =
        0.0f;


    latch_zero_reason_both(
        reason);

    command_zero_current();
}


// ============================================================================
// 8. CAN feedback
// ============================================================================

static void request_motor_feedback()
{
    left_motor.request_realtime();
    right_motor.request_realtime();
}


static void request_motor_status()
{
    left_motor.request_status();
    right_motor.request_status();
}


static bool reply_contains_motion_feedback(
    uint8_t command)
{
    return
        command == 0xA4 ||
        command == 0xA3 ||
        command == 0xC2 ||
        command == 0xC3 ||
        command == 0xC4 ||
        command == 0xF1;
}


static bool reply_contains_torque_feedback(
    uint8_t command)
{
    return
        command == 0xC0 ||
        command == 0xA1 ||
        command == 0xA4 ||
        command == 0xF1;
}


static void drain_can_feedback()
{
    while (
        Motor_Control_Tmotor::Can3.read(
            can_message))
    {
        const uint32_t now_us =
            micros();

        const uint8_t reply_command =
            can_message.len > 0
                ? can_message.buf[0]
                : 0x00;


        // -------------------------------------------------------------------
        // Left motor
        // -------------------------------------------------------------------

        if (
            can_message.id ==
            Config::LEFT_MOTOR_ID)
        {
            left_motor.handle_reply(
                can_message);


            rt.left_feedback.velocity =
                Config::LEFT_VELOCITY_SIGN *
                left_motor.spe;


            if (
                reply_contains_motion_feedback(
                    reply_command))
            {
                rt.left_feedback.position =
                    left_motor.pos;

                rt.left_feedback.last_feedback_us =
                    now_us;
            }


            if (
                reply_contains_torque_feedback(
                    reply_command))
            {
                rt.left_feedback.actual_torque =
                    Config::LEFT_TORQUE_SIGN *
                    left_motor.torque;
            }
        }


        // -------------------------------------------------------------------
        // Right motor
        // -------------------------------------------------------------------

        else if (
            can_message.id ==
            Config::RIGHT_MOTOR_ID)
        {
            right_motor.handle_reply(
                can_message);


            rt.right_feedback.velocity =
                Config::RIGHT_VELOCITY_SIGN *
                right_motor.spe;


            if (
                reply_contains_motion_feedback(
                    reply_command))
            {
                rt.right_feedback.position =
                    right_motor.pos;

                rt.right_feedback.last_feedback_us =
                    now_us;
            }


            if (
                reply_contains_torque_feedback(
                    reply_command))
            {
                rt.right_feedback.actual_torque =
                    Config::RIGHT_TORQUE_SIGN *
                    right_motor.torque;
            }
        }
    }
}


// ============================================================================
// 9. Motor initialization
// ============================================================================

static bool initialize_motors()
{
    left_motor.initial_CAN(
        Config::CAN_BAUD);

    delay(200);


    // Clear possible startup faults.
    left_motor.error_clear();
    right_motor.error_clear();

    delay(50);


    // -----------------------------------------------------------------------
    // Request motor parameters
    // -----------------------------------------------------------------------

    const uint32_t parameter_deadline =
        millis() + 1000;


    while (
        millis() <
        parameter_deadline)
    {
        if (
            !left_motor.torque_constant_is_valid())
        {
            left_motor.request_motor_parameters();
        }


        if (
            !right_motor.torque_constant_is_valid())
        {
            right_motor.request_motor_parameters();
        }


        delay(5);

        drain_can_feedback();


        if (
            left_motor.torque_constant_is_valid() &&
            right_motor.torque_constant_is_valid())
        {
            break;
        }
    }


    if (
        !left_motor.torque_constant_is_valid() ||
        !right_motor.torque_constant_is_valid())
    {
        return false;
    }


    // -----------------------------------------------------------------------
    // Software limits
    // -----------------------------------------------------------------------

    left_motor.set_software_current_limit_A(
        Config::MAX_Q_CURRENT_A);

    right_motor.set_software_current_limit_A(
        Config::MAX_Q_CURRENT_A);


    left_motor.set_software_torque_limit_Nm(
        Config::MAX_TORQUE_NM);

    right_motor.set_software_torque_limit_Nm(
        Config::MAX_TORQUE_NM);


    command_zero_current();


    // -----------------------------------------------------------------------
    // Wait for realtime feedback
    // -----------------------------------------------------------------------

    const uint32_t feedback_deadline =
        millis() + 1000;


    while (
        millis() <
        feedback_deadline)
    {
        request_motor_feedback();

        delay(5);

        drain_can_feedback();


        if (
            rt.left_feedback.last_feedback_us != 0 &&
            rt.right_feedback.last_feedback_us != 0)
        {
            break;
        }
    }


    // Get status once.
    request_motor_status();

    delay(20);

    drain_can_feedback();


    command_zero_current();


    return
        rt.left_feedback.last_feedback_us != 0 &&
        rt.right_feedback.last_feedback_us != 0;
}


// ============================================================================
// 10. Torque execution
// ============================================================================

static float send_joint_torque(
    Motor_Control_Tmotor& motor,
    float joint_torque_nm,
    float torque_sign,
    bool& command_failed)
{
    command_failed =
        false;


    // -----------------------------------------------------------------------
    // Check torque constant
    // -----------------------------------------------------------------------

    if (
        !motor.torque_constant_is_valid() ||
        !isfinite(
            motor.torque_constant) ||
        fabsf(
            motor.torque_constant) <
            1.0e-8f)
    {
        motor.command_q_current_A(
            0.0f);

        command_failed =
            true;

        return 0.0f;
    }


    // -----------------------------------------------------------------------
    // Joint torque saturation
    // -----------------------------------------------------------------------

    const float safe_joint_torque =
        clampf(
            joint_torque_nm,
            -Config::MAX_TORQUE_NM,
             Config::MAX_TORQUE_NM);


    // Convert joint convention -> motor convention.
    const float motor_torque =
        torque_sign *
        safe_joint_torque;


    // -----------------------------------------------------------------------
    // Torque -> q current
    // -----------------------------------------------------------------------

    const float target_current =
        clampf(
            motor_torque /
                motor.torque_constant,

            -Config::MAX_Q_CURRENT_A,
             Config::MAX_Q_CURRENT_A);


    // -----------------------------------------------------------------------
    // Send motor command
    // -----------------------------------------------------------------------

    if (
        !motor.command_q_current_A(
            target_current))
    {
        command_failed =
            true;

        return 0.0f;
    }


    // Return actual commanded joint-side torque.
    return
        torque_sign *
        target_current *
        motor.torque_constant;
}


// ============================================================================
// 11. Velocity safety
// ============================================================================

static float apply_velocity_safety(
    float torque,
    float velocity,
    bool& velocity_limited)
{
    velocity_limited =
        false;


    if (
        !Config::ENABLE_VELOCITY_LIMIT)
    {
        return torque;
    }


    // Positive overspeed + positive torque
    if (
        velocity >=
            Config::MAX_ABS_VELOCITY_RAD_S &&
        torque > 0.0f)
    {
        velocity_limited =
            true;

        return 0.0f;
    }


    // Negative overspeed + negative torque
    if (
        velocity <=
            -Config::MAX_ABS_VELOCITY_RAD_S &&
        torque < 0.0f)
    {
        velocity_limited =
            true;

        return 0.0f;
    }


    return torque;
}


// ============================================================================
// 12. 1-kHz control loop
// ============================================================================

static void update_control(
    uint32_t now_us,
    float dt_s)
{

    // -----------------------------------------------------------------------
    // Command timeout
    // -----------------------------------------------------------------------

    if (
        rt.command_enabled &&
        (uint32_t)(
            now_us -
            rt.last_valid_command_us)
        >
        Config::COMMAND_TIMEOUT_US)
    {
        rt.command_enabled =
            false;

        rt.disable_reason =
            ZeroReason::COMMAND_TIMEOUT;


        rt.left_torque.requested =
            0.0f;

        rt.right_torque.requested =
            0.0f;


        latch_zero_reason_both(
            ZeroReason::COMMAND_TIMEOUT);
    }


    // -----------------------------------------------------------------------
    // Disabled state
    // -----------------------------------------------------------------------

    if (
        !rt.command_enabled &&
        rt.disable_reason !=
            ZeroReason::NONE)
    {
        latch_zero_reason_both(
            rt.disable_reason);
    }


    // -----------------------------------------------------------------------
    // Motor safety
    // -----------------------------------------------------------------------

    const uint16_t motor_reason =
        motor_system_zero_reason(
            now_us);


    if (
        motor_reason !=
        ZeroReason::NONE)
    {
        latch_zero_reason_both(
            motor_reason);

        command_zero_current();

        return;
    }


    // -----------------------------------------------------------------------
    // Target torque
    // -----------------------------------------------------------------------

    const float left_target =
        rt.command_enabled
            ? rt.left_torque.requested
            : 0.0f;


    const float right_target =
        rt.command_enabled
            ? rt.right_torque.requested
            : 0.0f;


    // -----------------------------------------------------------------------
    // Slew rate
    // -----------------------------------------------------------------------

    rt.left_torque.ramped =
        slew_limit(
            left_target,
            rt.left_torque.ramped,
            Config::MAX_TORQUE_SLEW_NM_S,
            dt_s);


    rt.right_torque.ramped =
        slew_limit(
            right_target,
            rt.right_torque.ramped,
            Config::MAX_TORQUE_SLEW_NM_S,
            dt_s);


    // -----------------------------------------------------------------------
    // Final torque clamp
    // -----------------------------------------------------------------------

    float left_safe =
        clampf(
            rt.left_torque.ramped,
            -Config::MAX_TORQUE_NM,
             Config::MAX_TORQUE_NM);


    float right_safe =
        clampf(
            rt.right_torque.ramped,
            -Config::MAX_TORQUE_NM,
             Config::MAX_TORQUE_NM);


    // -----------------------------------------------------------------------
    // Velocity safety
    // -----------------------------------------------------------------------

    bool left_velocity_limited =
        false;

    bool right_velocity_limited =
        false;


    left_safe =
        apply_velocity_safety(
            left_safe,
            rt.left_feedback.velocity,
            left_velocity_limited);


    right_safe =
        apply_velocity_safety(
            right_safe,
            rt.right_feedback.velocity,
            right_velocity_limited);


    if (
        left_velocity_limited)
    {
        rt.left_zero_reason_latched |=
            ZeroReason::VELOCITY_LIMIT;
    }


    if (
        right_velocity_limited)
    {
        rt.right_zero_reason_latched |=
            ZeroReason::VELOCITY_LIMIT;
    }


    // -----------------------------------------------------------------------
    // Near-zero cleanup
    // -----------------------------------------------------------------------

    if (
        left_safe != 0.0f &&
        fabsf(
            left_safe)
        <
        Config::ZERO_EPS_NM)
    {
        left_safe =
            0.0f;

        rt.left_zero_reason_latched |=
            ZeroReason::ZERO_EPS;
    }


    if (
        right_safe != 0.0f &&
        fabsf(
            right_safe)
        <
        Config::ZERO_EPS_NM)
    {
        right_safe =
            0.0f;

        rt.right_zero_reason_latched |=
            ZeroReason::ZERO_EPS;
    }


    // -----------------------------------------------------------------------
    // Execute
    // -----------------------------------------------------------------------

    bool left_command_failed =
        false;

    bool right_command_failed =
        false;


    rt.left_torque.applied =
        send_joint_torque(
            left_motor,
            left_safe,
            Config::LEFT_TORQUE_SIGN,
            left_command_failed);


    rt.right_torque.applied =
        send_joint_torque(
            right_motor,
            right_safe,
            Config::RIGHT_TORQUE_SIGN,
            right_command_failed);


    if (
        left_command_failed)
    {
        rt.left_zero_reason_latched |=
            ZeroReason::MOTOR_COMMAND_FAILED;
    }


    if (
        right_command_failed)
    {
        rt.right_zero_reason_latched |=
            ZeroReason::MOTOR_COMMAND_FAILED;
    }
}


// ============================================================================
// 13. UART RX parser
// ============================================================================

enum class RxState : uint8_t
{
    HEADER1,
    HEADER2,
    COMMAND,
    PAYLOAD,
    CRC
};


struct RxParser
{
    RxState state =
        RxState::HEADER1;


    uint8_t command =
        0;


    uint8_t payload[
        Protocol::TORQUE_PAYLOAD_SIZE] = {};


    size_t expected_payload =
        0;

    size_t index =
        0;


    uint8_t crc =
        0;


    void reset()
    {
        state =
            RxState::HEADER1;

        command =
            0;

        expected_payload =
            0;

        index =
            0;

        crc =
            0;
    }
};


RxParser rx;


// ============================================================================
// 14. Torque command handling
// ============================================================================

static void handle_torque_packet()
{
    uint16_t sequence =
        0;

    float left_tau =
        0.0f;

    float right_tau =
        0.0f;

    uint8_t enable =
        0;


    // seq
    memcpy(
        &sequence,
        rx.payload + 0,
        sizeof(sequence));


    // left torque
    memcpy(
        &left_tau,
        rx.payload + 2,
        sizeof(left_tau));


    // right torque
    memcpy(
        &right_tau,
        rx.payload + 6,
        sizeof(right_tau));


    // enable
    memcpy(
        &enable,
        rx.payload + 10,
        sizeof(enable));


    // -----------------------------------------------------------------------
    // NaN / Inf protection
    // -----------------------------------------------------------------------

    if (
        !isfinite(left_tau) ||
        !isfinite(right_tau))
    {
        rt.rx_error_flags_latched |=
            RxError::MALFORMED_TORQUE;

        return;
    }


    // -----------------------------------------------------------------------
    // Clamp incoming torque
    //
    // Important:
    // finite commands outside the allowed range are CLAMPED,
    // not rejected.
    //
    // This prevents a policy spike from causing command timeout.
    // -----------------------------------------------------------------------

    left_tau =
        clampf(
            left_tau,
            -Config::MAX_INPUT_TORQUE_NM,
             Config::MAX_INPUT_TORQUE_NM);


    right_tau =
        clampf(
            right_tau,
            -Config::MAX_INPUT_TORQUE_NM,
             Config::MAX_INPUT_TORQUE_NM);


    // -----------------------------------------------------------------------
    // Store command
    // -----------------------------------------------------------------------

    rt.last_command_sequence =
        sequence;


    rt.last_valid_command_us =
        micros();


    rt.command_enabled =
        (enable != 0);


    rt.disable_reason =
        rt.command_enabled
            ? ZeroReason::NONE
            : ZeroReason::COMMAND_DISABLED;


    if (
        !rt.command_enabled)
    {
        latch_zero_reason_both(
            ZeroReason::COMMAND_DISABLED);
    }


    rt.left_torque.requested =
        rt.command_enabled
            ? left_tau
            : 0.0f;


    rt.right_torque.requested =
        rt.command_enabled
            ? right_tau
            : 0.0f;
}


// ============================================================================
// 15. STOP / CLEAR FAULT
// ============================================================================

static void handle_clear_fault()
{
    stop_immediately(
        ZeroReason::CLEAR_FAULT);


    left_motor.error_clear();
    right_motor.error_clear();


    delayMicroseconds(
        100);


    request_motor_status();
}


static void handle_complete_packet()
{

    if (
        rx.command ==
        Protocol::CMD_TORQUE)
    {
        handle_torque_packet();
    }


    else if (
        rx.command ==
        Protocol::CMD_STOP)
    {
        stop_immediately(
            ZeroReason::STOP_COMMAND);


        rt.last_valid_command_us =
            micros();
    }


    else if (
        rx.command ==
        Protocol::CMD_CLEAR_FAULT)
    {
        handle_clear_fault();
    }
}


// ============================================================================
// 16. Serial receive
// ============================================================================

static void process_serial_rx()
{
    while (
        LINK_SERIAL.available() > 0)
    {
        const uint8_t byte_in =
            static_cast<uint8_t>(
                LINK_SERIAL.read());


        switch (
            rx.state)
        {

            // ---------------------------------------------------------------
            // HEADER1
            // ---------------------------------------------------------------

            case RxState::HEADER1:
            {
                if (
                    byte_in ==
                    Protocol::HEADER1)
                {
                    rx.state =
                        RxState::HEADER2;
                }

                break;
            }


            // ---------------------------------------------------------------
            // HEADER2
            // ---------------------------------------------------------------

            case RxState::HEADER2:
            {
                if (
                    byte_in ==
                    Protocol::HEADER2)
                {
                    rx.state =
                        RxState::COMMAND;
                }

                else if (
                    byte_in !=
                    Protocol::HEADER1)
                {
                    rx.state =
                        RxState::HEADER1;
                }

                break;
            }


            // ---------------------------------------------------------------
            // COMMAND
            // ---------------------------------------------------------------

            case RxState::COMMAND:
            {
                rx.command =
                    byte_in;


                rx.crc =
                    crc8_update(
                        0x00,
                        rx.command);


                rx.index =
                    0;


                if (
                    rx.command ==
                    Protocol::CMD_TORQUE)
                {
                    rx.expected_payload =
                        Protocol::TORQUE_PAYLOAD_SIZE;

                    rx.state =
                        RxState::PAYLOAD;
                }


                else if (
                    rx.command ==
                        Protocol::CMD_STOP ||
                    rx.command ==
                        Protocol::CMD_CLEAR_FAULT)
                {
                    rx.expected_payload =
                        0;

                    rx.state =
                        RxState::CRC;
                }


                else
                {
                    rt.rx_error_flags_latched |=
                        RxError::UNKNOWN_COMMAND;

                    rx.reset();
                }

                break;
            }


            // ---------------------------------------------------------------
            // PAYLOAD
            // ---------------------------------------------------------------

            case RxState::PAYLOAD:
            {
                if (
                    rx.index >=
                    sizeof(rx.payload))
                {
                    rt.rx_error_flags_latched |=
                        RxError::PAYLOAD_OVERFLOW;

                    rx.reset();

                    break;
                }


                rx.payload[
                    rx.index++] =
                    byte_in;


                rx.crc =
                    crc8_update(
                        rx.crc,
                        byte_in);


                if (
                    rx.index >=
                    rx.expected_payload)
                {
                    rx.state =
                        RxState::CRC;
                }

                break;
            }


            // ---------------------------------------------------------------
            // CRC
            // ---------------------------------------------------------------

            case RxState::CRC:
            {
                if (
                    byte_in ==
                    rx.crc)
                {
                    handle_complete_packet();
                }

                else
                {
                    rt.rx_error_flags_latched |=
                        RxError::CRC_ERROR;
                }


                rx.reset();

                break;
            }
        }
    }
}


// ============================================================================
// 17. 22-byte STATE telemetry
// ============================================================================
//
// Frame:
//
// [0]      A5
// [1]      5A
// [2]      44
//
// payload:
//
// [3:4]    seq uint16
// [5:8]    left actual torque float32
// [9:12]   right actual torque float32
// [13:16]  left motor position float32
// [17:20]  right motor position float32
//
// [21]     CRC8
//
// CRC covers:
//   CMD_STATE + 18-byte payload
//
// ============================================================================

static void send_torque_feedback()
{
    uint8_t payload[
        Protocol::STATE_PAYLOAD_SIZE];


    size_t payload_index =
        0;


    // -----------------------------------------------------------------------
    // Sequence
    // -----------------------------------------------------------------------

    const uint16_t sequence =
        rt.telemetry_sequence++;


    append_bytes(
        payload,
        payload_index,
        &sequence,
        sizeof(sequence));


    // -----------------------------------------------------------------------
    // Actual torque
    // -----------------------------------------------------------------------

    append_bytes(
        payload,
        payload_index,
        &rt.left_feedback.actual_torque,
        sizeof(float));


    append_bytes(
        payload,
        payload_index,
        &rt.right_feedback.actual_torque,
        sizeof(float));


    // -----------------------------------------------------------------------
    // Motor position
    // -----------------------------------------------------------------------

    append_bytes(
        payload,
        payload_index,
        &rt.left_feedback.position,
        sizeof(float));


    append_bytes(
        payload,
        payload_index,
        &rt.right_feedback.position,
        sizeof(float));


    // Verify payload length.
    if (
        payload_index !=
        Protocol::STATE_PAYLOAD_SIZE)
    {
        return;
    }


    // -----------------------------------------------------------------------
    // Construct full frame
    // -----------------------------------------------------------------------

    uint8_t frame[
        Protocol::STATE_FRAME_SIZE];


    size_t frame_index =
        0;


    frame[
        frame_index++] =
        Protocol::HEADER1;


    frame[
        frame_index++] =
        Protocol::HEADER2;


    frame[
        frame_index++] =
        Protocol::CMD_STATE;


    memcpy(
        frame + frame_index,
        payload,
        payload_index);


    frame_index +=
        payload_index;


    // -----------------------------------------------------------------------
    // CRC
    // -----------------------------------------------------------------------

    frame[
        frame_index++] =
        crc8_compute(
            Protocol::CMD_STATE,
            payload,
            payload_index);


    // Safety check.
    if (
        frame_index !=
        Protocol::STATE_FRAME_SIZE)
    {
        return;
    }


    // -----------------------------------------------------------------------
    // Send whole 22-byte frame
    //
    // Do not send partial packets.
    // -----------------------------------------------------------------------

    if (
        LINK_SERIAL.availableForWrite() >=
        static_cast<int>(
            Protocol::STATE_FRAME_SIZE))
    {
        LINK_SERIAL.write(
            frame,
            Protocol::STATE_FRAME_SIZE);


        // Diagnostics are internal only in the 22-byte protocol.
        // Clear after each successfully transmitted STATE packet.
        rt.left_zero_reason_latched =
            ZeroReason::NONE;

        rt.right_zero_reason_latched =
            ZeroReason::NONE;

        rt.rx_error_flags_latched =
            RxError::NONE;
    }
}


// ============================================================================
// 18. Setup
// ============================================================================

void setup()
{
    // Allow motor drivers / system power to settle.
    delay(
        3000);


    // -----------------------------------------------------------------------
    // UART
    // -----------------------------------------------------------------------

    LINK_SERIAL.begin(
        Config::LINK_BAUD);


    // Reset runtime.
    rt =
        Runtime{};


    // -----------------------------------------------------------------------
    // Motor initialization
    // -----------------------------------------------------------------------

    const bool motors_initialized =
        initialize_motors();


    const uint32_t now_us =
        micros();


    // Initialize timing references.
    rt.previous_control_us =
        now_us;

    rt.previous_feedback_request_us =
        now_us;

    rt.previous_status_request_us =
        now_us;

    rt.previous_telemetry_us =
        now_us;

    rt.last_valid_command_us =
        now_us;


    // -----------------------------------------------------------------------
    // Never automatically enable assistance
    // -----------------------------------------------------------------------

    rt.command_enabled =
        false;

    rt.disable_reason =
        ZeroReason::COMMAND_DISABLED;


    // Keep zero torque if initialization failed.
    if (
        !motors_initialized)
    {
        command_zero_current();
    }
}


// ============================================================================
// 19. Main loop
// ============================================================================

void loop()
{
    // -----------------------------------------------------------------------
    // Communication first
    // -----------------------------------------------------------------------

    process_serial_rx();

    drain_can_feedback();


    const uint32_t now_us =
        micros();


    // -----------------------------------------------------------------------
    // Request realtime motor feedback @ 250 Hz
    // -----------------------------------------------------------------------

    if (
        (uint32_t)(
            now_us -
            rt.previous_feedback_request_us)
        >=
        Config::FEEDBACK_REQUEST_PERIOD_US)
    {
        rt.previous_feedback_request_us =
            now_us;


        request_motor_feedback();
    }


    // -----------------------------------------------------------------------
    // Request motor status @ 20 Hz
    // -----------------------------------------------------------------------

    if (
        (uint32_t)(
            now_us -
            rt.previous_status_request_us)
        >=
        Config::STATUS_REQUEST_PERIOD_US)
    {
        rt.previous_status_request_us =
            now_us;


        request_motor_status();
    }


    // -----------------------------------------------------------------------
    // Torque control @ 1 kHz
    // -----------------------------------------------------------------------

    if (
        (uint32_t)(
            now_us -
            rt.previous_control_us)
        >=
        Config::CONTROL_PERIOD_US)
    {
        const uint32_t elapsed_us =
            now_us -
            rt.previous_control_us;


        rt.previous_control_us =
            now_us;


        const float dt_s =
            clampf(
                elapsed_us *
                    1.0e-6f,

                0.0001f,
                0.02f);


        update_control(
            now_us,
            dt_s);
    }


    // -----------------------------------------------------------------------
    // STATE telemetry @ 100 Hz
    // -----------------------------------------------------------------------

    if (
        (uint32_t)(
            now_us -
            rt.previous_telemetry_us)
        >=
        Config::TELEMETRY_PERIOD_US)
    {
        rt.previous_telemetry_us =
            now_us;


        send_torque_feedback();
    }
}