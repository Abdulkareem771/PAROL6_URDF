# Firmware Architecture & Data Flow

This document visualizes the hard real-time architecture of the PAROL6 Teensy 4.1 firmware.

## 1. Top-Level Module Relationships

The firmware is designed around a strict separation of concerns, decoupling the asynchronous ROS 2 communication from the highly deterministic 1 kHz control mathematics.

```mermaid
graph TD
    classDef hardware fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef transport fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef control fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef safety fill:#ffebee,stroke:#b71c1c,stroke-width:2px;

    Host[ROS 2 Host PC] <-->|115200 Baud ASCII| SerialTransport:::transport
    
    subgraph MCU["Teensy 4.1 (i.MXRT1062)"]
        SerialTransport -->|Push push| RXQueue[(RosCommand Queue)]:::transport
        
        RXQueue -->|Pop| MainLoop[Main loop]:::transport
        MainLoop -->|Set Target| Interpolator:::control
        MainLoop -->|Feed Watchdog| Supervisor:::safety
        
        subgraph ISR["1 kHz Hardware Timer ISR (IntervalTimer)"]
            EncoderHAL:::hardware -->|Read| AlphaBetaFilter:::control
            AlphaBetaFilter -->|Pos/Vel Estimate| ControlLaw[Control Law: P + FF]:::control
            Interpolator -->|Pos/Vel Setpoint| ControlLaw
            
            ControlLaw -->|Velocity Command| OutputSaturation[Output Clamping]:::control
            OutputSaturation -->|Command| MotorOutput[Motor Velocity Output]:::hardware
            
            AlphaBetaFilter -->|Current Velocity| Supervisor
            Supervisor -->|Safe/Fault State| MotorOutput
        end
    end
```

## 2. The 1 kHz ISR Execution Pipeline

The 1 ms tick is the heart of the system. To guarantee jitter remains $< 50$ µs, the execution sequence inside the `run_control_loop_isr()` function is rigidly ordered to compute all math *before* making safety decisions and applying physical outputs.

```mermaid
sequenceDiagram
    participant Timer as IntervalTimer (1ms)
    participant HAL as EncoderHAL
    participant Filter as AlphaBetaFilter
    participant Interp as LinearInterpolator
    participant Safety as SafetySupervisor
    participant Motor as Motor Output
    
    Note over Timer, Motor: --- BEGIN 1ms INTERRUPT ---
    
    loop For Each Axis (0 to 5)
        Timer->>HAL: read_angle()
        HAL-->>Filter: Raw Position (rads)
        
        Filter->>Filter: Unwrap M_PI boundaries
        Filter->>Filter: Calculate Innovation
        Filter-->>Timer: estimated_pos, estimated_vel
        
        Timer->>Interp: tick_1ms()
        Interp-->>Timer: cmd_pos, cmd_vel_ff
        
        Note over Timer: Calculate: pos_error = cmd_pos - estimated_pos
        Note over Timer: Calculate: cmd_vel = cmd_vel_ff + (Kp * pos_error)
        Note over Timer: Clamp cmd_vel to MAX_VEL_CMD
    end
    
    Timer->>Safety: update(tick_ms, all_velocities)
    Note over Safety: Checks timeouts & runaway limits
    
    loop For Each Axis (0 to 5)
        Timer->>Safety: is_safe()?
        alt Nominal
            Timer->>Motor: Apply cmd_vel
        else Fault / E-Stop
            Timer->>Motor: Apply 0.0f (Fast Stop)
        end
    end
    
    Note over Timer, Motor: --- END 1ms INTERRUPT ---
```

## 3. Data Ownership & Thread Safety rules

Because the system blends an asynchronous background loop (serial parsing) with a pre-emptive foreground ISR (hardware timer), data ownership is strictly enforced to prevent race conditions without relying on heavy RTOS mutexes taking down the ISR.

*   **`CircularBuffer<RosCommand>`**: Acts as the sole locking boundary. The `MainLoop` owns pushing. `MainLoop` temporarily calls `noInterrupts()` to pop items safely before the ISR can strike. 
*   **`Interpolator`**: Owned by the ISR (`tick_1ms`). Setpoints (`set_target`) are injected by the `MainLoop` only during the safe periods between queue pops.
*   **`AlphaBetaFilter`**: Exclusively owned by the ISR. The `MainLoop` is allowed to *read* the state (for background 10 Hz telemetry) but must wrap the read in `noInterrupts()` to prevent reading a torn float if the 1ms tick interrupts the copy operation.
