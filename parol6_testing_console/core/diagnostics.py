import shutil

def check_tool(name: str) -> bool:
    """Check if a tool exists on the system PATH."""
    return shutil.which(name) is not None

def get_toolchain_diagnostics() -> dict[str, bool]:
    """Return an exhaustive dict of known toolchains and their availability."""
    return {
        "pio": check_tool("pio"),
        "platformio": check_tool("platformio"),
        "openocd": check_tool("openocd"),
        "dfu-util": check_tool("dfu-util"),
        "st-info": check_tool("st-info"),
        "st-flash": check_tool("st-flash"),
        "teensy_loader_cli": check_tool("teensy_loader_cli"),
        "ros2": check_tool("ros2")
    }

def get_required_tools_for_project(project: dict) -> list[str]:
    """
    Given a project dictionary, determine what tools it *probably* needs
    based on hints or environment. This prevents noisy warnings for unused tools.
    """
    reqs = ["pio"] # almost all need PlatformIO
    
    # Check flash hinting
    hint = project.get("flash", {}).get("tooling_hint", "").lower()
    
    if "st-link" in hint:
        reqs.append("openocd")  # PlatformIO stlink usually wraps openocd
    if "dfu-util" in hint or "dfu" in hint:
        reqs.append("dfu-util")
    if "teensy" in hint:
        reqs.append("teensy_loader_cli")
        
    return reqs

def build_diagnostic_report(project: dict) -> dict:
    """
    Builds a report stating which required tools are missing,
    and which tools are available.
    """
    tools = get_toolchain_diagnostics()
    reqs = get_required_tools_for_project(project)
    
    missing = [req for req in reqs if not tools.get(req, False)]
    
    return {
        "missing_required": missing,
        "all_tools_state": tools,
        "is_ok": len(missing) == 0
    }
