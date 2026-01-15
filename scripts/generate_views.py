#!/usr/bin/env python3

OWNER = "Abdulkareem771"
PROJECT_NUMBER = 1

BASE_URL = f"https://github.com/users/{OWNER}/projects/{PROJECT_NUMBER}"

VIEWS = {
    "Hardware": "Domain:hardware",
    "ESP32": "Domain:esp32",
    "Control": "Domain:control",
    "FOC": "Domain:foc",
    "Vision": "Domain:vision",
    "ROS": "Domain:ros",
    "Robotics": "Domain:robotics",
    "AI": "Domain:ai",
    "Other": "Domain:other",

    "Firmware Layer": "Layer:firmware",
    "Perception Layer": "Layer:perception",
    "Planning Layer": "Layer:planning",
    "Integration": "Layer:integration",

    "Prototype": "Maturity:prototype",
    "Validated": "Maturity:validated",
}

def main():
    print("\nPAROL6 Project View Links\n")

    for name, filter_expr in VIEWS.items():
        encoded = filter_expr.replace(":", "%3A")
        url = f"{BASE_URL}?query={encoded}"
        print(f"{name:20s} → {url}")

    print("\nOpen any link to instantly apply the filter.")

if __name__ == "__main__":
    main()
