#!/usr/bin/env python3
import pygame
import time

def test_controller():
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("No controller found!")
        return
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"Controller: {joystick.get_name()}")
    print(f"Axes: {joystick.get_numaxes()}")
    print(f"Buttons: {joystick.get_numbuttons()}")
    
    try:
        while True:
            pygame.event.pump()
            
            # Read axes
            axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
            
            # Read buttons
            buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
            
            print(f"Axes: {[f'{a:.2f}' for a in axes]} | Buttons: {buttons}", end='\r')
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nTest completed")

if __name__ == '__main__':
    test_controller()
