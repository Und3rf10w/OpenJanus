from collections.abc import Mapping
import logging
from time import sleep
from typing import List

import keyboard
import pydirectinput
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import KeyCode


LOGGER = logging.getLogger(__name__)
keyboard = KeyboardController()
mouse = MouseController()


class InvalidKeyError(Exception):
    """
    Exception raised in the event that an invalid key is found
    """
    def __init__(self, button: str, message: str):
        """
        Initalises the InvalidKeyError

        :param button: The name of the button that is invalid
        :param message: The message to log with the invalid key
        """
        self.button = button
        self.message = message
        LOGGER.error(f"{message}: {button}")


# Function to execute the mouse action
def mouse_action(button: Button, clicks: int, hold: bool):
    for _ in range(clicks):
        mouse.press(button)
        if not hold:
            mouse.release(button)

# Function to convert a string button name to a pynput button object
def get_button(button_name: str):
    try:
        match button_name.lower():
            case 'left mouse button':
                return Button.left
            case 'right mouse button':
                return Button.right
            case 'middle mouse button':
                return Button.middle
            # case 'mouse wheel up':
            #     return Button.scroll_up
            # case 'mouse wheel down':
            #     return Button.scroll_down
    except ValueError:
        raise InvalidKeyError(button=button_name, message="Unknown mouse button name")


# Function to perform key press or release
def handle_keys(keys: List[str], hold: bool):
    special_keys = {
        'left alt': Key.alt_l,
        'numpad 8': KeyCode.from_vk(104),
        'numpad 2': KeyCode.from_vk(98),
        'numpad 4': KeyCode.from_vk(100),
        'numpad 6': KeyCode.from_vk(102),
        'numpad 7': KeyCode.from_vk(103),
        'numpad 1': KeyCode.from_vk(97),
        'numpad 5': KeyCode.from_vk(101),
        'left control': Key.ctrl_l,
        'left shift': Key.shift_l,
        'right shift': Key.shift_r,
        'f1': Key.f1,
        'f2': Key.f2,
        'f3': Key.f3,
        'f4': Key.f4,
        'f5': Key.f5,
        'f6': Key.f6,
        'f7': Key.f7,
        'f8': Key.f8,
        'f9': Key.f9,
        'f10': Key.f10,
        'f11': Key.f11,
        'f12': Key.f12,
    }
    for key_name in keys:
        # Convert the key name to lowercase for consistent mapping
        key_name_lower = key_name.lower()
        # Check if the key is in the special keys mapping
        if key_name_lower in special_keys:
            key = special_keys[key_name_lower]
        else:
            # Assume the key is a single character and get a KeyCode for it
            key = KeyCode.from_char(key_name_lower)

        # Press or release the key based on the 'hold' flag
        if hold:
            # start_time = time()  # Get the current time
            # while time() - start_time < 0.3:  # Loop for 0.3 seconds
            #     keyboard.press(key)
            # keyboard.release(key)
            LOGGER.debug(f"Holding {key_name_lower}")
            pydirectinput.keyDown(key_name_lower)
            sleep(3)
            pydirectinput.keyUp(key_name_lower)
            
        else:
            keyboard.press(key)
            keyboard.release(key)


# Function to perform mouse and keyboard actions
def perform_action(action: dict):
    # Mouse actions
    if 'mouse' in action:
        button = get_button(action['mouse']['button'])
        clicks = action['mouse'].get('clicks', 1)
        hold = action['mouse'].get('hold', False)
        if button is not None:
            mouse_action(button, clicks, hold)

    # Keyboard actions
    if 'keys' in action:
        if isinstance(action['keys'], str):
            if action['keys'].startswith('hold '):
                hold = True
                key_name = action['keys'][5:]  # Get the key name after 'hold '
                handle_keys([key_name], hold)
        elif isinstance(action['keys'], list):
            for key_str in action['keys']:
                if key_str.lower().startswith('hold '):
                    hold = True
                    key_name = key_str[5:]  # Get the key name after 'hold '
                else:
                    if not 'hold' in action:
                        hold = False
                    else:
                        hold = action['hold']
                    key_name = key_str
                handle_keys([key_name], hold)


def run(keypresses: dict):
    if isinstance(keypresses, List):  # This is a list
        for action in keypresses:
            LOGGER.info(f"Onboard IA: Performing {action['action_name']}")
            perform_action(action)
            LOGGER.info(f"Onboard IA: {action['action_name']} completed")
            return f"Actions {[action['action_name'] for action in keypresses]} Completed Successfully"
    if isinstance(keypresses, Mapping):  # This is a dict
        if not "actions" in keypresses:
            LOGGER.info(f"Onboard IA: Performing {keypresses['action_name']}")
            perform_action(keypresses)
            LOGGER.info(f"Onboard IA: {keypresses['action_name']} completed")
            return f"Actions {keypresses['action_name']} completed."
        if "actions" in keypresses:
            if isinstance(keypresses['actions'], list):
                for action in keypresses['actions']:
                    LOGGER.info(f"Onboard IA: Performing {action['action_name']}")
                    perform_action(action)
                    LOGGER.info(f"Onboard IA: {action['action_name']} completed")
                return f"Actions {[action['action_name'] for action in keypresses['actions']]} Completed Successfully"
            LOGGER.info(f"Onboard IA: Performing {keypresses['actions']['action_name']}")
            perform_action(keypresses['actions'])
            LOGGER.info(f"Onboard IA: {keypresses['actions']['action_name']} completed")
            return f"Actions {keypresses['actions']['action_name']} Completed Successfully."



async def arun(keypresses: dict):
    if not isinstance(keypresses, list):  # This is a list
        for action in keypresses:
            LOGGER.info(f"Onboard IA: Performing {action['action_name']}")
            perform_action(action)
            LOGGER.info(f"Onboard IA: {action['action_name']} completed")
            return f"Actions {[action['action_name'] for action in keypresses]} Completed Successfully"
    if isinstance(keypresses, Mapping):  # This is a dict
        if not "actions" in keypresses:
            LOGGER.info(f"Onboard IA: Performing {keypresses['action_name']}")
            perform_action(keypresses)
            LOGGER.info(f"Onboard IA: {keypresses['action_name']} completed")
            return f"Actions {keypresses['action_name']} completed."
        if "actions" in keypresses:
            LOGGER.info(f"Onboard IA: Performing {keypresses['actions']['action_name']}")
            perform_action(keypresses['actions'])
            LOGGER.info(f"Onboard IA: {keypresses['actions']['action_name']} completed")
            return f"Actions {keypresses['actions']['action_name']} Completed Successfully."
    

# # Example usage
# sample_input = {
#     "Toggle Ship Lights": {
#         "keys": ["l"],
#         "hold": False
#     },
#     "Landing Gear": {
#         "keys": ["n"],
#         "hold": False
#     }
# }

# for action_name, action in sample_input.items():
#     perform_action(action)