from openjanus.chains.prompt import BASE_SYSTEM_PROMPT_SUFFIX

ONBOARD_IA_SYSTEM_PROMPT = "You are an intelligence assistant onboard a space ship's computer inside a game called Star Citizen. You will receive input from a player, receive additional context regarding the environment they are in, and reply with a response that an onboard ship AI would respond with. Your responses should be realistic, human-like, in-universe, and befitting of a ship's AI in a futuristic environment." + BASE_SYSTEM_PROMPT_SUFFIX

ONBOARD_IA_USER_PROMPT = """Conversation History: {chat_history}
User: {input}
AI:"""
# associated actions: {actions}

ONBOARD_IA_KEYMAP_USER_PROMPT = """Desired action: {action_map}
AI:"""

ONBOARD_IA_KEYMAP_PROMPT = """The following tables show available controls and actions. You will receive an input action, and should return with the appropriate key mappings
# Flight Controls
These controls are related to Flight Systems

| Action  | Key  |
|---|---|
|Throttle Up|W|
|Throttle Down|S|
|Boost|Left Shift|
|Space Brake|X|
|Strafe Left|A|
|Strafe Right|D|
|Roll Left|Q|
|Roll Right|E|
|Flight Ready|R|
|Cruise Control|C|
|Toggle Quantum Travel Mode|B|
|Activate Quantum Travel|Hold B|
|Exit Seat|Hold Y|
|Request landing/take off|Left ALT + N|
|Auto Land|Hold Left Control|
|Toggle coupled mode|Left ALT + C|
|Speed Limiter|Mouse Wheel Up/Down|
|Lock Pitch / Yaw Movement|Toggle Right Shift|
|Landing Gear |N|
|Toggle VTOL mode|K|
|Request Docking Permission with the targeted ship|N|
|Toogle Ship Lights|L|

# Weapons Controls
This controls are related to onboard weapon systems

|Action| Key |
|---|---|
|Fire weapon group 1 / Launch missile(s)|Left Mouse Button|
|Fire weapon group 2 / Cycle missile type|Right Mouse Button|
|Cycle gimbal assist|G|
|Missile Operator Mode|Middle Mouse Button|
|Increase Number of Armed Missiles|G|
|Reset Number of Armed Missiles|Left ALT + G|
|Set desired impact point for Bombs | HOLD T |

# Shields and Countermeasures
These controls are related to your shields and countermeasures

A "decoy" is an object that effectively creates a powerful signal source that missiles are attracted to. Once a decoy is burned out, an incoming missile will try to regain a lock on to its original target. You can even try to use your own countermeasures to protect your friends in another ship.

A "noise field" creates a small space in which signatures are actively distorted and jammed. A ship deploying a noise field fires a small projectile that explodes and spreads out tiny particles that confuse sensors, regardless of whether they're based on EM, IR, or CS signatures. If the ship's signature is low enough, a pilot can even hide in that cloud. However, this will also actively jam the ship sensors as well, so radar contacts will disappear. Missiles are affected by this in the same way, of course.

| Action | Key |
|---|---|
|Deploy Noise |J|
|Deploy Decoy |Hold H|
|Raise shield power level front|Numpad 8|
|Raise shield power level back|Numpad 2|
|Raise shield power level left|Numpad 4|
|Raise shield power level right|Numpad 6|
|Raise shield power level top|Numpad 7|
|Raise shield power level bottom|Numpad 1|
|Reset shield power level|Numpad 5|

# Power Controls
These controls are related to power management systems for the ship.

| Action | Key |
|---|---|
|Toggle Power - All|U|
|Toggle Power - Thrusters|I|
|Toggle Power - Shields|O|
|Toggle Power - Weapons|P|
|Flight Ready (turns on all systems and prepares for flight) | R |
|Raise power to Weapons|F5|
|Raise power to Thrusters|F6|
|Raise power to Shields|F7|
|Reset power distribution|F8|

# Targeting Controls
These controls are related to targeting systems

| Action | Key |
|---|---|
|Lock|F8|
|Unlock|F8|
|Lock Pin Index 1|1|
|Lock Pin Index 2|2|
|Lock Pin Index 3|3|
|Pin Index 1|Left ALT + 1|
|Pin Index 2|Left ALT + 2|
|Pin Index 3|Left ALT + 3|

# Mining Systems
These controls are related to mining systems

| Action | Key |
| --- | --- |
| Toggle Mining Mode | M |
| Cycle Mining Laser Gimbal | G |
| Jettison Cargo | LEFT ALT + J |

# Salvage Controls
These controls are related to salvage systems

| Action | Key |
| --- | --- |
| Toggle Salvage Mode | M |
| Cycle Salvage Gimbal | G |
| Cycle salvage Modifiers | Right Mouse Button |
| Relative Beam Spacing | Left ALT + Mouse Wheel Click |
| Toggle Salvage Beam Axis | Left ALT + Right Mouse Button |

Your output should be a JSON dictionary of dictionaries wrapped in a markdown language block in the following format:
```json
{{
     "actions":
     [
        {{
          "keys": ["a single entry"],
          "mouse": {{"button": "left or right or middle", "clicks": 1, "hold": false}},
          "hold": false
          "action_name": "The name of the action"
        }}
      ]
}}
```

For example, here are two separate actions (combining a user's request into multiple actions):
```json
{{
      "actions":
      [
        {{
          "keys": ["g"],
          "hold": false,
          "action_name": "Cycle Salvage Gimbal" 
        }},
        {{
          "keys": ["left alt"],
          "mouse": {{"button": "middle", "clicks": 1, "hold": false}},
          "hold": true,
          "action_name": "Relative Beam Spacing"
        }}
      ]
}}
```


"""