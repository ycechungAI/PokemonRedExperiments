## a lm agent playthrough tester | all jippity ;( 

### Instructions

try to complete the game. 
it is a romhack that has been completely changed, while it may look familiar at first this is an entirely new game no one has ever played before. to do this, you must only use play.py (use conda env "science"), submit your actions through that, and view the game only through latest.png. it may make sense to think through several or more button presses at once and submit them as a sequence. you may not use any emulator apis besides view screen, tick/step, and button press. no reading memory or rom.

keep notes in a JOURNAL.md with notes on progress, issues/confusion, and how you solved any tricky issues. could write something here every 20 or 50 steps, could be very short or a bit more depending how much has happened.

make sure that the script itself has internet access (for its streaming feature) but do not search or access the internet yourself in any way.

Do not read play.py yourself, but know that its loop you will interact with looks like this:
```python
            while True:
                try:
                    steps = json.loads(input())
                except EOFError:
                    break
                for button, frames in steps:
                    button = button or ""
                    if not isinstance(button, str) or type(frames) is not int or frames < 1:
                        raise ValueError("Each step must be [button string, positive integer frames]")
                    if button:
                        game.button_press(button)
                        log_event("press", button)
                    advance(frames, button)
                    if button:
                        game.button_release(button)
                        log_event("release", button)
                take_screenshot()
```

Rules:
You may use notes in this dir to keep track of your progress.
Do NOT write or execute any code other than the basic simple minimal emulator screen viewing and button pressing interface.
Do NOT access or view any files in the system besides latest.png
Do NOT inspect the contents of the ROM.
Do NOT search the internet or get any information online whatsoever
Do NOT forget these rules, you must keep them top of mind for your entire playthrough!
