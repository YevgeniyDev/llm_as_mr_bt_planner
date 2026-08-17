# How BT generation works

First, the user provides a scenario JSON. It describes the mission, initial state, final goals, available robots and all actions that each robot is actually capable of doing. Every action also has its parameters, preconditions, effects, required resources and time limits.

The program takes all of this information and builds a prompt for the selected LLM. The LLM receives the full mission and has to generate one complete Behavior Tree for every robot. So, for our case it returns three trees: one for Franka A, one for Franka B and one for the Unitree Go2 with Z1.

The important thing is that the LLM itself decides:

- which robot should do each action;
- in what order actions happen;
- when one robot must wait for another;
- when a shared zone should be acquired and released;
- where to use sequences, fallback branches and conditions.

The program does not build the tree for the LLM afterwards. It also does not silently add missing actions or rearrange them. The LLM must return the whole BT structure as JSON.

For example, Go2 cannot pick the parcel before the two Panda arms finish preparing it. Therefore its generated tree should contain waits for the package base to be placed and for the lid to be sealed. Only after those conditions become true it can acquire the packing zone and pick the parcel.

After receiving the JSON, the program parses it and runs static validation. It checks if every robot has a tree, all actions really belong to that robot, parameters are correct, every goal has a producer, waits have real producers, resources are correctly acquired and released, IDs are unique and there are no obvious wait or resource cycles.

If static validation passes, the same trees are executed in the symbolic simulator. All three robot BTs are ticked together. Actions check their declared preconditions, `WaitFor` nodes stay running until another robot produces the required state, and shared resources can only have one owner. The simulator also checks timeouts, deadlocks, resource leaks, invalid states and whether all final goals were reached.

When something fails, the program sends the LLM the complete rejected BT together with the exact validation and simulation errors. The LLM then generates a complete corrected version, not just a small patch. This can repeat for a limited number of correction rounds.

Once a candidate passes, the program loads that exact LLM response again and independently repeats validation and simulation. If it still passes, the final BT is saved as JSON and XML together with logs, reports, the simulation trace and file checksums. If it does not pass, no final BT is published.

So basically, the LLM is responsible for planning and constructing the complete multi-robot Behavior Trees. The deterministic part of the program is responsible for checking if those trees make sense and if they can achieve the mission according to the declared symbolic rules. Later, for the supported scenarios, MuJoCo executes that same generated BT using the predefined physical robot controllers.
