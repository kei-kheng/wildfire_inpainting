# Description
This code loads two randomly chosen images from the inference dataset as the environment for a group of agents to explore, simulating a collective perception scenario in a dynamic environment. The agents store 3 maps:
    - `observed`: What the agent currently knows about the environment
    - `explored`: Regions the agent has explored
    - `confidence`: Models information decay for agents to adapt to dynamic environment
and are able to communicate with agents within its communication range, exchanging information. When the size of the payload is limited, agents prioritize transmitting observation with higher confidence values. `observed` and `explored` decay at a fixed rate and information below the confidence threshold are phased out. The Pygame window displays 5 agents' prediction based on their respective observations. Screenshots of the window are captured periodically. Simulation conditions are written to a text file and the evaluation metrics are plotted against the simulation time at the end of simulation.

**Evaluation metrics:** PSNR, SSIM. Calculated over **entire image** - _different from 6_inpainting_GAN_.

# Usage - main.py
## Parameters
```
--img_scaled_dim               The dimension of the longest side of the image after scaling
--model_path                   Path to load trained model from
--no_of_agents                 Number of agents to be deployed in environment, must be greater than 1
--agent_patch_size             Size of patch observable by agent in pixels
--agent_comm_range             Communication range of agents in pixels based on the Euclidean/Pythagorean distance
--max_payload_size             Maximum payload size in bytes (transmission of each pixel are assumed to cost 3 bytes)
--agent_confidence_reception   Confidence value assigned to received payload (want received information to stay long enough to be useful)
--agent_confidence_decay       Rate at which confidence decays, calculated as new_confidence = old_confidence * (1 - agent_confidence_decay) at each time step
--agent_confidence_threshold   Confidence threshold, information with confidence below this value will be phased out
--log_comm                     Option to log communication between agents, turning this on might slow down simulation
--steps                        Length of simulation in simulation time
--output_dir                   Folder in which results are stored
```

**Example**
```
python main.py --img_scaled_dim 320 --model_path models/test8/generator.pth --no_of_agents 20 --agent_patch_size 25 --agent_comm_range 30 --max_payload_size 270 --agent_confidence_reception 0.6 --agent_confidence_decay 0.001 --agent_confidence_threshold 0.15 --log_comm --steps 10000 output_dir test
```

# Goal
To find suitable values of `agent_confidence_reception`, `agent_confidence_decay` and `agent_confidence_threshold` for agents to be able to perceive the entire environment given fixed `agent_patch_size`, `agent_comm_range` and limited `max_payload_size`.