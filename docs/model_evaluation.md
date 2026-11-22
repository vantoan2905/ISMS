# Model evaluation

Use a site specific validation set with day, night, occlusion, multiple camera angles, and helmet variants. Measure person and helmet precision and recall, false alerts per camera hour, zone entry recall, and event latency. Evaluate helmet detection specifically on small or partially obscured heads. Tune confidence and zone polygons using held out footage. The default model has no helmet class, so helmet alerts are disabled until compatible weights are supplied. Treat camera alerts as prompts for human review, not as the sole basis for safety action.
