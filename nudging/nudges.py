import pandas as pd

class VisualNudge:
    def __init__(
            self,
            data: str,
            api_call: callable,
            seed: int,
            eval_model: str,
            initial_prompt: str
    ):
        # Load and shuffle data with reproducibility
        self.data = pd.read_csv(data).sample(frac=1, random_state=seed)
        self.api_call = api_call
        self.seed = seed
        self.eval_model = eval_model
        self.initial_prompt = initial_prompt


    def get_nudge_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "select",
                    "strict": True,
                    "description": "Call this to select an image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "boolean",
                                "description": "Return true to select the second image and false to select the first.",
                            },
                        },
                        "required": ["decision"],
                        "additionalProperties": False,
                    },
                }
            }
        ]