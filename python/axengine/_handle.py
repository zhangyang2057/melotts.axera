import numpy as np
from axengine import _C


class InferenceSession:
    def __init__(self) -> None:
        """
        InferenceSession Collection.

        Args:
            model_format (string, optional): Model format. Defaults to "axmodel".
            device (string, optional): Running device. Defaults to "CPU".
        """
        self._handle = _C.Runner()
        self._init_engine()

    def _init_engine(self):
        self._handle.init_device()

    def load_model(self, model_path: str):
        """
        Load model graph to InferenceSession.

        Args:
            model_path (string): Path to model
            input_size (Tuple[int, int]): Image size with (H, W) layout
        """
        return self._handle.load_model(model_path)

    def get_cmm_usage(self):
        return self._handle.get_cmm_usage()

    def feed_input_to_index(self, input_feed: np.ndarray, input_index: int):
        self._handle.feed_input_to_index(input_feed, input_index)

    def feed_inputs(self, input_feed: list[np.ndarray]):
        """
        Args:
            input_feed (np.ndarray): The input feed
        """
        for i, inp in enumerate(input_feed):
            self.feed_input_to_index(inp, i)

    def forward(self):
        """
        Returns:
            np.ndarray: Output of the networks.
        """
        self._handle.forward()

    def get_output_from_index(self, output_index: int):
        return self._handle.get_output_from_index(output_index)

    def get_outputs(self, output_names):
        output_data = {}
        for i, output_name in enumerate(output_names):
            output_data[output_name] = self.get_output_from_index(i)
        return output_data
