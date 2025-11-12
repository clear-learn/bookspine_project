import numpy as np
import cv2
import os
import onnxruntime
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.device_id = 0
        model_path = os.path.join(os.path.dirname(__file__), "arcbook_model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at {model_path}")
        providers = [('CUDAExecutionProvider', {'device_id': self.device_id}), 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(
            f"ArcBook Recognition model (ONNX, Optimized Preprocessing) initialized on {self.session.get_providers()[0]}.")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                in_tensor = pb_utils.get_input_tensor_by_name(request, "CROPPED_IMAGE")
                image_bytes = in_tensor.as_numpy()[0]

                batch_input = np.expand_dims(image_bytes, axis=0)
                features = self.session.run([self.output_name], {self.input_name: batch_input})[0]

                output_vector = features.squeeze()

                out_tensor = pb_utils.Tensor("OUTPUT_VECTOR", output_vector.astype(np.float32))

                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(inference_response)

            except Exception as e:
                import traceback
                error_message = f"{str(e)}\n{traceback.format_exc()}"
                error_response = pb_utils.InferenceResponse(output_tensors=[],
                                                            error=pb_utils.TritonError(error_message))
                responses.append(error_response)

        return responses

    def finalize(self):
        print('Cleaning up recognition model (ONNX, Optimized Preprocessing) resources...')
