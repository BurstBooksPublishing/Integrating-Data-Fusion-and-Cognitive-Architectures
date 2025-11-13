import os, glob, numpy as np, tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class NumpyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_files, batch_size, input_shape, cache_file):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.files = batch_files
        self.cur = 0
        self.device_input = cuda.mem_alloc(np.prod(input_shape)*batch_size*4)  # float32
    def get_batch_size(self):
        return self.batch_size
    def get_batch(self, names):
        if self.cur + self.batch_size > len(self.files):
            return None
        batch = np.stack([np.load(f).astype(np.float32) for f in self.files[self.cur:self.cur+self.batch_size]])
        self.cur += self.batch_size
        cuda.memcpy_htod(self.device_input, batch.ravel())
        return [int(self.device_input)]
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

def build_engine(onnx_path, engine_path, calib):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1<<30
        builder.max_batch_size = calib.batch_size
        if not parser.parse_from_file(onnx_path):
            raise RuntimeError("Failed to parse ONNX")
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
        engine = builder.build_engine(network, config)
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        return engine

# --- usage ---
batch_files = sorted(glob.glob("calib_batches/*.npy"))
calibrator = NumpyCalibrator(batch_files, batch_size=8, input_shape=(8,3,224,224), cache_file="calib.cache")
engine = build_engine("model.onnx", "model_int8.plan", calibrator)
# After build, load engine and run inference in runtime for verification (omitted for brevity).