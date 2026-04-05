# -----------------------------------------------------------------------------
# Base: your WORKING image (do not touch this layer)
# -----------------------------------------------------------------------------
FROM shudh25/runpod-serverless-ace-step-1.5:training-v14

WORKDIR /app

# -----------------------------------------------------------------------------
# Copy New Handler
# -----------------------------------------------------------------------------
COPY training_handler.py /app/training_handler.py

# Smoke-test imports at build time — catches missing packages immediately
RUN python -c "import torch, torchaudio, lightning, peft, safetensors, runpod; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}'); print(f'lightning={lightning.__version__}  peft={peft.__version__}'); print('All training dependencies import OK')"

# Validate ACE-Step training module is importable
RUN python -c "from acestep.training.dataset_builder import DatasetBuilder, AudioSample; from acestep.training.trainer import LoRATrainer; from acestep.training.configs import LoRAConfig, TrainingConfig; from acestep.training.data_module import PreprocessedTensorDataset; print('ACE-Step training module OK'); import inspect; lora_fields=[f.name for f in LoRAConfig.__dataclass_fields__.values()]; train_fields=[f.name for f in TrainingConfig.__dataclass_fields__.values()]; print(f'LoRAConfig fields:     {lora_fields}'); print(f'TrainingConfig fields: {train_fields}'); train_sig=inspect.signature(LoRATrainer.__init__); print(f'LoRATrainer.__init__ signature: {train_sig}')"


CMD ["python", "-u", "/app/training_handler.py"]