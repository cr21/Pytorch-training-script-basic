import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """
  save model to target dir
  """
  target_dir_path=Path(target_dir)

  target_dir_path.mkdir(parents=True, exist_ok=True)

  model_path=target_dir_path/model_name

  torch.save(obj=model.state_dict(), f=model_path)
