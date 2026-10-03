"""Train the hybrid model on ALL data and save it to models/hybrid_model.joblib (the app loads this file).
Run: python steps/train_and_save.py"""
import sys, warnings, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from cropds.data import FEATURES, TARGET, load_dataset
from cropds.model import HybridCropModel, save_model, MODEL_PATH

t = time.time(); df = load_dataset()
m = HybridCropModel().fit(df[FEATURES], df[TARGET]); save_model(m)
print(f"saved {MODEL_PATH}  ({MODEL_PATH.stat().st_size/1e6:.1f} MB, {time.time()-t:.0f}s)  alpha={m.alpha_} width={m.width_} gamma={m.gamma_}")
