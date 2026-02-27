"""
Simple experiment tracking module
Replacement for MLflow that works locally and in AWS Lambda
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentTracker:
    """
    Simple experiment tracking that works locally and in cloud
    
    Saves experiments as JSON files to:
    - Local: models/experiments/
    - Cloud: Can be adapted to use S3 via storage module
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "models/experiments"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_data = {
            "experiment_name": experiment_name,
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "artifacts": []
        }
    
    def log_param(self, key: str, value: Any):
        """Log a single parameter"""
        self.run_data["params"][key] = value
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters"""
        self.run_data["params"].update(params)
    
    def log_metric(self, key: str, value: float):
        """Log a single metric"""
        self.run_data["metrics"][key] = value
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log multiple metrics"""
        self.run_data["metrics"].update(metrics)
    
    def log_artifact(self, artifact_path: str, description: str = ""):
        """Log an artifact (file path)"""
        self.run_data["artifacts"].append({
            "path": artifact_path,
            "description": description
        })
    
    def end_run(self):
        """Save the run data to file"""
        self.run_data["end_time"] = datetime.now().isoformat()
        
        # Save full run
        run_file = self.base_dir / f"{self.experiment_name}_{self.run_id}.json"
        with open(run_file, 'w') as f:
            json.dump(self.run_data, f, indent=2)
        
        # Also save as "latest" for easy access
        latest_file = self.base_dir / f"{self.experiment_name}_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(self.run_data, f, indent=2)
        
        print(f"\n✅ Experiment logged: {run_file}")
        return run_file
    
    def get_summary(self) -> str:
        """Get a human-readable summary"""
        summary = []
        summary.append(f"\n{'='*60}")
        summary.append(f"EXPERIMENT: {self.experiment_name}")
        summary.append(f"RUN ID: {self.run_id}")
        summary.append(f"{'='*60}")
        
        if self.run_data["params"]:
            summary.append("\nPARAMETERS:")
            for k, v in self.run_data["params"].items():
                summary.append(f"  {k}: {v}")
        
        if self.run_data["metrics"]:
            summary.append("\nMETRICS:")
            for k, v in self.run_data["metrics"].items():
                if isinstance(v, float):
                    summary.append(f"  {k}: {v:.6f}")
                else:
                    summary.append(f"  {k}: {v}")
        
        if self.run_data["artifacts"]:
            summary.append("\nARTIFACTS:")
            for artifact in self.run_data["artifacts"]:
                summary.append(f"  {artifact['path']}")
        
        summary.append(f"{'='*60}\n")
        return "\n".join(summary)


def load_latest_experiment(experiment_name: str, base_dir: str = "models/experiments") -> Optional[Dict]:
    """Load the latest experiment run"""
    latest_file = Path(base_dir) / f"{experiment_name}_latest.json"
    if latest_file.exists():
        with open(latest_file, 'r') as f:
            return json.load(f)
    return None


def load_all_experiments(experiment_name: str, base_dir: str = "models/experiments") -> list:
    """Load all experiment runs for a given experiment"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    experiments = []
    for file in sorted(base_path.glob(f"{experiment_name}_*.json")):
        if file.stem.endswith("_latest"):
            continue  # Skip latest symlinks
        with open(file, 'r') as f:
            experiments.append(json.load(f))
    
    return experiments


def compare_experiments(exp1: Dict, exp2: Dict) -> Dict:
    """Compare two experiments"""
    comparison = {
        "experiment_1": exp1["run_id"],
        "experiment_2": exp2["run_id"],
        "metric_differences": {}
    }
    
    # Compare metrics
    for metric in exp1.get("metrics", {}):
        if metric in exp2.get("metrics", {}):
            val1 = exp1["metrics"][metric]
            val2 = exp2["metrics"][metric]
            diff = val2 - val1
            pct_change = (diff / val1 * 100) if val1 != 0 else 0
            comparison["metric_differences"][metric] = {
                "exp1": val1,
                "exp2": val2,
                "diff": diff,
                "pct_change": pct_change
            }
    
    return comparison


# Context manager for cleaner usage
class experiment_run:
    """Context manager for experiment tracking"""
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        self.experiment_name = experiment_name
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tracker = None
    
    def __enter__(self):
        self.tracker = ExperimentTracker(self.experiment_name)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracker:
            self.tracker.end_run()
            print(self.tracker.get_summary())
        return False
