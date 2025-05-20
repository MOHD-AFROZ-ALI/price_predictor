# service.py

import numpy as np
from typing import Any, Dict, List

import bentoml
from bentoml import Service
from bentoml.artifacts import PickleArtifact
from bentoml.io import JSON as _JSON  # imported only for type hints in the client
from bentoml import runners

# ────────────────────────────────────────────────────────────
# 1) Define the Service with an artifact for the sklearn model
# ────────────────────────────────────────────────────────────

@bentoml.service(
    name="prediction_service",
    runners=[],  # no runners here, we will load the model directly
)
@bentoml.artifacts([PickleArtifact("model")])
class PredictionService(Service):
    """BentoML Service for housing price prediction."""

    @bentoml.api(
        input_spec=_JSON,    # accept arbitrary JSON payloads
        output_spec=_JSON,   # return arbitrary JSON payloads
        route="/predict"     # exposes POST /predict
    )
    async def predict(
        self,
        parsed_json: Dict[str, Any],  # dict parsed from JSON
    ) -> Dict[str, Any]:
        """
        Expects:
          {
            "dataframe_records": [
              { <feature>: <value>, … },
              …
            ]
          }
        Returns:
          {
            "predictions": [ <float>, … ]
          }
        """
        # 1. Extract the feature dicts and convert to NumPy array
        records: List[Dict[str, Any]] = parsed_json["dataframe_records"]
        data = np.array([list(rec.values()) for rec in records])

        # 2. Run the model (loaded from the artifact) for prediction
        model = self.artifacts.model  # PickleArtifact bound at build time
        preds = model.predict(data)

        # 3. Return JSON-serializable dict
        return {"predictions": preds.tolist()}

# ────────────────────────────────────────────────────────────
# 2) Entry point to load+serve the service with BentoML CLI
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # This lets you run `bentoml serve service.py:PredictionService`
    bentoml.serve(PredictionService, production=False)
