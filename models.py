from enum import Enum, auto
from typing import List, Dict
import random
import pdb
from collections import Counter

class ModelCategory(Enum):
    TEXT_ONLY = "TEXT_ONLY"
    VISION_CAPABLE = "VISION_CAPABLE"
    RAG_CAPABLE = "RAG_CAPABLE"

class Models:
    MODELS: Dict[ModelCategory, List[str]] = {
        ModelCategory.TEXT_ONLY: [
            "openai/o3-mini",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        ModelCategory.VISION_CAPABLE: [
            "openai/o1",
            "openai/gpt-4o-2024-11-20",
            "google/gemini-2.0-flash-thinking-exp:free",
            "google/gemini-2.0-flash-001",
            "anthropic/claude-3.5-sonnet:beta",
            "meta-llama/llama-3.2-90b-vision-instruct:free",
        ],
        ModelCategory.RAG_CAPABLE: [
            "perplexity/llama-3.1-sonar-large-128k-online",
            "google/gemini-2.0-flash-001"
        ]
    }

    ALL_MODELS: List[str] = list(set(
        MODELS[ModelCategory.TEXT_ONLY] + 
        MODELS[ModelCategory.VISION_CAPABLE] + 
        MODELS[ModelCategory.RAG_CAPABLE]
    ))
    ALL_NON_RAG_MODELS: List[str] = list(set(
        MODELS[ModelCategory.TEXT_ONLY] + 
        MODELS[ModelCategory.VISION_CAPABLE]
    ))

    # Initialize all models with default weight of 1.0
    MODEL_WEIGHTS: Dict[str, float] = {model: 1.0 for model in ALL_MODELS}
    
    # Override specific models with 2.0 weight (2x more likely to be chosen)
    MODEL_WEIGHTS.update({
        "perplexity/llama-3.1-sonar-large-128k-online": 10,
        "openai/o3-mini": 10,
        "openai/gpt-4o-2024-11-20": 10,
        "google/gemini-2.0-flash-thinking-exp:free": 10,
        "anthropic/claude-3.5-sonnet:beta": 0.1,
    })

    USES_GEMINI_GROUNDING = [
        "google/gemini-2.0-flash-001",
    ]


    @staticmethod
    def get_presentable_model_name(model_id: str) -> str:
        """Convert model ID to a more human-readable format."""
        return Models.MODEL_DISPLAY_NAMES.get(model_id, model_id)

    @staticmethod
    def model_uses_rag(model_id: str) -> bool:
        return model_id in Models.MODELS[ModelCategory.RAG_CAPABLE]
    
    @staticmethod
    def select_models(requires_vision: bool = False):
        """Select two random models based on current settings and weights."""
        # models_use_rag = False
        models_support_vision = False

        available_models = (
            Models.MODELS[ModelCategory.VISION_CAPABLE] 
            if requires_vision 
            else Models.ALL_MODELS
        )
        
        if len(available_models) < 2:
            raise Exception("Not enough models available that support this functionality")
        
        # Create weighted list for random selection
        weighted_models = []
        for model in available_models:
            weight = Models.MODEL_WEIGHTS.get(model, 1.0)  # Default to 1.0 if not specified
            weighted_models.extend([model] * int(weight * 100))  # Multiply by 100 for better precision
            
        # Select two unique models
        selected_models = []
        while len(selected_models) < 2:
            model = random.choice(weighted_models)
            if model not in selected_models:
                selected_models.append(model)
                
        models_support_vision = True \
            if selected_models[0] in Models.MODELS[ModelCategory.VISION_CAPABLE] \
                and selected_models[1] in Models.MODELS[ModelCategory.VISION_CAPABLE] \
                    else False
        return selected_models[0], selected_models[1], models_support_vision
